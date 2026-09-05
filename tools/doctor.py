"""Check that the platform is actually working, and say what to do when it is not.

Three callers share one implementation. `make doctor` prints the report,
`make infra` waits on the services (`--wait --services`), and `install.sh` does
both. Keeping them together means the health rules cannot drift between the
thing that waits and the thing that reports.

The rows that matter most are the ones nothing checked before. The gateway
degrades silently in two places: `services/api_gateway/main.py#_build_generator`
falls back to `GeneratorStub` when Ollama is unreachable, and the retriever
falls back to `QdrantRetrieverStub`. Both fallbacks let the platform start,
answer, and produce numbers that mean nothing. A green `/health` said nothing
about either, so the old bootstrap could print success over a gateway serving
hash vectors. Those two rows now read the component names and fail loudly.

Exit codes: 0 everything passed, 1 something failed, 2 warnings only.

    python3 -m tools.doctor                  # the report
    python3 -m tools.doctor --json           # machine-readable
    python3 -m tools.doctor --wait --services  # block until the stack is up
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
LANGFUSE_URL = os.getenv("LANGFUSE_EXTERNAL_HOST", "http://localhost:3001")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
GATEWAY_URL = f"http://localhost:{os.getenv('PORT', '8081')}"
UI_URL = "http://localhost:5173"

OK, WARN, FAIL, OFF = "ok", "warn", "fail", "off"

_SYMBOL = {OK: "✓", WARN: "!", FAIL: "✗", OFF: "–"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", OFF: "\033[90m"}
_RESET = "\033[0m"


@dataclass
class Check:
    component: str
    status: str
    detail: str = ""
    fix: str = ""
    indent: int = 0
    children: list[Check] = field(default_factory=list)


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _http(url: str, timeout: float = 4.0) -> tuple[bool, Any]:
    try:
        import httpx
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        try:
            return True, resp.json()
        except Exception:
            return True, resp.text
    except Exception as exc:
        return False, str(exc)


# ── individual checks ────────────────────────────────────────────────────────

def check_docker() -> Check:
    if shutil.which("docker") is None:
        return Check("docker", FAIL, "not installed",
                     "https://docs.docker.com/get-docker/")
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=15)
    except Exception:
        return Check("docker", FAIL, "installed, daemon not running",
                     "Start Docker Desktop, or: sudo systemctl start docker")
    try:
        out = subprocess.run(["docker", "compose", "version", "--short"],
                             capture_output=True, text=True, timeout=15)
        version = out.stdout.strip() or "unknown"
    except Exception:
        return Check("docker", WARN, "daemon up, `docker compose` unavailable",
                     "Install the Compose v2 plugin")
    return Check("docker", OK, f"daemon up, compose {version}")


def check_qdrant() -> Check:
    ok, body = _http(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections")
    if not ok:
        return Check(f"qdrant :{QDRANT_PORT}", FAIL, "not answering", "make infra")
    n = len((body or {}).get("result", {}).get("collections", []))
    return Check(f"qdrant :{QDRANT_PORT}", OK, f"{n} collections")


def check_opensearch() -> Check:
    ok, body = _http(f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/_cluster/health")
    if not ok:
        return Check(f"opensearch :{OPENSEARCH_PORT}", FAIL, "not answering", "make infra")
    status = (body or {}).get("status", "unknown")
    return Check(f"opensearch :{OPENSEARCH_PORT}", OK if status in ("green", "yellow") else WARN,
                 f"cluster {status}")


def check_mongodb() -> Check:
    host = MONGODB_URL.split("//")[-1].split("/")[0]
    hostname, _, port = host.partition(":")
    if not _port_open(hostname or "localhost", int(port or 27017)):
        return Check("mongodb :27017", FAIL, "not answering", "make infra")
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
        names = client[os.getenv("MONGODB_DB", "ragplatform")].list_collection_names()
        return Check("mongodb :27017", OK, f"{len(names)} collections")
    except Exception:
        return Check("mongodb :27017", OK, "port open")


def check_redis() -> Check:
    host = REDIS_URL.split("//")[-1].split("/")[0]
    hostname, _, port = host.partition(":")
    up = _port_open(hostname or "localhost", int(port or 6379))
    return Check("redis :6379", OK if up else FAIL, "port open" if up else "not answering",
                 "" if up else "make infra")


def check_langfuse() -> Check:
    ok, _ = _http(f"{LANGFUSE_URL}/api/public/health")
    return Check("langfuse :3001", OK if ok else WARN,
                 "answering" if ok else "not answering (tracing only, optional)",
                 "" if ok else "make infra")


def check_neo4j() -> Check:
    host, _, port = NEO4J_URI.split("//")[-1].partition(":")
    if _port_open(host or "localhost", int(port or 7687), timeout=1.5):
        return Check("neo4j :7687", OK, "answering")
    return Check("neo4j :7687", OFF, "profile `graph` not started (optional)",
                 "docker compose --profile graph up -d neo4j")


def check_ollama() -> Check:
    """Ollama is a host prerequisite, so a miss here is a real failure."""
    ok, body = _http(f"{OLLAMA_BASE_URL}/api/tags")
    if not ok:
        return Check("ollama :11434", FAIL, "not answering",
                     "macOS: brew install ollama && ollama serve  ·  "
                     "Linux: curl -fsSL https://ollama.com/install.sh | sh")
    models = {m["name"] for m in (body or {}).get("models", [])}
    root = Check("ollama :11434", OK, f"host install, {len(models)} models")

    def _model_row(name: str, label: str, required: bool) -> Check:
        present = name in models or any(m.split(":")[0] == name.split(":")[0] for m in models)
        if present:
            return Check(name, OK, label, indent=1)
        return Check(name, FAIL if required else WARN, f"{label}, not pulled",
                     f"ollama pull {name}", indent=1)

    root.children.append(_model_row(OLLAMA_MODEL, "generator", required=True))
    try:
        from eval.judge_model import JUDGE_MODEL
        root.children.append(_model_row(JUDGE_MODEL, "judge, only for `make test-eval`", required=False))
    except Exception:
        pass
    if any(c.status == FAIL for c in root.children):
        root.status = WARN
    return root


def check_embedder_weights() -> Check:
    """A 2.3 GB download that starts at the worst possible moment is the
    single most common reason a first run looks like a hang."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return Check("bge-m3 weights", WARN, "huggingface_hub not installed",
                     'pip install -e ".[integration]"')
    hit = try_to_load_from_cache("BAAI/bge-m3", "config.json")
    if isinstance(hit, str):
        return Check("bge-m3 weights", OK, "cached")
    return Check("bge-m3 weights", WARN, "not cached, first run will download ~2.3 GB",
                 "./install.sh  (prefetches it with visible progress)")


def check_gateway() -> Check:
    """The two silent-fallback rows: a stub generator or retriever means every
    number the platform produces is meaningless, and nothing else says so."""
    ok, body = _http(f"{GATEWAY_URL}/health")
    if not ok:
        return Check("gateway :8081", FAIL, "not answering", "make api  (or ./install.sh)")

    components = (body or {}).get("components", {}) or {}
    root = Check("gateway :8081", OK, "answering")

    def _flat(kind: str) -> str:
        value = components.get(kind)
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value or "")

    generator = _flat("generator")
    if "ollama" in generator:
        root.children.append(Check("generator", OK, generator, indent=1))
    else:
        root.children.append(Check(
            "generator", FAIL, f"{generator or 'none'} (stub: answers are placeholder text)",
            "Start Ollama, then restart the gateway", indent=1))

    retriever = _flat("retriever")
    if "qdrant" in retriever and "stub" not in retriever:
        root.children.append(Check("retriever", OK, retriever, indent=1))
    else:
        root.children.append(Check(
            "retriever", FAIL, f"{retriever or 'none'} (stub: retrieval returns nothing real)",
            "Start Qdrant (make infra), then restart the gateway", indent=1))

    # Asked of the gateway rather than read from this process's environment.
    # The doctor may run in a different shell than the one that started the
    # gateway, and the answer that matters is the gateway's own: `/settings`
    # derives embedder_mode from USE_REAL_BGE_M3 as that process saw it.
    settings_ok, settings = _http(f"{GATEWAY_URL}/settings")
    mode = (settings or {}).get("embedder_mode") if settings_ok else None
    if mode == "real":
        root.children.append(Check("embedder", OK, "real BGE-M3", indent=1))
    elif mode == "stub":
        root.children.append(Check(
            "embedder", FAIL, "stub vectors, so every similarity number is noise",
            "Restart the gateway with USE_REAL_BGE_M3=true", indent=1))
    else:
        root.children.append(Check(
            "embedder", WARN, "mode not reported", indent=1))

    pipelines = _flat("pipeline")
    root.children.append(Check("pipelines", OK if pipelines else WARN, pipelines or "none", indent=1))

    if any(c.status == FAIL for c in root.children):
        root.status = FAIL
    return root


def check_realms() -> Check:
    ok, body = _http(f"{GATEWAY_URL}/realms")
    if not ok:
        return Check("realms", WARN, "gateway not answering")
    realms = body or []
    if not realms:
        return Check("realms", WARN, "none yet", "make demo  (creates the demo realm)")

    ids = sorted(str(r.get("id", "?")) for r in realms)
    root = Check("realms", OK, f"{len(ids)}: " + ", ".join(ids[:4]))

    # Which realm the corpus and dataset checks below are about. It used to be
    # `realms[0]`, that is whichever the database happened to return first, so
    # a second realm silently changed the subject of the installation check and
    # nothing in the output said which realm it had looked at. Sorting makes
    # the choice repeatable and naming it makes the answer readable.
    subject = _subject_realm(realms)

    ok, corpora = _http(f"{GATEWAY_URL}/corpus/collections?realm_id={subject}")
    root.children.append(Check(
        f"corpus ({subject})", OK if ok and corpora else WARN,
        f"{len(corpora)} registered" if ok and corpora else "none registered",
        "" if ok and corpora else _seed_hint(subject), indent=1))
    ok, datasets = _http(f"{GATEWAY_URL}/datasets?realm_id={subject}")
    root.children.append(Check(
        f"dataset ({subject})", OK if ok and datasets else WARN,
        f"{len(datasets)} available" if ok and datasets else "none",
        "" if ok and datasets else _seed_hint(subject), indent=1))
    return root


# A realm built to be broken on purpose is a poor subject for a check that asks
# whether the installation works. Preferred against, not excluded: on a machine
# that holds nothing else it is still better to report on it than on nothing.
#
# Read from the realm's own `purpose` and no longer from a list of identifiers
# kept here. The list was a second place to remember, and a realm named anything
# else would have been checked as if it were healthy.
_FAULTY_ON_PURPOSE = "proving_ground"


def _subject_realm(realms: list[dict]) -> str:
    """Sorts here instead of trusting the caller to have sorted.

    The first version left the ordering to `check_realms` and was itself still
    order-dependent, so the guarantee held only as long as every call site
    remembered. A function whose contract is "repeatable" has to be repeatable
    on its own arguments.
    """
    ordered = sorted(realms, key=lambda r: str(r.get("id", "")))
    for realm in ordered:
        if realm.get("purpose") != _FAULTY_ON_PURPOSE:
            return str(realm.get("id", "?"))
    return str(ordered[0].get("id", "?"))


def _seed_hint(realm_id: str) -> str:
    """What to run to fill this realm, and not what fills the demo one.

    The advice was the literal `make demo` whatever realm had been checked,
    which becomes wrong the moment the checked realm is not the demo.
    """
    if realm_id == "proving-ground":
        return "make proving-ground"
    if realm_id == "demo":
        return "make demo"
    return f"load a corpus and a question set into the {realm_id!r} realm"


def check_ui() -> Check:
    ok, _ = _http(UI_URL, timeout=2)
    return Check("ui :5173", OK if ok else OFF,
                 "vite dev server" if ok else "not running",
                 "" if ok else "make ui")


def check_python_env() -> Check:
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info < (3, 12):  # noqa: UP036 — this tool exists to tell a user their interpreter is too old
        return Check("python env", FAIL, f"{version}, need 3.12+",
                     "macOS: brew install python@3.12 · Ubuntu: apt install python3.12")
    in_venv = sys.prefix != sys.base_prefix
    missing = []
    for module in ("fastapi", "qdrant_client", "motor", "sentence_transformers"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        return Check("python env", WARN, f"{version}, missing: {', '.join(missing)}",
                     'pip install -e ".[dev,integration,report]"')
    return Check("python env", OK, f"{version}{'' if in_venv else ', not in a venv'}")


SERVICE_CHECKS = (check_docker, check_qdrant, check_opensearch, check_mongodb,
                  check_redis, check_langfuse, check_neo4j)
ALL_CHECKS = SERVICE_CHECKS + (check_ollama, check_embedder_weights, check_gateway,
                               check_realms, check_ui, check_python_env)


# ── rendering ────────────────────────────────────────────────────────────────

def _flatten(checks: list[Check]) -> list[Check]:
    out: list[Check] = []
    for c in checks:
        out.append(c)
        out.extend(_flatten(c.children))
    return out


def render(checks: list[Check], colour: bool = True) -> str:
    rows = _flatten(checks)
    width = max((len(c.component) + c.indent * 2) for c in rows) + 2
    lines = ["", "  " + "─" * (width + 46)]
    for c in rows:
        symbol = _SYMBOL[c.status]
        if colour:
            symbol = f"{_COLOR[c.status]}{symbol}{_RESET}"
        name = ("  " * c.indent) + ("· " if c.indent else "") + c.component
        lines.append(f"  {symbol} {name:<{width}} {c.detail}")
        if c.fix and c.status in (FAIL, WARN):
            lines.append(f"      {'':<{width}} → {c.fix}")
    lines.append("  " + "─" * (width + 46))

    tally = {s: sum(1 for c in rows if c.status == s) for s in (OK, WARN, FAIL, OFF)}
    lines.append(
        f"  {tally[OK]} ok · {tally[WARN]} warning · {tally[FAIL]} failed · {tally[OFF]} off"
    )
    return "\n".join(lines)


def exit_code(checks: list[Check]) -> int:
    rows = _flatten(checks)
    if any(c.status == FAIL for c in rows):
        return 1
    if any(c.status == WARN for c in rows):
        return 2
    return 0


# ── waiting ──────────────────────────────────────────────────────────────────

_WAIT_TIMEOUTS = {
    "qdrant": 60, "redis": 60, "mongodb": 90,
    "opensearch": 180, "langfuse": 180, "docker": 30, "neo4j": 30,
}


def wait_for_services(quiet: bool = False) -> int:
    """Block until each service answers, with its own timeout.

    Replaces the old Qdrant-only poll, which reported the whole stack ready
    while OpenSearch was still forty seconds from accepting a connection.
    """
    deadline_start = time.monotonic()
    for check in SERVICE_CHECKS:
        name = check.__name__.removeprefix("check_")
        timeout = _WAIT_TIMEOUTS.get(name, 60)
        started = time.monotonic()
        while True:
            result = check()
            if result.status in (OK, OFF):
                if not quiet:
                    waited = time.monotonic() - started
                    print(f"  ✓ {result.component:<22} {result.detail} ({waited:.0f}s)")
                break
            if time.monotonic() - started > timeout:
                print(f"  ✗ {result.component:<22} {result.detail} (gave up after {timeout}s)")
                if result.fix:
                    print(f"      → {result.fix}")
                _print_logs(name)
                return 1
            time.sleep(2)
    if not quiet:
        print(f"  infrastructure ready in {time.monotonic() - deadline_start:.0f}s")
    return 0


def _print_logs(service: str, lines: int = 40) -> None:
    """Print the service's own logs instead of leaving the reader to find them."""
    compose = ["docker", "compose", "-f", "deploy/compose/docker-compose.yml",
               "logs", f"--tail={lines}", service]
    try:
        out = subprocess.run(compose, capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            print(f"\n  last {lines} lines from `{service}`:")
            for line in out.stdout.strip().split("\n")[-lines:]:
                print(f"    {line}")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--wait", action="store_true", help="block until checks pass")
    parser.add_argument("--services", action="store_true", help="only the backing services")
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args(argv)

    if args.wait:
        return wait_for_services(quiet=args.json)

    checks = [c() for c in (SERVICE_CHECKS if args.services else ALL_CHECKS)]

    if args.json:
        print(json.dumps([asdict(c) for c in checks], ensure_ascii=False, indent=2))
    else:
        print(render(checks, colour=not args.no_colour and sys.stdout.isatty()))

    return exit_code(checks)


if __name__ == "__main__":
    raise SystemExit(main())
