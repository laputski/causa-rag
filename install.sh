#!/usr/bin/env bash
#
# Causa RAG — one command from a fresh clone to a working platform.
#
#   git clone https://github.com/laputski/causa-rag && cd causa-rag && ./install.sh
#
# What this does differently from the bootstrap it replaces:
#
#   * Reports every missing prerequisite in one pass instead of failing on the
#     first, so a clean machine needs one round of installs rather than five.
#   * Downloads models in the foreground with visible progress. The old script
#     polled /health in silence while 2.3 GB of embedder weights downloaded,
#     which reads as a hang.
#   * Asserts what the gateway actually built. It falls back to a stub
#     generator, a stub retriever and stub embedding vectors without saying so,
#     and the old script printed success over all three. A platform whose
#     purpose is catching silent degradation must not ship it.
#
# Safe to re-run: every step checks whether its work is already done.
#
# Bash 3.2 compatible, because that is what macOS ships. No associative
# arrays, no ${x^^}.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── options ───────────────────────────────────────────────────────────────────

PORT="${PORT:-8081}"
WITH_DEMO=1
WITH_UI=1
WITH_JUDGE=0
SKIP_MODELS=0
ASSUME_YES=0
DO_STOP=0
CHECK_ONLY=0

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --no-demo        do not create the demo realm
  --no-ui          do not install or start the UI
  --with-judge     also install the three LLM judges and pull the judge model
                   (only `make test-eval` and the deep diagnostics need them)
  --skip-models    assume models and weights are already present (air-gap)
  --port=N         run the gateway on N instead of 8081
  --yes            do not prompt
  --stop           stop the gateway and UI this script started, then exit
  --check          run the preflight and stop, changing nothing
EOF
}

for arg in "$@"; do
  case "$arg" in
    --no-demo) WITH_DEMO=0 ;;
    --no-ui) WITH_UI=0 ;;
    --with-judge) WITH_JUDGE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --stop) DO_STOP=1 ;;
    --check) CHECK_ONLY=1 ;;
    --port=*) PORT="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

RUN_DIR=".run"
API_LOG="$RUN_DIR/api.log"
UI_LOG="$RUN_DIR/ui.log"
API_PID="$RUN_DIR/api.pid"
UI_PID="$RUN_DIR/ui.pid"
COMPOSE_FILE="deploy/compose/docker-compose.yml"

if [ -t 1 ]; then
  G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[90m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=""; Y=""; R=""; D=""; B=""; N=""
fi

ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$1"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$R" "$N" "$1"; }
info() { printf '  %s·%s %s\n' "$D" "$N" "$1"; }
step() { printf '\n%s%s%s\n' "$B" "$1" "$N"; }

# ── --stop ────────────────────────────────────────────────────────────────────

stop_one() {
  pidfile="$1"; label="$2"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      ok "stopped $label (pid $pid)"
    else
      info "$label was not running"
    fi
    rm -f "$pidfile"
  else
    info "$label was not started by this script"
  fi
}

if [ "$DO_STOP" -eq 1 ]; then
  step "Stopping"
  stop_one "$API_PID" "gateway"
  stop_one "$UI_PID" "ui"
  info "infrastructure left running; stop it with: docker compose -f $COMPOSE_FILE down"
  exit 0
fi

# ── 0. preflight ──────────────────────────────────────────────────────────────
# Everything is checked before anything is installed, and every failure names
# the fix for this platform. A reader who is missing three things should learn
# all three now rather than one per attempt.

step "Preflight"
PROBLEMS=0

fail_with() { bad "$1"; printf '      → %s\n' "$2"; PROBLEMS=$((PROBLEMS + 1)); }

case "$(uname -s)" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *)      PLATFORM="other" ;;
esac

if command -v python3 >/dev/null 2>&1; then
  PY_VER=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)'; then
    ok "python3 $PY_VER"
  else
    if [ "$PLATFORM" = "macos" ]; then hint="brew install python@3.12"; else hint="sudo apt install python3.12 python3.12-venv"; fi
    fail_with "python3 $PY_VER is too old, 3.12+ required" "$hint"
  fi
else
  fail_with "python3 not found" "https://www.python.org/downloads/"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '')"
  else
    if [ "$PLATFORM" = "macos" ]; then hint="Start Docker Desktop"; else hint="sudo systemctl start docker"; fi
    fail_with "docker is installed but the daemon is not running" "$hint"
  fi
  docker compose version >/dev/null 2>&1 || \
    fail_with "\`docker compose\` (v2) not available" "Install the Compose v2 plugin"
else
  fail_with "docker not found" "https://docs.docker.com/get-docker/"
fi

if [ "$WITH_UI" -eq 1 ]; then
  if command -v node >/dev/null 2>&1; then
    ok "node $(node --version)"
  else
    fail_with "node not found (needed for the UI)" "https://nodejs.org/  ·  or re-run with --no-ui"
  fi
fi

# Ollama is a prerequisite rather than a service this repository starts. The
# compose service that used to provide it published 11434 unconditionally, so
# it broke `docker compose up` on any machine already running Ollama, and it
# had no GPU access on macOS by construction.
if curl -sf -m 3 "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  ok "ollama answering on :11434"
else
  if [ "$PLATFORM" = "macos" ]; then hint="brew install ollama && ollama serve"; else hint="curl -fsSL https://ollama.com/install.sh | sh"; fi
  fail_with "ollama not answering on :11434" "$hint"
fi

FREE_GB=$(df -Pg . 2>/dev/null | awk 'NR==2 {print $4}' || echo "?")
if [ "$FREE_GB" != "?" ] && [ "$FREE_GB" -lt 25 ] 2>/dev/null; then
  warn "only ${FREE_GB}G free; models and indices need roughly 25G"
else
  ok "disk ${FREE_GB}G free"
fi

# Ports. 11434 is deliberately absent: Ollama holding it is the expected state.
BUSY=""
for p in 6333 6379 9200 5601 27017 3001 "$PORT" 5173; do
  # `|| true` is load-bearing under `set -euo pipefail`: lsof exits non-zero
  # when nothing holds the port, pipefail passes that through the pipe, and the
  # assignment inherits it, so `set -e` killed the script silently. It only
  # happened when a port was FREE, which is the state a first install is in and
  # the one no developer machine is ever in.
  owner=$(lsof -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1" (pid "$2")"}' || true)
  if [ -n "$owner" ]; then BUSY="$BUSY $p:$owner"; fi
done
if [ -n "$BUSY" ]; then
  info "already listening:$BUSY"
  info "re-running is fine; those are reused rather than replaced"
else
  ok "ports free"
fi

if [ "$PROBLEMS" -gt 0 ]; then
  printf '\n%sInstall the %d item(s) above, then re-run ./install.sh%s\n\n' "$R" "$PROBLEMS" "$N"
  exit 1
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  printf '\n%sPreflight passed. Run ./install.sh to install.%s\n\n' "$G" "$N"
  exit 0
fi

# ── 1. configuration ──────────────────────────────────────────────────────────

step "Configuration"
mkdir -p "$RUN_DIR"
if [ -f .env ]; then
  ok ".env exists, left untouched"
else
  cp .env.example .env
  ok ".env created from .env.example"
fi
set -a; . ./.env; set +a
export PORT

# ── 2. python ─────────────────────────────────────────────────────────────────

step "Dependencies"
if [ -d .venv ]; then
  info "reusing .venv"
else
  python3 -m venv .venv
  ok "created .venv"
fi
# Resolving deepeval, ragas and trulens together takes minutes on a cold run.
# A stamp of what was installed last lets a re-run skip it entirely, and the
# install itself is not silenced: a multi-minute step with no output is the
# same "looks frozen" problem this script exists to remove.
DEPS_STAMP="$RUN_DIR/deps.sha"
WANT_STAMP=$(shasum pyproject.toml | awk '{print $1}')
if [ -f "$DEPS_STAMP" ] && [ "$(cat "$DEPS_STAMP")" = "$WANT_STAMP" ]; then
  info "python packages already match pyproject.toml"
else
  info "installing python packages (a few minutes on a first run)"
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e ".[dev,integration,report]"
  echo "$WANT_STAMP" > "$DEPS_STAMP"
  ok "python packages installed"
fi

# The three LLM judges live in their own extra because resolving them together
# is the slow half of a first install. Nothing on the default path imports them,
# so leaving them out costs a clear error from two endpoints and nothing else.
if [ "$WITH_JUDGE" -eq 1 ]; then
  info "installing the LLM judges (this is the slow part, ten minutes or more)"
  .venv/bin/pip install -e ".[judges]"
  ok "judges installed"
else
  info "judges skipped; \`make test-eval\` and the deep diagnostics need them (--with-judge)"
fi

if [ "$WITH_UI" -eq 1 ]; then
  if [ -d ui/node_modules ] && [ ui/node_modules -nt ui/package-lock.json ]; then
    info "reusing ui/node_modules"
  else
    if [ -f ui/package-lock.json ]; then (cd ui && npm ci --silent); else (cd ui && npm install --silent); fi
    ok "ui packages installed"
  fi
fi

# ── 3. infrastructure ─────────────────────────────────────────────────────────

step "Infrastructure"
docker compose -f "$COMPOSE_FILE" up -d >/dev/null
.venv/bin/python -m tools.doctor --wait --services

# ── 4. models ─────────────────────────────────────────────────────────────────

step "Models"
if [ "$SKIP_MODELS" -eq 1 ]; then
  info "skipped (--skip-models)"
else
  GEN_MODEL="${OLLAMA_MODEL:-qwen3:8b}"
  if ollama list 2>/dev/null | grep -q "^${GEN_MODEL%%:*}"; then
    ok "generator $GEN_MODEL already pulled"
  else
    info "pulling generator $GEN_MODEL, this takes a few minutes"
    ollama pull "$GEN_MODEL"
    ok "generator $GEN_MODEL"
  fi

  if [ "$WITH_JUDGE" -eq 1 ]; then
    JUDGE_MODEL=$(.venv/bin/python -c 'from eval.judge_model import JUDGE_MODEL; print(JUDGE_MODEL)')
    if ollama list 2>/dev/null | grep -q "^${JUDGE_MODEL%%:*}"; then
      ok "judge $JUDGE_MODEL already pulled"
    else
      info "pulling judge $JUDGE_MODEL"
      ollama pull "$JUDGE_MODEL"
      ok "judge $JUDGE_MODEL"
    fi
  else
    info "judge model skipped, matching the judges themselves"
  fi

  # Warmed with the same call adapters/bge_m3.py makes later, so the cache it
  # fills is exactly the one the gateway reads. Progress bars stay visible:
  # a silent 2.3 GB download is why the old bootstrap looked frozen.
  if .venv/bin/python -c "
from huggingface_hub import try_to_load_from_cache
import sys
sys.exit(0 if isinstance(try_to_load_from_cache('BAAI/bge-m3', 'config.json'), str) else 1)
" 2>/dev/null; then
    ok "embedder weights already cached"
  else
    info "downloading embedder weights (BAAI/bge-m3, ~2.3 GB, once)"
    .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" >/dev/null
    ok "embedder weights ready"
  fi
fi

# ── 5. gateway ────────────────────────────────────────────────────────────────

step "Gateway"

# Two separate questions: is something answering on the port, and did this
# script start it. A gateway started by hand (`make api`) answers but has no
# pidfile here, and starting a second one would just fail on the bind. Adopt
# it instead, then let the degradation assertions below judge it on the same
# terms as one we started ourselves.
STARTED_BY_US=0
if [ -f "$API_PID" ] && kill -0 "$(cat "$API_PID")" 2>/dev/null; then STARTED_BY_US=1; fi

if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  if [ "$STARTED_BY_US" -eq 1 ]; then
    info "already running (pid $(cat "$API_PID"))"
  else
    info "a gateway is already answering on :$PORT, started outside this script"
    info "adopting it; stop it yourself if you wanted a fresh one"
  fi
else
  USE_REAL_BGE_M3=true .venv/bin/uvicorn services.api_gateway.main:app \
    --host 0.0.0.0 --port "$PORT" > "$API_LOG" 2>&1 &
  echo $! > "$API_PID"
  printf '  '
  for _ in $(seq 1 90); do
    if curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    printf '.'; sleep 2
  done
  printf '\n'
  if ! curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    bad "gateway did not come up within three minutes"
    printf '      → last 30 lines of %s:\n\n' "$API_LOG"
    tail -30 "$API_LOG" | sed 's/^/      /'
    exit 1
  fi
  ok "started on :$PORT (pid $(cat "$API_PID"))"
fi

# The assertion the old bootstrap lacked. A gateway that fell back to stubs
# answers /health exactly like a working one.
HEALTH=$(curl -s "http://localhost:$PORT/health")
SETTINGS=$(curl -s "http://localhost:$PORT/settings")
DEGRADED=0
case "$HEALTH" in
  *ollama*) ok "generator: ollama" ;;
  *) bad "generator fell back to a stub, so answers are placeholder text"; DEGRADED=1 ;;
esac
case "$HEALTH" in
  *qdrant_dense*) ok "retriever: qdrant" ;;
  *) bad "retriever fell back to a stub, so retrieval returns nothing real"; DEGRADED=1 ;;
esac
case "$SETTINGS" in
  *'"embedder_mode":"real"'*) ok "embedder: real BGE-M3" ;;
  *) bad "embedder is serving stub vectors, so every similarity number is noise"; DEGRADED=1 ;;
esac

if [ "$DEGRADED" -eq 1 ]; then
  printf '\n%sThe gateway started but is degraded. Numbers it produces would be meaningless.%s\n' "$R" "$N"
  printf '  Logs: %s\n  Retry: ./install.sh\n\n' "$API_LOG"
  exit 1
fi

# ── 6. demo ───────────────────────────────────────────────────────────────────

if [ "$WITH_DEMO" -eq 1 ]; then
  step "Demo realm"
  USE_REAL_BGE_M3=true .venv/bin/python -m tools.seed_demo 2>&1 | grep -v '^20' || true
else
  info "demo skipped (--no-demo)"
fi

# ── 7. ui ─────────────────────────────────────────────────────────────────────

if [ "$WITH_UI" -eq 1 ]; then
  step "UI"
  if [ -f "$UI_PID" ] && kill -0 "$(cat "$UI_PID")" 2>/dev/null; then
    info "already running (pid $(cat "$UI_PID"))"
  else
    (cd ui && npm run dev > "../$UI_LOG" 2>&1 & echo $! > "../$UI_PID")
    for _ in $(seq 1 30); do
      curl -sf -m 2 "http://localhost:5173" >/dev/null 2>&1 && break
      sleep 1
    done
    ok "started on :5173"
  fi
fi

# ── 8. verify and report ──────────────────────────────────────────────────────

step "Check"
.venv/bin/python -m tools.doctor || true

open_url() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1"
  elif command -v wslview >/dev/null 2>&1; then wslview "$1"
  else printf '  open %s\n' "$1"; fi
}

cat <<EOF

${B}Ready.${N}

  UI          http://localhost:5173
  API docs    http://localhost:$PORT/docs
  Qdrant      http://localhost:6333/dashboard
  Langfuse    http://localhost:3001

EOF

if [ "$WITH_DEMO" -eq 1 ]; then
  cat <<'EOF'
  Open the UI, pick the Demo realm and press "New run". It scores 15 questions
  over an eight-document handbook, which takes about a minute.

EOF
fi

cat <<EOF
  Stop        ./install.sh --stop
  Check       make doctor
  Logs        $API_LOG  $UI_LOG

EOF

if [ "$WITH_UI" -eq 1 ] && [ "$ASSUME_YES" -eq 0 ]; then
  open_url "http://localhost:5173" 2>/dev/null || true
fi
