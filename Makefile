.PHONY: help types quickstart doctor stop demo up down infra api ui ingest install install-judges test test-unit test-int test-eval test-e2e load logs ps clean bootstrap publish-check publish-export miracl-fetch miracl-ingest miracl-report
COMPOSE = docker compose -f deploy/compose/docker-compose.yml
PORT    ?= 8081

help:           ## Show the list of commands
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z0-9_-]+:.*##/{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

# ── Infrastructure ─────────────────────────────────────────────────────────────

up: infra       ## Bring up infrastructure + API + UI (the full dev stack)
	$(MAKE) -j2 api ui

infra:          ## Bring up the Docker services (Qdrant, OpenSearch, MongoDB, Redis, Langfuse)
	$(COMPOSE) up -d
	@python3 -m tools.doctor --wait --services

down:           ## Stop every Docker container
	$(COMPOSE) down

ps:             ## Container status
	$(COMPOSE) ps

logs:           ## Logs from every container (Ctrl-C to leave)
	$(COMPOSE) logs -f

# ── Python services ────────────────────────────────────────────────────────────

api:            ## Run the API Gateway on port 8081 (with the real BGE-M3)
	USE_REAL_BGE_M3=true uvicorn services.api_gateway.main:app --host 0.0.0.0 --port $(PORT) --reload

# ── UI ─────────────────────────────────────────────────────────────────────────

ui:             ## Run the UI (Vite dev server, port 5173)
	cd ui && npm run dev

ui-install:     ## Install the UI's npm dependencies
	cd ui && npm install

ui-build:       ## Build the UI into static assets
	cd ui && npm run build

# ── Data ───────────────────────────────────────────────────────────────────────

demo:           ## Seed the demo realm: corpus, dataset, prompt, preset
	USE_REAL_BGE_M3=true python3 -m tools.seed_demo

ingest:         ## Ingest the demo corpus (with the real BGE-M3)
	USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest corpus/demo_handbook/ \
	  --strategy structure_aware --corpus-id handbook --realm-id demo

migrate-db:     ## Move flat files into MongoDB (run once)
	python3 -m tools.migrate_to_mongodb

ollama-pull:    ## Pull the LLM into Ollama
	ollama pull qwen3:8b

# ── Tests ──────────────────────────────────────────────────────────────────────

types:          ## Type errors against the pinned per-file budget
	python3 -m tools.check_types

test:           ## Run the unit and contract tests
	python3 -m pytest tests/unit/ tests/contract/ -q

test-unit:      ## Unit tests only
	python3 -m pytest tests/unit/ -q

test-int:       ## Integration tests (needs docker compose up)
	python3 -m pytest tests/integration/ -m integration -v

openapi:        ## Regenerate the governance/openapi.json snapshot (the API surface guard fitness test reads it)
	python3 -c "import json,sys;sys.path.insert(0,'.');from services.api_gateway.main import app;f=open('governance/openapi.json','w');json.dump(app.openapi(),f,indent=2,ensure_ascii=False)" 2>/dev/null
	@echo "✅  governance/openapi.json updated: $$(python3 -c 'import json;print(len(json.load(open("governance/openapi.json"))["paths"]))' 2>/dev/null) paths"

test-eval:      ## DeepEval tests (needs Ollama, Qdrant and the `judges` extra)
	@python3 -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('deepeval') else 1)" \
	  || { echo "The judges are not installed. Install them: pip install -e '.[judges]'"; exit 1; }
	USE_REAL_BGE_M3=true python3 -m pytest tests/eval/ -m deepeval -v

test-e2e:       ## The full walkthrough over the demo realm (needs the whole stack)
	python3 -m pytest tests/e2e/ -m e2e -v

test-all:       ## Every test, with an HTML report (needs the `report` extra)
	python3 -m pytest tests/ -q -m "" \
	  --html=test-reports/pytest/report.html --self-contained-html \
	  --junitxml=test-reports/pytest/junit.xml
	@echo "📄 Report: test-reports/pytest/report.html"

load:           ## The k6 load test (needs the API Gateway on port 8081)
	bash tests/load/run_k6.sh

# ── Installation ───────────────────────────────────────────────────────────────

install:        ## Install the Python dependencies (without the judges; see install-judges)
	pip install -e ".[dev,integration,report]"

install-judges: ## The three LLM judges: needed by `make test-eval` and the deep diagnostics alone
	pip install -e ".[judges]"

install-dev:    ## Unit tests and linters alone, without the heavy ML dependencies
	pip install -e ".[dev]"

venv:           ## Create the venv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,integration,report]"
	@echo "✅  Activate it: source .venv/bin/activate"

# The one command that takes an external-RAG developer from
# nothing: venv, Python dependencies, the UI's npm dependencies, infrastructure
# (Qdrant/Redis/Ollama/Langfuse), the services themselves in the background, a
# gateway health check, and the UI opened. It does not replace `make up`, which
# already brought up infra, API and UI in the foreground for everyday work. This
# exists for the very first run on a clean machine, when even the venv does not
# exist yet.
quickstart:     ## From nothing in one command: dependencies, infra, models, services, demo
	@./install.sh

doctor:         ## Check that everything works, and say what to fix
	@python3 -m tools.doctor

stop:           ## Stop the gateway and UI that install.sh started
	@./install.sh --stop

# Everything bootstrap did is now done by install.sh, and done more honestly:
# it reports every missing dependency at once, pulls the models in the
# foreground, and checks that the gateway did not fall back to stubs.
bootstrap: quickstart  ## An alias for quickstart (its historical name)

# ── Other ──────────────────────────────────────────────────────────────────────

clean:          ## Remove the pytest cache, __pycache__ and test-reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	rm -rf test-reports/pytest test-reports/k6

open:           ## Open every web interface in a browser
	open http://localhost:5173    # UI chat + Admin
	open http://localhost:8081    # API Gateway
	open http://localhost:3001    # Langfuse
	open http://localhost:6333/dashboard  # Qdrant

# ── Configuration Report on MIRACL ─────────────────────────────────────────────
# Three steps, run in order, all on this machine. The grid needs Qdrant,
# OpenSearch and the real BGE-M3, so none of it runs in CI.

miracl-fetch:   ## Download the MIRACL dev slices (ar, ru, en) and lay them out
	@for l in ar ru en; do python3 -m eval.miracl.fetch --lang $$l; done

miracl-ingest:  ## Index the MIRACL corpora, one per language and analyser
	@for l in ar ru en; do \
	  USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest corpus/miracl-$$l/ \
	    --strategy fixed --chunk-size 3000 --overlap 0 \
	    --corpus-id miracl-$$l --language $$l || exit 1; \
	done

miracl-report:  ## Run the configuration grid, retrieval only (LIMIT=200 for a smoke run)
	USE_REAL_BGE_M3=true python3 -m eval.miracl.report $(if $(LIMIT),--limit $(LIMIT))

# The publication workflow: targets that copy this tree into the public
# repository and gate that copy. They call `tools/publish_export.py` and
# `tools/check_publishable.py`, neither of which is published, so they live in a
# file that is not published either. The leading `-` means a tree without it
# simply has no such targets, which is the case in the public repository.
-include Makefile.publish
