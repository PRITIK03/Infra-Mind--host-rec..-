# InfraMind — AWS Instance Advisor

An AI agent that gathers your workload requirements, reasons about system design, researches live AWS instance data across compute, database, and cache tiers, and recommends the optimal setup — with deployable Terraform files generated automatically.

The project has two parts that work together:

```
aws-instance-advisor/       ← Python backend (FastAPI + LangGraph agent)
../aws-advisor-ui/          ← canonical Next.js frontend (chat UI + results dashboard)
```

The frontend deployment root is the sibling `aws-advisor-ui` directory at the
workspace root. Do not deploy or recreate a second frontend under this backend
directory.

---

## How it works

1. You describe your application and expected workload (via CLI or the web UI).
2. The agent asks clarifying questions until it has enough context.
3. It reasons about system design — concurrency, resource profile, traffic pattern, scaling strategy.
4. It researches live EC2, RDS, and ElastiCache instance data.
5. It returns a full architecture recommendation (compute + database + cache + load balancer) with confidence scores and trade-offs.
6. Terraform files for the recommended setup are written to `terraform_output/`.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| Node.js | 18 |
| npm | 9 |

---

## Quick start (local — both services)

### 1. Clone the repo

```bash
git clone https://github.com/PRITIK03/Infra-Mind--host-rec..-.git
cd Infra-Mind--host-rec..-
```

### 2. Set up the backend

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
cp .env.example .env
```

Open `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Yes | Your OpenRouter API key |
| `BASE_URL` | Yes | OpenRouter base URL (`https://openrouter.ai/api/v1`) |
| `MODEL_NAME` | Yes | Model slug e.g. `anthropic/claude-3.5-sonnet` |
| `VANTAGE_API_KEY` | Yes | [Vantage](https://www.vantage.sh/) API key for live EC2 pricing data |
| `TAVILY_API_KEY` | No | Enables a web-search round during reasoning. Agent works without it. |
| `API_KEY_2` | No | Second OpenRouter key for automatic rate-limit failover |
| `CORS_ALLOWED_ORIGIN` | Yes | Explicit frontend origin; use `http://localhost:3000` locally and the real Vercel domain in production |
| `PORT` | No | FastAPI port (default `8000`) |
| `RATE_LIMIT_MAX_REQUESTS` | No | Recommendation requests allowed per client IP per window (default `5`) |
| `RATE_LIMIT_WINDOW_SECONDS` | No | Recommendation rate-limit window (default `60`) |

### 3. Start the backend API

From the repository workspace root (`D:\! Sciqus Internship\AWS Agent` on
Windows), run:

```powershell
Set-Location .\aws-instance-advisor
& .\.venv\Scripts\Activate.ps1
python -m uvicorn app.api.main:app --reload --port 8000
```

Do not run `uvicorn main:app` from the workspace root: there is no root
`main.py`; the ASGI application is `app.api.main:app`.

Alternatively, after activating the environment, run the following while
already inside `aws-instance-advisor`:

```bash
uvicorn app.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. You can verify it with:

```bash
curl http://localhost:8000/api/health
```

### 4. Set up the frontend

```bash
cd aws-advisor-ui

# Install dependencies
npm install

# Copy the env template
cp .env.example .env.local
```

`.env.local` only needs one variable:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 5. Start the frontend

In a second terminal, from the workspace root:

```powershell
Set-Location .\aws-advisor-ui
npm install
npm run dev
```

Or, if you are already inside `aws-advisor-ui`, run:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## CLI mode (no frontend needed)

If you just want the terminal experience:

```bash
python -m app.main
```

The agent will ask questions interactively and print the full recommendation + write Terraform files to `./terraform_output/`.

---

## Running tests

```bash
pytest -v
```

---

## Docker (backend only)

The Dockerfile builds and runs the FastAPI backend:

```bash
docker build -t infra-mind .
docker run -p 8000:8000 --env-file .env infra-mind
```

> The Docker image runs the API server only. For the frontend, run `npm run dev` locally or deploy `aws-advisor-ui` separately (e.g. Vercel — set `NEXT_PUBLIC_API_URL` to your deployed backend URL in the Vercel project settings).

The recommendation endpoint has a small process-local per-IP rate limit because
each request consumes LLM and live-data quota. The job store is also intentionally
process-local for now: jobs are lost on server restart and are not shared across
workers or instances. Before horizontal scaling or durable job history, replace it
with a shared Redis-backed store.

---

## Project structure

```
aws-instance-advisor/
├── app/
│   ├── agent/
│   │   ├── graph.py               # LangGraph graph definition
│   │   ├── state.py               # Shared agent state schema
│   │   └── nodes/                 # One file per graph node
│   │       ├── requirement_collector.py
│   │       ├── requirement_validator.py
│   │       ├── system_design_reasoner.py
│   │       ├── instance_researcher.py
│   │       ├── database_researcher.py
│   │       ├── cache_researcher.py
│   │       ├── holistic_recommender.py
│   │       ├── recommender.py
│   │       └── terraform_generator.py
│   ├── api/
│   │   ├── main.py                # FastAPI app + job endpoints
│   │   └── jobs.py                # In-memory job store
│   ├── llm/
│   │   └── client.py              # OpenRouter LLM client
│   ├── models/
│   │   └── schemas.py             # Pydantic data models
│   ├── tools/
│   │   ├── aws_instance_data.py   # Vantage API integration
│   │   └── web_search.py          # Tavily search tool
│   ├── config.py                  # Settings loaded from .env
│   └── main.py                    # CLI entrypoint
├── aws-advisor-ui/                # Next.js frontend (see its own README)
├── tests/                         # pytest test suite
├── terraform_output/              # Generated Terraform files (git-ignored)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/recommend` | Start a new job `{ "message": "..." }` |
| `GET` | `/api/recommend/{job_id}` | Poll job status / result |
| `POST` | `/api/recommend/{job_id}/answer` | Reply to a follow-up question `{ "answer": "..." }` |

Job status values: `collecting` → `running` → `awaiting_input` → `done` / `error`
