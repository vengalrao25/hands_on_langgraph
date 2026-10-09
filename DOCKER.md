# Docker Instructions

## Prerequisites

Make sure Docker is installed:

```bash
docker --version
```

---

## Step 1 — Add your API Key

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=your_key_here
```

---

## Step 2 — Build Image & Start Container (Once only)

Builds the image and starts a long-running container in the background. Do this
once (or whenever you change the `Dockerfile` / `requirements.txt`).

```bash
./run.sh
```

- Builds the image (`langgraph-exp`)
- Starts the container (`langgraph-container`) in the background and keeps it alive
- API key is loaded automatically from `.env`

---

## Step 3 — Run Any Lesson

```bash
./start.sh stage1_fundamentals/01_why_langgraph.py
./start.sh stage2_agents_and_state/05_tool_agent_from_scratch.py
```

- No startup delay — the container is already running
- Edits to your `.py` files are picked up immediately (the folder is mounted)
- Output appears directly in your terminal

---

## Installed Packages

| Package | What it does |
|---------|-------------|
| `langgraph` | The graph engine — state, nodes, edges, checkpointers |
| `langgraph-checkpoint-sqlite` | Durable on-disk state (`SqliteSaver`) |
| `langchain` | Provides `create_agent` (referenced in lessons) |
| `langchain-openai` | OpenAI integration (`ChatOpenAI`) |
| `langchain-core` | Messages, tools, runnables |

To add packages, edit `requirements.txt`, then rebuild with `./run.sh`.

---

## Managing the Container

| Task | Command |
|------|---------|
| Build & start | `./run.sh` |
| Run a lesson | `./start.sh <path/to/file.py>` |
| List running containers | `docker ps` |
| Stop | `docker stop langgraph-container` |
| Start again (no rebuild) | `docker start langgraph-container` |
| Remove container | `docker rm langgraph-container` |
| Remove image | `docker rmi langgraph-exp` |
| Remove everything | `docker stop langgraph-container && docker rm langgraph-container && docker rmi langgraph-exp` |

---

## When to rebuild (`./run.sh` again)

| Change made | Action needed |
|-------------|---------------|
| Changed a `.py` file | Nothing — just `./start.sh` |
| Added a package to `requirements.txt` | `./run.sh` — rebuild required |
| First time setup | `./run.sh` |
