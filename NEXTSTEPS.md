# Next Steps — After the Roadmap ("Actually Ship It")

The `ROADMAP.md` teaches the LangGraph **engine** (state, loops, tools, persistence,
HITL, streaming, multi-agent). That's the hard core of a production agent. But a few
things live *outside* the graph and are needed before real traffic. Do these once the
3 stages are done.

> Verdict: after the roadmap you can build a reliable agent in a controlled setting.
> These steps make it survive production.

---

## Stage 4 — Production Hardening

- [ ] **Deployment / serving** — wrap the graph in an API (FastAPI or LangGraph
  Platform). Handle concurrency, timeouts, and multiple simultaneous threads.
- [ ] **Real persistence backend** — swap the in-memory checkpointer for a durable one
  (Postgres or Redis checkpointer). Test crash-and-resume.
- [ ] **Observability** — tracing + monitoring (LangSmith or equivalent). Track tokens,
  cost, latency, and failure rates per run.
- [ ] **Evaluation & testing** — build an eval dataset + regression tests. Know *why*
  the agent is good, not just that it ran.
- [ ] **Security / guardrails** — prompt-injection defense, tool-permission boundaries,
  output validation before actions fire.
- [ ] **Cost & latency tuning** — model routing (cheap model for easy steps), prompt
  caching, response caching.

---

## Reminder

Think of the roadmap as building the engine + transmission. Stage 4 is the wheels,
brakes, and dashboard — needed before driving in traffic.

*(We'll flesh these out with real lessons/code when we get here.)*
