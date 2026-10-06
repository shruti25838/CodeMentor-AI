# Monitoring

The backend serves Prometheus metrics at `GET /metrics`. **`/metrics` needs the API key**
(`X-API-Key` header matching `CODEATLAS_API_KEY`), like the other admin endpoints. Without a key
configured on the server it answers 503; with a wrong or missing key, 401.

## Metrics

| Metric | Type | Labels | What it measures |
|---|---|---|---|
| `codeatlas_request_total` | counter | `method`, `path`, `status` | Requests. `path` is the route template (`/files/content`), or `unmatched` for unknown URLs |
| `codeatlas_request_latency_seconds` | histogram | `path` | Time until the response started. For `/ask/stream` this is when the stream opened, not when it closed |
| `codeatlas_errors_total` | counter | `path`, `kind` | `http_5xx` (5xx response), `exception` (unhandled; also counted as status 500 above), `stream_error` (error event inside a 200 chat stream) |
| `codeatlas_stage_duration_seconds` | histogram | `endpoint`, `stage` | Per request, time in each stage. `endpoint` is `analyze`, `ask` or `ask_stream`. Indexing stages: `clone`, `parse`, `graph`, `read_files`, `embed`, `index_write`, `save_state` (cache hits: `cache_lookup`, `remote_head`). Chat stages: `embed_query`, `search`, `rerank`, `model`. Every request also records `total` |
| `codeatlas_retrieval_hit_rate` | gauge | `split`, `k` | Retrieval eval hit rate at k = 1, 3, 5 |
| `codeatlas_retrieval_mrr` | gauge | `split` | Retrieval eval MRR@10 |
| `codeatlas_retrieval_eval_info` | gauge (always 1) | `split`, `commit`, `embedder` | Which eval run the two gauges above come from |

Histogram buckets run from 1 ms to 120 s (the indexing time limit). A stage that runs several times
in one request is summed: a chat question calls the model 3 times, and `model` holds the sum, plus a
follow-up rewrite if there was one. The general (no repository) chat streams straight from the model
and records no `model` stage. Failed indexing requests (500/504) record no stages, only errors.

**The retrieval gauges are not live traffic.** They are read at startup from the JSON files in
`eval/results/` (`CODEATLAS_EVAL_RESULTS_DIR`), which `scripts/eval_retrieval.py --json` writes.
To refresh them, rerun the eval and commit the files:

```powershell
python scripts/eval_retrieval.py --split dev --json eval/results/dev.json
python scripts/eval_retrieval.py --split test --json eval/results/test.json
python scripts/eval_retrieval.py --split all --json eval/results/all.json
```

Labels never hold raw URLs, repository names, questions or answers.

## Local: Prometheus and Grafana with Docker Compose

`docker-compose.monitoring.yml` runs the app with the API key on, Prometheus, and Grafana with
the dashboard already loaded. It needs a key in `CODEATLAS_API_KEY`, either in `.env` next to the
compose file or in the shell:

```powershell
$env:CODEATLAS_API_KEY = -join ((48..57) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
docker compose -f docker-compose.monitoring.yml up --build
```

- App: http://localhost:8000
- Prometheus: http://localhost:9090 (Status > Targets should show `codeatlas` as UP)
- Grafana: http://localhost:3000 (first login admin / admin), dashboard **CodeMentor > CodeMentor AI**

Compose passes the same key to Prometheus as a secret file (`/run/secrets/codeatlas_api_key`);
`ops/prometheus.yml` sends it with `http_headers`. The key is never written
into a config file in the repository. Without the variable, compose stops with a message saying so;
that includes `docker compose -f docker-compose.monitoring.yml down`, so keep it set (or in `.env`)
for that too. Chat in this stack needs `OPENAI_API_KEY` (it sets `CODEATLAS_LLM_PROVIDER=openai`);
without it, chat requests fail with 500 and the chat panels stay empty, while indexing works.

Files: `ops/prometheus.yml` (scrape config), `ops/grafana/provisioning/` (data source and
dashboard provider), `ops/grafana/dashboards/codementor.json` (the dashboard).

## A deployed Prometheus (for example against Render)

Point Prometheus at the public backend over HTTPS and give it the key from a file that only
Prometheus can read:

```yaml
scrape_configs:
  - job_name: codementor
    scheme: https
    metrics_path: /metrics
    http_headers:
      X-API-Key:
        files: [/etc/prometheus/secrets/codeatlas_api_key]   # contains only the key
    static_configs:
      - targets: ["<your-backend-host>"]
```

- Store the key in your secret manager or a mounted secret (Kubernetes Secret, Docker secret,
  systemd credential), not in the YAML and not in this repository.
- Use the same value as `CODEATLAS_API_KEY` on the backend. To rotate it, change both and reload
  Prometheus (`/-/reload` or restart); scrapes fail with 401 in between.
- `http_headers` is accepted by Prometheus 2.53 and 3.5 (checked with `promtool check config`);
  older releases may not have it. The backend reads only `X-API-Key`, not `Authorization: Bearer`,
  so a scraper that can only send a bearer token cannot read `/metrics`.
- Each scrape is a request to the backend. It is not rate limited, but on a free Render instance
  that sleeps when idle, a 15 s scrape interval keeps it awake.
- Counters and histograms live in one process: they reset when the server restarts, and with more
  than one uvicorn worker each worker has its own.
