# 📈 InvoiceAI Tracking

Observability for InvoiceAI: **metrics and alerts** (Prometheus), **logs** (Loki), **dashboards** (Grafana) and **LLM cost and latency per request** (Langfuse).

This branch holds only the monitoring stack. The services it watches report their own data:

| Branch | Exposes |
|--------|---------|
| `fastapi` | `GET /metrics`: requests and latency per route, jobs and documents submitted, queue publish failures, worker results, end-to-end job duration |
| `worker` | `:9100/metrics`: documents processed by status (`ok` / `error` / `retry`), jobs retried and dead-lettered, time per document and per stage (OCR / LLM / validation), documents in progress, LLM calls, latency, tokens and estimated USD cost. Plus **Langfuse** traces when `LANGFUSE_*` keys are set |
| RabbitMQ | Built-in Prometheus plugin on `:15692`: queue depth for `invoice.jobs`, `invoice.jobs.retry`, `invoice.jobs.dlq` and `invoice.results` |

The *Queue depth* panel covers every `invoice.*` queue, including the retry queue and the DLQ. `invoiceai_worker_jobs_retried_total` and `invoiceai_worker_jobs_dead_lettered_total` aren't on the dashboard or in any alert yet; query them in Prometheus or Grafana Explore. Every other metric the dashboard and alerts reference is exported by the API or the worker, all at zero from startup.

---

## 🧭 What goes where

    API ──┐                        ┌──> Prometheus ──> alert rules
    workers (1..N) ──── /metrics ──┤                     │
    RabbitMQ ─────┘                │                     v
                                   │                  Grafana  <── Loki <── Alloy <── container logs
    workers ── Gemini spans ───────┴──> Langfuse (cost, tokens, latency per call)

- **Prometheus** answers "how is the system doing": throughput, error rates, latency percentiles, queue backlog, spend per hour.
- **Loki** stores every container's logs (collected by Grafana Alloy, labelled by compose service and log level), searchable next to the metrics in Grafana.
- **Langfuse** answers "what happened on this request": one trace per PDF, grouped into a session per job, with OCR / LLM / validation spans and the exact Gemini tokens, cost and latency.
- Prometheus doesn't store logs; that's why Loki is here.

## 🖥️ Grafana dashboard: *InvoiceAI overview*

| Row | Panels |
|-----|--------|
| Overview | PDFs processed (1h), failure rate, p95 time per PDF, LLM cost (24h), queue backlog, workers up |
| API | Requests/s by route, error responses, p95 latency by route |
| Pipeline | PDFs/min ok vs error, p95 per stage (where the time goes), job p50/p95 end to end, queue depth, PDFs in progress per worker (load balance) |
| LLM cost & latency | Spend per hour, tokens/min, LLM p50/p95, calls ok vs error, average cost per PDF, link to Langfuse |
| Logs | Errors and warnings across services, full log search filtered by service |

## 🚨 Alerts (`prometheus/alerts.yml`)

| Alert | Fires when |
|-------|-----------|
| `InvoiceAIApiDown` | API scrape fails for 1 min |
| `InvoiceAINoWorkers` | No worker is up for 2 min (PDFs will pile up) |
| `InvoiceAIApiHighErrorRate` | > 5% of requests return 5xx for 5 min |
| `InvoiceAIApiSlow` | p95 > 1 s on any route except `/process` for 10 min |
| `InvoiceAIQueueUnavailable` | API failed to publish to RabbitMQ (users get 503) |
| `InvoiceAIQueueBacklog` | > 50 PDFs waiting for 10 min |
| `InvoiceAIPdfFailureRate` | > 20% of PDFs fail for 15 min |
| `InvoiceAISlowExtraction` | p95 per PDF > 5 min for 15 min |
| `InvoiceAILlmErrors` | Gemini calls failing for 10 min |
| `InvoiceAILlmCostSpike` | Estimated spend > $5 in the last hour (edit to your budget) |

Rules have unit tests in `prometheus/alerts.test.yml` (`promtool test rules`). Alerts show in Prometheus and Grafana. To get notified, add an Alertmanager and point Prometheus at it.

---

## 🚀 Run with Docker Compose

1. Start the app from the `fastapi` branch (creates the `invoiceai` network):

        docker compose up -d

2. From this branch:

        cp .env.example .env
        docker compose up -d                      # Prometheus, Loki, Alloy, Grafana
        docker compose --profile langfuse up -d   # optional: self-hosted Langfuse

3. Open:

| Service | URL | Login |
|---------|-----|-------|
| Grafana | http://localhost:3001 | `admin` / `admin` (from `.env`) |
| Prometheus | http://localhost:9090 | – |
| Langfuse | http://localhost:3002 | `LANGFUSE_ADMIN_EMAIL` / `LANGFUSE_ADMIN_PASSWORD` |

4. Point the workers at Langfuse by adding to the `fastapi` branch's `.env` (the worker service reads it):

        LANGFUSE_PUBLIC_KEY=pk-lf-invoiceai-local
        LANGFUSE_SECRET_KEY=sk-lf-invoiceai-local
        LANGFUSE_HOST=http://langfuse-web:3000

   With **Langfuse Cloud** instead, skip the `langfuse` profile and use your project's keys with `LANGFUSE_HOST=https://cloud.langfuse.com`.

Langfuse computes cost from its model price table. The Prometheus cost metric is an estimate from token counts at the worker's `LLM_INPUT_USD_PER_1M` / `LLM_OUTPUT_USD_PER_1M` (Gemini 2.5 Flash list prices by default).

## ☸️ Kubernetes

With [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack):

    kubectl apply -f k8s/podmonitors.yaml -f k8s/prometheus-rule.yaml
    kubectl create configmap invoiceai-dashboard --from-file=grafana/dashboards/invoiceai-overview.json
    kubectl label configmap invoiceai-dashboard grafana_dashboard=1

Logs: install the `grafana/loki` and `grafana/alloy` Helm charts. Langfuse: the `langfuse/langfuse` Helm chart or Langfuse Cloud, with the keys in the `invoiceai-secrets` Secret (see the `worker` branch).

`k8s/prometheus-rule.yaml` is generated from `prometheus/alerts.yml`: run `python scripts/gen_k8s_rules.py` after editing the alerts.

## 🧪 CI

`.github/workflows/ci.yml` on `tracking`: Prometheus config check and alert unit tests (`promtool`), Loki and Alloy config validation, dashboard JSON and k8s rules sync, compose validation; then starts Prometheus, Loki and Grafana and checks that rules, data sources and the dashboard are loaded.

## 📂 Structure

    ├── docker-compose.yml           # Prometheus, Loki, Alloy, Grafana (+ Langfuse profile)
    ├── prometheus/
    │   ├── prometheus.yml           # scrape targets: api, workers (DNS), rabbitmq
    │   ├── alerts.yml               # alert rules
    │   └── alerts.test.yml          # promtool unit tests for the rules
    ├── loki/config.yml
    ├── alloy/config.alloy           # Docker log collection -> Loki
    ├── grafana/
    │   ├── provisioning/            # data sources + dashboard provider
    │   └── dashboards/invoiceai-overview.json
    ├── k8s/                         # PodMonitors, PrometheusRule
    └── scripts/
        ├── gen_k8s_rules.py         # alerts.yml -> k8s/prometheus-rule.yaml
        └── build_dashboard.py       # regenerates grafana/dashboards/invoiceai-overview.json
