"""Generates grafana/dashboards/invoiceai-overview.json. Usage: python scripts/build_dashboard.py grafana/dashboards/invoiceai-overview.json"""
import json
import sys

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}

panels = []
_id = [0]
y = [0]


def nid():
    _id[0] += 1
    return _id[0]


def row(title):
    panels.append({"type": "row", "title": title, "id": nid(), "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y[0]}, "panels": []})
    y[0] += 1


def target(expr, legend="", ref="A", instant=False):
    t = {"datasource": PROM, "expr": expr, "legendFormat": legend, "refId": ref}
    if instant:
        t["instant"] = True
        t["range"] = False
    return t


def stat(title, expr, x, w, unit="short", decimals=None, thresholds=None, desc=""):
    p = {
        "type": "stat", "title": title, "id": nid(), "description": desc, "datasource": PROM,
        "gridPos": {"h": 4, "w": w, "x": x, "y": y[0]},
        "targets": [target(expr, instant=True)],
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "value", "graphMode": "none", "textMode": "value"},
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "thresholds"},
                                     "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "green", "value": None}]}},
                        "overrides": []},
    }
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    panels.append(p)


STATUS_COLORS = {"ok": "green", "error": "red", "input": "blue", "output": "purple"}


def ts(title, targets, x, w, unit="short", h=8, desc="", stack=False):
    # Fixed colours for status-like series so "error" is always red.
    overrides = [
        {"matcher": {"id": "byName", "options": name},
         "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
        for name, color in STATUS_COLORS.items()
    ]
    panels.append({
        "type": "timeseries", "title": title, "id": nid(), "description": desc, "datasource": PROM,
        "gridPos": {"h": h, "w": w, "x": x, "y": y[0]},
        "targets": [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 15,
                                                               "stacking": {"mode": "normal" if stack else "none"}}},
                        "overrides": overrides},
    })


# ---- Overview -------------------------------------------------------------
row("Overview")
stat("PDFs processed (1h)", 'sum(increase(invoiceai_worker_pdfs_total[1h]))', 0, 4, decimals=0)
stat("PDF failure rate (1h)",
     'sum(increase(invoiceai_worker_pdfs_total{status="error"}[1h])) / clamp_min(sum(increase(invoiceai_worker_pdfs_total[1h])), 1)',
     4, 4, unit="percentunit",
     thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.05}, {"color": "red", "value": 0.2}])
stat("p95 time per PDF", 'histogram_quantile(0.95, sum by (le) (rate(invoiceai_worker_pdf_duration_seconds_bucket[1h])))',
     8, 4, unit="s")
stat("LLM cost (24h)", 'sum(increase(invoiceai_llm_cost_usd_total[24h]))', 12, 4, unit="currencyUSD", decimals=2,
     desc="Estimated from Gemini token usage at the worker's LLM_*_USD_PER_1M prices. Exact per-call cost is in Langfuse.")
stat("Queue backlog", 'sum(rabbitmq_queue_messages_ready{queue="invoice.jobs"})', 16, 4, decimals=0,
     thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 20}, {"color": "red", "value": 50}],
     desc="PDFs waiting in invoice.jobs for a free worker.")
stat("Workers up", 'sum(up{job="invoiceai-worker"})', 20, 4, decimals=0,
     thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}])
y[0] += 4

# ---- API ------------------------------------------------------------------
row("API")
ts("Requests / s by route", [('sum by (route) (rate(invoiceai_api_requests_total{route!="/metrics"}[$__rate_interval]))', "{{route}}")],
   0, 8, unit="reqps")
ts("Error responses / s", [('sum by (route, status) (rate(invoiceai_api_requests_total{status=~"[45].."}[$__rate_interval]))',
                            "{{route}} {{status}}")], 8, 8, unit="reqps")
ts("Latency p95 by route", [('histogram_quantile(0.95, sum by (le, route) (rate(invoiceai_api_request_duration_seconds_bucket{route!="/metrics"}[$__rate_interval])))',
                             "{{route}}")], 16, 8, unit="s",
   desc="/process includes waiting for the workers, so it tracks end-to-end extraction time.")
y[0] += 8

# ---- Pipeline ---------------------------------------------------------------
row("Pipeline (workers)")
ts("PDFs processed / min", [('sum by (status) (rate(invoiceai_worker_pdfs_total[$__rate_interval])) * 60', "{{status}}")],
   0, 8, unit="short", stack=True)
ts("Stage latency p95", [('histogram_quantile(0.95, sum by (le, stage) (rate(invoiceai_worker_stage_duration_seconds_bucket[$__rate_interval])))',
                          "{{stage}}")], 8, 8, unit="s", desc="Where the time goes: OCR vs LLM vs schema validation.")
ts("Job end-to-end p50 / p95", [
    ('histogram_quantile(0.50, sum by (le) (rate(invoiceai_api_job_duration_seconds_bucket[$__rate_interval])))', "p50"),
    ('histogram_quantile(0.95, sum by (le) (rate(invoiceai_api_job_duration_seconds_bucket[$__rate_interval])))', "p95"),
], 16, 8, unit="s", desc="From submission until the last PDF of the job has a result.")
y[0] += 8
ts("Queue depth", [('sum by (queue) (rabbitmq_queue_messages_ready{queue=~"invoice.*"})', "{{queue}} ready"),
                   ('sum by (queue) (rabbitmq_queue_messages_unacked{queue=~"invoice.*"})', "{{queue}} in progress")],
   0, 12, unit="short")
ts("PDFs in progress per worker", [('invoiceai_worker_in_progress', "{{instance}}")], 12, 12, unit="short",
   desc="Shows how evenly RabbitMQ spreads work across worker pods.")
y[0] += 8

# ---- LLM ----------------------------------------------------------------------
row("LLM cost & latency")
ts("Spend / hour (USD)", [('sum by (model) (rate(invoiceai_llm_cost_usd_total[$__rate_interval])) * 3600', "{{model}}")],
   0, 6, unit="currencyUSD")
ts("Tokens / min", [('sum by (type) (rate(invoiceai_llm_tokens_total[$__rate_interval])) * 60', "{{type}}")],
   6, 6, unit="short", stack=True)
ts("LLM latency p50 / p95", [
    ('histogram_quantile(0.50, sum by (le) (rate(invoiceai_llm_latency_seconds_bucket[$__rate_interval])))', "p50"),
    ('histogram_quantile(0.95, sum by (le) (rate(invoiceai_llm_latency_seconds_bucket[$__rate_interval])))', "p95"),
], 12, 6, unit="s")
ts("LLM calls / min", [('sum by (status) (rate(invoiceai_llm_calls_total[$__rate_interval])) * 60', "{{status}}")],
   18, 6, unit="short", stack=True)
y[0] += 8
stat("Avg cost per PDF (24h)",
     'sum(increase(invoiceai_llm_cost_usd_total[24h])) / clamp_min(sum(increase(invoiceai_worker_pdfs_total{status="ok"}[24h])), 1)',
     0, 6, unit="currencyUSD", decimals=4)
stat("Tokens (24h)", 'sum(increase(invoiceai_llm_tokens_total[24h]))', 6, 6, unit="short", decimals=0)
stat("LLM error rate (1h)",
     'sum(increase(invoiceai_llm_calls_total{status="error"}[1h])) / clamp_min(sum(increase(invoiceai_llm_calls_total[1h])), 1)',
     12, 6, unit="percentunit",
     thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.05}])
panels.append({
    "type": "text", "title": "Per-request traces", "id": nid(),
    "gridPos": {"h": 4, "w": 6, "x": 18, "y": y[0]},
    "options": {"mode": "markdown", "content":
                "Every PDF is a **Langfuse** trace (session = job id) with OCR / LLM spans and exact Gemini tokens, "
                "cost and latency.\n\n[Open Langfuse](${langfuse_url})"},
})
y[0] += 4

# ---- Logs ---------------------------------------------------------------------
row("Logs")
panels.append({
    "type": "logs", "title": "Errors and warnings", "id": nid(), "datasource": LOKI,
    "gridPos": {"h": 10, "w": 24, "x": 0, "y": y[0]},
    "targets": [{"datasource": LOKI, "refId": "A",
                 "expr": '{service=~"$service", level=~"ERROR|WARNING|CRITICAL"}'}],
    "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending", "enableLogDetails": True},
})
y[0] += 10
panels.append({
    "type": "logs", "title": "All logs", "id": nid(), "datasource": LOKI,
    "gridPos": {"h": 12, "w": 24, "x": 0, "y": y[0]},
    "targets": [{"datasource": LOKI, "refId": "A", "expr": '{service=~"$service"} |= "$search"'}],
    "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending", "enableLogDetails": True},
})

dashboard = {
    "uid": "invoiceai-overview",
    "title": "InvoiceAI overview",
    "tags": ["invoiceai"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "editable": True,
    "templating": {"list": [
        {"name": "service", "label": "Service", "type": "query", "datasource": LOKI,
         "query": {"label": "service", "stream": "", "type": 1, "refId": "LokiVariableQueryEditor-VariableQuery"},
         "definition": "label_values(service)", "includeAll": True, "multi": True, "allValue": ".+",
         "current": {"selected": True, "text": ["All"], "value": ["$__all"]}, "refresh": 2},
        {"name": "search", "label": "Search logs", "type": "textbox", "query": "", "current": {"text": "", "value": ""}},
        {"name": "langfuse_url", "label": "Langfuse URL", "type": "constant", "hide": 2,
         "query": "http://localhost:3002"},
    ]},
    "panels": panels,
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2)
    f.write("\n")
print(f"{len(panels)} panels -> {sys.argv[1]}")
