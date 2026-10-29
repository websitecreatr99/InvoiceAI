"""Wrap prometheus/alerts.yml in a PrometheusRule for the Prometheus Operator.

  python scripts/gen_k8s_rules.py          # rewrite k8s/prometheus-rule.yaml
  python scripts/gen_k8s_rules.py --check  # fail if it's out of date (CI)
"""
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "prometheus" / "alerts.yml"
TARGET = ROOT / "k8s" / "prometheus-rule.yaml"

rule = {
    "apiVersion": "monitoring.coreos.com/v1",
    "kind": "PrometheusRule",
    "metadata": {"name": "invoiceai", "labels": {"release": "kube-prometheus-stack"}},
    "spec": yaml.safe_load(SOURCE.read_text(encoding="utf-8")),
}
text = "# Generated from prometheus/alerts.yml by scripts/gen_k8s_rules.py; do not edit.\n" + yaml.safe_dump(
    rule, sort_keys=False, width=120
)

if "--check" in sys.argv:
    if TARGET.read_text(encoding="utf-8") != text:
        sys.exit("k8s/prometheus-rule.yaml is out of date: run python scripts/gen_k8s_rules.py")
    print("k8s/prometheus-rule.yaml is up to date")
else:
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT)}")
