"""Train the Assignment-2 classifiers (DT / SVM / MLP) on period 1, test on period 2."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsearch import config
from simsearch.documents import load_jsonl
from simsearch.risk import train

report = train(load_jsonl(config.DOCS_PERIOD1), load_jsonl(config.DOCS_PERIOD2))
out = ROOT / "evals" / "results"; out.mkdir(parents=True, exist_ok=True)
(out / "risk_model.json").write_text(json.dumps(report, indent=2))
for name, r in report["models"].items():
    v, t = r["validation_period1"], r["test_period2"]
    print(f"{name:15s} val AUC {v['roc_auc']:.3f} | period-2 AUC {t['roc_auc']:.3f} acc {t['accuracy']:.3f} f1 {t['f1']:.3f}")
print("selected:", report["selected_model"])
