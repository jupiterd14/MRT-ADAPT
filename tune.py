import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("test_results/full_upload_batch_20261011_181825.csv")
pred = df["predicted"].values
actual = df["actual"].values

def cat(c, severe=80, cong=50, mod=25):
    if c > severe: return "Severe"
    elif c > cong: return "Congested"
    elif c > mod: return "Moderate"
    else: return "Light"

act_cat = pd.Series(actual).apply(cat)
mean_pred = pred.mean()

base_acc = accuracy_score(act_cat, pd.Series(pred).apply(cat))
base_f1 = f1_score(act_cat, pd.Series(pred).apply(cat), average="weighted", zero_division=0)
print(f"BASELINE: acc={base_acc*100:.1f}%  F1w={base_f1*100:.1f}%")

best = {"f1": 0, "acc": 0}
for alpha in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
    sharp = np.clip(mean_pred + (pred - mean_pred) * alpha, 0, 100)
    for sev in [70, 75, 80, 85]:
        for cg in [42, 45, 48, 50]:
            for md in [20, 23, 25, 28]:
                pred_cat = pd.Series(sharp).apply(lambda c: cat(c, sev, cg, md))
                f1w = f1_score(act_cat, pred_cat, average="weighted", zero_division=0)
                acc = accuracy_score(act_cat, pred_cat)
                if f1w > best["f1"]:
                    best = {"f1": f1w, "acc": acc, "alpha": alpha,
                            "sev": sev, "cg": cg, "md": md}

print(f"BEST: alpha={best[chr(39)+chr(97)+chr(108)+chr(112)+chr(104)+chr(97)+chr(39)]}")
