#!/usr/bin/env python3
"""
nPMI + k=100 screening gate (2026-06-16), stdlib-only so it runs on the login node.
Restricts to the 6-group Stage-A screening subset (the only groups with npmi rows)
so every metric is compared on the SAME groups (apples-to-apples).

Reads  results/bassez2021/stageA/bio_auc/bio_auc_collected.csv
Writes results/bassez2021/stageA/bio_auc/npmi_gate_comparison.csv
Prints assoc x (strat,k) bio_mean pivot + per-module @k50 + STOP/GO(npmi vs cosine)
       + k-saturation (k5/50/100) for npmi & the magnitude winners.
"""
import csv, re, statistics as st
from pathlib import Path

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
BIO = ROOT / "results/bassez2021/stageA/bio_auc"
SUBSET = {"BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell",
          "BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"}
pat = re.compile(r"^([a-z]+)_(star|bidirectional)_w(\d+)_k(\d+)_")
MODS = ["auc_S_phase","auc_G2M","auc_IFN_alpha","auc_IFN_gamma"]

rows = []
with open(BIO / "bio_auc_collected.csv") as f:
    for r in csv.DictReader(f):
        if r["group"] not in SUBSET:
            continue
        m = pat.match(r["config_tag"])
        if not m:
            continue
        assoc, strat, w, k = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        rec = {"group": r["group"], "assoc": assoc, "strat": strat, "k": k,
               "mean_bio_auc": float(r["mean_bio_auc"])}
        for mod in MODS:
            rec[mod] = float(r[mod]) if r.get(mod) not in (None,"","nan") else float("nan")
        rows.append(rec)

# dedup to one row per (group,assoc,strat,k) (keep last)
uniq = {}
for r in rows:
    uniq[(r["group"], r["assoc"], r["strat"], r["k"])] = r
rows = list(uniq.values())

with open(BIO / "npmi_gate_comparison.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=["group","assoc","strat","k","mean_bio_auc"]+MODS)
    wtr.writeheader()
    for r in sorted(rows, key=lambda x:(x["group"],x["assoc"],x["strat"],x["k"])):
        wtr.writerow(r)

def mean(xs): xs=[x for x in xs if x==x]; return sum(xs)/len(xs) if xs else float("nan")

assocs = sorted({r["assoc"] for r in rows})
cells  = sorted({(r["strat"], r["k"]) for r in rows})

print("=== bio_mean_auc by assoc x (strat,k)  [6 screening groups] ===")
hdr = "assoc".ljust(9) + "".join(f"{s[:5]}_k{k}".rjust(12) for (s,k) in cells)
print(hdr)
for a in assocs:
    line = a.ljust(9)
    for (s,k) in cells:
        v = mean([r["mean_bio_auc"] for r in rows if r["assoc"]==a and r["strat"]==s and r["k"]==k])
        line += (f"{v:.3f}" if v==v else "  -  ").rjust(12)
    print(line)

print("\n=== per-module mean @k=50 (assoc x strat) ===")
for mod in MODS:
    print(f"\n{mod}:")
    for a in assocs:
        line = "  "+a.ljust(9)
        for s in ["star","bidirectional"]:
            v = mean([r[mod] for r in rows if r["assoc"]==a and r["strat"]==s and r["k"]==50])
            line += f"{s}={v:.3f} " if v==v else f"{s}=  -  "
        print(line)

print("\n=== STOP/GO: npmi vs matched cosine, per (strat,k) [gate: delta>0 AND wins>=2] ===")
for (s,k) in cells:
    cos = {r["group"]: r["mean_bio_auc"] for r in rows if r["assoc"]=="cosine" and r["strat"]==s and r["k"]==k}
    npmi= {r["group"]: r["mean_bio_auc"] for r in rows if r["assoc"]=="npmi"   and r["strat"]==s and r["k"]==k}
    common = set(cos)&set(npmi)
    if not common: continue
    delta = mean([npmi[g]-cos[g] for g in common])
    wins  = sum(1 for g in common if npmi[g]>cos[g])
    verdict = "GO" if (delta>0 and wins>=2) else "no"
    print(f"  {s:13s} k={k:<3d} npmi: mean_delta_vs_cosine={delta:+.3f}  wins={wins}/{len(common)}  -> {verdict}")

print("\n=== k-saturation (mean bio_auc over 6 groups): k5 -> k50 -> k100 ===")
for a in ["cosine","spearman","propr","npmi"]:
    for s in ["star","bidirectional"]:
        vals = {k: mean([r["mean_bio_auc"] for r in rows if r["assoc"]==a and r["strat"]==s and r["k"]==k]) for k in (5,50,100)}
        if all(v!=v for v in vals.values()): continue
        def fmt(v): return f"{v:.3f}" if v==v else "  -  "
        d100 = vals[100]-vals[50] if (vals[100]==vals[100] and vals[50]==vals[50]) else float("nan")
        print(f"  {a:9s} {s:13s}  k5={fmt(vals[5])}  k50={fmt(vals[50])}  k100={fmt(vals[100])}  (k100-k50={d100:+.3f})")
print(f"\nWrote {BIO/'npmi_gate_comparison.csv'}")
