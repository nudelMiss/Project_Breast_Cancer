#!/usr/bin/env python3
"""Inventory: cell counts per (patient x celltype) group for the 4 multi-type cohorts,
mapped to the 6 strong cell types, restricted to groups that have a trained propr model.
Reports the per cohort x celltype distribution to inform the top-K choice. Read-only."""
import re
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG  = "propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS = ["bassez2021","wu2021","qian2020","pal2021"]
EXP = {"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","qian2020":"exports_qian","pal2021":"exports_pal2021"}
# strong cell types only (drop NK + normal Epithelial + Mast + equivocal + anything else)
BROAD = {'Malignant':'Malignant','T_cell':'T / Lymphoid','Lymphoid':'T / Lymphoid',
         'B_cell':'B cells','Plasmablast':'B cells',
         'Myeloid':'Myeloid','Macrophage':'Myeloid','Monocyte':'Myeloid',
         'Fibroblast':'Fibroblast / Stroma','Pericyte':'Fibroblast / Stroma',
         'Endothelial':'Endothelial'}

def ncells(d):
    cf = d/"cells.csv"
    if not cf.exists(): return None
    with open(cf) as f:
        return sum(1 for _ in f)

rows=[]
for ds in DATASETS:
    for d in sorted((REPO/EXP[ds]).glob("patient=*__celltype=*")):
        m=re.match(r"patient=(.+)__celltype=(.+)$", d.name)
        if not m: continue
        patient, ct = m.group(1), m.group(2)
        broad = BROAD.get(ct)
        if broad is None: continue          # not a strong cell type
        model = REPO/f"results/{ds}/stageA/models/{patient}_{ct}/{TAG}/gene_embeddings.model"
        n = ncells(d)
        rows.append(dict(ds=ds, patient=patient, ct=ct, broad=broad,
                         ncells=n, has_model=model.exists()))
df=pd.DataFrame(rows)
out=REPO/"results/multidataset/pcoa/cellcount_inventory_strong.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out}  ({len(df)} strong-celltype groups across 4 cohorts)\n")

dfm = df[df.has_model].copy()
print(f"groups with a trained propr model: {len(dfm)} / {len(df)}\n")

print("=== n groups (with model) per cohort x broad cell type ===")
piv = dfm.pivot_table(index='ds', columns='broad', values='patient', aggfunc='count', fill_value=0)
print(piv.to_string()); print()

print("=== how many cohorts have >=5 / >=3 groups, per broad cell type ===")
for k in (5,3):
    cnt = (piv>=k).sum(axis=0)
    print(f" K={k}: ", {c:int(cnt[c]) for c in piv.columns})
print()

print("=== cell-count distribution per cohort x broad (min / median / max) ===")
g = dfm.groupby(['ds','broad'])['ncells'].agg(['count','min','median','max'])
print(g.to_string())
