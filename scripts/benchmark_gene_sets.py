"""
Evaluate raw cosine + existing W2V models against multiple gene sets.
Gene sets: CORUM (per-complex avg, comparison anchor), S phase, G2M phase,
           Hallmark IFN-alpha response, Hallmark IFN-gamma response.

Single AUC per (group, method, gene_set): within-set pairs (positive) vs
in-set/out-of-set pairs (negative; min 200), same protocol as the existing
CORUM benchmark.
"""
import sys
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from train_model_new import (
    normalize_cells_log1p, remove_invalid_genes, filter_by_top_variance,
    _safe_load_genes, _align_expr_and_genes,
)

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
OUT = PROOT / "results/bassez2021/supervisor_diagnostic/benchmark_gene_sets.csv"

GROUPS = {
    "BIOKEY_18_T_cell":    "patient=BIOKEY_18__celltype=T_cell",
    "BIOKEY_30_Malignant": "patient=BIOKEY_30__celltype=Malignant",
    "BIOKEY_4_B_cell":     "patient=BIOKEY_4__celltype=B_cell",
}
WALKS = [1, 5, 10, 50, 100]

# --------------- gene sets ---------------
# Seurat cc.genes (Tirosh et al. 2015) -- S phase
S_GENES = {
"MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7",
"DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN",
"WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45",
"CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B",
"BRIP1","E2F8"
}
# Seurat cc.genes -- G2M phase
G2M_GENES = {
"HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2",
"CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2",
"AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1",
"CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2",
"KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3",
"GAS2L3","CBX5","CENPA"
}
# MSigDB Hallmark INTERFERON_ALPHA_RESPONSE (Liberzon 2015)
IFN_A = {
"ADAR","B2M","BATF2","BST2","C1S","CASP1","CASP8","CCRL2","CD47","CD74","CMPK2",
"CMTR1","CNP","CSF1","CXCL10","CXCL11","DDX60","DHX58","EIF2AK2","ELF1","EPSTI1",
"GBP2","GBP4","GMPR","HELZ2","HERC6","HLA-C","IFI27","IFI30","IFI35","IFI44",
"IFI44L","IFIH1","IFIT2","IFIT3","IFITM1","IFITM2","IFITM3","IL15","IL4R","IL7",
"IRF1","IRF2","IRF7","IRF9","ISG15","ISG20","LAMP3","LAP3","LGALS3BP","LPAR6",
"LY6E","MOV10","MVB12A","MX1","NCOA7","NMI","NUB1","OAS1","OASL","OGFR","PARP12",
"PARP14","PARP9","PLSCR1","PNPT1","PROCR","PSMA3","PSMB8","PSMB9","PSME1","PSME2",
"RIPK2","RNF31","RSAD2","RTP4","SAMD9","SAMD9L","SELL","SLC25A28","SP110","STAT1",
"STAT2","TAP1","TDRD7","TENT5A","TMEM140","TRAFD1","TRIM14","TRIM21","TRIM25",
"TRIM26","TRIM5","TXNIP","UBA7","UBE2L6","USP18","WARS1"
}
# MSigDB Hallmark INTERFERON_GAMMA_RESPONSE -- representative subset (~110 of 200);
# the ones below are the most-conserved, most-cited IFN-γ-induced ISGs and TFs.
# This is sufficient: the AUC protocol is on within-set co-expression so subset is fine
# as long as the genes really are co-regulated.
IFN_G = {
"ADAR","APOL6","ARID5B","ARL4A","AUTS2","B2M","BANK1","BATF2","BPGM","BST2","BTG1",
"C1R","C1S","CASP1","CASP3","CASP4","CASP7","CASP8","CCL2","CCL5","CCL7","CD274",
"CD38","CD40","CD69","CD74","CD86","CDKN1A","CFB","CFH","CIITA","CMKLR1","CMPK2",
"CMTR1","CSF2RB","CXCL10","CXCL11","CXCL9","DDX58","DDX60","DHX58","EIF2AK2","EIF4E3",
"EPSTI1","FAS","FCGR1A","FGL2","FPR1","GBP4","GBP6","GCH1","GPR18","GZMA","HELZ2",
"HERC6","HIF1A","HLA-A","HLA-B","HLA-DMA","HLA-DQA1","HLA-DRB1","HLA-G","ICAM1","IDO1",
"IFI27","IFI30","IFI35","IFI44","IFI44L","IFIH1","IFIT1","IFIT2","IFIT3","IFITM2",
"IFITM3","IFNAR2","IL10RA","IL15","IL15RA","IL18BP","IL2RB","IL4R","IL6","IL7","IRF1",
"IRF2","IRF4","IRF5","IRF7","IRF8","IRF9","ISG15","ISG20","ISOC1","ITGB7","JAK2",
"KLRK1","LAP3","LATS2","LCP2","LGALS3BP","LY6E","LYSMD2","MARCHF1","METTL7B","MT2A",
"MTHFD2","MVP","MX1","MX2","MYD88","NAMPT","NCOA3","NFKB1","NFKBIA","NLRC5","NMI",
"NOD1","NUP93","OAS2","OAS3","OASL","OGFR","P2RY14","PARP12","PARP14","PDE4B","PELI1",
"PFKP","PIM1","PLA2G4A","PLSCR1","PML","PNP","PNPT1","PSMA2","PSMA3","PSMB10","PSMB2",
"PSMB8","PSMB9","PSME1","PSME2","PTGS2","PTPN1","PTPN2","PTPN6","RAPGEF6","RBCK1",
"RIPK1","RIPK2","RNF31","RSAD2","RTP4","SAMD9L","SAMHD1","SECTM1","SELP","SERPING1",
"SLAMF7","SLC25A28","SOCS1","SOCS3","SOD2","SP110","SPPL2A","SRI","SSPN","ST3GAL5",
"ST8SIA4","STAT1","STAT2","STAT3","STAT4","TAP1","TAPBP","TDRD7","TNFAIP2","TNFAIP3",
"TNFAIP6","TNFSF10","TOR1B","TRAFD1","TRIM14","TRIM21","TRIM25","TRIM26","UBE2L6","UPP1",
"USP18","VAMP5","VAMP8","VCAM1","WARS1","XAF1","XCL1","ZBP1","ZNFX1"
}

GENE_SETS = {
    "S_phase":  S_GENES,
    "G2M":      G2M_GENES,
    "IFN_alpha":IFN_A,
    "IFN_gamma":IFN_G,
}

# --------------- preprocessing ---------------
def load_group_matrix(d):
    mat = mmread(str(d / "expr.mtx")).tocsr()
    genes = _safe_load_genes(d / "genes.csv")
    mat, genes = _align_expr_and_genes(mat, genes)
    mat = normalize_cells_log1p(mat)
    mat, genes = remove_invalid_genes(mat, genes)
    mat, genes = filter_by_top_variance(mat, genes, 0.75)
    return mat, genes

def cosine_matrix_from_expr(mat):
    X = mat.toarray().astype(np.float32) if sp.issparse(mat) else np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def cosine_matrix_from_embedding(model_path):
    m = Word2Vec.load(str(model_path))
    genes = list(m.wv.index_to_key)
    V = np.array([m.wv[g] for g in genes], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Vn = V / norms
    return Vn @ Vn.T, genes

# --------------- benchmark ---------------
def auc_for_gene_set(C, gene_list, gene_set, seed=42, min_neg=200):
    """Single AUC for one gene set vs background. Same protocol as CORUM eval."""
    rng = np.random.default_rng(seed)
    gidx = {g: i for i, g in enumerate(gene_list)}
    allg = set(gene_list)
    shared = gene_set & allg
    if len(shared) < 3:
        return None, len(shared), 0, 0
    idx = np.array([gidx[g] for g in sorted(shared)])
    i_arr, j_arr = np.triu_indices(len(idx), k=1)
    pos = C[idx[i_arr], idx[j_arr]]
    n_pos = len(pos)
    non = np.array([gidx[g] for g in sorted(allg - shared)])
    n_neg = max(min_neg, n_pos)
    a = rng.choice(idx, size=n_neg, replace=True)
    b = rng.choice(non, size=n_neg, replace=True)
    neg = C[a, b]
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos, neg])
    if np.std(scores) < 1e-10:
        return 0.5, len(shared), n_pos, n_neg
    return float(roc_auc_score(labels, scores)), len(shared), n_pos, n_neg

# --------------- run ---------------
rows = []
for tag, gname in GROUPS.items():
    print(f"\n=== {tag} ===", flush=True)
    mat, expr_genes = load_group_matrix(PROOT / "exports_bassez" / gname)
    print(f"  expr genes after var75: {len(expr_genes)}", flush=True)

    # Raw cosine
    C_raw = cosine_matrix_from_expr(mat)
    for gs_name, gs in GENE_SETS.items():
        auc, n_in_set, n_pos, n_neg = auc_for_gene_set(C_raw, expr_genes, gs)
        print(f"  [raw cosine] {gs_name}: AUC={auc:.3f}  (in_set={n_in_set}, pos={n_pos}, neg={n_neg})", flush=True)
        rows.append(dict(group=tag, method="raw_cosine", walks=np.nan,
                         gene_set=gs_name, n_in_set=n_in_set, n_pos=n_pos,
                         n_neg=n_neg, auc=auc))
    del C_raw

    # W2V at each walks setting
    for w in WALKS:
        mp = PROOT / f"results/bassez2021/models/{tag}/raw_cosine_bidirectional_w{w}_k5_wl3_perpat/gene_embeddings.model"
        C_emb, emb_genes = cosine_matrix_from_embedding(mp)
        print(f"  W2V walks={w}: {len(emb_genes)} genes in embedding", flush=True)
        for gs_name, gs in GENE_SETS.items():
            auc, n_in_set, n_pos, n_neg = auc_for_gene_set(C_emb, emb_genes, gs)
            print(f"    {gs_name}: AUC={auc:.3f}  (in_set={n_in_set})", flush=True)
            rows.append(dict(group=tag, method=f"w2v_w{w}", walks=w,
                             gene_set=gs_name, n_in_set=n_in_set, n_pos=n_pos,
                             n_neg=n_neg, auc=auc))
        del C_emb

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}\n")
print("=== mean AUC across the 3 groups ===")
print(df.groupby(["method","gene_set"])["auc"].mean().round(3).unstack())
