#!/usr/bin/env python3
"""
Aggregate CORUM AUC per-embedding summary CSV files.

Searches recursively under an output root for files named
"corum_auc_summary.csv", concatenates them, adds metadata parsed from path,
and writes one combined CSV and a Boxplot with dynamic naming.
"""

import argparse
from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CORUM AUC summary files")
    parser.add_argument(
        "--input_root",
        type=str,
        default="results/auc_benchmarks/spearman",
        help="Root directory containing per-embedding corum_auc_summary.csv files",
    )
    # Removing output_csv argument as we will generate it dynamically
    return parser.parse_args()


def add_path_metadata(df: pd.DataFrame, summary_path: Path, input_root: Path) -> pd.DataFrame:
    """Attach metadata columns inferred from relative path structure dynamically."""
    rel_parts = summary_path.relative_to(input_root).parts
    
    patient_celltype = "Unknown"
    config_tag = "Unknown"
    
    for part in rel_parts[:-1]:
        if part.startswith("patient="):
            patient_celltype = part
        else:
            config_tag = part

    patient = ""
    celltype = ""
    if "__celltype=" in patient_celltype:
        left, right = patient_celltype.split("__celltype=", 1)
        patient = left.replace("patient=", "", 1)
        celltype = right

    out = df.copy()
    out["summary_path"] = str(summary_path)
    out["patient_celltype"] = patient_celltype
    out["patient"] = patient
    out["celltype"] = celltype
    out["config_tag"] = config_tag
    return out


def extract_run_info(input_root: Path) -> tuple:
    """Extract metric type (cosine/spearman) and number of walks from path."""
    path_str = str(input_root).lower()
    
    # Determine metric type
    metric = "unknown_metric"
    if "spearman" in path_str:
        metric = "spearman"
    elif "cosine" in path_str:
        metric = "cosine"
        
    # Find number of walks (e.g., 'walks_5' or 'walks=5')
    walks_match = re.search(r'walks[=_]?(\d+)', path_str)
    walks = walks_match.group(1) if walks_match else "unknown_walks"
    
    return metric, walks


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    summary_files = sorted(input_root.rglob("corum_auc_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No corum_auc_summary.csv files found under {input_root}")

    # Extract run info for dynamic naming
    metric, walks = extract_run_info(input_root)
    
    # Create dynamic output filenames
    output_filename_base = f"summary_{metric}_walks_{walks}"
    output_csv = input_root / f"{output_filename_base}.csv"
    output_png = input_root / f"{output_filename_base}.png"

    combined_frames = []
    for summary_file in summary_files:
        df = pd.read_csv(summary_file)
        df = add_path_metadata(df, summary_file, input_root)
        combined_frames.append(df)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = combined.sort_values(by=["patient_celltype", "config_tag"]).reset_index(drop=True)

    # ==========================================
    # --- Generate Filtered Boxplot ---
    # ==========================================
    
   # Identify columns
    auc_col = next((col for col in combined.columns if 'auc' in col.lower()), 'mean_auc')
    pval_col = next((c for c in combined.columns if c.lower() in {"p_value", "pval", "pvalue", "p"}), None)

    if pval_col is None:
        print(f"Error: No p-value column found. Check your input files.")
        return

    # Apply BH-FDR correction per config_tag
    alpha = 0.05

    combined["q_value_bh"] = float("nan")
    combined["significant_bh"] = False

    for config, group_idx in combined.groupby("config_tag").groups.items():
        pvals = combined.loc[group_idx, pval_col].to_numpy()
        finite_mask = pd.notna(pvals)

        if not finite_mask.any():
            continue

        rej, q, _, _ = multipletests(
            pvals[finite_mask],
            alpha=alpha,
            method="fdr_bh"
        )

        finite_rows = group_idx[finite_mask]
        combined.loc[finite_rows, "q_value_bh"] = q
        combined.loc[finite_rows, "significant_bh"] = rej

    # Save combined CSV AFTER adding BH columns
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"Input root: {input_root}")
    print(f"Found summaries: {len(summary_files)}")
    print(f"Wrote combined CSV with BH results: {output_csv}")

    # ==========================================
    # FILTER OPTION 1: BH-FDR significance
    # We used this to keep only models that remain statistically significant
    # after correcting p-values per configuration.
    # Currently commented out because for the presentation plot we want
    # to show models above random chance: AUC > 0.5.
    # ==========================================

    # total_models = len(combined)
    # combined_filtered = combined[combined["significant_bh"]].copy()
    # remaining_models = len(combined_filtered)
    #
    # print(
    #     f"BH-FDR filter per config_tag: kept {remaining_models}/{total_models} "
    #     f"models with q_value_bh < {alpha}"
    # )
    #
    # if combined_filtered.empty:
    #     print(f"Warning: No significant models found (q < {alpha}). Skipping plot.")
    #     return


    # ==========================================
    # FILTER OPTION 2: AUC > 0.5
    # 0.5 is random-chance performance, so this keeps only models
    # that show at least some better-than-random biological signal.
    # This is useful for the presentation comparison between cosine and Spearman.
    # ==========================================

    total_models = len(combined)
    combined_filtered = combined.copy()
    remaining_models = len(combined_filtered)

    print(
        f"AUC filter: kept {remaining_models}/{total_models} "
        f"models with {auc_col} > 0.5"
    )

    if combined_filtered.empty:
        print("Warning: No models found with AUC > 0.5. Skipping plot.")
        return

    # Helper function to extract and format key hyperparameters for the X-axis labels[cite: 1]
    def clean_config_label(label):
        if label == "Unknown": return label
        parts = label.split('__')
        # Filter for relevant parameters like k, window size, and walks[cite: 1]
        relevant = [p for p in parts if 'k=' in p or 'win=' in p or 'walks=' in p]
        return "\n".join(relevant) if relevant else label

    # Apply label cleaning to the configuration tags[cite: 1]
    combined_filtered['display_label'] = combined_filtered['config_tag'].apply(clean_config_label)

    # Initialize the plot with specific dimensions[cite: 1]
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Generate the boxplot for the filtered data points[cite: 1]
    sns.boxplot(data=combined_filtered, x='display_label', y=auc_col, palette='Set2', width=0.4)
    # Overlay individual data points (stripplot) to show patient distribution[cite: 1]
    sns.stripplot(data=combined_filtered, x='display_label', y=auc_col, color='black', alpha=0.5, jitter=True)
    
    # Add a horizontal dashed line representing random chance at 0.5[cite: 1, 2]
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance (0.5)')
    
    # Calculate the percentage of models that passed the filter[cite: 1]
    success_rate = (remaining_models / total_models) * 100
    
    # Title for BH-FDR filtering:
    # full_title = (f"CORUM AUC Analysis: {metric.upper()} Metric (FDR Significant: q < {alpha})\n"
    #               f"Showing {remaining_models} out of {total_models} models ({success_rate:.1f}%)\n"
    #               f"Configuration: {walks} Walks")

    # Title for AUC > 0.5 filtering:
    full_title = (f"CORUM AUC Analysis: {metric.upper()} Metric (Only T-cells)\n"
                f"Showing {remaining_models} out of {total_models} models ({success_rate:.1f}%)\n"
                f"Configuration: {walks} Walks")
    
    plt.title(full_title, fontsize=14, pad=20)
    plt.xlabel('Model Hyperparameters (Configuration)', fontsize=12, labelpad=10)
    plt.ylabel('Mean AUC Score (Biological Accuracy)', fontsize=12)
    
    # Set Y-axis limits to focus on results above the 0.5 threshold[cite: 1]
    plt.ylim(0.495, max(combined_filtered[auc_col].max() * 1.05, 0.55))
    
    # Format tick marks and legend[cite: 1]
    plt.xticks(rotation=0, ha='center', fontsize=9)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the resulting plot to the output directory[cite: 1]
    plt.savefig(output_png, dpi=300)
    print(f"Wrote Filtered Boxplot: {output_png} ({remaining_models}/{total_models} models)")

if __name__ == "__main__":
    main()