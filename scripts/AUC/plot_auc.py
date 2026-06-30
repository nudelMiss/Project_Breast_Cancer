import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration settings ---
WALKS = "5"  # Options: "5", "50", "100", "500"
METRIC = "spearman"  # Options: "spearman", "cosine"
# ------------------------------

def generate_boxplot():
    # Define directory paths using a relative path (removed the leading slash)
    input_root = Path(f"results/auc_benchmarks/{METRIC}/walks_{WALKS}_variance")
    output_png = input_root / f"boxplot_auc_{METRIC}_w{WALKS}_variance.png"
    
    # Locate all summary files in the directory tree
    summary_files = list(input_root.rglob("corum_auc_summary.csv"))
    
    if not summary_files:
        print(f"No results found in {input_root}. Make sure you ran aggregation/runs first.")
        return

    # Load and combine all datasets
    all_data = []
    for f in summary_files:
        df = pd.read_csv(f)
        # Extract patient name and cell type from folder path
        folder_name = f.parent.name
        df['patient_celltype'] = folder_name.replace('patient=', '')
        all_data.append(df)
    
    df_raw = pd.concat(all_data, ignore_index=True)
    total_patients = len(df_raw)

    # Validate that data is not empty
    if df_raw.empty:
        print(f"Warning: No data found in the summary files.")
        return

    # Create the visualization
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Generate boxplot and stripplot on the complete, unfiltered dataset
    sns.boxplot(data=df_raw, y='mean_auc', color='#A8DADC', width=0.3)
    sns.stripplot(data=df_raw, y='mean_auc', color='#1D3557', alpha=0.6, jitter=True)
    
    # Add red horizontal reference line at random chance (0.5)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance (0.5)')
    
    # Generate dynamic title reflecting all analyzed models
    title = (f"CORUM AUC Distribution \n"
             f"Metric: {METRIC.capitalize()} | Walks: {WALKS}\n"
             f"Showing all {total_patients} models")
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Mean AUC Score", fontsize=12)
    plt.ylim(0.48, 0.56)  # Focused axis range tailored to your results
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    
    # Output execution summary to console
    print(f"--- Process Complete ---")
    print(f"Total models analyzed: {total_patients}")
    print(f"Boxplot saved to: {output_png}")

if __name__ == "__main__":
    generate_boxplot()