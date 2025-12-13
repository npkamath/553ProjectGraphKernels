import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import numpy as np
np.float_ = np.float64
np.int_ = np.int64

# Ensure we can import from src/
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.processing import process_graphs
from src.trainer import train_pipeline

def run_tuning_sweep():
    print("========================================")
    print("      PARAMETER TUNING SWEEP            ")
    print("========================================")

    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        with open(RAW_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        graphs = data['graphs']
        metadata = data['metadata']
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    # --- PARAMETERS TO TEST ---
    # Customize these ranges as needed
    radii_to_test = [5,3,2,1]       # Neighbor Radius (Note: 2 is significantly slower)
    bins_to_test = [2]   # Discretization Bins
    wl_iters_to_test = [2] # WL Kernel Depth
    noise_val_to_test = [0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22]

    results = []

    # 2. Outer Loop: Structure Parameters (Requires re-processing graphs)
    for radius in radii_to_test:
        for noise_val in noise_val_to_test:
            print(f"\n[Processing] Radius={radius}, noise_threshold={noise_val}...")
            n_bins = 2
            # This is the slow step: extracting new subgraphs
            subgraphs, labels, groups = process_graphs(
                graphs, 
                metadata, 
                n_bins=n_bins, 
                radius=radius,
                noise_threshold=noise_val
            )
            
            # 3. Inner Loop: Kernel Parameters (Fast, just re-training SVM)
            for wl_iter in wl_iters_to_test:
                print(f"   > Training WL_Iter={wl_iter}...", end="")
                
                # Run training silently (verbose=False) to keep output clean
                acc = train_pipeline(
                    subgraphs, 
                    labels, 
                    groups, 
                    n_iter=wl_iter, 
                    verbose=False
                )
                
                print(f" Accuracy: {acc:.4f}")
                
                results.append({
                    'Neighbor Radius': radius,
                    'Bins': n_bins,
                    'noise_threshold': noise_val,
                    'WL Iterations': wl_iter,
                    'Accuracy': acc
                })

    # 4. Display & Save Results
    df = pd.DataFrame(results)
    
    print("\n========================================")
    print("           TOP 5 CONFIGURATIONS         ")
    print("========================================")
    print(df.sort_values(by='Accuracy', ascending=False).head(5))
    
    # Save CSV for record
    df.to_csv('tuning_results.csv', index=False)
    print("\nFull results saved to 'tuning_results.csv'")

    # 5. Visualization
    plot_results(df)

def plot_results(df):
    """
    Generates a performance plot comparing different radii and bins.
    """
    radii = df['Neighbor Radius'].unique()
    noise_threshold = df['noise_threshold'].unique()
    
    plt.figure(figsize=(12, 6))
    
    # Create a line for every combination of Radius + Bins
    for r in radii:
        for n in noise_threshold:
            subset = df[(df['Neighbor Radius'] == r) & (df['noise_threshold'] == n)]
            subset = subset.sort_values(by='WL Iterations')
            
            label = f"Radius {r}"
            plt.plot(subset['noise_threshold'], subset['Accuracy'], marker='o', label=label)
    
    plt.title("Parameter Tuning Results")
    plt.xlabel("noise_threshold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('tuning_plot.png')
    print("Performance plot saved to 'tuning_plot.png'")
    plt.show()

if __name__ == "__main__":
    run_tuning_sweep()