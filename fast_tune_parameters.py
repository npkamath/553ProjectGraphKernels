import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score
)
from sklearn.svm import SVC

# Ensure we can import from src/
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH, RANDOM_SEED, SVM_C, N_JOBS
from src.processing import extract_raw_samples, to_grakel
from src.kernels import get_kernel

# ==========================================
# CONFIGURATION
# ==========================================
MAX_SAMPLES = None  # Set to None to use the full dataset

# --- PIVOT CONFIGURATION (OFAT) ---
# We vary one parameter while holding these fixed.
DEF_RADIUS = 6
DEF_BINS = 2
DEF_WL = 2

def get_tasks():
    """
    Generates the list of specific (R, B, WL) combinations to test.
    Strategy: One-Factor-At-A-Time (OFAT) around the default pivot.
    """
    # 1. Define Ranges
    radii_range = [1, 2, 3, 4, 5, 6, 7]
    bins_range = [2, 3, 4, 5, 10, 20]
    wl_range = [1, 2, 3, 4, 5]

    tasks = []
    
    # Sweep 1: Vary Radius (Fix Bins & WL)
    for r in radii_range:
        tasks.append((r, DEF_BINS, DEF_WL))
        
    # Sweep 2: Vary Bins (Fix Radius & WL)
    for b in bins_range:
        # Avoid re-running the default case if already added
        if b != DEF_BINS: 
            tasks.append((DEF_RADIUS, b, DEF_WL))
            
    # Sweep 3: Vary WL Iterations (Fix Radius & Bins)
    for wl in wl_range:
        if wl != DEF_WL: 
            tasks.append((DEF_RADIUS, DEF_BINS, wl))
            
    # Optimization: Sort by Radius to minimize expensive graph extraction calls
    tasks.sort(key=lambda x: x[0])
    return tasks

def run_tuning_sweep():
    print("========================================")
    print("      PARAMETER TUNING (OFAT)           ")
    print(f"      Pivot: R={DEF_RADIUS}, B={DEF_BINS}, WL={DEF_WL}")
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

    tasks = get_tasks()
    print(f"Scheduled {len(tasks)} specific experiments.")
    
    results = []
    current_radius = -1
    raw_samples_all = None

    # Loop through our specific tasks
    for i, (radius, n_bins, wl_iter) in enumerate(tasks):
        
        # --- OPTIMIZATION: Only extract samples if Radius changes ---
        if radius != current_radius:
            print(f"\n[{i+1}/{len(tasks)}] Changing Radius to {radius}... Extracting Graphs...")
            current_radius = radius
            raw_samples_all = extract_raw_samples(graphs, metadata, radius=radius)
            
            if MAX_SAMPLES and MAX_SAMPLES < len(raw_samples_all):
                raw_samples_all = raw_samples_all[:MAX_SAMPLES]
        
        print(f"   Running: R={radius}, Bins={n_bins}, WL={wl_iter}...", end=" ")

        # --- Standard Pipeline ---
        groups_all = [s['group'] for s in raw_samples_all]
        dummy_y = [s['label'] for s in raw_samples_all]
        
        # CV Split (Stratified Group)
        try:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            train_idx, test_idx = next(sgkf.split(raw_samples_all, dummy_y, groups_all))
        except ValueError:
            print("CV Error (too few samples).")
            continue
        
        raw_train = [raw_samples_all[i] for i in train_idx]
        raw_test = [raw_samples_all[i] for i in test_idx]
        
        # Discretize
        # Note: Bins=10 or 20 might fail if dataset is tiny (empty bins), but usually fine.
        X_train, y_train_list, _, fitted_disc = to_grakel(raw_train, discretizer=None, n_bins=n_bins)
        X_test, y_test_list, _, _ = to_grakel(raw_test, discretizer=fitted_disc)
        y_train = np.array(y_train_list)
        y_test = np.array(y_test_list)
        
        # Kernel Calculation (GraKeL parallelized)
        gk = get_kernel('WL-OA', n_iter=wl_iter) 
        K_train = gk.fit_transform(X_train)
        K_test = gk.transform(X_test)
        
        # SVM Training
        clf = SVC(kernel='precomputed', C=SVM_C, class_weight='balanced')
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        
        # --- METRICS COMPUTATION (ALL 4) ---
        known_classes = np.unique(y_train)
        
        # 1. Accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # 2. Balanced Accuracy
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        # 3. F1 Score (Weighted)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=known_classes)
        
        # 4. Sensitivity (Recall Macro)
        sens = recall_score(y_test, y_pred, average='macro', zero_division=0, labels=known_classes)
        
        # Print all 4 live
        print(f"-> Acc: {acc:.3f} | Bal: {bal_acc:.3f} | F1: {f1:.3f} | Sens: {sens:.3f}")

        results.append({
            'Neighbor Radius': radius,
            'Bins': n_bins,
            'WL Iterations': wl_iter,
            'Accuracy': acc,
            'Balanced Accuracy': bal_acc,
            'F1 Score': f1,
            'Sensitivity': sens
        })
        
        # --- INCREMENTAL SAVE ---
        pd.DataFrame(results).to_csv('tuning_results.csv', index=False)

    # Plotting Logic (Updated for OFAT)
    print("\nRun complete. Generating plots...")
    df = pd.DataFrame(results)
    plot_ofat_results(df)

def plot_ofat_results(df):
    Path('figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Radius Plot: Filter where Bins & WL are at Default
    df_rad = df[(df['Bins'] == DEF_BINS) & (df['WL Iterations'] == DEF_WL)]
    if not df_rad.empty:
        sns.lineplot(data=df_rad, x='Neighbor Radius', y='Balanced Accuracy', marker='o', ax=axes[0], color='navy')
        axes[0].set_title(f"Impact of Radius\n(Fixed B={DEF_BINS}, WL={DEF_WL})")
        axes[0].grid(True, linestyle='--')

    # 2. Bins Plot: Filter where Radius & WL are at Default
    df_bins = df[(df['Neighbor Radius'] == DEF_RADIUS) & (df['WL Iterations'] == DEF_WL)]
    if not df_bins.empty:
        sns.lineplot(data=df_bins, x='Bins', y='Balanced Accuracy', marker='s', ax=axes[1], color='darkgreen')
        axes[1].set_title(f"Impact of Bins\n(Fixed R={DEF_RADIUS}, WL={DEF_WL})")
        axes[1].xaxis.get_major_locator().set_params(integer=True)
        axes[1].grid(True, linestyle='--')

    # 3. WL Plot: Filter where Radius & Bins are at Default
    df_wl = df[(df['Neighbor Radius'] == DEF_RADIUS) & (df['Bins'] == DEF_BINS)]
    if not df_wl.empty:
        sns.lineplot(data=df_wl, x='WL Iterations', y='Balanced Accuracy', marker='^', ax=axes[2], color='darkred')
        axes[2].set_title(f"Impact of WL Depth\n(Fixed R={DEF_RADIUS}, B={DEF_BINS})")
        axes[2].xaxis.get_major_locator().set_params(integer=True)
        axes[2].grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig('figures/fig_parameter_tuning_3panel.png', dpi=300)
    print("Saved 'figures/fig_parameter_tuning_3panel.png'")
    plt.show()

if __name__ == "__main__":
    run_tuning_sweep()