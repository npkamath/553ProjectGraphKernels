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
    radii_to_test = [1,2,3,4, 5, 6, 7]      
    bins_to_test = [2, 3, 4, 5]       
    wl_iters_to_test = [1, 2, 3, 4,5]   

    results = []
    
    # Track the best model for final visualization
    best_score = -1.0
    best_cm = None
    best_params = {}

    # 2. Outer Loop: Structure Parameters (Radius)
    # This is the most expensive loop (Graph Extraction), so we do it first/least often.
    for radius in radii_to_test:
        print(f"\n\n=== [Structure Group] Radius={radius} ===")
        
        # Extract raw physics samples (geometry)
        raw_samples_all = extract_raw_samples(graphs, metadata, radius=radius)
        
        # 3. Middle Loop: Discretization (Bins)
        for n_bins in bins_to_test:
            print(f"   > Processing Bins={n_bins}...")
            
            # --- STRATIFIED GROUP SPLIT ---
            # We must split raw samples to ensure Discretizer is fit ONLY on Train
            groups_all = [s['group'] for s in raw_samples_all]
            dummy_y = [s['label'] for s in raw_samples_all]
            
            # Use StratifiedGroupKFold (n_splits=5 means 20% test set)
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            train_idx, test_idx = next(sgkf.split(raw_samples_all, dummy_y, groups_all))
            
            raw_train = [raw_samples_all[i] for i in train_idx]
            raw_test = [raw_samples_all[i] for i in test_idx]
            
            # Fit Discretizer on Train, Transform Test
            X_train, y_train_list, _, fitted_disc = to_grakel(raw_train, discretizer=None, n_bins=n_bins)
            X_test, y_test_list, _, _ = to_grakel(raw_test, discretizer=fitted_disc)
            
            y_train = np.array(y_train_list)
            y_test = np.array(y_test_list)

            # 4. Inner Loop: Kernel Parameters
            for wl_iter in wl_iters_to_test:
                print(f"     >>> WL_Iter={wl_iter}...", end=" ")
                
                # Compute Kernel
                # Note: N_JOBS=-1 ensures GraKeL uses all cores for the matrix calculation
                gk = get_kernel('WL-OA', n_iter=wl_iter) 
                
                # IMPORTANT: GraKeL parallelization happens here
                K_train = gk.fit_transform(X_train)
                K_test = gk.transform(X_test)
                
                # Train SVM 
                clf = SVC(kernel='precomputed', C=SVM_C, class_weight='balanced')
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
                
                # Metrics
                known_classes = np.unique(y_train)
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                # Pass labels=known_classes to prevent warnings if Test set misses a rare class
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=known_classes)
                sens = recall_score(y_test, y_pred, average='macro', zero_division=0, labels=known_classes)
                
                # UPDATED PRINT STATEMENT
                print(f"-> Acc: {acc:.4f} | Bal Acc: {bal_acc:.4f} | F1: {f1:.4f} | Sens: {sens:.4f}")

                # Store results
                results.append({
                    'Neighbor Radius': radius,
                    'Bins': n_bins,
                    'WL Iterations': wl_iter,
                    'Accuracy': acc,
                    'Balanced Accuracy': bal_acc,
                    'F1 Score': f1,
                    'Sensitivity': sens
                })
                
                # Check for best model
                if bal_acc > best_score:
                    best_score = bal_acc
                    cm = confusion_matrix(y_test, y_pred, labels=known_classes)
                    best_cm = pd.DataFrame(cm, index=known_classes, columns=known_classes)
                    best_params = {'Radius': radius, 'Bins': n_bins, 'WL': wl_iter}

    # 5. Display & Save Results
    df = pd.DataFrame(results)
    
    print("\n========================================")
    print(f"   BEST CONFIGURATION (Bal Acc: {best_score:.4f})")
    print(f"   {best_params}")
    print("========================================")
    
    df.to_csv('tuning_results.csv', index=False)
    print("Results saved to 'tuning_results.csv'")

    plot_results(df, best_cm, best_params)

def plot_results(df, best_cm, best_params):
    """
    Plots tuning curves for Radius, Bins, and WL Iterations, 
    plus the confusion matrix of the best model.
    """
    Path('figures').mkdir(exist_ok=True)
    
    # Setup the figure for 3 side-by-side tuning plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Impact of Neighbor Radius
    sns.lineplot(data=df, x='Neighbor Radius', y='Balanced Accuracy', marker='o', ax=axes[0], color='navy')
    axes[0].set_title("Impact of Neighbor Radius")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Impact of Number of Bins
    sns.lineplot(data=df, x='Bins', y='Balanced Accuracy', marker='s', ax=axes[1], color='darkgreen')
    axes[1].set_title("Impact of Discretization Bins")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].xaxis.get_major_locator().set_params(integer=True) # Ensure integer ticks
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Impact of WL Iterations
    sns.lineplot(data=df, x='WL Iterations', y='Balanced Accuracy', marker='^', ax=axes[2], color='darkred')
    axes[2].set_title("Impact of WL Kernel Depth")
    axes[2].set_ylabel("Balanced Accuracy")
    axes[2].xaxis.get_major_locator().set_params(integer=True) # Ensure integer ticks
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('figures/fig_parameter_tuning_3panel.png', dpi=300)
    print("Tuning panel saved to 'figures/fig_parameter_tuning_3panel.png'")
    
    # --- Plot 4: Best Confusion Matrix ---
    if best_cm is not None:
        plt.figure(figsize=(7, 6))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Best Model CM\n(R={best_params['Radius']}, B={best_params['Bins']}, WL={best_params['WL']})")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('figures/fig_best_confusion_matrix.png', dpi=300)
        print("Confusion Matrix saved to 'figures/fig_best_confusion_matrix.png'")
    
    plt.show()

if __name__ == "__main__":
    run_tuning_sweep()