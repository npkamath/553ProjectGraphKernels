import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score, 
    roc_auc_score
)
from sklearn.svm import SVC

# Ensure src is in path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH, RANDOM_SEED, SVM_C
from src.processing import extract_raw_samples, to_grakel
from src.kernels import get_kernel
from src.custom_model import CustomKernelSVM

def run_model_comparison():
    print("========================================")
    print("       MODEL COMPARISON TOURNAMENT       ")
    print("========================================")

    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # 2. Process Data (Leakage-Proof)
    OPTIMAL_BINS = 2
    OPTIMAL_RADIUS = 6
    
    print(f"Extracting Raw Samples (Radius={OPTIMAL_RADIUS})...")
    raw_samples_all = extract_raw_samples(data['graphs'], data['metadata'], radius=OPTIMAL_RADIUS)
    
    # Split
    print("Splitting Data (Stratified Group)...")
    groups_all = [s['group'] for s in raw_samples_all]
    dummy_y = [s['label'] for s in raw_samples_all]
    
    # FIX: Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf.split(raw_samples_all, dummy_y, groups_all))
    
    raw_train = [raw_samples_all[i] for i in train_idx]
    raw_test = [raw_samples_all[i] for i in test_idx]
    
    # Discretize (Fit on Train, Transform Test)
    print(f"Discretizing (Bins={OPTIMAL_BINS})...")
    X_train, y_train_list, _, fitted_disc = to_grakel(raw_train, discretizer=None, n_bins=OPTIMAL_BINS)
    X_test, y_test_list, _, _ = to_grakel(raw_test, discretizer=fitted_disc)
    
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    
    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")

    # 3. Define the Contenders
    experiments = [
        # (Kernel Name, SVM Type, Display Name)
        ('WL-OA',        'sklearn', 'GraKeL Kernel + Sklearn SVM (Baseline)'),
        ('CUSTOM-WL-OA', 'sklearn', 'Custom Kernel + Sklearn SVM'),
        ('WL-OA',        'custom',  'GraKeL Kernel + Custom SVM'),
        ('CUSTOM-WL-OA', 'custom',  'Custom Kernel + Custom SVM (Full Custom)')
    ]

    results = []

    # 4. Run the Tournament
    for kernel_name, svm_type, name in experiments:
        print(f"\n--- Testing: {name} ---")
        try:
            # A. Compute Kernel
            print("  > Computing Kernel...")
            gk = get_kernel(kernel_name, n_iter=2)
            K_train = gk.fit_transform(X_train)
            K_test = gk.transform(X_test)
            
            # B. Train Model
            clf = None
            print(f"  > Training SVM ({svm_type})...")
            if svm_type == 'custom':
                clf = CustomKernelSVM(C=SVM_C)
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
            else:
                clf = SVC(kernel='precomputed', C=SVM_C, class_weight='balanced', decision_function_shape='ovr')
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
            
            # C. Metrics
            # Define known classes to prevent warnings
            known_classes = np.unique(y_train)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=known_classes)
            sens = recall_score(y_test, y_pred, average='macro', zero_division=0, labels=known_classes)
            
            # Try AUC
            auc = "N/A"
            if hasattr(clf, "decision_function"):
                try:
                    y_scores = clf.decision_function(K_test)
                    if len(np.unique(y_train)) > 2:
                        auc = roc_auc_score(y_test, y_scores, multi_class='ovr', average='weighted')
                    else:
                        auc = roc_auc_score(y_test, y_scores)
                except:
                    pass

            entry = {
                'Name': name, 
                'Status': 'Success',
                'Accuracy': acc,
                'Balanced Accuracy': bal_acc,
                'F1 Score': f1,
                'Sensitivity': sens,
                'AUC': auc
            }
            results.append(entry)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'Name': name, 
                'Status': 'Failed',
                'Accuracy': 0.0, 'Balanced Accuracy': 0.0, 'F1 Score': 0.0, 'Sensitivity': 0.0
            })

    # 5. Show Final Standings
    df = pd.DataFrame(results)
    
    cols = ['Name', 'Accuracy', 'Balanced Accuracy', 'F1 Score', 'Sensitivity', 'AUC', 'Status']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n========================================")
    print("           TOURNAMENT RESULTS           ")
    print("========================================")
    print(df.sort_values(by='Accuracy', ascending=False))
    
    df.to_csv('model_comparison_results.csv', index=False)
    print("\nFull results saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    run_model_comparison()