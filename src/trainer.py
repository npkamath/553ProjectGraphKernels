import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score, 
    roc_auc_score
)
import pandas as pd
from .kernels import get_kernel
from .custom_model import CustomKernelSVM
from .config import KERNEL_TYPE, SVM_C, RANDOM_SEED, WL_ITERATIONS, SVM_IMPLEMENTATION
from .processing import extract_raw_samples, to_grakel

def train_pipeline(graphs, metadata, 
                   n_iter=None, 
                   kernel_type=None, 
                   svm_impl=None, 
                   verbose=True):
    """
    Executes the training and evaluation pipeline with Strict Leakage Prevention.
    """
    # Resolve actual parameters using overrides
    actual_iter = n_iter if n_iter is not None else WL_ITERATIONS
    actual_kernel = kernel_type if kernel_type is not None else KERNEL_TYPE
    actual_svm = svm_impl if svm_impl is not None else SVM_IMPLEMENTATION

    if verbose: 
        print(f"Training with Kernel={actual_kernel}, SVM={actual_svm}, Iterations={actual_iter}")
    
    # --- 1. STRATIFIED GROUP SPLIT ON RAW SIMULATIONS ---
    n_sims = len(graphs)
    dummy_y = [m['crystal_type'] for m in metadata]
    groups = np.arange(n_sims) # Each simulation is its own group
    
    # FIX: Use StratifiedGroupKFold to ensure classes are balanced
    # n_splits=5 gives us a 20% test set (1/5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf.split(graphs, dummy_y, groups))
    
    if verbose: 
        print(f"Split: {len(train_idx)} Train Sims vs {len(test_idx)} Test Sims")

    # Helper to subset lists
    train_graphs = [graphs[i] for i in train_idx]
    train_meta = [metadata[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_meta = [metadata[i] for i in test_idx]

    # --- 2. EXTRACT RAW SAMPLES ---
    if verbose: print("Extracting raw samples (Ego Graphs + Raw Features)...")
    raw_train = extract_raw_samples(train_graphs, train_meta)
    raw_test = extract_raw_samples(test_graphs, test_meta)

    # --- 3. DISCRETIZE (PREVENT LEAKAGE) ---
    if verbose: print("Discretizing features...")
    
    # A. FIT on Train
    X_train, y_train_list, _, fitted_disc = to_grakel(raw_train, discretizer=None)
    
    # B. TRANSFORM Test (using fitted discretizer)
    X_test, y_test_list, _, _ = to_grakel(raw_test, discretizer=fitted_disc)
    
    # Convert labels to numpy for metric calculation
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    if verbose:
        print(f"Processed Samples -> Train: {len(X_train)} | Test: {len(X_test)}")

    # --- 4. COMPUTE KERNELS ---
    gk = get_kernel(actual_kernel, n_iter=actual_iter)
    
    if verbose: print("Computing Kernel Matrix...")
    K_train = gk.fit_transform(X_train)
    K_test = gk.transform(X_test)
    
    # --- 5. MODEL TRAINING ---
    clf = None
    if actual_svm == 'custom':
        if verbose: print(f"Fitting CUSTOM SVM (SMO Algorithm, C={SVM_C})...")
        clf = CustomKernelSVM(C=SVM_C)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        
    else:
        if verbose: print(f"Fitting SKLEARN SVM (C={SVM_C}, Weights='balanced')...")
        clf = SVC(kernel='precomputed', C=SVM_C, class_weight='balanced', decision_function_shape='ovo')
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
    
    # --- 6. CALCULATE METRICS ---
    # Define known classes to prevent warnings if Test set is missing a rare class
    known_classes = np.unique(y_train)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=known_classes)
    sensitivity = recall_score(y_test, y_pred, average='macro', zero_division=0, labels=known_classes)
    
    # Try AUC
    auc = "N/A"
    try:
        if hasattr(clf, "decision_function"):
            y_scores = clf.decision_function(K_test)
            if len(np.unique(y_train)) > 2:
                auc = roc_auc_score(y_test, y_scores, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, y_scores)
    except Exception:
        pass 

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=known_classes)
    cm_df = pd.DataFrame(cm, index=known_classes, columns=known_classes)

    if verbose:
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"F1 Score (W):      {f1:.4f}")
        print(f"Sensitivity (Mac): {sensitivity:.4f}")
        print(f"AUC:               {auc}")
        print("\n--- CONFUSION MATRIX ---")
        print(cm_df)
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred, zero_division=0, labels=known_classes))
        
    return {
        'Accuracy': acc,
        'Balanced Accuracy': bal_acc,
        'F1 Score': f1,
        'Sensitivity': sensitivity,
        'AUC': auc,
        'Confusion Matrix': cm_df
    }