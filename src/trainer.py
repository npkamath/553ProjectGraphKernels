import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from .kernels import get_kernel
from .custom_model import CustomKernelSVM
from .config import KERNEL_TYPE, SVM_C, RANDOM_SEED, WL_ITERATIONS, SVM_IMPLEMENTATION

def train_pipeline(subgraphs, labels, groups, n_iter=None, verbose=True):
    """
    Executes the training and evaluation pipeline using StratifiedGroupKFold.
    
    This function handles:
    1. Splitting data into Train/Test while ensuring NO DATA LEAKAGE (Group Split).
    2. Computing the Kernel Matrix (Gram Matrix) using either GraKeL or Custom Logic.
    3. Training an SVM classifier (Standard Sklearn or Custom SMO) based on config.
    4. Evaluating accuracy and precision/recall.
    
    Args:
        subgraphs (list): List of grakel.Graph objects (the inputs).
        labels (list): List of target strings (e.g., 'fcc', 'bcc').
        groups (list): List of parent simulation IDs. 
                       CRITICAL: This ensures that subgraphs from the same 
                       simulation file are kept together (all in train OR all in test).
        n_iter (int): Optional override for WL kernel iterations (depth).
        verbose (bool): Whether to print progress logs.
        
    Returns:
        float: The accuracy score on the test set.
    """
    actual_iter = n_iter if n_iter is not None else WL_ITERATIONS

    if verbose: 
        print(f"Training with Kernel={KERNEL_TYPE}, Iterations={actual_iter}")
        print(f"SVM Implementation: {SVM_IMPLEMENTATION}")
    
    # Convert lists to numpy arrays for advanced indexing
    subgraphs_arr = np.array(subgraphs, dtype=object)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    # --- STRATIFIED GROUP SPLIT ---
    # StratifiedGroupKFold ensures no leakage + balanced classes
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf.split(subgraphs_arr, labels_arr, groups_arr))
    
    X_train = subgraphs_arr[train_idx]
    X_test = subgraphs_arr[test_idx]
    y_train = labels_arr[train_idx]
    y_test = labels_arr[test_idx]

    if verbose:
        n_train_sims = len(np.unique(groups_arr[train_idx]))
        n_test_sims = len(np.unique(groups_arr[test_idx]))
        print(f"Train size: {len(X_train)} samples ({n_train_sims} simulations)")
        print(f"Test size:  {len(X_test)} samples ({n_test_sims} simulations)")

    # 1. Compute Kernels (Uses Factory)
    gk = get_kernel(KERNEL_TYPE, n_iter=actual_iter)
    
    if verbose: print("Computing Training Kernel Matrix...")
    K_train = gk.fit_transform(X_train)
    
    if verbose: print("Computing Test Kernel Matrix...")
    # NOTE: Custom SVM needs the full Test x Train matrix to look up support vectors
    # K_test will be shape (n_test, n_train)
    K_test = gk.transform(X_test)
    
    # 2. Select & Train Model
    if SVM_IMPLEMENTATION == 'custom':
        if verbose: print(f"Fitting CUSTOM SVM (SMO Algorithm, C={SVM_C})...")
        clf = CustomKernelSVM(C=SVM_C)
        clf.fit(K_train, y_train)
        
        # Predict needs full kernel matrix to project test points onto support vectors
        y_pred = clf.predict(K_test)
        
    else:
        if verbose: print(f"Fitting SKLEARN SVM (C={SVM_C})...")
        # Standard Sklearn with class balancing
        # Note: Weights should match your labels exactly (e.g. 'fcc' vs 'FCC')
        weights = {'Disordered': 1, 'fcc': 5, 'bcc': 5, 'hcp': 5, 'sc': 5}
        clf = SVC(kernel='precomputed', C=SVM_C, class_weight=weights)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
    return acc