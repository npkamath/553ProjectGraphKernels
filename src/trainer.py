import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from .kernels import get_kernel
from .config import KERNEL_TYPE, SVM_C, RANDOM_SEED

def train_pipeline(subgraphs, labels, groups, n_iter=None, verbose=True):
    """
    Executes the training and evaluation pipeline using GroupShuffleSplit to prevent leakage.
    
    Args:
        subgraphs (list): List of grakel.Graph objects.
        labels (list): List of target strings ('fcc', 'bcc', etc.).
        groups (list): List of parent simulation IDs corresponding to each subgraph.
                       This ensures that all subgraphs from the same parent stay together.
        n_iter (int): Optional override for WL kernel iterations.
        verbose (bool): Whether to print progress and reports to stdout.
        
    Returns:
        float: The accuracy score on the test set.
    """
    if verbose: print(f"Training with Kernel={KERNEL_TYPE}, Iterations={n_iter}")
    
    # Convert to numpy arrays for advanced indexing
    subgraphs_arr = np.array(subgraphs, dtype=object)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    # --- LEAKAGE PREVENTION ---
    # GroupShuffleSplit splits based on 'groups', not individual samples.
    # If Simulation #42 is in Train, ALL its subgraphs are in Train.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(subgraphs_arr, labels_arr, groups_arr))
    
    X_train = subgraphs_arr[train_idx]
    X_test = subgraphs_arr[test_idx]
    y_train = labels_arr[train_idx]
    y_test = labels_arr[test_idx]

    if verbose:
        n_train_sims = len(np.unique(groups_arr[train_idx]))
        n_test_sims = len(np.unique(groups_arr[test_idx]))
        print(f"Train size: {len(X_train)} subgraphs (from {n_train_sims} distinct simulations)")
        print(f"Test size:  {len(X_test)} subgraphs (from {n_test_sims} distinct simulations)")

    # Initialize Kernel
    gk = get_kernel(KERNEL_TYPE, n_iter=n_iter)
    
    # Compute Gram Matrices
    if verbose: print("Computing Training Kernel Matrix...")
    K_train = gk.fit_transform(X_train)
    
    if verbose: print("Computing Test Kernel Matrix...")
    K_test = gk.transform(X_test)
    
    # Train SVM
    if verbose: print(f"Fitting SVM Classifier (C={SVM_C})...")
    # Custom class weights to penalize missing a Crystal more than missing Disordered
    # (Adjust keys 'sc', 'fcc' etc. to match your exact label strings)
    weights = {'Disordered': 1, 'fcc': 1, 'bcc': 1, 'hcp': 1, 'sc': 1}
    clf = SVC(kernel='precomputed', C=SVM_C, class_weight=weights)
    clf.fit(K_train, y_train)
    
    # Predict & Evaluate
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred))
        
    return acc