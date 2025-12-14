from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
import pickle
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent))

from src.processing import extract_raw_samples, to_grakel
from src.kernels import get_kernel
from src.config import RAW_DATA_PATH, RANDOM_SEED, KERNEL_TYPE

def main():
    with open(RAW_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    # 1. Split RAW data first (Train/Test)
    # We use optimal radius/bins here as defaults for the 'inner' SVM tuning
    RADIUS = 6
    BINS = 2
    
    print("Extracting raw samples...")
    raw_samples_all = extract_raw_samples(data["graphs"], data["metadata"], radius=RADIUS)
    
    groups_all = np.array([s['group'] for s in raw_samples_all])
    dummy_y = np.array([s['label'] for s in raw_samples_all])
    
    # Outer Split
    # FIX: Use StratifiedGroupKFold instead of GroupShuffleSplit
    # n_splits=5 gives us an 80/20 split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf.split(raw_samples_all, dummy_y, groups_all))
    
    raw_train = [raw_samples_all[i] for i in train_idx]
    
    # 2. Process Training Data Only (Fit Discretizer)
    print("Discretizing training set...")
    X_train, y_train_list, groups_train, _ = to_grakel(raw_train, discretizer=None, n_bins=BINS)
    
    X_train = np.array(X_train, dtype=object)
    y_train = np.array(y_train_list)
    groups_train = np.array(groups_train)

    # 3. Define Grid Search Wrapper
    # This wrapper computes the kernel matrix on the fly for the CV folds
    class KWrapper:
        def __init__(self, C=1.0, wl_iterations=2):
            self.C = C
            self.wl_iterations = wl_iterations
            self.model = None
            self.K_train = None
            self.gk = None

        def fit(self, X, y):
            # Re-initialize kernel with current param
            self.gk = get_kernel(KERNEL_TYPE, n_iter=self.wl_iterations)
            self.K_train = self.gk.fit_transform(X)

            # Use balanced weighting for tuning
            weights = {'Disordered': 1, 'fcc': 2, 'bcc': 2, 'hcp': 2, 'sc': 2}
            self.model = SVC(kernel="precomputed", C=self.C, class_weight=weights)
            self.model.fit(self.K_train, y)
            return self

        def predict(self, X):
            K_test = self.gk.transform(X)
            return self.model.predict(K_test)
        
        def get_params(self, deep=True):
            return {"C": self.C, "wl_iterations": self.wl_iterations}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    # 4. Run Grid Search
    # Use StratifiedGroupKFold to prevent leakage during CV
    print("Running GridSearchCV...")
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "wl_iterations": [2] 
    }

    cv_strategy = StratifiedGroupKFold(n_splits=3)
    
    grid = GridSearchCV(
        KWrapper(), 
        param_grid, 
        cv=cv_strategy, 
        scoring="accuracy", 
        n_jobs=-1
    )
    
    # Pass groups to .fit() for the CV splitter
    grid.fit(X_train, y_train, groups=groups_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV score: {grid.best_score_:.4f}\n")

    print("=== Detailed results for all parameter combinations ===")
    for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                 grid.cv_results_["std_test_score"],
                                 grid.cv_results_["params"]):
        print(f"Params={params}: mean accuracy={mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()