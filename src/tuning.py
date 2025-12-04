from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
import pickle
import numpy as np
from processing import process_graphs
from kernels import get_kernel
from config import RAW_DATA_PATH, RANDOM_SEED, KERNEL_TYPE

def main():
    with open(RAW_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    subgraphs, labels, groups = process_graphs(data["graphs"], data["metadata"])

    subgraphs_arr = np.array(subgraphs, dtype=object)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(subgraphs_arr, labels_arr, groups_arr))
    
    X_train = subgraphs_arr[train_idx]
    X_test = subgraphs_arr[test_idx]
    y_train = labels_arr[train_idx]
    y_test = labels_arr[test_idx]

    gk = get_kernel(KERNEL_TYPE, n_iter=2)

    param_grid = {
        "C": [0.01, 0.01, 1, 10]
    }

    class KWrapper:
        def __init__(self, C=1.0, wl_iterations=2):
            self.C = C
            self.wl_iterations = wl_iterations

        def fit(self, X, y):
            gk.wl_iterations = self.wl_iterations
            self.K_train = gk.fit_transform(X)
            self.model = SVC(kernel="precomputed", C=self.C)
            self.model.fit(self.K_train, y)
            return self

        def predict(self, X):
            K_test = gk.transform(X)
            return self.model.predict(K_test)
        
        def get_params(self, deep=True):
            return {"C": self.C, "wl_iterations": self.wl_iterations}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    grid = GridSearchCV(
        KWrapper(), param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV score: {grid.best_score_:.4f}\n")

    print("=== Detailed results for all parameter combinations ===")
    for mean, std, params in zip(grid.cv_results_["mean_test_score"],
                                 grid.cv_results_["std_test_score"],
                                 grid.cv_results_["params"]):
        print(f"C={params['C']}: mean accuracy={mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
