from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import pickle
from processing import process_graphs
from kernels import get_graph_kernel
from config import RAW_DATA_PATH

def main():
    with open(RAW_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    graphs, labels = process_graphs(data["graphs"], data["metadata"])

    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, stratify=labels, test_size=0.2, random_state=42
    )

    gk = get_graph_kernel()

    param_grid = {
        "C": [0.1, 1, 10],
        "wl_iterations": [1, 2, 3],
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

    grid = GridSearchCV(
        KWrapper(), param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)

if __name__ == "__main__":
    main()
