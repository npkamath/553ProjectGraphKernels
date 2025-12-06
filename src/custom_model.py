import numpy as np
import random

class BinarySMO:
    """
    Binary SVM trained via Simplified Sequential Minimal Optimization (SMO).
    
    This implements Platt's algorithm (1998) to solve the SVM Quadratic Programming 
    problem without using a numerical optimization library.
    
    Objective:
    Maximize: sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
    Subject to: 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0
    """
    def __init__(self, C=1.0, tol=0.01, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.support_vectors_idx = None
        self.support_labels = None
        self.support_alphas = None

    def fit(self, K, y):
        """
        Trains the binary SVM on a precomputed kernel matrix K.
        y must contain labels {-1, 1}.
        """
        n_samples = len(y)
        self.alphas = np.zeros(n_samples)
        self.b = 0
        passes = 0
        
        # The SMO Optimization Loop
        # Iterates until alphas converge (KKT conditions met)
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Calculate margin error for point i
                f_i = np.dot(self.alphas * y, K[i]) + self.b
                E_i = f_i - y[i]

                # Check KKT conditions (if violated, we optimize)
                if ((y[i] * E_i < -self.tol and self.alphas[i] < self.C) or 
                    (y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Select a second alpha j randomly to optimize jointly
                    j = i
                    while j == i:
                        j = random.randint(0, n_samples - 1)
                    
                    f_j = np.dot(self.alphas * y, K[j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Compute optimization bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H: continue

                    # Compute 2nd derivative (eta)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: continue

                    # Update alpha_j
                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    if self.alphas[j] > H: self.alphas[j] = H
                    elif self.alphas[j] < L: self.alphas[j] = L

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5: continue

                    # Update alpha_i based on change in alpha_j
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    # Update threshold b
                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C: self.b = b1
                    elif 0 < self.alphas[j] < self.C: self.b = b2
                    else: self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0: passes += 1
            else: passes = 0
        
        # Save Support Vectors (where alpha > 0) to save compute later
        sv_indices = self.alphas > 1e-5
        self.support_vectors_idx = np.where(sv_indices)[0]
        self.support_alphas = self.alphas[sv_indices]
        self.support_labels = y[sv_indices]

    def predict_decision(self, K_test_subset):
        """Returns distance from hyperplane. K_test_subset is (n_test, n_support_vectors)."""
        return np.dot(self.support_alphas * self.support_labels, K_test_subset.T) + self.b

class CustomKernelSVM:
    """
    Multiclass SVM Wrapper implementing One-vs-Rest strategy.
    
    Since SMO is inherently binary, we train one binary classifier per class
    (e.g., FCC vs Rest) and predict the class with the highest confidence score.
    """
    def __init__(self, C=1.0):
        self.C = C
        self.models = []
        self.classes = []

    def fit(self, K_train, y_train):
        self.classes = np.unique(y_train)
        print(f"  > Custom SVM: Training One-vs-Rest for {len(self.classes)} classes...")
        self.models = []
        for cls in self.classes:
            # Binary target: 1 for current class, -1 for others
            y_binary = np.where(y_train == cls, 1, -1)
            
            model = BinarySMO(C=self.C)
            model.fit(K_train, y_binary)
            self.models.append(model)
            
    def predict(self, K_test_full):
        """
        Predicts class based on highest confidence score.
        Args:
            K_test_full: The full (n_test, n_train) kernel matrix.
        """
        n_test = K_test_full.shape[0]
        scores = np.zeros((n_test, len(self.classes)))
        
        for i, model in enumerate(self.models):
            # Extract columns for support vectors
            sv_idx = model.support_vectors_idx
            if len(sv_idx) == 0:
                scores[:, i] = -9999
            else:
                K_subset = K_test_full[:, sv_idx]
                scores[:, i] = model.predict_decision(K_subset)
            
        predictions_idx = np.argmax(scores, axis=1)
        return self.classes[predictions_idx]