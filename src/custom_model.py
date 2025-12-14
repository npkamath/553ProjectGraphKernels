import numpy as np
import random
# Using this Scikit-Learn utility to compute weights automatically, 
# mimicking the 'balanced' option in a standard library model.
from sklearn.utils.class_weight import compute_class_weight 

class BinarySMO:
    """
    Binary SVM trained via Simplified Sequential Minimal Optimization (SMO).
    
    This implements Platt's algorithm (1998) to solve the SVM Quadratic Programming 
    problem, supporting custom regularization bounds (C_i) for class balancing.
    
    Objective:
    Maximize: sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
    Subject to: 0 <= alpha_i <= C_i, sum(alpha_i * y_i) = 0
    """
    def __init__(self, C=1.0, tol=0.01, max_passes=5, class_weight=None):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        # class_weight is a dictionary {1: w_pos, -1: w_neg} used to adjust C
        self.class_weight = class_weight 
        self.alphas = None
        self.b = 0
        self.support_vectors_idx = None
        self.support_labels = None
        self.support_alphas = None

    def fit(self, K, y):
        """
        Trains the binary SVM on a precomputed kernel matrix K.
        y must contain labels {-1, 1}.
        
        Args:
            K (np.array): The training kernel matrix (K_train x K_train).
            y (np.array): The binary target labels (+1 or -1).
        """
        n_samples = len(y)
        self.alphas = np.zeros(n_samples)
        self.b = 0
        passes = 0
        
        # --- CALCULATE C_i for every sample based on class_weight ---
        C_vector = np.full(n_samples, self.C)
        if self.class_weight is not None:
            for i in range(n_samples):
                # Use the weight corresponding to the sample's label (1 or -1)
                label = int(y[i])
                C_vector[i] = self.C * self.class_weight.get(label, 1.0) # Default weight is 1.0
        
        # The SMO Optimization Loop
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Calculate margin error for point i
                f_i = np.dot(self.alphas * y, K[i]) + self.b
                E_i = f_i - y[i]

                # Use sample-specific C_i for the upper bound check
                C_i = C_vector[i]

                # Check KKT conditions (if violated, we optimize)
                if ((y[i] * E_i < -self.tol and self.alphas[i] < C_i) or 
                    (y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Select a second alpha j randomly to optimize jointly
                    j = i
                    while j == i:
                        j = random.randint(0, n_samples - 1)
                    
                    f_j = np.dot(self.alphas * y, K[j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Use sample-specific C_j
                    C_j = C_vector[j]

                    # Compute optimization bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(C_j, C_i + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - C_i)
                        H = min(C_j, self.alphas[i] + self.alphas[j])

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

                    if 0 < self.alphas[i] < C_i: self.b = b1
                    elif 0 < self.alphas[j] < C_j: self.b = b2
                    else: self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0: passes += 1
            else: passes = 0
        
        # Save Support Vectors (where alpha > 0)
        sv_indices = self.alphas > 1e-5
        self.support_vectors_idx = np.where(sv_indices)[0]
        self.support_alphas = self.alphas[sv_indices]
        self.support_labels = y[sv_indices]

    def predict_decision(self, K_test_subset):
        """
        Calculates distance from hyperplane (raw confidence score).
        
        Args:
            K_test_subset (np.array): The kernel matrix between Test points and 
                                      this model's Support Vectors (shape: n_test x n_support_vectors).
                                      
        Returns:
            np.array: Decision scores.
        """
        return np.dot(self.support_alphas * self.support_labels, K_test_subset.T) + self.b

class CustomKernelSVM:
    """
    Multiclass SVM Wrapper implementing One-vs-Rest strategy with Balancing Logic.
    
    Manages the training and prediction of multiple BinarySMO models 
    for multiclass classification. This ensures rare classes are weighted equally 
    to the majority class, improving Sensitivity (Recall).
    """
    def __init__(self, C=1.0):
        self.C = C
        self.models = []
        self.classes = []

    def fit(self, K_train, y_train):
        """
        Trains one BinarySMO model for each class using dynamically computed weights.
        
        Args:
            K_train (np.array): Training kernel matrix.
            y_train (np.array): Training labels (e.g., ['fcc', 'bcc', 'fcc']).
        """
        self.classes = np.unique(y_train)
        print(f"  > Custom SVM: Training One-vs-Rest for {len(self.classes)} classes...")
        self.models = []
        
        # Calculate weights automatically (mimicking sklearn.utils.class_weight='balanced')
        # This gives a baseline weight for every unique true class label.
        weights_array = compute_class_weight('balanced', classes=self.classes, y=y_train)
        weight_dict_multiclass = dict(zip(self.classes, weights_array))
        
        print(f"    - Computed Multiclass Weights: {weight_dict_multiclass}")

        for cls in self.classes:
            # Binary target: 1 for current class, -1 for others
            y_binary = np.where(y_train == cls, 1, -1)
            
            # The balancing trick:
            # Positive weight (+1) is set to the importance of the rare class.
            # Negative weight (-1) is set to 1.0 (the baseline for the majority group).
            binary_weights = {
                1: weight_dict_multiclass[cls], 
                -1: 1.0
            }
            
            model = BinarySMO(C=self.C, class_weight=binary_weights)
            model.fit(K_train, y_binary)
            self.models.append(model)
            
    def predict(self, K_test_full):
        """
        Predicts class based on highest confidence score (max margin).
        
        Args:
            K_test_full (np.array): The full (n_test, n_train) kernel matrix.
            
        Returns:
            np.array: Predicted class labels.
        """
        n_test = K_test_full.shape[0]
        scores = np.zeros((n_test, len(self.classes)))
        
        for i, model in enumerate(self.models):
            # Extract columns corresponding to support vectors
            sv_idx = model.support_vectors_idx
            if len(sv_idx) == 0:
                scores[:, i] = -9999
            else:
                K_subset = K_test_full[:, sv_idx]
                scores[:, i] = model.predict_decision(K_subset)
            
        # The class with the highest score wins
        predictions_idx = np.argmax(scores, axis=1)
        return self.classes[predictions_idx]