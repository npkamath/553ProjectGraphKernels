# src/trainer.py
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from grakel.kernels import WeisfeilerLehmanOptimalAssignment
from .config import WL_ITERATIONS, SVM_C, RANDOM_SEED, N_JOBS

def train_pipeline(subgraphs, labels):
    print(f"\n--- TRAINING START ---")
    print(f"Total samples: {len(subgraphs)}")
    
    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        subgraphs, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=RANDOM_SEED
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 2. Initialize WL-OA Kernel
    # n_jobs=N_JOBS enables parallel processing (multicore)
    print(f"Initializing WL-OA Kernel (n_iter={WL_ITERATIONS}, n_jobs={N_JOBS})...")
    gk = WeisfeilerLehmanOptimalAssignment(n_iter=WL_ITERATIONS, normalize=True, n_jobs=N_JOBS)
    
    # 3. Compute Gram Matrices
    print("Computing Training Kernel Matrix (this uses all cores)...")
    K_train = gk.fit_transform(X_train)
    
    print("Computing Test Kernel Matrix...")
    K_test = gk.transform(X_test)
    
    # 4. Train SVM
    print("Fitting SVM Classifier...")
    # class_weight='balanced' handles any imbalance between crystal types
    clf = SVC(kernel='precomputed', C=SVM_C, class_weight='balanced')
    clf.fit(K_train, y_train)
    
    # 5. Evaluation
    print("Predicting on Test Set...")
    y_pred = clf.predict(K_test)
    
    print("\n--- RESULTS ---")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))