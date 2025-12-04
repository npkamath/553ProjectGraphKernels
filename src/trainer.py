# src/trainer.py
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Import your new kernel factory and the config setting
from .kernels import get_kernel
from .config import KERNEL_TYPE, SVM_C, RANDOM_SEED

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
    
    # 2. Get Kernel from Factory
    gk = get_kernel(KERNEL_TYPE)
    
    # 3. Compute Gram Matrices
    print("Computing Training Kernel Matrix...")
    K_train = gk.fit_transform(X_train)
    
    print("Computing Test Kernel Matrix...")
    K_test = gk.transform(X_test)
    
    # 4. Train SVM
    print(f"Fitting SVM Classifier (C={SVM_C})...")
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