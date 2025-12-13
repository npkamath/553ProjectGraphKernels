import sys
import pickle
import pandas as pd
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.processing import process_graphs
from src.trainer import train_pipeline

def run_model_comparison():
    print("========================================")
    print("      MODEL COMPARISON TOURNAMENT       ")
    print("========================================")

    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # 2. Process Data ONCE (Standardize the input)
    # Using your optimal parameters from previous tuning
    OPTIMAL_BINS = 2
    OPTIMAL_RADIUS = 6
    
    print(f"Processing graphs (Radius={OPTIMAL_RADIUS}, Bins={OPTIMAL_BINS})...")
    subgraphs, labels, groups = process_graphs(
        data['graphs'], 
        data['metadata'], 
        n_bins=OPTIMAL_BINS, 
        radius=OPTIMAL_RADIUS
    )
    print(f"Generated {len(subgraphs)} subgraphs.")

    # 3. Define the Contenders
    experiments = [
        # (Kernel Name, SVM Type, Display Name)
        ('WL-OA',        'sklearn', 'GraKeL Kernel + Sklearn SVM (Baseline)'),
        ('CUSTOM-WL-OA', 'sklearn', 'Custom Kernel + Sklearn SVM'),
        ('WL-OA',        'custom',  'GraKeL Kernel + Custom SVM'),
        ('CUSTOM-WL-OA', 'custom',  'Custom Kernel + Custom SVM (Full Custom)')
    ]

    results = []

    # 4. Run the Tournament
    for kernel, svm, name in experiments:
        print(f"\n--- Testing: {name} ---")
        try:
            # The pipeline now returns a dictionary of metrics
            metrics = train_pipeline(
                subgraphs, 
                labels, 
                groups, 
                kernel_type=kernel,
                svm_impl=svm,
                verbose=True
            )
            
            # Combine name with metrics
            entry = {'Name': name, 'Status': 'Success'}
            entry.update(metrics)
            # Drop the confusion matrix object from the CSV report (too big for a cell)
            if 'Confusion Matrix' in entry:
                del entry['Confusion Matrix']
                
            results.append(entry)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'Name': name, 
                'Accuracy': 0.0, 
                'Balanced Accuracy': 0.0,
                'F1 Score': 0.0,
                'Sensitivity': 0.0,
                'AUC': 'N/A',
                'Status': 'Failed'
            })

    # 5. Show Final Standings
    df = pd.DataFrame(results)
    
    # Sort columns for readability
    cols = ['Name', 'Accuracy', 'Balanced Accuracy', 'F1 Score', 'Sensitivity', 'AUC', 'Status']
    # Filter only existing columns just in case
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n========================================")
    print("           TOURNAMENT RESULTS           ")
    print("========================================")
    print(df.sort_values(by='Accuracy', ascending=False))
    
    df.to_csv('model_comparison_results.csv', index=False)
    print("\nFull results saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    run_model_comparison()