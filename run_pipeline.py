import pickle
import sys
from pathlib import Path

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.trainer import train_pipeline

def main():
    """
    Entry point for the Crystal Structure Classification pipeline.
    
    Steps:
    1. Loads raw .pkl data from data/raw/
    2. Runs the training pipeline (which now handles processing and leakage-proof splitting internally).
    """
    print("========================================")
    print("   CRYSTAL STRUCTURE CLASSIFICATION     ")
    print("========================================")

    # 1. Load Data
    print(f"\n[1/2] Loading raw data from: {RAW_DATA_PATH}")
    try:
        with open(RAW_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"      Loaded {len(data['graphs'])} simulation graphs.")
    except FileNotFoundError:
        print(f"ERROR: Could not find file at {RAW_DATA_PATH}")
        return

    # 2. Train Model
    # Note: We no longer call 'process_graphs' here. 
    # We pass the raw data to train_pipeline so it can split 
    # the data BEFORE processing to prevent data leakage.
    print("\n[2/2] Training and Evaluating Model...")
    
    metrics = train_pipeline(data['graphs'], data['metadata'])
    
    # Optional: Print final summary from the returned metrics dict
    print("\nPipeline Complete.")
    print(f"Final Test Accuracy: {metrics['Accuracy']:.4f}")

if __name__ == "__main__":
    main()