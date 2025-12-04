import pickle
import sys
from pathlib import Path

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.processing import process_graphs
from src.trainer import train_pipeline

def main():
    print("========================================")
    print("   CRYSTAL STRUCTURE CLASSIFICATION     ")
    print("========================================")

    # 1. Load Data
    print(f"\n[1/3] Loading raw data from: {RAW_DATA_PATH}")
    try:
        with open(RAW_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"      Loaded {len(data['graphs'])} simulation graphs.")
    except FileNotFoundError:
        print(f"ERROR: Could not find file at {RAW_DATA_PATH}")
        return

    # 2. Process Data (Discretize -> Relabel -> Extract Ego Graphs)
    print("\n[2/3] Processing graphs...")
    # This function handles all the heavy lifting defined in src/processing.py
    subgraphs, labels = process_graphs(data['graphs'], data['metadata'])
    print(f"      Generated {len(subgraphs)} labeled subgraphs.")

    # 3. Train Model (WL Kernel + SVM)
    print("\n[3/3] Training and Evaluating Model...")
    # This uses the settings in src/config.py to pick the kernel (WL-OA vs WL)
    train_pipeline(subgraphs, labels)

if __name__ == "__main__":
    main()