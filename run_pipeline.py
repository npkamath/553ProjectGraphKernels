import pickle
import sys
from pathlib import Path

import numpy as np
np.float_ = np.float64
np.int_ = np.int64

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.processing import process_graphs
from src.trainer import train_pipeline

def main():
    """
    Entry point for the Crystal Structure Classification pipeline.
    
    Steps:
    1. Loads raw .pkl data from data/raw/
    2. Runs processing to generate discretized, labeled subgraphs.
    3. Runs the training pipeline with leakage-proof group splitting.
    """
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

    # 2. Process Data
    print("\n[2/3] Processing graphs...")
    # Now returns groups as well
    subgraphs, labels, groups = process_graphs(data['graphs'], data['metadata'])
    print(f"      Generated {len(subgraphs)} labeled subgraphs.")

    # 3. Train Model
    print("\n[3/3] Training and Evaluating Model...")
    # Pass groups to trainer to ensure proper splitting
    train_pipeline(subgraphs, labels, groups)

if __name__ == "__main__":
    main()