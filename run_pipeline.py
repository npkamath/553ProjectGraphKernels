# run_pipeline.py
import pickle
import sys
from src.config import RAW_DATA_PATH
from src.graph_processing import process_graphs
from src.trainer import train_pipeline

def main():
    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # NetworkX graphs are usually stored under 'graphs' key
    # based on your inspection output
    graphs = data['graphs'] 
    metadata = data['metadata']
    
    # 2. Process
    subgraphs, labels = process_graphs(graphs, metadata)
    
    # 3. Train
    train_pipeline(subgraphs, labels)

if __name__ == "__main__":
    main()