# src/config.py
from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'crystal_graphs_dataset_smaller.pkl'

# --- PREPROCESSING ---
NOISE_THRESHOLD = 0.2     
NEIGHBOR_RADIUS = 7       
N_BINS = 5               

# --- SAMPLING ---
SAMPLES_PER_GRAPH = 3  
RANDOM_SEED = 44

# --- KERNEL SELECTION ---
# Options: 'WL-OA' (Optimal Assignment) or 'WL' (Standard Subtree)
KERNEL_TYPE = 'WL-OA'     

# --- MODEL PARAMETERS ---
WL_ITERATIONS = 2         # Depth of refinement (h)
N_JOBS = -1               # Use all cores
SVM_C = 1.0              # Regularization