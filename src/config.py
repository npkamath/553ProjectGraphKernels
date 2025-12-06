from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'crystal_graphs_dataset.pkl'

# --- PREPROCESSING ---
NOISE_THRESHOLD = 0.2     
NEIGHBOR_RADIUS = 4     
N_BINS = 2              

# --- SAMPLING ---
SAMPLES_PER_GRAPH = 3    
RANDOM_SEED = 42

# --- KERNEL SELECTION ---
# Options: 
# 'WL-OA'        -> GraKeL Optimal Assignment (Fast, Library)
# 'WL'           -> GraKeL Subtree (Fastest, Library)
# 'CUSTOM-WL-OA' -> Your Hand-coded OA Kernel (Slower, High Effort)
# 'CUSTOM-WL'    -> Your Hand-coded Subtree Kernel (Medium, High Effort)
KERNEL_TYPE = 'CUSTOM-WL-OA'     

# --- MODEL SELECTION ---
# Options: 
# 'sklearn' -> Standard Library (One-vs-One strategy)
# 'custom'  -> Your Hand-coded SMO (One-vs-Rest strategy)
SVM_IMPLEMENTATION = 'custom'

# --- MODEL PARAMETERS ---
WL_ITERATIONS = 4         
N_JOBS = -1               
SVM_C = 10.0