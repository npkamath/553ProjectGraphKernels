# src/kernels.py
from grakel.kernels import WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment
from .config import WL_ITERATIONS, N_JOBS

def get_kernel(kernel_name):
    """
    Factory function to initialize a kernel based on the config name.
    
    Args:
        kernel_name (str): 'WL-OA' or 'WL'
        
    Returns:
        grakel.Kernel: An initialized kernel object ready for fitting.
    """
    print(f"Initializing Kernel: {kernel_name} (n_iter={WL_ITERATIONS}, n_jobs={N_JOBS})")

    if kernel_name == 'WL-OA':
        #The complex, high-accuracy kernel (Optimal Assignment)
        return WeisfeilerLehmanOptimalAssignment(
            n_iter=WL_ITERATIONS, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    
    elif kernel_name == 'WL':
      
        # Useful if WL-OA is taking too long to compute.
        return WeisfeilerLehman(
            n_iter=WL_ITERATIONS, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_name}. Valid options are 'WL-OA', 'WL'.")