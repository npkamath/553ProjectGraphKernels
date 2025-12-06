from grakel.kernels import WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment
from .custom_kernels import CustomWLOptimalAssignment, CustomWLSubtree
from .config import WL_ITERATIONS, N_JOBS

def get_kernel(kernel_name, n_iter=WL_ITERATIONS):
    """
    Factory function to initialize a GraKeL kernel object based on config.
    
    Args:
        kernel_name (str): The type of kernel to initialize. 
                           Options: 'WL-OA', 'WL', 'CUSTOM-WL-OA', 'CUSTOM-WL'.
        n_iter (int): Override for the number of WL iterations (depth). 
                      If None, uses the default WL_ITERATIONS from config.
        
    Returns:
        object: An initialized kernel object ready for fitting (Standard or Custom).
        
    Raises:
        ValueError: If an unknown kernel_name is provided.
    """
    # Use the provided n_iter, or fall back to config default if somehow None is passed
    iterations = n_iter if n_iter is not None else WL_ITERATIONS
    
    print(f"Initializing Kernel: {kernel_name} (n_iter={iterations})")

    # --- Standard GraKeL Kernels (C++ Optimized) ---
    if kernel_name == 'WL-OA':
        return WeisfeilerLehmanOptimalAssignment(
            n_iter=iterations, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    elif kernel_name == 'WL':
        return WeisfeilerLehman(
            n_iter=iterations, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    
    # --- Custom Hand-Coded Kernels (Pure Python) ---
    elif kernel_name == 'CUSTOM-WL-OA':
        return CustomWLOptimalAssignment(n_iter=iterations)
    elif kernel_name == 'CUSTOM-WL':
        return CustomWLSubtree(n_iter=iterations)
        
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")