from grakel.kernels import WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment, NeighborhoodSubgraphPairwiseDistance, SubgraphMatching
from .config import WL_ITERATIONS, N_JOBS

def get_kernel(kernel_name, n_iter=WL_ITERATIONS):
    """
    Factory function to initialize a GraKeL kernel object based on config.
    
    Args:
        kernel_name (str): The type of kernel to initialize. 
                           Options: 'WL-OA' (Optimal Assignment) or 'WL' (Subtree).
        n_iter (int): Override for the number of WL iterations (depth). 
                      If None, uses the default WL_ITERATIONS from config.
        
    Returns:
        grakel.Kernel: An initialized kernel object ready for fitting.
        
    Raises:
        ValueError: If an unknown kernel_name is provided.
    """
    # Use the provided n_iter, or fall back to config default if somehow None is passed
    iterations = n_iter if n_iter is not None else WL_ITERATIONS
    
    print(f"Initializing Kernel: {kernel_name} (n_iter={iterations}, n_jobs={N_JOBS})")

    if kernel_name == 'WL-OA':
        # The complex, high-accuracy kernel (Optimal Assignment)
        return WeisfeilerLehmanOptimalAssignment(
            n_iter=iterations, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    elif kernel_name == 'WL':
        # The fast, standard kernel (Subtree)
        return WeisfeilerLehman(
            n_iter=iterations, 
            normalize=True, 
            n_jobs=N_JOBS
        )
    elif kernel_name == "NSPD":
        return NeighborhoodSubgraphPairwiseDistance(
            normalize=True
        )
    elif kernel_name == "SM":
        return SubgraphMatching(
            normalize=True
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}. Valid options are 'WL-OA', 'WL'.")