import numpy as np
import hashlib
from collections import Counter

class BaseCustomWL:
    """
    Base class containing the shared Weisfeiler-Lehman relabeling logic.
    
    This class handles the iterative color refinement process (hashing node labels 
    based on neighbors) but does not implement the final similarity calculation.
    """
    def __init__(self, n_iter=4):
        self.n_iter = n_iter
        self.train_histograms = [] # Stores feature counts for training graphs
        self.train_norms = None    # Stores normalization factors

    def _hash_label(self, label_str):
        """Creates a deterministic MD5 hash for a label string."""
        return hashlib.md5(label_str.encode('utf-8')).hexdigest()

    def _get_histograms(self, graphs):
        """
        Runs the Weisfeiler-Lehman Relabeling process.
        
        Args:
            graphs (list): List of GraKeL graph objects.
            
        Returns:
            list: A list of lists, where histograms_by_iter[i] contains 
                  the label counts for all graphs at iteration i.
        """
        # 1. Parse initial labels from GraKeL graphs into string format
        current_graphs_labels = []
        for G in graphs:
            labels = G.get_labels(purpose='dictionary')
            current_graphs_labels.append({k: str(v) for k, v in labels.items()})

        histograms_by_iter = []
        
        for it in range(self.n_iter + 1):
            # A. Count labels at this iteration (The Feature Vector)
            current_iter_hists = []
            for G_labels in current_graphs_labels:
                current_iter_hists.append(Counter(G_labels.values()))
            histograms_by_iter.append(current_iter_hists)
            
            # B. Relabel for next iteration (if needed)
            if it < self.n_iter:
                new_graphs_labels = []
                for i, G in enumerate(graphs):
                    old_labels = current_graphs_labels[i]
                    new_labels = {}
                    
                    # Safe access to adjacency list
                    adj = G.get_edge_dictionary()
                    
                    for node in old_labels:
                        neighbors = list(adj.get(node, {}).keys())
                        # WL Requirement: Neighbors must be sorted to ensure canonical string
                        neighbor_labels = sorted([old_labels[n] for n in neighbors if n in old_labels])
                        
                        # Form string: "MyLabel + NeighborLabel1 + NeighborLabel2..."
                        long_str = old_labels[node] + "_" + "".join(neighbor_labels)
                        
                        # Hash it to get the new discrete label
                        new_labels[node] = self._hash_label(long_str)
                    new_graphs_labels.append(new_labels)
                current_graphs_labels = new_graphs_labels
                
        return histograms_by_iter

    def _flatten_histograms(self, histograms_by_iter, n_graphs):
        """
        Combines histograms from all iterations into one single feature counter per graph.
        This represents the "Bag of Colors" accumulated over all depths.
        """
        flat_counters = []
        for i in range(n_graphs):
            total_counter = Counter()
            for it in range(self.n_iter + 1):
                total_counter.update(histograms_by_iter[it][i])
            flat_counters.append(total_counter)
        return flat_counters

class CustomWLSubtree(BaseCustomWL):
    """
    Standard Weisfeiler-Lehman Subtree Kernel (Hand-coded).
    
    Mathematical Definition:
    K(G1, G2) = <Phi(G1), Phi(G2)>
    Where Phi(G) is the vector of counts of all labels at all iterations.
    """
    def fit_transform(self, graphs):
        """Computes the normalized kernel matrix for training data."""
        print(f"  > Custom WL-Subtree: Generating histograms (Depth {self.n_iter})...")
        histograms = self._get_histograms(graphs)
        self.train_histograms = self._flatten_histograms(histograms, len(graphs))
        
        n = len(graphs)
        K = np.zeros((n, n))
        
        print("  > Computing Kernel Matrix (Dot Product)...")
        for i in range(n):
            for j in range(i, n):
                c1 = self.train_histograms[i]
                c2 = self.train_histograms[j]
                
                # Dot product: sum(count1 * count2) for common labels
                common = set(c1.keys()) & set(c2.keys())
                score = sum(c1[k] * c2[k] for k in common)
                
                K[i, j] = score
                K[j, i] = score
        
        # Normalize: K(x,y) / sqrt(K(x,x)*K(y,y))
        d = np.diag(K)
        self.train_norms = np.sqrt(d)
        norm_mat = np.outer(self.train_norms, self.train_norms)
        return K / (norm_mat + 1e-10)

    def transform(self, graphs):
        """Computes kernel matrix between new graphs and training graphs."""
        histograms = self._get_histograms(graphs)
        test_histograms = self._flatten_histograms(histograms, len(graphs))
        
        n_test = len(graphs)
        n_train = len(self.train_histograms)
        K = np.zeros((n_test, n_train))
        
        test_norms = []
        for c in test_histograms:
            self_score = sum(v*v for v in c.values())
            test_norms.append(np.sqrt(self_score))
            
        print("  > Computing Test Kernel Matrix...")
        for i in range(n_test):
            for j in range(n_train):
                c1 = test_histograms[i]
                c2 = self.train_histograms[j]
                common = set(c1.keys()) & set(c2.keys())
                score = sum(c1[k] * c2[k] for k in common)
                
                norm = test_norms[i] * self.train_norms[j]
                K[i, j] = score / (norm + 1e-10)
        return K

class CustomWLOptimalAssignment(BaseCustomWL):
    """
    Weisfeiler-Lehman Optimal Assignment Kernel (Hand-coded).
    
    Mathematical Definition:
    K(G1, G2) = Sum(Min(Count_G1(label), Count_G2(label))) for all labels.
    This corresponds to the Histogram Intersection kernel.
    """
    def fit_transform(self, graphs):
        print(f"  > Custom WL-OA: Generating histograms (Depth {self.n_iter})...")
        histograms = self._get_histograms(graphs)
        self.train_histograms = self._flatten_histograms(histograms, len(graphs))
        
        n = len(graphs)
        K = np.zeros((n, n))
        
        print("  > Computing Kernel Matrix (Histogram Intersection)...")
        for i in range(n):
            for j in range(i, n):
                c1 = self.train_histograms[i]
                c2 = self.train_histograms[j]
                
                # Intersection: sum(min(count1, count2))
                common = set(c1.keys()) & set(c2.keys())
                score = sum(min(c1[k], c2[k]) for k in common)
                
                K[i, j] = score
                K[j, i] = score
        
        d = np.diag(K)
        self.train_norms = np.sqrt(d)
        norm_mat = np.outer(self.train_norms, self.train_norms)
        return K / (norm_mat + 1e-10)

    def transform(self, graphs):
        histograms = self._get_histograms(graphs)
        test_histograms = self._flatten_histograms(histograms, len(graphs))
        
        n_test = len(graphs)
        n_train = len(self.train_histograms)
        K = np.zeros((n_test, n_train))
        
        test_norms = []
        for c in test_histograms:
            # Self intersection is just sum of all counts
            self_score = sum(c.values())
            test_norms.append(np.sqrt(self_score))
            
        print("  > Computing Test Kernel Matrix...")
        for i in range(n_test):
            for j in range(n_train):
                c1 = test_histograms[i]
                c2 = self.train_histograms[j]
                common = set(c1.keys()) & set(c2.keys())
                score = sum(min(c1[k], c2[k]) for k in common)
                
                norm = test_norms[i] * self.train_norms[j]
                K[i, j] = score / (norm + 1e-10)
        return K