import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
from grakel import Graph
import networkx as nx
from .kernels import get_kernel
from .processing import get_node_feature_vector
from .config import (NEIGHBOR_RADIUS, N_BINS, KERNEL_TYPE, SVM_C, RANDOM_SEED)
from sklearn.svm import SVC


class TrajectoryClassifier:
    """
    A classifier that can be trained once and then applied to predict
    crystal structure at each node in new simulation trajectories.
    Supports saving and loading trained models.
    """
    
    def __init__(self, n_iter=None):
        """
        Initialize the classifier.
        
        Args:
            n_iter (int): Number of WL iterations for the kernel (overrides config).
        """
        self.n_iter = n_iter
        self.discretizer = None
        self.kernel = None
        self.svm = None
        self.is_fitted = False
        self.kernel_type = KERNEL_TYPE
        self.radius = NEIGHBOR_RADIUS
        self.n_bins = N_BINS
        
    def fit(self, subgraphs, labels, groups, verbose=True):
        """
        Train the classifier on prepared subgraphs.
        
        Args:
            subgraphs (list): List of grakel.Graph objects (from processing.py).
            labels (list): List of target strings ('fcc', 'bcc', etc.).
            groups (list): Parent simulation IDs (for proper train/test split).
            verbose (bool): Whether to print progress.
            
        Returns:
            self: The fitted classifier.
        """
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import accuracy_score, classification_report
        
        if verbose:
            print(f"Training Classifier (Kernel={self.kernel_type}, Iterations={self.n_iter})")
        
        # Convert to arrays
        subgraphs_arr = np.array(subgraphs, dtype=object)
        labels_arr = np.array(labels)
        groups_arr = np.array(groups)
        
        # Split with leakage prevention
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(subgraphs_arr, labels_arr, groups_arr))
        
        X_train = subgraphs_arr[train_idx]
        X_test = subgraphs_arr[test_idx]
        y_train = labels_arr[train_idx]
        y_test = labels_arr[test_idx]
        
        if verbose:
            print(f"Train: {len(X_train)} subgraphs, Test: {len(X_test)} subgraphs")
        
        # Initialize and fit kernel
        self.kernel = get_kernel(self.kernel_type, n_iter=self.n_iter)
        
        if verbose: print("Computing Kernel Matrices...")
        K_train = self.kernel.fit_transform(X_train)
        K_test = self.kernel.transform(X_test)
        
        # Train SVM with class weights
        if verbose: print(f"Training SVM (C={SVM_C})...")
        weights = {'Disordered': 1, 'fcc': 2, 'bcc': 2, 'hcp': 2, 'sc': 2}
        self.svm = SVC(kernel='precomputed', C=SVM_C, class_weight=weights, 
                       probability=True)  # Enable probability estimates
        self.svm.fit(K_train, y_train)
        
        # Evaluate
        y_pred = self.svm.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"\nTest Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        self.is_fitted = True
        return self
    
    def fit_from_raw_graphs(self, graphs, metadata, verbose=True):
        """
        Train directly from raw simulation graphs (convenience method).
        
        Args:
            graphs (list): NetworkX graph objects.
            metadata (list): Metadata dicts with 'crystal_type', 'noise_level'.
            verbose (bool): Print progress.
            
        Returns:
            self: The fitted classifier.
        """
        from .processing import process_graphs
        
        if verbose: print("Processing training graphs...")
        subgraphs, labels, groups = process_graphs(graphs, metadata)
        
        # Store the discretizer for later use in prediction
        # We need to refit it here to have it available
        all_features = []
        for G in graphs:
            for _, data in G.nodes(data=True):
                vec = get_node_feature_vector(data)
                all_features.append(vec)
        all_features = np.array(all_features)
        
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', 
                                           strategy='kmeans')
        self.discretizer.fit(all_features)
        
        return self.fit(subgraphs, labels, groups, verbose=verbose)
    
    def save(self, filepath):
        """
        Save the trained classifier to disk.
        
        Args:
            filepath (str or Path): Path where the model should be saved.
                                   Automatically adds .pkl extension if missing.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted classifier!")
        
        filepath = Path(filepath)
        if filepath.suffix != '.pkl':
            filepath = filepath.with_suffix('.pkl')
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print("Preparing model for saving...")
        
        # Save all kernel attributes needed for prediction
        kernel_state = {
            'X': self.kernel.X,
        }
        
        # Save all private attributes that start with _ (fitted state)
        for attr in dir(self.kernel):
            if attr.startswith('_') and not attr.startswith('__'):
                if hasattr(self.kernel, attr):
                    try:
                        val = getattr(self.kernel, attr)
                        # Only save if it's not a method
                        if not callable(val):
                            kernel_state[attr] = val
                    except:
                        pass
        
        model_data = {
            'discretizer': self.discretizer,
            'kernel_state': kernel_state,
            'svm': self.svm,
            'n_iter': self.n_iter,
            'kernel_type': self.kernel_type,
            'radius': self.radius,
            'n_bins': self.n_bins,
            'is_fitted': self.is_fitted
        }
        
        print("Writing model to disk...")
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to: {filepath}")
        
        # Clear large objects to free memory
        import gc
        del model_data
        gc.collect()
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained classifier from disk.
        
        Args:
            filepath (str or Path): Path to the saved model file.
            
        Returns:
            TrajectoryClassifier: The loaded classifier, ready for prediction.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance and restore all attributes
        instance = cls(n_iter=model_data['n_iter'])
        instance.discretizer = model_data['discretizer']
        instance.svm = model_data['svm']
        instance.kernel_type = model_data['kernel_type']
        instance.radius = model_data['radius']
        instance.n_bins = model_data['n_bins']
        instance.is_fitted = model_data['is_fitted']
        
        # Recreate the kernel and restore its fitted state
        instance.kernel = get_kernel(instance.kernel_type, n_iter=instance.n_iter)
        
        # Restore all kernel state attributes
        kernel_state = model_data['kernel_state']
        for attr, value in kernel_state.items():
            setattr(instance.kernel, attr, value)
        
        print(f"Model loaded from: {filepath}")
        print(f"  Kernel: {instance.kernel_type}, Iterations: {instance.n_iter}")
        print(f"  Ready for prediction!")
        
        return instance
    
    def predict_node(self, graph, node_id, radius=None):
        """
        Predict the crystal structure class for a single node.
        
        Args:
            graph (nx.Graph): The full simulation graph (NetworkX format).
            node_id: The target node to classify.
            radius (int): Neighborhood radius for ego graph extraction.
                         If None, uses the value from training.
            
        Returns:
            str: Predicted class label (e.g., 'fcc', 'bcc', 'Disordered').
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction!")
        
        if radius is None:
            radius = self.radius
        
        # Extract ego graph
        ego = nx.ego_graph(graph, node_id, radius=radius)
        
        # Convert to GraKeL format with discretized labels
        gk_labels = {}
        node_ids = list(ego.nodes())
        
        raw_feats = [get_node_feature_vector(graph.nodes[n]) for n in node_ids]
        discrete_codes = self.discretizer.transform(raw_feats).astype(int)
        
        for i, n_id in enumerate(node_ids):
            label_str = "-".join(map(str, discrete_codes[i]))
            gk_labels[n_id] = label_str
        
        gk_edges = list(ego.edges())
        gk_graph = Graph(gk_edges, node_labels=gk_labels)
        
        # Compute kernel and predict
        K = self.kernel.transform([gk_graph])
        prediction = self.svm.predict(K)[0]
        
        return prediction
    
    def predict_trajectory(self, graph, node_ids=None, radius=None, 
                          verbose=False):
        """
        Predict crystal structure for multiple nodes in a trajectory.
        
        Args:
            graph (nx.Graph): The full simulation graph.
            node_ids (list): Specific nodes to classify. If None, classifies all nodes.
            radius (int): Neighborhood radius. If None, uses training value.
            verbose (bool): Print progress.
            
        Returns:
            dict: {node_id: predicted_label} mapping for all requested nodes.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction!")
        
        if radius is None:
            radius = self.radius
        
        if node_ids is None:
            node_ids = list(graph.nodes())
        
        predictions = {}
        total = len(node_ids)
        
        for idx, node_id in enumerate(node_ids):
            if verbose and (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total} nodes...")
            
            try:
                pred = self.predict_node(graph, node_id, radius=radius)
                predictions[node_id] = pred
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to classify node {node_id}: {e}")
                predictions[node_id] = 'Unknown'
        
        if verbose:
            print(f"\nCompleted classification of {total} nodes.")
            # Print summary statistics
            from collections import Counter
            counts = Counter(predictions.values())
            print("\nPrediction Summary:")
            for label, count in counts.most_common():
                print(f"  {label}: {count} ({100*count/total:.1f}%)")
        
        return predictions
    
    def predict_trajectory_batch(self, graph, node_ids=None, radius=None,
                                batch_size=100, verbose=True):
        """
        Batch prediction for better performance on large trajectories.
        
        Args:
            graph (nx.Graph): The simulation graph.
            node_ids (list): Nodes to classify (all nodes if None).
            radius (int): Neighborhood radius. If None, uses training value.
            batch_size (int): Number of nodes to process in each batch.
            verbose (bool): Print progress.
            
        Returns:
            dict: {node_id: predicted_label} mapping.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction!")
        
        if radius is None:
            radius = self.radius
        
        if node_ids is None:
            node_ids = list(graph.nodes())
        
        predictions = {}
        total = len(node_ids)
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_nodes = node_ids[batch_start:batch_end]
            
            if verbose:
                print(f"Processing batch {batch_start//batch_size + 1}: "
                      f"nodes {batch_start+1}-{batch_end}/{total}")
            
            # Prepare all subgraphs in the batch
            batch_graphs = []
            valid_nodes = []
            
            for node_id in batch_nodes:
                try:
                    ego = nx.ego_graph(graph, node_id, radius=radius)
                    
                    gk_labels = {}
                    node_list = list(ego.nodes())
                    raw_feats = [get_node_feature_vector(graph.nodes[n]) 
                                for n in node_list]
                    discrete_codes = self.discretizer.transform(raw_feats).astype(int)
                    
                    for i, n_id in enumerate(node_list):
                        label_str = "-".join(map(str, discrete_codes[i]))
                        gk_labels[n_id] = label_str
                    
                    gk_edges = list(ego.edges())
                    batch_graphs.append(Graph(gk_edges, node_labels=gk_labels))
                    valid_nodes.append(node_id)
                    
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Skipping node {node_id}: {e}")
            
            # Batch prediction
            if batch_graphs:
                K_batch = self.kernel.transform(batch_graphs)
                preds = self.svm.predict(K_batch)
                
                for node_id, pred in zip(valid_nodes, preds):
                    predictions[node_id] = pred
        
        if verbose:
            print(f"\nCompleted: {len(predictions)}/{total} nodes classified.")
            from collections import Counter
            counts = Counter(predictions.values())
            print("\nPrediction Summary:")
            for label, count in counts.most_common():
                print(f"  {label}: {count} ({100*count/total:.1f}%)")
        
        return predictions


# Convenience function for training and saving
def train_and_save_model(graphs, metadata, save_path, n_iter=None, verbose=True):
    """
    Train a classifier and save it to disk.
    
    Args:
        graphs (list): Training NetworkX graphs.
        metadata (list): Training metadata dicts.
        save_path (str or Path): Where to save the model.
        n_iter (int): WL iterations (optional).
        verbose (bool): Print progress.
        
    Returns:
        TrajectoryClassifier: The trained classifier.
    """
    print("="*60)
    print("Training Crystal Structure Classifier")
    print("="*60)
    
    clf = TrajectoryClassifier(n_iter=n_iter)
    clf.fit_from_raw_graphs(graphs, metadata, verbose=verbose)
    clf.save(save_path)
    
    return clf


# Convenience function for loading and predicting
def load_and_predict(model_path, graph, node_ids=None, verbose=True):
    """
    Load a trained model and classify a simulation trajectory.
    
    Args:
        model_path (str or Path): Path to the saved model.
        graph (nx.Graph): Simulation to classify.
        node_ids (list): Specific nodes to classify (optional).
        verbose (bool): Print progress.
        
    Returns:
        dict: {node_id: predicted_label} for the simulation.
    """
    print("="*60)
    print("Loading Model and Classifying Trajectory")
    print("="*60)
    
    clf = TrajectoryClassifier.load(model_path)
    
    print("\nClassifying nodes...")
    predictions = clf.predict_trajectory_batch(graph, node_ids=node_ids, 
                                              verbose=verbose)
    
    return predictions