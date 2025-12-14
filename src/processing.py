import random
import numpy as np
import networkx as nx
from grakel import Graph
from sklearn.preprocessing import KBinsDiscretizer
from .config import (NEIGHBOR_RADIUS, NOISE_THRESHOLD, N_BINS, 
                     SAMPLES_PER_GRAPH, RANDOM_SEED, KERNEL_TYPE)

def get_node_feature_vector(node_attributes):
    """
    Extracts specific Minkowski order parameters from a node's attribute dictionary.
    
    Args:
        node_attributes (dict): The dictionary of attributes for a single node.
                                Expected keys: 'M4', 'M5', 'M6', 'M8', 'M10', 'M12'.
        
    Returns:
        np.array: A feature vector of shape (6,) containing the float values.
                  Missing keys default to 0.0.
    """
    feature_keys = ['M4', 'M5', 'M6', 'M8', 'M10', 'M12']
    features = []
    for key in feature_keys:
        features.append(node_attributes.get(key, 0.0))
    return np.array(features)

def process_graphs(graphs, metadata, n_bins=N_BINS, radius=NEIGHBOR_RADIUS):
    """
    The main data processing pipeline. Converts raw simulation graphs into 
    labeled subgraphs ready for kernel classification.
    
    Steps:
    1. Collects all Minkowski features from all nodes to train a global Discretizer.
    2. Applies K-Means discretization to convert continuous floats to integer bins.
    3. Iterates through each simulation graph:
       a. Checks noise level in metadata. If > NOISE_THRESHOLD, forces label to 'Disordered'.
       b. Randomly samples SAMPLES_PER_GRAPH nodes.
       c. Extracts the Ego Graph (local neighborhood) for each sampled node.
       d. Generates discrete labels (strings) for the GraKeL graph format.
       
    Args:
        graphs (list): List of NetworkX graph objects from the raw pickle file.
        metadata (list): List of metadata dictionaries corresponding to the graphs.
        n_bins (int): Number of bins for K-Means discretization (overrides config).
        radius (int): Radius for ego-graph extraction (overrides config).
        
    Returns:
        tuple: (final_subgraphs, final_labels, final_groups)
            - final_subgraphs (list): List of grakel.Graph objects.
            - final_labels (list): Target labels (e.g., 'fcc', 'bcc').
            - final_groups (list): Integer IDs indicating which parent simulation 
                                   each subgraph came from (used to prevent leakage).
    """
    random.seed(RANDOM_SEED)
    
    # 1. Feature Collection
    # We must see all data first to determine the global bin edges
    all_features = []
    for G in graphs:
        for _, data in G.nodes(data=True):
            vec = get_node_feature_vector(data)
            all_features.append(vec)
    all_features = np.array(all_features)
    
    # 2. Discretization
    # Train K-Means to find natural clusters in the feature space
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    discretizer.fit(all_features)
    
    # 3. Extraction & Labeling
    final_subgraphs = []
    final_labels = []
    final_groups = []  # Tracks the parent simulation ID
    
    for graph_idx, (G, meta) in enumerate(zip(graphs, metadata)):
        # Apply physics-based filtering logic
        if meta['noise_level'] > NOISE_THRESHOLD:
            label = 'Disordered'
        else:
            label = meta['crystal_type']
            
        all_nodes = list(G.nodes())
        if not all_nodes: continue
            
        # Downsample large graphs to save compute
        selected_nodes = random.sample(all_nodes, min(len(all_nodes), SAMPLES_PER_GRAPH))
        
        for node_id in selected_nodes:
            # Extract local neighborhood
            ego = nx.ego_graph(G, node_id, radius=radius)
            
            gk_labels = {}
            node_ids = list(ego.nodes())
            
            # Convert features to discrete 'barcodes' for the kernel
            raw_feats = [get_node_feature_vector(G.nodes[n]) for n in node_ids]
            discrete_codes = discretizer.transform(raw_feats).astype(int)
            
            for i, n_id in enumerate(node_ids):
                # e.g., "5-12-0-1"
                label_str = "-".join(map(str, discrete_codes[i]))
                gk_labels[n_id] = label_str
            
            gk_edges = list(ego.edges())

            if (KERNEL_TYPE == "SM") or (KERNEL_TYPE == "NSPD"):
                edge_labels = { (i, j): 1 for i, j in gk_edges }
                final_subgraphs.append(Graph(gk_edges, node_labels=gk_labels, edge_labels=edge_labels))
            else:
                final_subgraphs.append(Graph(gk_edges, node_labels=gk_labels))
                
            final_labels.append(label)
            # Store the index of the parent graph (0, 1, 2...) so we can group split later
            final_groups.append(graph_idx)

    return final_subgraphs, final_labels, final_groups