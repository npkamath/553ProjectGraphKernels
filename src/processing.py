import random
import numpy as np
import networkx as nx
from grakel import Graph
from sklearn.preprocessing import KBinsDiscretizer
from .config import (NEIGHBOR_RADIUS, NOISE_THRESHOLD, N_BINS, 
                     SAMPLES_PER_GRAPH, RANDOM_SEED)

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

def extract_raw_samples(graphs, metadata, radius=NEIGHBOR_RADIUS):
    """
    Stage 1: Extract Ego Graphs and Raw Features (No Discretization yet).
    
    This function isolates the physics extraction from the machine learning 
    preprocessing, allowing us to split Train/Test BEFORE learning the bin edges.

    Returns:
        list: A list of dictionaries, where each dictionary contains the raw 
              edges, node IDs, and float feature vectors for a single sample.
    """
    random.seed(RANDOM_SEED)
    raw_samples = []
    
    for graph_idx, (G, meta) in enumerate(zip(graphs, metadata)):
        # 1. Determine Label
        if meta['noise_level'] > NOISE_THRESHOLD:
            label = 'Disordered'
        else:
            label = meta['crystal_type']
            
        all_nodes = list(G.nodes())
        if not all_nodes: continue
            
        # 2. Subsample
        selected_nodes = random.sample(all_nodes, min(len(all_nodes), SAMPLES_PER_GRAPH))
        
        for node_id in selected_nodes:
            # 3. Extract Ego Graph
            ego = nx.ego_graph(G, node_id, radius=radius)
            node_ids = list(ego.nodes())
            edges = list(ego.edges())
            
            # 4. Extract Raw Floats (Features)
            # Shape: (n_nodes_in_ego, 6)
            raw_feats = np.array([get_node_feature_vector(G.nodes[n]) for n in node_ids])
            
            raw_samples.append({
                'edges': edges,
                'node_ids': node_ids,
                'raw_features': raw_feats,
                'label': label,
                'group': graph_idx
            })
            
    return raw_samples

def to_grakel(raw_samples, discretizer=None, n_bins=N_BINS):
    """
    Stage 2: Discretize Features and Build GraKeL Objects.
    
    This function handles the conversion of continuous physics features into
    discrete string labels required by the Weisfeiler-Lehman kernel.
    
    Args:
        raw_samples (list): Output from extract_raw_samples.
        discretizer (KBinsDiscretizer): A fitted discretizer. If None, a new one 
                                        will be created and fitted (Training Mode).
        n_bins (int): Number of bins for K-Means (only used if fitting new).

    Returns:
        tuple: (subgraphs, labels, groups, discretizer)
    """
    # 1. Collect all features to fit the discretizer (if needed)
    # We stack them all to process efficiently in batch
    all_feats_stacked = np.vstack([s['raw_features'] for s in raw_samples])
    
    if discretizer is None:
        print(f"  > Fitting new KBinsDiscretizer (Bins={n_bins}, Strategy='kmeans')...")
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        discretizer.fit(all_feats_stacked)
    
    # 2. Transform all features
    all_discrete_codes = discretizer.transform(all_feats_stacked).astype(int)
    
    final_subgraphs = []
    final_labels = []
    final_groups = []
    
    cursor = 0
    for sample in raw_samples:
        n_nodes = len(sample['node_ids'])
        
        # Slice the transformed codes corresponding to this specific graph
        codes = all_discrete_codes[cursor : cursor + n_nodes]
        cursor += n_nodes
        
        # Create Label Dictionary: {node_id: "2-0-1-5-..."}
        gk_labels = {}
        for i, n_id in enumerate(sample['node_ids']):
            gk_labels[n_id] = "-".join(map(str, codes[i]))
            
        # Build GraKeL Graph
        final_subgraphs.append(Graph(sample['edges'], node_labels=gk_labels))
        final_labels.append(sample['label'])
        final_groups.append(sample['group'])
        
    return final_subgraphs, final_labels, final_groups, discretizer