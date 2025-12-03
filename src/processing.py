# src/processing.py
import random
import numpy as np
import networkx as nx
from grakel import Graph
from sklearn.preprocessing import KBinsDiscretizer
from .config import (NEIGHBOR_RADIUS, NOISE_THRESHOLD, N_BINS, 
                     SAMPLES_PER_GRAPH, RANDOM_SEED)

def get_node_feature_vector(node_attributes):
    """
    Manually extracts M4, M5, M6, M8, M10, M12 into a single numpy array.
    Used to fix the issue where features were stored as separate keys.
    """
    # The exact keys found in your dataset inspection
    feature_keys = ['M4', 'M5', 'M6', 'M8', 'M10', 'M12']
    
    features = []
    for key in feature_keys:
        # Get the value, default to 0.0 if missing
        val = node_attributes.get(key, 0.0)
        features.append(val)
            
    return np.array(features)

def process_graphs(graphs, metadata):
    """
    Main pipeline stage:
    1. Collects all features to train the Discretizer.
    2. Relabels noisy graphs.
    3. Extracts ego-subgraphs and assigns discrete labels.
    """
    random.seed(RANDOM_SEED)
    
    # --- STEP 1: COLLECT ALL FEATURES ---
    print("Step 1: Collecting features from all graphs...")
    all_features = []

    for G in graphs:
        for _, data in G.nodes(data=True):
            vec = get_node_feature_vector(data)
            all_features.append(vec)
            
    all_features = np.array(all_features)
    
    # --- STEP 2: TRAIN DISCRETIZER ---
    print(f"Step 2: Discretizing features into {N_BINS} bins...")
    discretizer = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='kmeans')
    discretizer.fit(all_features)
    
    # --- STEP 3: RELABEL & EXTRACT SUBGRAPHS ---
    print("Step 3: Relabeling and extracting subgraphs...")
    final_subgraphs = []
    final_labels = []
    
    for G, meta in zip(graphs, metadata):
        # A. Relabeling Logic
        # If noise is high, override the label to 'Disordered'
        if meta['noise_level'] > NOISE_THRESHOLD:
            label = 'Disordered'
        else:
            label = meta['crystal_type']
            
        # B. Random Sampling
        all_nodes = list(G.nodes())
        if not all_nodes: continue
            
        # Pick random nodes to sample
        selected_nodes = random.sample(all_nodes, min(len(all_nodes), SAMPLES_PER_GRAPH))
        
        for node_id in selected_nodes:
            # Extract neighborhood (Ego Graph)
            ego = nx.ego_graph(G, node_id, radius=NEIGHBOR_RADIUS)
            
            # Prepare GraKeL data
            gk_labels = {}
            node_ids = list(ego.nodes())
            
            # C. Discretize Features for this Subgraph
            # Get raw vectors for all nodes in the subgraph
            raw_feats = [get_node_feature_vector(G.nodes[n]) for n in node_ids]
            
            # Convert to bin indices (integers)
            discrete_codes = discretizer.transform(raw_feats).astype(int)
            
            # Convert integers to string labels (e.g., "5-12-0-1")
            for i, n_id in enumerate(node_ids):
                label_str = "-".join(map(str, discrete_codes[i]))
                gk_labels[n_id] = label_str
            
            # Create GraKeL Graph
            gk_edges = list(ego.edges())
            final_subgraphs.append(Graph(gk_edges, node_labels=gk_labels))
            final_labels.append(label)

    print(f"   > Extraction complete. Created {len(final_subgraphs)} subgraphs.")
    return final_subgraphs, final_labels