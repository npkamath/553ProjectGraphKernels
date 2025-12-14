import pickle
import sys
from pathlib import Path
import numpy as np
np.float_ = np.float64
np.int_ = np.int64

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_PATH
from src.prediction import train_and_save_model, load_and_predict

def main():
    """
    Entry point for training and prediction pipeline.
    
    Steps:
    1. Checks if model exists - if yes, skip training unless FORCE_RETRAIN=True
    2. If training needed: loads raw training data and trains model
    3. Loads test data from data/raw/simTest.pkl
    4. Runs prediction on all test simulations
    5. Saves predictions to results/
    """
    print("="*60)
    print("   CRYSTAL STRUCTURE CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Configuration
    MAX_NODES_PER_SIM = None  # Set to e.g. 500 to subsample large graphs
    BATCH_SIZE = 1000  # Smaller = less memory, slower. Larger = more memory, faster
    FORCE_RETRAIN = True  # Set True to retrain even if model exists
    
    # Define paths
    model_path = Path("models/crystal_classifier_radius_1.pkl")
    test_data_path = Path("data/raw/simTest3.pkl")
    results_path = Path("results/test_predictions_radius_1.pkl")
    
    # ============================================================
    # CHECK: Does model already exist?
    # ============================================================
    model_exists = model_path.exists()
    
    if model_exists and not FORCE_RETRAIN:
        print(f"\n✓ Found existing model at: {model_path}")
        print("  Skipping training. Set FORCE_RETRAIN=True to retrain.")
        print("  Loading model for prediction...")
        
        from src.prediction import TrajectoryClassifier
        try:
            clf = TrajectoryClassifier.load(model_path)
            needs_training = False
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print("  Will retrain from scratch...")
            needs_training = True
    else:
        if FORCE_RETRAIN:
            print(f"\nFORCE_RETRAIN=True: Retraining model...")
        else:
            print(f"\nNo existing model found. Training new model...")
        needs_training = True
    
    # ============================================================
    # STEP 1: Load Training Data (only if needed)
    # ============================================================
    if needs_training:
        print(f"\n[1/4] Loading training data from: {RAW_DATA_PATH}")
        try:
            with open(RAW_DATA_PATH, 'rb') as f:
                train_data = pickle.load(f)
            print(f"      ✓ Loaded {len(train_data['graphs'])} training graphs")
        except FileNotFoundError:
            print(f"ERROR: Could not find training file at {RAW_DATA_PATH}")
            return
    else:
        print(f"\n[1/4] Skipped (using existing model)")
    
    # ============================================================
    # STEP 2: Train and Save Model (only if needed)
    # ============================================================
    if needs_training:
        print(f"\n[2/4] Training model...")
        try:
            from src.prediction import TrajectoryClassifier
            clf = TrajectoryClassifier()
            clf.fit_from_raw_graphs(
                graphs=train_data['graphs'],
                metadata=train_data['metadata'],
                verbose=True
            )
            print(f"      ✓ Model successfully trained!")
            
            # Free training data from memory
            import gc
            del train_data
            gc.collect()
            
            print(f"\nSaving model to: {model_path}")
            clf.save(model_path)
            print(f"      ✓ Model saved!")
                
        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"\n[2/4] Skipped (using existing model)")
    
    # ============================================================
    # STEP 3: Load Test Data
    # ============================================================
    print(f"\n[3/4] Loading test data from: {test_data_path}")
    try:
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Handle both single graph and list of graphs
        if isinstance(test_data, dict) and 'graphs' in test_data:
            test_graphs = test_data['graphs']
            test_metadata = test_data.get('metadata', [{}] * len(test_graphs))
        elif isinstance(test_data, list):
            test_graphs = test_data
            test_metadata = [{}] * len(test_graphs)
        else:
            # Single graph
            test_graphs = [test_data]
            test_metadata = [{}]
        
        print(f"      ✓ Loaded {len(test_graphs)} test simulation(s)")
    except FileNotFoundError:
        print(f"ERROR: Could not find test file at {test_data_path}")
        print("Skipping prediction step.")
        return
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # STEP 4: Run Predictions
    # ============================================================
    print(f"\n[4/4] Running predictions on test simulations...")
    
    # clf is already loaded from training step
    
    all_predictions = []
    
    for idx, test_graph in enumerate(test_graphs):
        print(f"\n--- Simulation {idx + 1}/{len(test_graphs)} ---")
        print(f"    Nodes: {test_graph.number_of_nodes()}")
        print(f"    Edges: {test_graph.number_of_edges()}")
        
        try:
            # Optionally subsample nodes for very large graphs
            all_nodes = list(test_graph.nodes())
            
            if MAX_NODES_PER_SIM and len(all_nodes) > MAX_NODES_PER_SIM:
                import random
                random.seed(42)
                selected_nodes = random.sample(all_nodes, MAX_NODES_PER_SIM)
                print(f"    Subsampling {MAX_NODES_PER_SIM} nodes (out of {len(all_nodes)})")
            else:
                selected_nodes = None  # Predict all nodes
            
            # Use smaller batch size to reduce memory usage
            predictions = clf.predict_trajectory_batch(
                graph=test_graph,
                node_ids=selected_nodes,
                batch_size=BATCH_SIZE,
                verbose=True
            )
            
            # Store predictions with metadata
            result = {
                'simulation_id': idx,
                'n_nodes': test_graph.number_of_nodes(),
                'n_edges': test_graph.number_of_edges(),
                'predictions': predictions,
                'metadata': test_metadata[idx] if idx < len(test_metadata) else {}
            }
            all_predictions.append(result)
            
            # Force garbage collection after each simulation
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"    ERROR predicting simulation {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # STEP 5: Save Results
    # ============================================================
    if all_predictions:
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")
        
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'wb') as f:
            pickle.dump(all_predictions, f)
        
        print(f"✓ Results saved to: {results_path}")
        print(f"  Total simulations processed: {len(all_predictions)}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for result in all_predictions:
            from collections import Counter
            counts = Counter(result['predictions'].values())
            print(f"\nSimulation {result['simulation_id']}:")
            for label, count in counts.most_common():
                pct = 100 * count / result['n_nodes']
                print(f"  {label:>12}: {count:>5} nodes ({pct:>5.1f}%)")
    else:
        print("\nNo predictions were generated.")
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()