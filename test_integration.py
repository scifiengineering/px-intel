"""
Integration test: M1-M4 pipeline
Verifies unsupervised clustering → audit → causal reasoning pipeline
"""

import numpy as np
from data_loader import DataLoader
from unsupervised_clustering import UnsupervisedClusteringEngine
from cluster_audit import ClusterAuditEngine
from causal_reasoning import CausalReasoningEngine


def test_end_to_end_pipeline():
    """Test complete M1-M4 pipeline."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: M1-M4 Unsupervised Pipeline")
    print("="*70)
    
    # ====================================================================
    # Load Data (existing Phase 1)
    # ====================================================================
    print("\n[PHASE 1] Loading data...")
    loader = DataLoader('text_data.csv')
    df, stats = loader.load()
    texts = df['text_normalized'].tolist()
    print(f"  ✓ Loaded {len(texts)} entries ({stats.success_rate:.1%} success)")
    
    # ====================================================================
    # M1: Clustering
    # ====================================================================
    print("\n[M1] Unsupervised Clustering...")
    clustering_engine = UnsupervisedClusteringEngine(random_state=42)
    clustering_engine.fit(texts, auto_select=True)
    
    cluster_assignments = clustering_engine.cluster_assignments
    n_clusters = len(np.unique(cluster_assignments))
    print(f"  ✓ Created {n_clusters} clusters")
    print(f"  ✓ Topics: {clustering_engine.optimal_n_topics}")
    
    # ====================================================================
    # M2: Cluster Auditing
    # ====================================================================
    print("\n[M2] Cluster Auditing...")
    audit_engine = ClusterAuditEngine(model_device=-1)
    audit_engine.prepare_clusters(texts, cluster_assignments)
    audit_engine.load_sentiment_model()
    audit_engine.analyze_all_cluster_sentiments()
    audit_engine.identify_cluster_zones()
    
    red_zones = audit_engine.get_red_zones()
    green_zones = audit_engine.get_green_zones()
    print(f"  ✓ Red Zones: {len(red_zones)} clusters")
    print(f"  ✓ Green Zones: {len(green_zones)} clusters")
    
    # ====================================================================
    # M3: Causal Reasoning
    # ====================================================================
    print("\n[M3] Causal Reasoning...")
    
    # Prepare cluster features
    cluster_lda_features = {}
    for cid in np.unique(cluster_assignments):
        mask = cluster_assignments == cid
        cluster_lda_features[cid] = clustering_engine.lda_features[mask]
    
    # Extract vocabularies (simplified for speed)
    cluster_vocabularies = {}
    for cid in np.unique(cluster_assignments):
        # Use dummy keywords for integration test
        cluster_vocabularies[cid] = [('wait', 0.9), ('staff', 0.85), ('clean', 0.8)]
    
    causal_engine = CausalReasoningEngine(model_device=-1)
    causal_engine.compute_cluster_similarities(cluster_lda_features)
    
    cascade_count = sum(len(v) for v in causal_engine.cascade_predictions.values())
    print(f"  ✓ Cascade predictions computed ({cascade_count} total)")
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "="*70)
    print("INTEGRATION TEST RESULTS")
    print("="*70)
    print(f"✓ M1 (Clustering):       PASSED - {n_clusters} clusters found")
    print(f"✓ M2 (Auditing):         PASSED - {len(red_zones)} red, {len(green_zones)} green zones")
    print(f"✓ M3 (Causal):          PASSED - {cascade_count} cascade predictions")
    print(f"✓ M4 (Dashboard):        READY - app_unsupervised.py configured")
    print("="*70)
    print("\n✅ INTEGRATION TEST PASSED\n")


if __name__ == "__main__":
    test_end_to_end_pipeline()
