# PX-Intel: Unsupervised-First Approach (M1-M4)

## Overview

PX-Intel has transitioned from a **supervised/zero-shot** model approach (RoBERTa + DeBERTa) to an **unsupervised-first discovery system** that automatically discovers service gaps without pre-defined categories.

### Architecture: 4-Phase Pipeline

```
Text Data (Normalized)
    ↓
[M1] Discovery: LDA + K-Means + t-SNE
    → Automatic cluster discovery
    → Jensen-Shannon Divergence (Paradox Score)
    ↓
[M2] Audit: RoBERTa + KeyBERT
    → Per-cluster sentiment analysis
    → Red/Green/Neutral zone classification
    → Vocabulary extraction
    ↓
[M3] Reasoning: DeBERTa NLI
    → Causal linkage validation
    → Cross-cluster similarity
    → Cascade effect prediction
    ↓
[M4] Dashboard: Streamlit
    → Interactive t-SNE landscape
    → Cluster audit reports
    → Causal insights
    → Data export
```

---

## M1: Discovery (Unsupervised Clustering)

**Module**: `unsupervised_clustering.py`
**Class**: `UnsupervisedClusteringEngine`

### What it does

- **TF-IDF Vectorization**: Converts texts to numerical features
- **LDA Topic Modeling**: Discovers latent themes (auto-selects optimal topics)
- **K-Means Clustering**: Groups texts by topic similarity (auto-selects optimal K)
- **Jensen-Shannon Divergence**: Computes "Paradox Score" between clusters
- **t-SNE Projection**: 2D visualization of cluster landscape

### Key Outputs

```python
engine = UnsupervisedClusteringEngine()
engine.fit(texts, auto_select=True)

# Access results

engine.cluster_assignments  # Cluster ID per document
engine.tsne_projection      # 2D coordinates for visualization
engine.paradox_scores       # JS divergence between clusters
engine.cluster_statistics   # Size, composition per cluster
```

### Usage Example

```python
from unsupervised_clustering import UnsupervisedClusteringEngine

engine = UnsupervisedClusteringEngine(random_state=42)
engine.fit(normalized_texts, auto_select=True)
print(engine.get_cluster_summary())
```

---

## M2: Audit (Cluster Sentiment & Vocabulary)

**Module**: `cluster_audit.py`
**Class**: `ClusterAuditEngine`

### What it does

- **Sentiment Analysis**: RoBERTa per-cluster sentiment (POSITIVE/NEGATIVE/NEUTRAL)
- **Sentiment Density**: Concentration score per cluster
- **Zone Classification**: Red (distress) / Green (satisfaction) / Neutral
- **KeyBERT Extraction**: Top keywords/issues per cluster
- **Audit Reports**: Human-readable summary per cluster

### Key Outputs

```python
audit_engine = ClusterAuditEngine()
audit_engine.audit(texts, cluster_assignments)

# Access results

audit_engine.get_red_zones()          # High-distress clusters
audit_engine.get_green_zones()        # High-satisfaction clusters
audit_engine.cluster_audit_reports    # Formatted reports
audit_engine.export_to_dataframe()    # CSV-ready data
```

### Usage Example

```python
from cluster_audit import ClusterAuditEngine

audit = ClusterAuditEngine(model_device=-1)
audit.audit(texts, cluster_assignments, n_keywords=10)
print(audit.get_summary())

df = audit.export_to_dataframe()
df.to_csv('audit_results.csv', index=False)
```

---

## M3: Reasoning (Causal Analysis & Predictions)

**Module**: `causal_reasoning.py`
**Class**: `CausalReasoningEngine`

### What it does

- **Causal Validation**: Uses DeBERTa NLI to verify keyword → sentiment entailment
- **Cluster Similarity**: Cosine similarity on LDA topic distributions
- **Cascade Prediction**: If Cluster A fixed, predict impact on Cluster B
- **If-Then Statements**: Generate actionable insights

### Key Outputs

```python
causal_engine = CausalReasoningEngine()
causal_engine.reason(lda_features, vocabularies, sentiment_results)

# Access results

causal_engine.cascade_predictions     # Cross-cluster impacts
causal_engine.causal_validations      # Keyword entailments
causal_engine.cluster_similarities    # Pairwise similarities
```

### Usage Example

```python
from causal_reasoning import CausalReasoningEngine

causal = CausalReasoningEngine(model_device=-1)
causal.reason(cluster_lda_dict, vocabularies, sentiments)
print(causal.get_summary())
```

---

## M4: Dashboard (Streamlit App)

**Module**: `app_unsupervised.py`

### Launch

```bash
streamlit run app_unsupervised.py
```

### Features

- **🗺️ Landscape Tab**: Interactive t-SNE scatter plot
  - Color-coded by cluster zone (Red/Green/Neutral)
  - Hover for entry details
  - Zone distribution summary

- **📊 Cluster Audit Tab**: Per-cluster reports
  - Sentiment breakdown
  - Top keywords
  - Distress/satisfaction levels

- **⚡ Causal Analysis Tab**: Cascade predictions
  - If-Then statements
  - Cross-cluster impacts
  - Causal confirmations

- **📋 Data Export Tab**: Download results
  - Enriched CSV with cluster assignments
  - Clustering results (pickle)

---

## Testing

### Unit Tests (25 tests for M1)

```bash
pytest test_unsupervised_clustering.py -v
```

### Cluster Audit Tests

```bash
pytest test_cluster_audit.py -v
```

### Integration Test (M1-M4)

```bash
python test_integration.py
```

---

## API Reference

### UnsupervisedClusteringEngine

```python
engine = UnsupervisedClusteringEngine(random_state=42)
engine.fit(texts, auto_select=True)                      # Run pipeline
engine.export_to_dataframe(texts)                        # Get enriched DF
engine.save('clustering.pkl')                            # Save results
engine.load('clustering.pkl')                            # Load results
engine.get_cluster_summary()                             # Print summary
```

### ClusterAuditEngine

```python
audit = ClusterAuditEngine(model_device=-1)
audit.audit(texts, cluster_assignments, n_keywords=10)   # Full audit
audit.export_to_dataframe()                              # Get DF
audit.get_red_zones()                                    # High-distress clusters
audit.get_green_zones()                                  # High-satisfaction clusters
audit.get_summary()                                      # Print summary
```

### CausalReasoningEngine

```python
causal = CausalReasoningEngine(model_device=-1)
causal.reason(lda_dict, vocabularies, sentiments)        # Run reasoning
causal.export_to_dataframe()                             # Get DF
causal.get_summary()                                     # Print summary
```

---

## Key Differences vs Supervised Approach

| Aspect | Old (Supervised) | New (Unsupervised) |
|--------|------------------|--------------------|
| **Categorization** | 6 fixed RoBERTa hypotheses | Data-driven LDA topics |
| **Clusters** | Zero-shot + Phase 3 multi-class | K-Means on LDA features |
| **Discovery** | Pre-defined issues | Automatic via unsupervised learning |
| **Validation** | Sentiment + cause extraction | NLI causal validation |
| **Visualization** | Distribution charts | Interactive t-SNE landscape |
| **Flexibility** | Requires label engineering | Adapts to data patterns |

---

## Performance Characteristics

- **Data Loading**: < 1s (Phase 1)
- **LDA + K-Means**: 2-3 minutes (M1)
- **Sentiment Analysis**: 30-60 seconds (M2)
- **Causal Reasoning**: 10-20 seconds (M3)
- **Dashboard Load**: <2 seconds (M4, cached)

---

## Files Structure

```
.
├── unsupervised_clustering.py   # M1: Clustering engine
├── cluster_audit.py             # M2: Audit functions
├── causal_reasoning.py          # M3: Causal reasoning
├── app_unsupervised.py          # M4: Streamlit dashboard
├── test_unsupervised_clustering.py  # M1 tests (25)
├── test_cluster_audit.py        # M2 tests
├── test_integration.py          # M1-M4 integration test
├── data_loader.py               # Existing Phase 1 (unchanged)
├── app.py                       # Old supervised app (for reference)
└── text_data.csv                # Input data
```

---

## Future Enhancements

- [ ] Temporal tracking (trends over time)
- [ ] Recommendation engine (T5/BART for action generation)
- [ ] Custom clustering parameters (UI controls)
- [ ] Department-specific analysis
- [ ] Real-time streaming updates
- [ ] Alert thresholds (sentiment drop detection)

---

## Support

- **Questions**: See app_unsupervised.py docstrings
- **Debugging**: Run test_integration.py for diagnostics
- **Data Format**: CSV with 'content' column (see text_data.csv example)
