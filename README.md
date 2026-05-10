# PX-Intel: Service Feedback Analysis System

A comprehensive sentiment and causal analysis system for service feedback using advanced NLP models.

## Overview

PX-Intel processes service feedback data through four phases:

- **Phase 1**: Robust data loading with error handling (10,301+ entries)
- **Phase 2**: Streamlit prototype with sentiment analysis (RoBERTa)
- **Phase 3**: Advanced NLI-based causal analysis (DeBERTa) - Archived
- **Phase 4**: Unsupervised clustering pipeline (LDA + K-Means + t-SNE) - **Active**

The current emphasis is on plain-language insights that help explain how feedback relates to service experience, regardless of whether the domain is healthcare, customer support, or another client-facing setting.

## Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment

### Setup

```bash

# Create and activate virtual environment

python3 -m venv .venv
source .venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Run the unsupervised analysis dashboard

streamlit run app_unsupervised.py
```

## Features

### Phase 4: Unsupervised Clustering (Active)

#### M1 - Clustering Engine

- Automatic topic discovery with LDA (coherence optimization)
- K-Means clustering with silhouette scoring
- Jensen-Shannon Divergence (Paradox Score) for cluster diversity
- t-SNE projection for 2D visualization

#### M2 - Cluster Auditing

- Per-cluster sentiment analysis (RoBERTa)
- Vocabulary extraction (KeyBERT)
- Red/Green/Neutral zone classification
- Sentiment density scoring

#### M3 - Causal Reasoning

- DeBERTa NLI validation for keyword entailment
- Cross-cluster similarity computation
- Cascade effect prediction (if-then statements)
- Multi-hop causal chains
- Plain-language operational summaries

Current limitation: cascade suggestions are still broad in some clusters. The next iteration will make recommendations more specific and coherent, such as linking repeated wait-time or frustration signals to staffing or equipment improvements when the evidence supports that step.

#### M4 - Dashboard

- **Landscape Tab**: Interactive t-SNE scatter with zone coloring
- **Cluster Audit Tab**: Sentiment breakdown and top keywords per cluster
- **Causal Analysis Tab**: Cascade predictions, causal chains, and service-oriented follow-up guidance
- **Data Export Tab**: CSV and pickle download

### Phase 2-3: Supervised Analysis (Archived)

Still available in `app.py`:

- RoBERTa sentiment classification (3 classes)
- DeBERTa zero-shot causal analysis
- 6 issue categories + 5 cause factors
- Confidence-based filtering (50% / 40%)

## Project Structure

```text
.
├── app.py                              # Supervised Streamlit app (archived)
├── app_unsupervised.py                 # Unsupervised Streamlit dashboard (active)
├── data_loader.py                      # Robust CSV data loader
├── unsupervised_clustering.py           # M1: Clustering engine (566 lines)
├── cluster_audit.py                    # M2: Audit engine (441 lines)
├── causal_reasoning.py                 # M3: Reasoning engine (389 lines)
├── test_loader.py                      # Phase 1 unit tests (16 tests)
├── test_unsupervised_clustering.py     # Phase 4 M1 tests (25 tests)
├── test_cluster_audit.py               # Phase 4 M2 tests
├── test_integration.py                 # End-to-end M1-M4 test
├── requirements.txt                    # Python dependencies
├── text_data.csv                       # Hospital feedback data (10,334 rows)
├── README.md                           # This file
├── PROJECT_STATUS.md                   # Complete project overview
├── UNSUPERVISED_APPROACH.md            # Phase 4 technical documentation
├── PHASE_3_SUMMARY.md                  # Phase 3 technical details
└── GIT_SETUP_COMPLETE.md               # Git configuration
```text

## Key Statistics

- **Data**: 10,301 successfully loaded entries (99.68% success rate)
- **Phase 1 Tests**: 16 unit tests (all passing)
- **Phase 4 Tests**: 28 unit tests + integration test (all passing)
- **Performance**: <1s startup with Streamlit caching
- **Models**: RoBERTa (sentiment) + KeyBERT (vocabulary) + DeBERTa (causal validation)
- **Interpretation**: Plain-language summaries are preferred over technical model labels

## Documentation

- **README.md** - Project overview and quick start
- **PROJECT_STATUS.md** - Complete project overview and status
- **UNSUPERVISED_APPROACH.md** - Phase 4 detailed implementation guide
- **PHASE_3_SUMMARY.md** - Phase 3 supervised approach (archived)
- **GIT_SETUP_COMPLETE.md** - Git configuration docs

## Running Tests

```bash

# Phase 1 data loader tests

python -m pytest test_loader.py -v

# Phase 4 unsupervised clustering tests (M1)

python -m pytest test_unsupervised_clustering.py -v

# End-to-end integration test (M1-M4)

python test_integration.py
```

## Troubleshooting

See **PROJECT_STATUS.md** or **UNSUPERVISED_APPROACH.md** for detailed troubleshooting.

### Common Issues

- **Model loading timeout**: Increase initial load time (models cache after first load)
- **Token overflow**: Texts auto-truncated to 512 chars
- **Cluster selection**: Auto-optimized via coherence/silhouette scores
- **GPU vs CPU**: Set device before initialization

## Technology Stack

- **Data**: pandas, numpy
- **ML**: torch, transformers, scikit-learn, gensim, keybert
- **UI**: streamlit, plotly
- **Tokenization**: tiktoken, sentencepiece
- **Testing**: pytest

## Author

PX-Intel Development Team
