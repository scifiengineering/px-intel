# PX-Intel: Hospital Feedback Analysis System

A comprehensive sentiment and causal analysis system for hospital patient feedback using advanced NLP models.

## Overview

PX-Intel processes hospital feedback data through three phases:
- **Phase 1**: Robust data loading with error handling (10,301+ entries)
- **Phase 2**: Streamlit prototype with sentiment analysis and causal classification
- **Phase 3**: Advanced NLI-based causal analysis with multi-class classification

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

# Run the Streamlit app
streamlit run app.py
```

## Features

### Sentiment Analysis (RoBERTa)
- 3-class classification: Positive, Negative, Neutral
- Pre-trained Cardiff NLP model
- Confidence scores for each prediction

### Causal Analysis (DeBERTa)
- NLI-based multi-class classification
- 6 issue categories identification
- Confidence-based filtering (50% threshold)

### Cause-and-Effect Analysis
- Extracted for negative sentiment only
- 5 cause-specific hypotheses
- 40% confidence threshold

## Project Structure

```
.
├── app.py                          # Streamlit application
├── data_loader.py                  # Robust CSV data loader
├── test_loader.py                  # Unit tests (16 tests)
├── requirements.txt                # Python dependencies
├── text_data.csv                   # Hospital feedback data (10,334 rows)
├── README.md                       # This file
├── PROJECT_STATUS.md               # Complete project status
└── PHASE_3_SUMMARY.md              # Technical details - Phase 3
```

## Key Statistics

- **Data**: 10,301 successfully loaded entries (99.68% success rate)
- **Models**: RoBERTa (sentiment) + DeBERTa (causal analysis)
- **Tests**: 16 unit tests (all passing)
- **Performance**: <1s startup with caching

## Documentation

- **PROJECT_STATUS.md** - Complete project overview and status
- **PHASE_3_SUMMARY.md** - Advanced causal analysis implementation
- **STREAMLIT_PROTOTYPE_SUMMARY.md** - System architecture and capabilities

## Running Tests

```bash
python -m pytest test_loader.py -v
```

## Troubleshooting

See **FINAL_STATUS_REPORT.md** for common issues and solutions.

## Technology Stack

- **Data**: pandas, numpy
- **ML**: torch, transformers, scikit-learn
- **UI**: streamlit, plotly
- **Tokenization**: tiktoken, sentencepiece

## Author

PX-Intel Development Team
