# Copilot Instructions for PX-Intel

## Quick Reference

**Repository**: PX-Intel - Hospital Feedback Analysis System  
**Purpose**: NLP-based sentiment and causal analysis of hospital patient feedback  
**Tech Stack**: Python 3.11, Streamlit, RoBERTa, DeBERTa, PyTorch, Transformers

---

## Build, Test & Run Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests (16 unit tests)
pytest test_loader.py -v

# Run specific test
pytest test_loader.py::TestDataLoader::test_normalize_text_lowercase -v

# Run tests with coverage
pytest test_loader.py --cov=data_loader
```

### Running the Application
```bash
# Start Streamlit app (opens at http://localhost:8501)
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port=8080

# Run without browser auto-open
streamlit run app.py --logger.level=warning
```

### Data Processing
```bash
# Load and process data programmatically
python -c "
from data_loader import DataLoader
loader = DataLoader('text_data.csv')
df, stats = loader.load()
loader.print_stats()
print(f'Loaded {len(df)} entries')
"
```

---

## High-Level Architecture

### Three-Phase System

**Phase 1: Data Loading** (`data_loader.py`)
- Robust CSV parsing with mixed line terminator support
- Unicode NFKD normalization for consistent text representation
- Statistics tracking: success rate, text length distribution
- Fallback encoding (utf-8 → latin-1) for corrupted files
- Output: Normalized DataFrame with text_normalized column

**Phase 2: Sentiment Analysis** (Streamlit UI)
- RoBERTa model (Cardiff NLP) for 3-class sentiment classification
- Batch processing with 512-char text truncation
- Confidence scores included for each prediction
- Cached model loading (first run ~2 min, subsequent <1s)

**Phase 3: Causal Analysis** (Phase 3 enhancements)
- DeBERTa zero-shot multi-class classification
- **NLI Hypotheses** for issue categorization (6 categories):
  - Wait times/scheduling
  - Staff behavior/communication
  - Facility cleanliness/conditions
  - Treatment quality/outcomes
  - Costs/billing
  - General feedback/other
- **Cause-Effect Extraction** (negative sentiment only, 5 factors):
  - Inadequate staffing
  - Poor communication
  - Facility maintenance issues
  - Long wait times
  - Lack of empathy/care
- Confidence filtering: 50% for categories, 40% for causes

### Data Flow
```
text_data.csv (10,334 rows)
    ↓
DataLoader (Phase 1)
    → Unicode normalization
    → Error handling & statistics
    ↓
RoBERTa Sentiment (Phase 2)
    → 3-class classification (POS/NEG/NEU)
    → Confidence scores
    ↓
DeBERTa Causal Analysis (Phase 3)
    → Multi-class issue detection
    → Cause-effect extraction (if negative)
    ↓
Streamlit UI
    → Distribution charts
    → Deep Dive analysis
    → Interactive entry selection
```

### Key Statistics
- **Success Rate**: 99.68% (10,301/10,334 entries loaded)
- **Sentiment Distribution**: Typically 30-40% negative, 40-50% neutral, 20-30% positive
- **Startup Time**: <1s with Streamlit caching
- **Sentiment Analysis**: ~100ms per entry
- **Causal Analysis**: ~200ms per entry (multi-class overhead)

---

## Code Conventions & Patterns

### Text Processing
- **Always normalize text**: Use `DataLoader._normalize_text()` for consistency
  - Converts to lowercase
  - Applies Unicode NFKD normalization
  - Collapses whitespace
  ```python
  from data_loader import DataLoader
  normalized = DataLoader._normalize_text(raw_text)
  ```

### Model Loading
- **Always use Streamlit caching**: `@st.cache_resource` for models, `@st.cache_data` for data
  - Models load once per session (~2 min for both models)
  - Prevents re-initialization on app refresh
  ```python
  @st.cache_resource
  def load_sentiment_model():
      return pipeline("sentiment-analysis", model="cardiffnlp/...")
  ```

### Text Length Safety
- **Always truncate to 512 characters** before model inference
  - RoBERTa max sequence length: 512 tokens
  - BPE tokenization can exceed token limit with longer texts
  - Use explicit parameters: `truncation=True, max_length=512`
  ```python
  text_truncated = text[:512]
  result = model(text_truncated, truncation=True, max_length=512)
  ```

### Multi-Class Classification (Phase 3)
- **Use `multi_class=True`** for DeBERTa to detect multiple issues in one feedback
  ```python
  result = deberta_model(
      text,
      hypotheses,
      multi_class=True,  # Capture all detected issues
      truncation=True,
      max_length=512
  )
  ```
- **Apply confidence thresholds**: 50% for categories, 40% for causes
- **Cause-effect only for NEGATIVE sentiment**: Optimization to reduce compute

### Error Handling
- **CSV Loading**: Engine='python' for mixed line terminators, fallback to latin-1 encoding
- **Text Processing**: Return empty string for None/invalid input
- **Model Inference**: Wrap in try-except, fallback to NEUTRAL for errors
- **Statistics**: Check for empty arrays before computing min/max/mean

### Data Structure
- **DataLoader output**: DataFrame with columns:
  - `content`: Original text
  - `text_normalized`: Lowercase, NFKD, whitespace-collapsed
  - Additional columns (id, category, etc.) preserved from input
- **LoaderStats dataclass**: Tracks total_rows, successful_rows, failed_rows, success_rate, min/max/avg length

### Testing Patterns
- **Unit tests use fixtures**: Temporary CSV files for isolation
- **Test edge cases**: Empty strings, None, whitespace-only, special characters, Unicode
- **Test both success and error paths**: Normal loading + encoding fallback
- **Use pytest markers** for categorization if needed

---

## Important Design Decisions

### Why NLI Hypotheses Over Generic Labels?
Phase 2 initial approach with generic zero-shot labels ("cause of problem") resulted in 60%+ "General Feedback" classifications. Phase 3 switched to specific NLI hypotheses ("This feedback mentions wait times...") for precision and actionability.

### Why Multi-Class Classification?
Single-class mode missed multi-faceted complaints. Multi-class enables detecting "Staff rude AND facility dirty AND wait long" simultaneously—critical for root cause analysis.

### Why Separate Cause-Effect Pipeline?
Cause extraction is optimized for negative sentiment only (30-40% of dataset). Running on all entries wastes compute; negative-only filtering improves efficiency without losing actionable insight.

### Why 512-Char Truncation?
RoBERTa's max 512 tokens + BPE tokenization's character→token overhead requires aggressive truncation. 512 chars provides safe margin (typical expansion factor: 1.2-1.5x).

---

## Common Tasks

### Adding a New Issue Category
1. Update `hypotheses` list in `analyze_causal_with_nli()` (app.py line ~90)
2. Re-run Streamlit (no model retraining needed—zero-shot classification)
3. Test with sample feedback

### Adjusting Confidence Thresholds
- Categories: Line ~105 in `analyze_causal_with_nli()` (currently 0.50)
- Causes: Line ~135 in `extract_cause_effect()` (currently 0.40)
- Lower → more detections, higher false positives
- Higher → fewer detections, higher precision

### Troubleshooting Model Loading
- First Streamlit load: Check HuggingFace cache (~2.5 GB total)
- Token overflow errors: Verify text truncation to 512 chars
- GPU vs CPU: Set `device` in session state before loading

### Testing with Custom Data
```python
from data_loader import DataLoader

loader = DataLoader("custom_data.csv")
df, stats = loader.load(text_column="feedback")  # Adjust column name
print(stats)
```

---

## File Ownership & Responsibilities

| File | Purpose | Key Functions | Test File |
|------|---------|---|---|
| `data_loader.py` | CSV parsing & normalization | `DataLoader`, `LoaderStats`, `_normalize_text()` | `test_loader.py` |
| `app.py` | Streamlit UI & orchestration | `load_sentiment_model()`, `analyze_causal_with_nli()`, `extract_cause_effect()` | Run: `streamlit run app.py` |
| `test_loader.py` | Data loading tests | 16 unit tests (normalization, CSV load, stats) | `pytest test_loader.py -v` |
| `requirements.txt` | Dependencies | 14 core packages (pandas, torch, transformers, streamlit, etc.) | `pip install -r` |

---

## Git & Repository Conventions

### Branch Strategy
- Main: `main` (production-ready)
- Development: Feature branches as needed

### Commits
- Include Co-authored-by trailer: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- Example: `git commit -m "Message here"`

### .gitignore Protection
- `.venv/` (400-500 MB excluded)
- `huggingface/` model cache (500+ MB per model)
- `.env`, `*.key`, credentials (security)
- IDE settings (.vscode/, .idea/)
- Python cache (__pycache__/, *.pyc)

---

## Known Limitations & Gotchas

1. **First Model Load Slow**: 2-3 minutes on first run (model download). Subsequent runs <1s with caching.
2. **Text Truncation Required**: Texts >512 chars are truncated. Longer texts may lose context.
3. **Negative-Only Cause Analysis**: Positive/neutral feedback skips cause extraction (by design).
4. **Multi-Class Scores Don't Sum to 1.0**: Each hypothesis evaluated independently; sum can exceed 1.0.
5. **NFKD Normalization**: Decomposes accented characters (é → e). May affect non-English text nuances.

---

## Documentation References

- **PROJECT_STATUS.md** — Complete project status, all 3 phases, production checklist
- **PHASE_3_SUMMARY.md** — NLI implementation details, confidence thresholds, architecture
- **GIT_SETUP_COMPLETE.md** — Git configuration and .gitignore rationale
- **README.md** — Quick start, features, tech stack
