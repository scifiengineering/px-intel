"""
PX-Intel: Hospital Feedback Analysis System
Sentiment Analysis (RoBERTa) + Causal Analysis (DeBERTa) with Phase 3 NLI enhancements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from data_loader import DataLoader
import pickle
from pathlib import Path


# ============================================================================
# Model Loading (Cached)
# ============================================================================

@st.cache_resource
def load_sentiment_model():
    """Load RoBERTa sentiment analysis model"""
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if st.session_state.get("device", "cpu") == "gpu" else -1
    )


@st.cache_resource
def load_deberta_model():
    """Load DeBERTa zero-shot classification model"""
    return pipeline(
        "zero-shot-classification",
        model="microsoft/deberta-v3-large",
        device=0 if st.session_state.get("device", "cpu") == "gpu" else -1
    )


# ============================================================================
# Causal Analysis Functions (Phase 3)
# ============================================================================

def analyze_causal_with_nli(text: str, deberta_model) -> dict:
    """
    Analyze causal factors using NLI hypotheses (Phase 3)
    Uses multi_class=True for multi-faceted issue detection
    """
    if not text or len(text) < 10:
        return {}
    
    # Truncate to 512 chars for safety
    text = text[:512]
    
    # NLI-based category hypotheses (more specific than zero-shot)
    hypotheses = [
        "This feedback mentions wait times or scheduling issues",
        "This feedback mentions staff behavior or communication",
        "This feedback mentions cleanliness or facility conditions",
        "This feedback mentions treatment quality or outcomes",
        "This feedback mentions costs or billing issues",
        "This feedback is general feedback or other issues"
    ]
    
    try:
        result = deberta_model(
            text,
            hypotheses,
            multi_class=True,
            truncation=True,
            max_length=512
        )
        
        categories = {}
        for label, score in zip(result['labels'], result['scores']):
            if score >= 0.50:  # 50% confidence threshold
                categories[label] = float(score)
        
        return categories
    except Exception as e:
        return {"error": str(e)}


def extract_cause_effect(text: str, sentiment: str, deberta_model) -> dict:
    """
    Extract cause-and-effect factors for negative sentiment (Phase 3)
    """
    if sentiment != "NEGATIVE" or not text or len(text) < 10:
        return {}
    
    text = text[:512]
    
    cause_hypotheses = [
        "The reason for dissatisfaction is inadequate staffing",
        "The reason for dissatisfaction is poor communication",
        "The reason for dissatisfaction is facility maintenance issues",
        "The reason for dissatisfaction is long wait times",
        "The reason for dissatisfaction is lack of empathy or care"
    ]
    
    try:
        result = deberta_model(
            text,
            cause_hypotheses,
            multi_class=True,
            truncation=True,
            max_length=512
        )
        
        causes = {}
        for label, score in zip(result['labels'], result['scores']):
            if score >= 0.40:  # 40% confidence threshold for causes
                causes[label] = float(score)
        
        return causes
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_data(filepath: str = "text_data.csv"):
    """Load and cache processed feedback data"""
    # Try loading from cache first
    cache_path = Path("data/processed_feedback.pkl")
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pd.read_pickle(f)
    
    # Otherwise load from CSV
    loader = DataLoader(filepath)
    df, stats = loader.load(text_column="content")
    return df


# ============================================================================
# Sentiment Analysis
# ============================================================================

def analyze_sentiment_batch(texts: list, model) -> list:
    """Analyze sentiment for batch of texts"""
    results = []
    
    for i, text in enumerate(texts):
        try:
            if not text or len(str(text).strip()) < 5:
                results.append({"label": "NEUTRAL", "score": 0.0})
                continue
            
            # Truncate to 512 chars
            text_truncated = str(text)[:512]
            
            result = model(text_truncated, truncation=True, max_length=512)
            results.append({
                "label": result[0]["label"],
                "score": result[0]["score"]
            })
        except Exception as e:
            results.append({"label": "NEUTRAL", "score": 0.0, "error": str(e)})
    
    return results


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="PX-Intel Sentiment Analysis", layout="wide")
    
    st.title("🏥 PX-Intel: Hospital Feedback Analysis System")
    st.markdown("**Sentiment Analysis + Causal Analysis using RoBERTa & DeBERTa**")
    
    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "sentiment_results" not in st.session_state:
        st.session_state.sentiment_results = None
    
    # Load data
    with st.spinner("Loading data and models..."):
        try:
            data = load_data()
            st.session_state.data = data
            
            sentiment_model = load_sentiment_model()
            deberta_model = load_deberta_model()
            
            st.success(f"✅ Loaded {len(data)} feedback entries")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Sidebar options
    st.sidebar.header("⚙️ Options")
    
    max_entries = st.sidebar.slider(
        "Number of entries to analyze",
        min_value=10,
        max_value=min(500, len(data)),
        value=100,
        step=10
    )
    
    # Run analysis
    if st.sidebar.button("🚀 Run Analysis", key="run_analysis"):
        with st.spinner("Analyzing sentiment..."):
            data_subset = data.head(max_entries).copy()
            
            # Sentiment analysis
            texts = data_subset['content'].fillna("").tolist()
            sentiment_results = analyze_sentiment_batch(texts, sentiment_model)
            
            data_subset['sentiment'] = [r['label'] for r in sentiment_results]
            data_subset['confidence'] = [r['score'] for r in sentiment_results]
            
            st.session_state.sentiment_results = data_subset
            st.success(f"✅ Analyzed {len(data_subset)} entries")
    
    # Display results
    if st.session_state.sentiment_results is not None:
        results_df = st.session_state.sentiment_results
        
        # ====================================================================
        # Sentiment Distribution Chart
        # ====================================================================
        st.header("📊 Sentiment Distribution")
        
        sentiment_counts = results_df['sentiment'].value_counts()
        colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=[colors.get(s, '#3498db') for s in sentiment_counts.index],
                text=sentiment_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Sentiment Classification Results",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", sentiment_counts.get('POSITIVE', 0))
        with col2:
            st.metric("Negative", sentiment_counts.get('NEGATIVE', 0))
        with col3:
            st.metric("Neutral", sentiment_counts.get('NEUTRAL', 0))
        
        # ====================================================================
        # Deep Dive Analysis (Phase 3)
        # ====================================================================
        st.header("🔍 Deep Dive Analysis")
        
        selected_idx = st.selectbox(
            "Select a feedback entry to analyze",
            range(len(results_df)),
            format_func=lambda i: f"Entry {i}: {results_df.iloc[i]['content'][:50]}..."
        )
        
        if selected_idx is not None:
            entry = results_df.iloc[selected_idx]
            
            st.subheader(f"Entry #{selected_idx}")
            st.write(f"**Feedback**: {entry['content']}")
            st.write(f"**Sentiment**: {entry['sentiment']} (confidence: {entry['confidence']:.2%})")
            
            # Phase 3: Causal Analysis with NLI
            st.markdown("---")
            st.subheader("📋 Issue Category Analysis (NLI)")
            
            with st.spinner("Analyzing causal factors..."):
                categories = analyze_causal_with_nli(
                    str(entry['content'])[:512],
                    deberta_model
                )
                
                if categories and "error" not in categories:
                    # Display as bar chart
                    cat_names = list(categories.keys())
                    cat_scores = list(categories.values())
                    
                    fig_cat = go.Figure(data=[
                        go.Bar(
                            y=cat_names,
                            x=cat_scores,
                            orientation='h',
                            marker_color='#3498db'
                        )
                    ])
                    
                    fig_cat.update_layout(
                        title="Issue Categories (Confidence Scores)",
                        xaxis_title="Confidence Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig_cat, use_container_width=True)
                    
                    for cat, score in categories.items():
                        st.write(f"• {cat}: {score:.2%}")
                else:
                    st.info("No issues detected in this feedback")
            
            # Phase 3: Cause-Effect Analysis (negative only)
            if entry['sentiment'] == 'NEGATIVE':
                st.markdown("---")
                st.subheader("🔗 Cause-and-Effect Analysis")
                
                with st.spinner("Extracting causes..."):
                    causes = extract_cause_effect(
                        str(entry['content'])[:512],
                        entry['sentiment'],
                        deberta_model
                    )
                    
                    if causes and "error" not in causes:
                        for cause, score in causes.items():
                            st.write(f"• {cause}: {score:.2%}")
                    else:
                        st.info("Could not extract specific causes from this feedback")
    
    else:
        st.info("👆 Click 'Run Analysis' in the sidebar to start analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**PX-Intel v3.0** | RoBERTa + DeBERTa | "
        "Phase 3: NLI Causal Analysis | "
        "[GitHub](https://github.com/scifiengineering/px-intel)"
    )


if __name__ == "__main__":
    main()
