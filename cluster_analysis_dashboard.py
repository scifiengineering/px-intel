"""
CX-Intel Cluster Analysis Dashboard
Professional SaaS Analytics for Customer Experience Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="CX-Intel | Cluster Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DARK THEME STYLING (CRITICAL)
# ============================================================================

st.markdown("""
<style>
    :root {
        /* BACKGROUND COLORS */
        --bg-primary: #0B1220;
        --bg-secondary: #111827;
        --bg-tertiary: #1F2937;
        
        /* TEXT COLORS */
        --text-primary: #E5E7EB;
        --text-secondary: #9CA3AF;
        
        /* ACCENT COLORS - MEANING DRIVEN */
        --positive: #22C55E;
        --negative: #EF4444;
        --neutral: #9CA3AF;
        --warning: #F59E0B;
        --primary-accent: #3B82F6;
        --secondary-accent: #8B5CF6;
        
        /* GLOW EFFECTS */
        --glow-red: 0 0 20px rgba(239, 68, 68, 0.3);
        --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
        --glow-green: 0 0 20px rgba(34, 197, 94, 0.3);
        --glow-purple: 0 0 20px rgba(139, 92, 246, 0.3);
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }

    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    .main {
        background-color: var(--bg-primary);
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--bg-tertiary);
    }

    /* HEADER STYLING */
    h1, h2, h3, h4 {
        color: var(--text-primary);
        font-weight: 700;
    }

    h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        font-size: 1.5rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bg-tertiary);
    }

    h3 {
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }

    /* CARD STYLING */
    .custom-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .custom-card:hover {
        border-color: var(--primary-accent);
        box-shadow: var(--glow-blue), 0 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }

    .custom-card.active {
        border-color: var(--primary-accent);
        box-shadow: var(--glow-blue);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        border-color: var(--primary-accent);
        box-shadow: var(--glow-blue);
    }

    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-accent);
        margin: 0.5rem 0;
    }

    /* BADGE STYLING */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.25rem;
    }

    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid #DC2626;
        box-shadow: var(--glow-red);
    }

    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FCD34D;
        border: 1px solid #D97706;
    }

    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #86EFAC;
        border: 1px solid #16A34A;
    }

    /* PRIORITY PILL */
    .priority-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .priority-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        color: #FCA5A5;
        border: 1px solid #DC2626;
        box-shadow: var(--glow-red);
    }

    .priority-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
        color: #FCD34D;
        border: 1px solid #D97706;
    }

    /* KEYWORD TAGS */
    .keyword-tag {
        display: inline-block;
        background-color: var(--bg-tertiary);
        border: 1px solid var(--bg-tertiary);
        border-radius: 20px;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .keyword-tag:hover {
        border-color: var(--primary-accent);
        color: var(--primary-accent);
    }

    .keyword-tag.sentiment-positive {
        color: var(--positive);
        border-color: var(--positive);
    }

    .keyword-tag.sentiment-negative {
        color: var(--negative);
        border-color: var(--negative);
    }

    .keyword-tag.sentiment-neutral {
        color: var(--neutral);
        border-color: var(--neutral);
    }

    /* TABLE STYLING */
    .cluster-table {
        width: 100%;
        border-collapse: collapse;
    }

    .cluster-table thead {
        border-bottom: 2px solid var(--bg-tertiary);
    }

    .cluster-table th {
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .cluster-table td {
        padding: 1rem;
        border-bottom: 1px solid var(--bg-tertiary);
        color: var(--text-primary);
    }

    .cluster-table tr:hover {
        background-color: var(--bg-tertiary);
    }

    .cluster-table tr.active {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 3px solid var(--primary-accent);
    }

    /* SENTIMENT INDICATOR */
    .sentiment-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .sentiment-positive {
        background: rgba(34, 197, 94, 0.2);
        color: var(--positive);
    }

    .sentiment-negative {
        background: rgba(239, 68, 68, 0.2);
        color: var(--negative);
    }

    .sentiment-neutral {
        background: rgba(156, 163, 175, 0.2);
        color: var(--neutral);
    }

    /* METRIC CARDS IN DETAIL VIEW */
    .metric-mini {
        text-align: center;
        padding: 1rem;
        background-color: var(--bg-tertiary);
        border-radius: 8px;
        border: 1px solid var(--bg-tertiary);
    }

    .metric-mini-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .metric-mini-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-accent);
    }

    /* FEEDBACK QUOTE CARD */
    .feedback-quote {
        background-color: var(--bg-tertiary);
        border-left: 3px solid var(--primary-accent);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-style: italic;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* CASCADE DIAGRAM */
    .cascade-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0 0.5rem;
        color: white;
        text-align: center;
        min-width: 120px;
    }

    .cascade-step.problem {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        box-shadow: var(--glow-red);
    }

    .cascade-step.cause {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
    }

    .cascade-step.system {
        background: linear-gradient(135deg, #EAB308, #CA8A04);
        box-shadow: 0 0 20px rgba(234, 179, 8, 0.3);
    }

    .cascade-step.outcome {
        background: linear-gradient(135deg, #22C55E, #16A34A);
        box-shadow: var(--glow-green);
    }

    /* SIDEBAR */
    .sidebar-nav-item {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.95rem;
    }

    .sidebar-nav-item:hover {
        background-color: var(--bg-tertiary);
        color: var(--primary-accent);
    }

    .sidebar-nav-item.active {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        color: var(--primary-accent);
        border-left: 3px solid var(--primary-accent);
        font-weight: 600;
    }

    /* ICONS */
    .icon {
        font-size: 1.2rem;
        display: inline-block;
    }

    /* INSIGHT TEXT */
    .insight-text {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }

    /* PLOTLY CHARTS */
    .plotly {
        background-color: transparent !important;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: transparent !important;
    }

    /* RESPONSIVE */
    @media (max-width: 1200px) {
        h1 {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION & CACHING
# ============================================================================

@st.cache_data
def generate_cluster_data():
    """Generate realistic cluster analysis data"""
    np.random.seed(42)
    
    clusters = [
        {
            'id': 1,
            'name': 'Wait Time / Delays',
            'icon': '⏰',
            'count': 1850,
            'positive': 120,
            'neutral': 180,
            'negative': 1550,
            'sentiment_score': 2.1,
            'priority': 'High',
            'keywords': ['wait', 'time', 'delay', 'long', 'queue', 'rushed'],
            'feedback_samples': [
                '"Waited 3 hours past appointment time with no updates"',
                '"Long wait despite early arrival"',
                '"Queue management needs improvement"'
            ],
            'root_cause': 'Understaffing and inefficient scheduling systems',
            'cascade': [
                {'label': 'Long Wait Times', 'type': 'problem', 'icon': '⏰'},
                {'label': 'Inadequate Staffing', 'type': 'cause', 'icon': '👥'},
                {'label': 'Scheduling Issues', 'type': 'system', 'icon': '📅'},
                {'label': 'Patient Frustration', 'type': 'outcome', 'icon': '😞'},
            ],
            'recommendations': [
                '✓ Implement AI-driven queue management system',
                '✓ Hire 15-20% more staff during peak hours',
                '✓ Create virtual queue with wait time estimates'
            ],
            'color': '#EF4444'
        },
        {
            'id': 2,
            'name': 'Staff Communication',
            'icon': '💬',
            'count': 1420,
            'positive': 280,
            'neutral': 350,
            'negative': 790,
            'sentiment_score': 1.8,
            'priority': 'High',
            'keywords': ['communication', 'clarity', 'explain', 'listen', 'understand', 'rude'],
            'feedback_samples': [
                '"Staff didn\'t explain the procedure clearly"',
                '"Felt rushed and not heard"',
                '"Great communication skills shown by Dr. Chen"'
            ],
            'root_cause': 'Lack of structured communication training and high workload stress',
            'cascade': [
                {'label': 'Poor Communication', 'type': 'problem', 'icon': '🗣️'},
                {'label': 'Insufficient Training', 'type': 'cause', 'icon': '📚'},
                {'label': 'Staff Burnout', 'type': 'system', 'icon': '😰'},
                {'label': 'Low Patient Trust', 'type': 'outcome', 'icon': '⚠️'},
            ],
            'recommendations': [
                '✓ Implement quarterly empathy training programs',
                '✓ Create communication protocol handbook',
                '✓ Establish feedback loop with staff'
            ],
            'color': '#F59E0B'
        },
        {
            'id': 3,
            'name': 'Billing / Charges',
            'icon': '💳',
            'count': 980,
            'positive': 95,
            'neutral': 185,
            'negative': 700,
            'sentiment_score': 1.6,
            'priority': 'Medium',
            'keywords': ['billing', 'charge', 'cost', 'fee', 'insurance', 'unexpected'],
            'feedback_samples': [
                '"Why wasn\'t I informed about additional charges?"',
                '"Insurance explanation was confusing"',
                '"Itemized bill would be helpful"'
            ],
            'root_cause': 'Lack of billing transparency and poor insurance verification process',
            'cascade': [
                {'label': 'Unexpected Charges', 'type': 'problem', 'icon': '💰'},
                {'label': 'Poor Documentation', 'type': 'cause', 'icon': '📄'},
                {'label': 'Payment Disputes', 'type': 'system', 'icon': '⚠️'},
                {'label': 'Lost Revenue', 'type': 'outcome', 'icon': '📉'},
            ],
            'recommendations': [
                '✓ Create automated billing transparency system',
                '✓ Improve insurance pre-verification process',
                '✓ Generate itemized receipts immediately'
            ],
            'color': '#8B5CF6'
        },
        {
            'id': 4,
            'name': 'Service Quality',
            'icon': '⭐',
            'count': 756,
            'positive': 450,
            'neutral': 180,
            'negative': 126,
            'sentiment_score': 1.4,
            'priority': 'Medium',
            'keywords': ['quality', 'service', 'excellent', 'professional', 'care', 'thorough'],
            'feedback_samples': [
                '"Excellent care and attention to detail"',
                '"Professional staff, well-organized"',
                '"Could improve follow-up process"'
            ],
            'root_cause': 'Overall positive but some process improvements needed',
            'cascade': [
                {'label': 'Good Foundation', 'type': 'cause', 'icon': '✓'},
                {'label': 'Process Gaps', 'type': 'system', 'icon': '🔧'},
                {'label': 'Minor Issues', 'type': 'problem', 'icon': '⚙️'},
                {'label': 'Good Outcomes', 'type': 'outcome', 'icon': '👍'},
            ],
            'recommendations': [
                '✓ Standardize care protocols across all staff',
                '✓ Enhance follow-up communication system',
                '✓ Regular quality assurance audits'
            ],
            'color': '#3B82F6'
        },
        {
            'id': 5,
            'name': 'Cleanliness',
            'icon': '🧹',
            'count': 642,
            'positive': 590,
            'neutral': 40,
            'negative': 12,
            'sentiment_score': 1.2,
            'priority': 'Low',
            'keywords': ['clean', 'hygiene', 'sanitary', 'neat', 'tidy', 'spotless'],
            'feedback_samples': [
                '"Facilities were impeccably clean and well-maintained"',
                '"Great attention to hygiene standards"',
                '"Restrooms could be checked more frequently"'
            ],
            'root_cause': 'Strong housekeeping team with minor scheduling improvements possible',
            'cascade': [
                {'label': 'Strong Operations', 'type': 'cause', 'icon': '✓'},
                {'label': 'Good Processes', 'type': 'system', 'icon': '🏗️'},
                {'label': 'Minor Gaps', 'type': 'problem', 'icon': '🔍'},
                {'label': 'Positive Feedback', 'type': 'outcome', 'icon': '😊'},
            ],
            'recommendations': [
                '✓ Increase restroom check frequency',
                '✓ Maintain current housekeeping standards',
                '✓ Regular staff appreciation for excellence'
            ],
            'color': '#22C55E'
        }
    ]
    
    return clusters

@st.cache_data
def generate_tsne_data():
    """Generate t-SNE projection data for cluster visualization"""
    np.random.seed(42)
    
    clusters_centers = np.array([
        [5, 8],
        [-3, 6],
        [2, -5],
        [-4, -3],
        [8, -2]
    ])
    
    tsne_data = []
    cluster_names = ['Wait Time', 'Communication', 'Billing', 'Quality', 'Cleanliness']
    cluster_colors = ['#EF4444', '#F59E0B', '#8B5CF6', '#3B82F6', '#22C55E']
    
    for i, center in enumerate(clusters_centers):
        points = center + np.random.normal(0, 1.5, (150, 2))
        for x, y in points:
            tsne_data.append({
                'x': x,
                'y': y,
                'cluster': cluster_names[i],
                'color': cluster_colors[i]
            })
    
    return pd.DataFrame(tsne_data)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 CX-Intel")
    st.markdown("Cluster Analysis Dashboard")
    st.markdown("---")
    
    # Navigation
    pages = [
        ("📈 Overview", "overview"),
        ("🔍 Cluster Analysis", "cluster"),
        ("😊 Sentiment", "sentiment"),
        ("⚡ Priorities", "priority"),
        ("🔗 Cascades", "cascade"),
        ("💡 Recommendations", "recommendations"),
        ("🔔 Alerts", "alerts"),
        ("🤖 AI Assistant", "ai"),
    ]
    
    st.markdown("### Navigation")
    for label, page_id in pages:
        st.markdown(f"""
        <div class="sidebar-nav-item">
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Agents")
    agents = ["Clustering Agent", "Sentiment Analyzer", "Priority Scorer", "Cascade Mapper", "Insight Generator"]
    for agent in agents:
        st.markdown(f"- {agent}")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.markdown("# 📊 Cluster Analysis Dashboard")
st.markdown("Semantic clustering of customer feedback into actionable topic groups")
st.markdown("---")

# Load data
clusters = generate_cluster_data()
tsne_df = generate_tsne_data()

# Session state for selected cluster
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = 0

# ============================================================================
# TOP STATS ROW
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Total Clusters</div>
        <div class="kpi-value">5</div>
        <div style="font-size: 0.8rem; color: var(--text-secondary);">Active</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Total Feedback</div>
        <div class="kpi-value">6,648</div>
        <div style="font-size: 0.8rem; color: var(--positive);">↑ 12% vs last period</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Avg Sentiment</div>
        <div class="kpi-value">1.62</div>
        <div style="font-size: 0.8rem; color: var(--text-secondary);">Scale: 0-3</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">High Priority</div>
        <div class="kpi-value" style="color: var(--negative);">2</div>
        <div style="font-size: 0.8rem; color: var(--negative);">⚠️ Requires Action</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Positive Ratio</div>
        <div class="kpi-value" style="color: var(--positive);">28%</div>
        <div style="font-size: 0.8rem; color: var(--text-secondary);">Of total feedback</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

col_left, col_right = st.columns([2, 1.5], gap="large")

# LEFT PANEL - CLUSTER OVERVIEW & SELECTION
with col_left:
    st.markdown("### Cluster Distribution & Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 All Clusters", "🎯 T-SNE Projection", "📈 Sentiment Breakdown"])
    
    with tab1:
        # Cluster selection table
        st.markdown("#### Select a Cluster to View Details")
        
        table_html = """
        <table class="cluster-table">
            <thead>
                <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                    <th>Sentiment</th>
                    <th>Priority</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, cluster in enumerate(clusters):
            sentiment_color = {
                'Positive': 'sentiment-positive',
                'Neutral': 'sentiment-neutral',
                'Negative': 'sentiment-negative'
            }
            
            priority_color = {
                'High': 'badge-high',
                'Medium': 'badge-medium',
                'Low': 'badge-low'
            }
            
            is_active = "active" if idx == st.session_state.selected_cluster else ""
            
            # Determine sentiment label
            if cluster['negative'] > cluster['positive']:
                sentiment_label = "Negative"
                sentiment_class = "sentiment-negative"
            elif cluster['positive'] > cluster['negative']:
                sentiment_label = "Positive"
                sentiment_class = "sentiment-positive"
            else:
                sentiment_label = "Neutral"
                sentiment_class = "sentiment-neutral"
            
            table_html += f"""
                <tr class="{is_active}" onclick="selectCluster({idx})">
                    <td>
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{cluster['icon']}</span>
                        <strong>{cluster['name']}</strong>
                    </td>
                    <td>{cluster['count']:,}</td>
                    <td>
                        <span class="sentiment-indicator {sentiment_class}">
                            {sentiment_label}
                        </span>
                    </td>
                    <td>
                        <span class="badge {priority_color[cluster['priority']]}">
                            {cluster['priority']}
                        </span>
                    </td>
                    <td>
                        <span style="font-size: 0.8rem; color: var(--text-secondary);">
                            {cluster['sentiment_score']:.1f}/3.0
                        </span>
                    </td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Cluster selection buttons
        st.markdown("#### Quick Select")
        cols = st.columns(5)
        for idx, cluster in enumerate(clusters):
            with cols[idx]:
                if st.button(f"{cluster['icon']} {cluster['name']}", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.selected_cluster = idx
                    st.rerun()
    
    with tab2:
        # T-SNE scatter plot
        st.markdown("#### Cluster Semantic Space")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=tsne_df[tsne_df['cluster'] == cluster_name]['x'],
                y=tsne_df[tsne_df['cluster'] == cluster_name]['y'],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate=f'<b>{cluster_name}</b><br>Semantic Position<extra></extra>'
            )
            for cluster_name, color in zip(
                ['Wait Time', 'Communication', 'Billing', 'Quality', 'Cleanliness'],
                ['#EF4444', '#F59E0B', '#8B5CF6', '#3B82F6', '#22C55E']
            )
        ])
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#E5E7EB", size=11),
            height=400,
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hovermode='closest',
            margin=dict(l=50, r=20, t=20, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with tab3:
        # Sentiment breakdown by cluster
        st.markdown("#### Sentiment Distribution Across Clusters")
        
        sentiment_data = {
            'Cluster': [c['name'] for c in clusters],
            'Positive': [c['positive'] for c in clusters],
            'Neutral': [c['neutral'] for c in clusters],
            'Negative': [c['negative'] for c in clusters]
        }
        
        fig = go.Figure(data=[
            go.Bar(name='Positive', x=sentiment_data['Cluster'], y=sentiment_data['Positive'], marker_color='#22C55E'),
            go.Bar(name='Neutral', x=sentiment_data['Cluster'], y=sentiment_data['Neutral'], marker_color='#9CA3AF'),
            go.Bar(name='Negative', x=sentiment_data['Cluster'], y=sentiment_data['Negative'], marker_color='#EF4444')
        ])
        
        fig.update_layout(
            barmode='stack',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color="#E5E7EB", size=11),
            height=400,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# RIGHT PANEL - CLUSTER DETAILS
with col_right:
    cluster = clusters[st.session_state.selected_cluster]
    
    st.markdown("### Cluster Details")
    
    # Cluster header with priority
    st.markdown(f"""
    <div class="custom-card">
        <h3 style="margin: 0 0 0.5rem 0;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{cluster['icon']}</span>
            {cluster['name']}
        </h3>
        <div class="priority-pill priority-{cluster['priority'].lower()}">
            <span>●</span> {cluster['priority']} Priority
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("#### Key Metrics")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-mini">
            <div class="metric-mini-label">Total Mentions</div>
            <div class="metric-mini-value">{cluster['count']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-mini">
            <div class="metric-mini-label">Sentiment Score</div>
            <div class="metric-mini-value" style="color: var(--warning);">{cluster['sentiment_score']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment breakdown
    st.markdown("#### Sentiment Breakdown")
    
    sentiment_pct = {
        'Positive': (cluster['positive'] / cluster['count']) * 100,
        'Neutral': (cluster['neutral'] / cluster['count']) * 100,
        'Negative': (cluster['negative'] / cluster['count']) * 100
    }
    
    for sentiment, pct in sentiment_pct.items():
        color = {'Positive': '#22C55E', 'Neutral': '#9CA3AF', 'Negative': '#EF4444'}[sentiment]
        st.markdown(f"""
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.9rem;">{sentiment}</span>
                <span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>
            </div>
            <div style="background-color: var(--bg-tertiary); border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="background-color: {color}; height: 100%; width: {pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Keywords/Tags
    st.markdown("#### Top Keywords")
    
    keywords_html = '<div style="margin-bottom: 1rem;">'
    for keyword in cluster['keywords']:
        keywords_html += f'<span class="keyword-tag sentiment-negative">{keyword}</span>'
    keywords_html += '</div>'
    
    st.markdown(keywords_html, unsafe_allow_html=True)
    
    # Root cause
    st.markdown("#### Root Cause")
    st.markdown(f'<p class="insight-text">{cluster["root_cause"]}</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# CASCADE ANALYSIS
# ============================================================================

st.markdown("### Cause-Effect Cascade")

cluster = clusters[st.session_state.selected_cluster]

cascade_html = '<div style="display: flex; align-items: center; justify-content: center; gap: 1rem; flex-wrap: wrap; padding: 2rem 0;">'

for i, step in enumerate(cluster['cascade']):
    cascade_html += f"""
    <div class="cascade-step {step['type']}" title="{step['label']}">
        <span style="margin-right: 0.5rem;">{step['icon']}</span>
        {step['label']}
    </div>
    """
    
    if i < len(cluster['cascade']) - 1:
        cascade_html += '<span style="font-size: 1.5rem; color: var(--primary-accent);">→</span>'

cascade_html += '</div>'

st.markdown(cascade_html, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FEEDBACK SAMPLES & INSIGHTS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sample Feedback")
    
    for i, feedback in enumerate(cluster['feedback_samples']):
        st.markdown(f"""
        <div class="feedback-quote">
            {feedback}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Recommended Actions")
    
    for rec in cluster['recommendations']:
        st.markdown(f"""
        <div style="padding: 0.75rem; margin-bottom: 0.75rem; background-color: var(--bg-tertiary); 
                    border-left: 3px solid {cluster['color']}; border-radius: 8px;">
            <p style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem; margin-top: 2rem; padding: 1.5rem 0;">
    <p>CX-Intel Cluster Analysis v1.0.0 | © 2024 Customer Experience Intelligence Platform</p>
    <p>
        <a href="#" style="color: var(--primary-accent); text-decoration: none; margin: 0 1rem;">Privacy</a>
        <a href="#" style="color: var(--primary-accent); text-decoration: none; margin: 0 1rem;">Support</a>
        <a href="#" style="color: var(--primary-accent); text-decoration: none; margin: 0 1rem;">Documentation</a>
    </p>
</div>
""", unsafe_allow_html=True)
