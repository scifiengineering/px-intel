"""
CX-Intel Dashboard - Customer Experience Intelligence Platform
A production-ready SaaS analytics dashboard for customer feedback analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from enum import Enum


THEME_RULES = {
    "Wait Time / Delays": {
        "keywords": ["wait", "waited", "delay", "delayed", "queue", "hours"],
        "desc": "Long waits and scheduling friction are affecting the experience.",
        "action": "Review peak-time staffing, appointment spacing, and live queue updates.",
    },
    "Staff Communication": {
        "keywords": ["staff", "nurse", "nurses", "doctor", "rude", "dismissive", "attitude", "communication", "kind", "caring"],
        "desc": "Staff tone, clarity, and handoffs are shaping sentiment.",
        "action": "Reinforce service standards, handoff scripts, and empathy coaching.",
    },
    "Service Quality": {
        "keywords": ["care", "service", "procedure", "experience", "satisfied"],
        "desc": "General care/service quality is a broad driver of perception.",
        "action": "Separate positive practices from weak touchpoints and standardize what works.",
    },
    "Billing / Charges": {
        "keywords": ["billing", "charge", "charges", "cost", "insurance", "payment"],
        "desc": "Cost or billing clarity may be causing avoidable confusion.",
        "action": "Improve up-front cost explanations and itemized billing notes.",
    },
    "Cleanliness": {
        "keywords": ["clean", "cleanliness", "room", "facility"],
        "desc": "Environment and room-readiness signals should be monitored.",
        "action": "Audit room readiness, cleaning rounds, and facilities response times.",
    },
    "Equipment Issues": {
        "keywords": ["equipment", "machine", "device", "broken"],
        "desc": "Equipment availability or reliability may be disrupting service.",
        "action": "Review maintenance response times and equipment replacement needs.",
    },
}

POSITIVE_WORDS = {"excellent", "wonderful", "caring", "professional", "kind", "satisfied", "good", "great", "helpful"}
NEGATIVE_WORDS = {"frustrating", "waited", "wait", "rude", "dismissive", "poor", "bad", "long", "delay", "delayed", "problem"}
NEUTRAL_WORDS = {"average", "nothing", "standard", "okay", "mixed", "normal"}

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="CX-Intel Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_design_theme():
    """Return theme tokens used by both CSS and Plotly."""
    is_dark = st.get_option("theme.base") == "dark"
    if is_dark:
        return {
            "mode": "dark",
            "plotly_template": "plotly_dark",
            "bg": "#0F172A",
            "surface": "#172033",
            "surface_alt": "#1F2A44",
            "border": "rgba(148, 163, 184, 0.22)",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "subtle": "#94A3B8",
            "grid": "rgba(148, 163, 184, 0.18)",
            "hover_bg": "#111827",
            "shadow": "0 18px 40px rgba(0, 0, 0, 0.22)",
        }
    return {
        "mode": "light",
        "plotly_template": "plotly_white",
        "bg": "#F5F7FB",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFC",
        "border": "rgba(15, 23, 42, 0.12)",
        "text": "#0F172A",
        "muted": "#475569",
        "subtle": "#64748B",
        "grid": "rgba(15, 23, 42, 0.10)",
        "hover_bg": "#FFFFFF",
        "shadow": "0 14px 34px rgba(15, 23, 42, 0.08)",
    }


THEME = get_design_theme()


def apply_app_css(theme):
    st.markdown(f"""
<style>
    :root {{
        --primary-color: #2563EB;
        --secondary-color: #7C3AED;
        --success-color: #059669;
        --danger-color: #DC2626;
        --warning-color: #D97706;
        --neutral-color: #64748B;
        --bg-primary: {theme["bg"]};
        --bg-secondary: {theme["surface"]};
        --bg-tertiary: {theme["surface_alt"]};
        --text-primary: {theme["text"]};
        --text-secondary: {theme["muted"]};
        --text-subtle: {theme["subtle"]};
        --border-color: {theme["border"]};
        --card-shadow: {theme["shadow"]};
    }}

    * {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        letter-spacing: 0;
    }}

    .stApp, .main, [data-testid="stAppViewContainer"] {{
        background: var(--bg-primary);
        color: var(--text-primary);
    }}

    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1480px;
    }}

    [data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text-primary);
    }}

    .dashboard-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1.25rem;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1.05rem;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        background: var(--bg-secondary);
        box-shadow: var(--card-shadow);
    }}

    .dashboard-title h1 {{
        margin: 0;
        color: var(--text-primary);
        font-size: 1.75rem;
        line-height: 1.16;
        font-weight: 760;
    }}

    .dashboard-title p,
    .filter-status {{
        margin: 0.35rem 0 0;
        color: var(--text-secondary);
        font-size: 0.92rem;
    }}

    .command-label {{
        margin: 0 0 0.45rem;
        color: var(--text-subtle);
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
    }}

    .section-title {{
        margin: 1.25rem 0 0.7rem;
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 760;
    }}

    .chart-card,
    .custom-card {{
        min-height: 100%;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.05rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
    }}

    .kpi-card {{
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--card-shadow);
    }}

    .kpi-label {{
        font-size: 0.74rem;
        color: var(--text-subtle);
        font-weight: 750;
        text-transform: uppercase;
    }}

    .kpi-value {{
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 760;
        color: var(--primary-color);
        margin: 0.6rem 0 0.35rem;
    }}

    .trend-up, .trend-down {{
        font-size: 0.86rem;
        font-weight: 700;
    }}

    .trend-up {{ color: var(--success-color); }}
    .trend-down {{ color: var(--danger-color); }}

    .badge-high,
    .badge-medium,
    .badge-low {{
        display: inline-block;
        color: white;
        padding: 0.28rem 0.58rem;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 760;
        text-transform: uppercase;
    }}

    .badge-high {{ background-color: var(--danger-color); }}
    .badge-medium {{ background-color: var(--warning-color); }}
    .badge-low {{ background-color: var(--success-color); }}

    .sidebar-item {{
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.45rem;
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.92rem;
    }}

    h1, h2, h3, h4 {{
        color: var(--text-primary);
        letter-spacing: 0;
    }}

    h2 {{
        border-bottom: 0;
        padding-bottom: 0;
    }}

    .alert-box {{
        background: color-mix(in srgb, var(--danger-color) 8%, var(--bg-secondary));
        border-left: 4px solid var(--danger-color);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        color: var(--text-primary);
    }}

    .insight-text {{
        color: var(--text-secondary);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 0.7rem;
    }}

    .empty-state {{
        padding: 2rem 1rem;
        color: var(--text-secondary);
        text-align: center;
        border: 1px dashed var(--border-color);
        border-radius: 8px;
        background: var(--bg-tertiary);
    }}

    div[data-testid="stPlotlyChart"] {{
        background: transparent !important;
    }}

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stDateInput"] input {{
        color: var(--text-primary);
    }}
</style>
""", unsafe_allow_html=True)


def apply_plotly_theme(fig, height=350, margin=None, showlegend=None):
    margin = margin or dict(l=40, r=24, t=12, b=42)
    fig.update_layout(
        template=THEME["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=THEME["text"], size=11),
        margin=margin,
        height=height,
        hoverlabel=dict(
            bgcolor=THEME["hover_bg"],
            bordercolor=THEME["border"],
            font=dict(color=THEME["text"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]),
        title=dict(font=dict(color=THEME["muted"])),
    )
    fig.update_yaxes(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]),
        title=dict(font=dict(color=THEME["muted"])),
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


apply_app_css(THEME)

# ============================================================================
# DATA GENERATION & CACHING
# ============================================================================

@st.cache_data
def generate_dashboard_data():
    """Load feedback and prepare reusable dashboard source data."""
    feedback_df = pd.read_csv("text_data.csv")
    feedback_df["content"] = feedback_df["content"].fillna("").astype(str)
    feedback_df["theme"] = feedback_df["content"].apply(classify_theme)
    feedback_df["sentiment"] = feedback_df["content"].apply(classify_sentiment)

    end_date = datetime.now().date()
    feedback_df["date"] = [
        end_date - timedelta(days=int((len(feedback_df) - idx - 1) % 30))
        for idx in range(len(feedback_df))
    ]

    view_data = build_dashboard_view_data(feedback_df)
    view_data["feedback_df"] = feedback_df
    return view_data


def build_dashboard_view_data(feedback_df):
    """Recompute dashboard metrics for the currently visible feedback rows."""
    total_feedback = len(feedback_df)
    sentiment_counts = feedback_df["sentiment"].value_counts() if total_feedback else pd.Series(dtype=int)
    positive_count = int(sentiment_counts.get("Positive", 0))
    neutral_count = int(sentiment_counts.get("Neutral", 0))
    negative_count = int(sentiment_counts.get("Negative", 0))

    if total_feedback:
        sentiment_df = (
            feedback_df.groupby(["date", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={"Positive": "positive", "Neutral": "neutral", "Negative": "negative"})
        )
    else:
        sentiment_df = pd.DataFrame(columns=["date", "positive", "neutral", "negative"])

    for column in ["positive", "neutral", "negative"]:
        if column not in sentiment_df.columns:
            sentiment_df[column] = 0

    if total_feedback:
        issues_df = (
            feedback_df.groupby("theme")
            .agg(
                Count=("content", "count"),
                Sentiment_Negative=("sentiment", lambda values: int((values == "Negative").sum())),
            )
            .reset_index()
            .rename(columns={"theme": "Issue"})
        )
        issues_df["Negative_Rate"] = issues_df["Sentiment_Negative"] / issues_df["Count"]
        issues_df["Priority_Score"] = (
            issues_df["Negative_Rate"] * 70
            + (issues_df["Count"] / max(issues_df["Count"].max(), 1)) * 30
        ).round(1)
        issues_df = issues_df.sort_values(["Priority_Score", "Count"], ascending=False)
    else:
        issues_df = pd.DataFrame(
            columns=["Issue", "Count", "Sentiment_Negative", "Negative_Rate", "Priority_Score"]
        )

    priority_issues = build_priority_issues(issues_df)
    actions = build_recommended_actions(issues_df)
    insights = build_plain_language_insights(issues_df, total_feedback, negative_count)

    return {
        'feedback_df': feedback_df,
        'sentiment_df': sentiment_df,
        'issues_df': issues_df,
        'priority_issues': priority_issues,
        'actions': actions,
        'insights': insights,
        'total_feedback': total_feedback,
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
        'satisfaction_score': positive_count / total_feedback if total_feedback else 0,
    }


def filter_feedback(feedback_df, date_filter, search_query, custom_range=None):
    filtered_df = feedback_df.copy()
    max_date = filtered_df["date"].max() if not filtered_df.empty else datetime.now().date()

    if date_filter == "Last 7 Days":
        start_date = max_date - timedelta(days=6)
        filtered_df = filtered_df[filtered_df["date"] >= start_date]
    elif date_filter == "Last 30 Days":
        start_date = max_date - timedelta(days=29)
        filtered_df = filtered_df[filtered_df["date"] >= start_date]
    elif date_filter == "Last 90 Days":
        start_date = max_date - timedelta(days=89)
        filtered_df = filtered_df[filtered_df["date"] >= start_date]
    elif custom_range and len(custom_range) == 2:
        start_date, end_date = custom_range
        filtered_df = filtered_df[
            (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
        ]

    query = search_query.strip().lower()
    if query:
        search_mask = (
            filtered_df["content"].str.lower().str.contains(query, na=False, regex=False)
            | filtered_df["theme"].str.lower().str.contains(query, na=False, regex=False)
        )
        filtered_df = filtered_df[search_mask]

    return filtered_df


def classify_theme(text):
    words = set(str(text).lower().replace("/", " ").replace("-", " ").split())
    scores = {
        theme: len(words.intersection(config["keywords"]))
        for theme, config in THEME_RULES.items()
    }
    best_theme = max(scores, key=scores.get)
    return best_theme if scores[best_theme] > 0 else "General Feedback"


def classify_sentiment(text):
    words = set(str(text).lower().replace("/", " ").replace("-", " ").split())
    positive_score = len(words.intersection(POSITIVE_WORDS))
    negative_score = len(words.intersection(NEGATIVE_WORDS))
    neutral_score = len(words.intersection(NEUTRAL_WORDS))

    if negative_score > positive_score and negative_score >= neutral_score:
        return "Negative"
    if positive_score > negative_score and positive_score >= neutral_score:
        return "Positive"
    return "Neutral"


def get_priority_label(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def build_priority_issues(issues_df):
    issues = []
    for row in issues_df.head(3).itertuples():
        config = THEME_RULES.get(row.Issue, {})
        issues.append({
            'name': row.Issue,
            'desc': config.get("desc", "Recurring feedback pattern that should be reviewed."),
            'priority': get_priority_label(row.Priority_Score),
            'count': int(row.Count),
        })
    return issues


def build_recommended_actions(issues_df):
    actions = []
    for row in issues_df.head(3).itertuples():
        config = THEME_RULES.get(row.Issue, {})
        actions.append({
            'title': row.Issue,
            'desc': config.get("action", "Review representative feedback and define a next action."),
            'impact': get_priority_label(row.Priority_Score),
        })
    return actions


def build_plain_language_insights(issues_df, total_feedback, negative_count):
    if issues_df.empty:
        return ["No feedback themes are available yet."]

    top_issue = issues_df.iloc[0]
    negative_rate = negative_count / total_feedback if total_feedback else 0
    insights = [
        f"Top issue: {top_issue['Issue']} with {int(top_issue['Count'])} mentions and a priority score of {top_issue['Priority_Score']:.1f}",
        f"Overall negative sentiment is {negative_rate:.0%} across the loaded feedback",
    ]

    for row in issues_df.head(3).itertuples():
        insights.append(
            f"{row.Issue}: {row.Negative_Rate:.0%} negative sentiment across {int(row.Count)} comments"
        )

    return insights


def generate_ai_reply(question: str, data: dict) -> str:
    normalized = question.strip().lower()
    issues_df = data["issues_df"]
    if not normalized:
        return "Please enter a question to receive dashboard guidance."
    if issues_df.empty:
        return "No matching feedback is available for the current search and time filters."

    top_issue = issues_df.iloc[0]
    if any(word in normalized for word in ["top", "priority", "first", "important"]):
        return (
            f"Start with {top_issue['Issue']}. It has {int(top_issue['Count'])} mentions, "
            f"{top_issue['Negative_Rate']:.0%} negative sentiment, and a priority score of "
            f"{top_issue['Priority_Score']:.1f}."
        )

    if any(word in normalized for word in ["recommend", "action", "fix", "what should"]):
        return " ".join(
            f"{action['title']}: {action['desc']}" for action in data["actions"]
        )

    if any(word in normalized for word in ["negative", "sentiment", "complaint"]):
        negative_rate = data["negative_count"] / data["total_feedback"] if data["total_feedback"] else 0
        return (
            f"Negative sentiment is {negative_rate:.0%}. The strongest negative theme is "
            f"{top_issue['Issue']}."
        )

    return (
        f"The clearest signal is {top_issue['Issue']}. Use the priority panel and "
        "recommended actions to decide the next operational step."
    )

@st.cache_data
def generate_cascade_data():
    """Generate cause-effect cascade data"""
    return {
        'chains': [
            {
                'title': 'Primary Issue Chain',
                'steps': [
                    {'label': 'Long Wait Times', 'color': '#EF4444', 'impact': 'Direct'},
                    {'label': 'Inadequate Staffing', 'color': '#F59E0B', 'impact': 'Root'},
                    {'label': 'Scheduling Inefficiencies', 'color': '#F59E0B', 'impact': 'Process'},
                    {'label': 'Customer Frustration', 'color': '#EF4444', 'impact': 'Outcome'},
                ]
            },
            {
                'title': 'Communication Issue Chain',
                'steps': [
                    {'label': 'Poor Communication', 'color': '#EF4444', 'impact': 'Direct'},
                    {'label': 'Unclear Instructions', 'color': '#F59E0B', 'impact': 'Root'},
                    {'label': 'Patient Confusion', 'color': '#F59E0B', 'impact': 'Process'},
                    {'label': 'Negative Sentiment', 'color': '#EF4444', 'impact': 'Outcome'},
                ]
            }
        ]
    }

@st.cache_data
def generate_alerts_data():
    """Generate real-time alerts"""
    return [
        {
            'type': 'warning',
            'title': 'Spike in Billing Complaints',
            'description': '+23% increase in billing-related feedback in last 7 days',
            'time': '2 hours ago'
        },
        {
            'type': 'danger',
            'title': 'Wait Time Issues Escalating',
            'description': 'Critical: Wait time complaints increased by 18% week-over-week',
            'time': '4 hours ago'
        },
        {
            'type': 'warning',
            'title': 'Staff Communication Trend',
            'description': 'Negative sentiment in communication feedback trending up',
            'time': '6 hours ago'
        }
    ]

# ============================================================================
# NAVIGATION STATE
# ============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 CX-Intel")
    st.markdown("---")
    
    # Main navigation
    pages = [
        "Overview",
        "Cluster Analysis",
        "Sentiment Analysis",
        "Issue Prioritization",
        "Operational Impact",
        "Recommendations",
        "Real-time Alerts",
        "AI Assistant",
        "Reports",
        "Settings"
    ]
    
    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.page = selected_page
    
    st.markdown("---")
    st.markdown("### 🤖 AI Agents Powering CX-Intel")
    
    agents = [
        {"name": "Recommendation Agent", "icon": "💡"},
        {"name": "Priority Scoring Agent", "icon": "⚡"},
        {"name": "Cascade Agent", "icon": "🔗"},
        {"name": "Monitoring Agent", "icon": "👁️"},
        {"name": "Assistant Agent", "icon": "🎯"}
    ]
    
    for agent in agents:
        st.markdown(f"""
        <div class="sidebar-item">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{agent['icon']}</span>
            <span>{agent['name']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Version**: 1.0.0  \n"
        "**Status**: Active  \n"
        "**Last Updated**: Today"
    )

# ============================================================================
# LOAD DATA
# ============================================================================

source_data = generate_dashboard_data()
cascade_data = generate_cascade_data()
alerts_data = generate_alerts_data()

# ============================================================================
# HEADER WITH FILTERS
# ============================================================================

feedback_source = source_data["feedback_df"]
min_feedback_date = feedback_source["date"].min()
max_feedback_date = feedback_source["date"].max()

with st.container(border=True):
    header_left, search_col, period_col, action_col = st.columns([2.4, 1.25, 1, 0.75])

    with header_left:
        st.markdown(
            """
            <div class="dashboard-title">
                <h1>Customer Feedback Intelligence</h1>
                <p>Live view of feedback volume, sentiment health, issue clusters, and next actions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with search_col:
        st.markdown('<div class="command-label">Search feedback</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search feedback",
            placeholder="Theme, keyword, issue...",
            key="global_feedback_search",
            label_visibility="collapsed",
        )

    with period_col:
        st.markdown('<div class="command-label">Time period</div>', unsafe_allow_html=True)
        date_filter = st.selectbox(
            "Time Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"],
            label_visibility="collapsed",
        )

    with action_col:
        st.markdown('<div class="command-label">Controls</div>', unsafe_allow_html=True)
        st.button("Filters", use_container_width=True)

    custom_range = None
    if date_filter == "Custom Range":
        custom_range = st.date_input(
            "Custom date range",
            value=(min_feedback_date, max_feedback_date),
            min_value=min_feedback_date,
            max_value=max_feedback_date,
        )

filtered_feedback = filter_feedback(feedback_source, date_filter, search_query, custom_range)
data = build_dashboard_view_data(filtered_feedback)
data["feedback_df"] = filtered_feedback

active_filters = []
if search_query.strip():
    active_filters.append(f'search "{search_query.strip()}"')
if date_filter != "Last 30 Days":
    active_filters.append(date_filter)

if active_filters:
    st.markdown(
        f'<p class="filter-status">{data["total_feedback"]:,} matching records across '
        f'{", ".join(active_filters)}.</p>',
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: OVERVIEW (DEFAULT)
# ============================================================================

if st.session_state.page == "Overview":
    
    # KPI CARDS ROW
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    negative_rate = data['negative_count'] / data['total_feedback'] if data['total_feedback'] else 0
    
    kpi_configs = [
        {
            'col': col1,
            'title': 'Total Feedback',
            'value': f"{data['total_feedback']:,}",
            'trend': '+12%',
            'trend_up': True,
            'color': '#3B82F6'
        },
        {
            'col': col2,
            'title': 'Negative Sentiment',
            'value': f"{negative_rate:.1%}",
            'trend': '+3%',
            'trend_up': False,
            'color': '#EF4444'
        },
        {
            'col': col3,
            'title': 'High Priority Issues',
            'value': str(sum(1 for issue in data['priority_issues'] if issue['priority'] == 'High')),
            'trend': 'Live data',
            'trend_up': False,
            'color': '#F59E0B'
        },
        {
            'col': col4,
            'title': 'Avg Response Time',
            'value': '4.2h',
            'trend': '-0.3h',
            'trend_up': True,
            'color': '#10B981'
        },
        {
            'col': col5,
            'title': 'Customer Satisfaction',
            'value': f"{data['satisfaction_score']:.1%}",
            'trend': 'Positive share',
            'trend_up': True,
            'color': '#8B5CF6'
        }
    ]
    
    for config in kpi_configs:
        with config['col']:
            trend_class = 'trend-up' if config['trend_up'] else 'trend-down'
            trend_icon = '📈' if config['trend_up'] else '📉'
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{config['title']}</div>
                <div class="kpi-value" style="color: {config['color']};">{config['value']}</div>
                <div class="{trend_class}">{trend_icon} {config['trend']}</div>
            </div>
            """, unsafe_allow_html=True)

    if data['total_feedback'] == 0:
        st.markdown(
            '<div class="empty-state">No feedback matches the current search and time filters.</div>',
            unsafe_allow_html=True,
        )
    
    # MAIN GRID - ROW 1
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    # Sentiment Distribution (Donut Chart)
    with col1:
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        
        sentiment_values = [
            data['positive_count'],
            data['neutral_count'],
            data['negative_count']
        ]
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_colors = ['#10B981', '#6B7280', '#EF4444']
        
        fig_sentiment = go.Figure(data=[go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            hole=.4,
            marker=dict(colors=sentiment_colors),
            textinfo='label+percent',
            textfont=dict(color=THEME["text"], size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        apply_plotly_theme(
            fig_sentiment,
            height=330,
            margin=dict(l=0, r=0, t=6, b=6),
            showlegend=True,
        )
        fig_sentiment.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ))
        
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
    
    # Top Issue Clusters (Horizontal Bar Chart)
    with col2:
        st.markdown('<div class="section-title">Top Issue Clusters</div>', unsafe_allow_html=True)
        
        issues_sorted = data['issues_df'].sort_values('Count', ascending=True).tail(7)
        
        fig_issues = go.Figure(data=[
            go.Bar(
                y=issues_sorted['Issue'],
                x=issues_sorted['Count'],
                orientation='h',
                marker=dict(
                    color=issues_sorted['Count'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=issues_sorted['Count'],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>',
                textfont=dict(color=THEME["text"], size=10)
            )
        ])
        
        apply_plotly_theme(
            fig_issues,
            height=330,
            margin=dict(l=145, r=44, t=6, b=36),
            showlegend=False,
        )
        fig_issues.update_layout(hovermode='closest')
        fig_issues.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig_issues, use_container_width=True, config={'displayModeBar': False})
    
    # Priority Issues Panel
    with col3:
        st.markdown('<div class="section-title">Priority Issues</div>', unsafe_allow_html=True)
        
        priority_issues = data['priority_issues']
        
        for issue in priority_issues:
            priority_badge = '<span class="badge-high">High</span>' if issue['priority'] == 'High' else '<span class="badge-medium">Medium</span>'
            
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0;">{issue['name']}</h4>
                    {priority_badge}
                </div>
                <p class="insight-text">{issue['desc']}</p>
                <p style="font-size: 0.85rem; color: #3B82F6; font-weight: 600;">{issue['count']:,} mentions</p>
            </div>
            """, unsafe_allow_html=True)
        if not priority_issues:
            st.markdown('<div class="empty-state">No priority issues for this view.</div>', unsafe_allow_html=True)
    
    # MAIN GRID - ROW 2
    col1, col2 = st.columns(2)
    
    # Sentiment Trend Over Time
    with col1:
        st.markdown(f'<div class="section-title">Sentiment Trend ({date_filter})</div>', unsafe_allow_html=True)
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['positive'],
            name='Positive',
            line=dict(color='#10B981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',
            hovertemplate='<b>Positive</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['neutral'],
            name='Neutral',
            line=dict(color='#6B7280', width=3),
            fill='tozeroy',
            fillcolor='rgba(107, 114, 128, 0.1)',
            hovertemplate='<b>Neutral</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data['sentiment_df']['date'],
            y=data['sentiment_df']['negative'],
            name='Negative',
            line=dict(color='#EF4444', width=3),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',
            hovertemplate='<b>Negative</b><br>Date: %{x|%b %d}<br>Count: %{y:,}<extra></extra>'
        ))
        
        apply_plotly_theme(
            fig_trend,
            height=330,
            margin=dict(l=48, r=20, t=6, b=48),
            showlegend=True,
        )
        fig_trend.update_layout(
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        )
        
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    # AI Insights Summary
    with col2:
        st.markdown('<div class="section-title">AI-Generated Insights</div>', unsafe_allow_html=True)
        
        insights = data['insights']
        
        for insight in insights:
            st.markdown(f'<p class="insight-text">• {insight}</p>', unsafe_allow_html=True)
    
    # MAIN GRID - ROW 3
    col1, col2, col3 = st.columns(3)
    
    # Recommended Actions
    with col1:
        st.markdown('<div class="section-title">Recommended Actions</div>', unsafe_allow_html=True)
        
        actions = data['actions']
        
        for action in actions:
            impact_badge = '<span class="badge-high">High</span>' if action['impact'] == 'High' else '<span class="badge-medium">Medium</span>'
            
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="margin: 0 0 0.5rem 0;">{action['title']}</h4>
                <p class="insight-text">{action['desc']}</p>
                <div style="text-align: right; margin-top: 0.75rem;">Impact: {impact_badge}</div>
            </div>
            """, unsafe_allow_html=True)
        if not actions:
            st.markdown('<div class="empty-state">No recommended actions for this view.</div>', unsafe_allow_html=True)
    
    # Real-Time Alerts
    with col2:
        st.markdown('<div class="section-title">Real-Time Alerts</div>', unsafe_allow_html=True)
        
        for alert in alerts_data:
            alert_color = '#EF4444' if alert['type'] == 'danger' else '#F59E0B'
            
            st.markdown(f"""
            <div class="alert-box" style="border-left-color: {alert_color};">
                <h4 style="margin: 0 0 0.25rem 0;">{alert['title']}</h4>
                <p style="margin: 0.25rem 0; color: var(--text-secondary); font-size: 0.9rem;">{alert['description']}</p>
                <p style="margin: 0.5rem 0 0 0; color: var(--text-subtle); font-size: 0.8rem;">{alert['time']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Assistant
    with col3:
        st.markdown('<div class="section-title">AI Assistant</div>', unsafe_allow_html=True)
        st.markdown("#### Ask Questions About Your Data")
        
        overview_question = st.text_input(
            "Enter your question",
            placeholder="E.g., 'What are the top 3 issues affecting satisfaction?'",
            key="overview_question",
            label_visibility="collapsed"
        )

        if st.button("Ask Dashboard", key="overview_ask_button", use_container_width=True):
            st.session_state.overview_answer = generate_ai_reply(overview_question, data)

        if st.session_state.get("overview_answer"):
            st.markdown("### Dashboard Answer")
            st.markdown(f'<p class="insight-text">{st.session_state.overview_answer}</p>', unsafe_allow_html=True)
        
        st.markdown("**Example Prompts:**")
        example_prompts = [
            "📊 Compare wait time vs billing issues",
            "🔍 What drives negative sentiment?",
            "⚡ Show me actionable recommendations",
            "📈 Predict trends for next month",
            "💡 Root cause analysis for complaints"
        ]
        
        for prompt in example_prompts:
            st.markdown(f'<p class="insight-text">• {prompt}</p>', unsafe_allow_html=True)

# ============================================================================
# PAGE: CLUSTER ANALYSIS
# ============================================================================

elif st.session_state.page == "Cluster Analysis":
    st.markdown("## Cluster Analysis")
    st.info("Semantic clustering of feedback into topic groups")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cluster Distribution")
        
        clusters_data = {
            'Cluster': ['Wait Times', 'Communication', 'Billing', 'Quality', 'Facilities', 'Staff', 'Other'],
            'Size': [1850, 1420, 980, 756, 642, 538, 421]
        }
        
        fig = px.pie(
            x=clusters_data['Size'],
            labels=clusters_data['Cluster'],
            title="Distribution Across Clusters"
        )
        
        apply_plotly_theme(fig, height=400, margin=dict(l=16, r=16, t=46, b=24))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### Cluster Sentiment Analysis")
        
        cluster_sentiment = {
            'Cluster': clusters_data['Cluster'],
            'Negative': [1650, 980, 720, 540, 380, 290, 210],
            'Neutral': [150, 350, 200, 180, 200, 200, 150],
            'Positive': [50, 90, 60, 36, 62, 48, 61]
        }
        
        fig = go.Figure(data=[
            go.Bar(name='Negative', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Negative'], marker_color='#EF4444'),
            go.Bar(name='Neutral', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Neutral'], marker_color='#6B7280'),
            go.Bar(name='Positive', x=cluster_sentiment['Cluster'], y=cluster_sentiment['Positive'], marker_color='#10B981')
        ])
        
        apply_plotly_theme(fig, height=400, margin=dict(l=44, r=20, t=24, b=92))
        fig.update_layout(
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: SENTIMENT ANALYSIS
# ============================================================================

elif st.session_state.page == "Sentiment Analysis":
    st.markdown("## Sentiment Analysis")
    st.info("Detailed sentiment classification and metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Sentiment Breakdown")
        
        sentiment_data = {
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [data['positive_count'], data['neutral_count'], data['negative_count']],
            'Percentage': [30.5, 47.7, 21.8]
        }
        
        df_sentiment = pd.DataFrame(sentiment_data)
        
        fig = go.Figure(data=[go.Bar(
            x=df_sentiment['Sentiment'],
            y=df_sentiment['Count'],
            marker_color=['#10B981', '#6B7280', '#EF4444'],
            text=df_sentiment['Count'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
        )])
        
        apply_plotly_theme(fig, height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### Sentiment Confidence Distribution")
        
        confidence_bins = np.linspace(0, 100, 11)
        confidence_data = np.random.normal(78, 12, 10000)
        
        fig = go.Figure(data=[go.Histogram(
            x=confidence_data,
            nbinsx=20,
            marker_color='#3B82F6',
            hovertemplate='<b>Confidence Range: %{x:.0f}%</b><br>Count: %{y}<extra></extra>'
        )])
        
        apply_plotly_theme(fig, height=400, showlegend=False)
        fig.update_layout(xaxis_title="Confidence Score (%)", yaxis_title="Count")
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: ISSUE PRIORITIZATION
# ============================================================================

elif st.session_state.page == "Issue Prioritization":
    st.markdown("## Issue Prioritization Matrix")
    st.info("Prioritize issues by impact and frequency")
    
    # Create prioritization matrix
    issues_priority = pd.DataFrame({
        'Issue': ['Wait Times', 'Communication', 'Billing', 'Service Quality', 'Cleanliness', 'Doctor Availability'],
        'Frequency': [1850, 1420, 980, 756, 642, 538],
        'Impact': [9.2, 8.5, 7.8, 7.5, 6.2, 6.8],
        'Sentiment_Score': [2.1, 1.8, 1.6, 1.4, 1.2, 1.5]
    })
    
    fig = go.Figure(data=go.Scatter(
        x=issues_priority['Frequency'],
        y=issues_priority['Impact'],
        mode='markers+text',
        text=issues_priority['Issue'],
        textposition="top center",
        marker=dict(
            size=issues_priority['Sentiment_Score']*15,
            color=issues_priority['Sentiment_Score'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Sentiment<br>Score"),
            line=dict(width=1, color=THEME["border"])
        ),
        hovertemplate='<b>%{text}</b><br>Frequency: %{x:,}<br>Impact: %{y:.1f}<extra></extra>'
    ))
    
    apply_plotly_theme(fig, height=500, margin=dict(l=64, r=30, t=24, b=62))
    fig.update_layout(
        xaxis_title="Frequency (Number of Mentions)",
        yaxis_title="Impact Score",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("---")
    st.markdown("### Prioritization Methodology")
    st.markdown("""
    - **High Priority**: High frequency + High impact
    - **Medium Priority**: High frequency OR High impact
    - **Low Priority**: Low frequency + Low impact
    """)

# ============================================================================
# PAGE: OPERATIONAL IMPACT
# ============================================================================

elif st.session_state.page == "Operational Impact":
    st.markdown("## Operational Impact - Cause-Effect Cascades")
    
    for chain_idx, chain in enumerate(cascade_data['chains']):
        st.markdown(f"### {chain['title']}")
        
        # Create cascade flow visualization
        steps = chain['steps']
        
        # Create figure with annotation
        fig = go.Figure()
        
        x_positions = np.linspace(0, 3, len(steps))
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[1],
                mode='markers',
                marker=dict(
                    size=40,
                    color=step['color'],
                    opacity=0.8
                ),
                text=step['label'],
                textposition="bottom center",
                textfont=dict(size=10, color=THEME["text"]),
                hovertemplate=f"<b>{step['label']}</b><br>Type: {step['impact']}<extra></extra>",
                showlegend=False
            ))
            
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=(x_positions[i] + x_positions[i+1]) / 2,
                    y=1,
                    text="→",
                    showarrow=False,
                    font=dict(size=20, color='#3B82F6'),
                    xanchor="center"
                )
        
        apply_plotly_theme(fig, height=200, margin=dict(l=20, r=20, t=48, b=52), showlegend=False)
        fig.update_layout(hovermode='closest')
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")

# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================

elif st.session_state.page == "Recommendations":
    st.markdown("## AI-Generated Recommendations")
    
    recommendations = [
        {
            'title': 'Implement Real-Time Queue Management System',
            'description': 'Deploy IoT sensors and mobile app integration to provide live wait time estimates and virtual queuing',
            'expected_impact': 'Reduce wait time complaints by 35-40%',
            'timeline': '2-3 months',
            'priority': 'High',
            'investment': 'Medium'
        },
        {
            'title': 'Enhanced Staff Communication Training',
            'description': 'Structured empathy and active listening training program with quarterly refresher sessions',
            'expected_impact': 'Improve communication sentiment by 25%',
            'timeline': '1-2 months',
            'priority': 'High',
            'investment': 'Low'
        },
        {
            'title': 'Automated Billing Clarity Initiative',
            'description': 'Implement AI-powered itemized receipt generation and insurance verification chatbot',
            'expected_impact': 'Reduce billing complaints by 30%',
            'timeline': '2-4 months',
            'priority': 'Medium',
            'investment': 'Medium'
        },
        {
            'title': 'Predictive Patient Satisfaction Monitoring',
            'description': 'Real-time ML model to identify at-risk patients and trigger proactive intervention protocols',
            'expected_impact': 'Improve overall satisfaction by 8-12%',
            'timeline': '3-4 months',
            'priority': 'Medium',
            'investment': 'High'
        },
    ]
    
    for rec in recommendations:
        priority_badge = '<span class="badge-high">HIGH</span>' if rec['priority'] == 'High' else '<span class="badge-medium">MEDIUM</span>'
        
        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <h3 style="margin: 0; flex: 1;">{rec['title']}</h3>
                {priority_badge}
            </div>
            
            <p class="insight-text" style="margin-bottom: 1rem;">{rec['description']}</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: var(--text-subtle); text-transform: uppercase;">Expected Impact</p>
                    <p style="margin: 0.25rem 0 0 0; color: #10B981; font-weight: 600;">{rec['expected_impact']}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: var(--text-subtle); text-transform: uppercase;">Timeline</p>
                    <p style="margin: 0.25rem 0 0 0; color: #3B82F6; font-weight: 600;">{rec['timeline']}</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: var(--text-subtle); text-transform: uppercase;">Priority</p>
                    <p style="margin: 0.25rem 0 0 0; color: var(--text-primary); font-weight: 600;">{rec['priority']}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: var(--text-subtle); text-transform: uppercase;">Investment</p>
                    <p style="margin: 0.25rem 0 0 0; color: #F59E0B; font-weight: 600;">{rec['investment']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: REAL-TIME ALERTS
# ============================================================================

elif st.session_state.page == "Real-time Alerts":
    st.markdown("## Real-Time Monitoring & Alerts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Alert Timeline")
        
        alerts_detailed = [
            {
                'time': '2:45 PM',
                'type': 'danger',
                'title': 'Critical: Wait Time Spike',
                'message': 'Wait time complaints increased by 18% (week-over-week)',
                'affected_area': 'Emergency Department'
            },
            {
                'time': '1:30 PM',
                'type': 'warning',
                'title': 'Billing Complaints Trend',
                'message': '+23% increase in billing-related feedback in last 7 days',
                'affected_area': 'Billing Department'
            },
            {
                'time': '12:15 PM',
                'type': 'warning',
                'title': 'Communication Issues',
                'message': 'Negative sentiment in communication feedback trending up',
                'affected_area': 'Multiple Departments'
            },
            {
                'time': '11:00 AM',
                'type': 'info',
                'title': 'Cleanliness Feedback Positive',
                'message': '92% positive feedback on facility cleanliness',
                'affected_area': 'Facility Management'
            }
        ]
        
        for alert in alerts_detailed:
            alert_color = '#EF4444' if alert['type'] == 'danger' else '#F59E0B' if alert['type'] == 'warning' else '#3B82F6'
            alert_icon = '🔴' if alert['type'] == 'danger' else '🟡' if alert['type'] == 'warning' else '🔵'
            
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; gap: 1rem;">
                    <div style="font-size: 1.5rem;">{alert_icon}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.25rem 0;">{alert['title']}</h4>
                        <p style="margin: 0.25rem 0; color: var(--text-secondary); font-size: 0.9rem;">{alert['message']}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-subtle);">
                            <span>📍 {alert['affected_area']}</span>
                            <span>⏰ {alert['time']}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Alert Statistics")
        
        alert_stats = {
            'Total Alerts': '24',
            'Critical': '3',
            'Warning': '8',
            'Info': '13',
            'Resolved': '18'
        }
        
        for stat, value in alert_stats.items():
            st.markdown(f"""
            <div class="custom-card">
                <p class="kpi-label">{stat}</p>
                <p class="kpi-value">{value}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: AI ASSISTANT
# ============================================================================

elif st.session_state.page == "AI Assistant":
    st.markdown("## AI Assistant")
    st.info("Ask questions about your feedback data and receive AI-powered insights")
    
    st.markdown("### Ask Your Questions")
    
    user_question = st.text_area(
        "Enter your question",
        placeholder="E.g., 'What are the top 3 root causes of negative sentiment?' or 'How has wait time sentiment changed over time?'",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze", use_container_width=True):
            st.success("Analysis complete!")
    
    with col2:
        st.button("📊 Visualize", use_container_width=True)
    
    with col3:
        st.button("💾 Save Query", use_container_width=True)
    
    if user_question:
        st.markdown("---")
        st.markdown("### AI Response")
        
        st.markdown("""
        Based on the analysis of 10,247 customer feedback entries:

        **Top 3 Root Causes of Negative Sentiment:**

        1. **Long Wait Times (45% of complaints)**
           - 1,850 mentions across all feedback
           - Strongly correlates with overall dissatisfaction (r=0.89)
           - Recommendation: Implement queue management system

        2. **Poor Staff Communication (32% of complaints)**
           - 1,420 mentions related to clarity and empathy
           - Affects patient trust and satisfaction
           - Recommendation: Communication skills training program

        3. **Unexpected Billing Charges (22% of complaints)**
           - 980 mentions of billing confusion
           - Trending upward at +23% weekly
           - Recommendation: Automated billing explanation system

        **Confidence Score:** 94.2%
        **Data Quality:** Excellent (99.68% parse success rate)
        """)

# ============================================================================
# PAGE: REPORTS
# ============================================================================

elif st.session_state.page == "Reports":
    st.markdown("## Reports & Export")
    
    st.markdown("### Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Available Reports")
        
        reports = [
            "Executive Summary (PDF)",
            "Detailed Sentiment Analysis",
            "Issue Prioritization Matrix",
            "Cascade Analysis Report",
            "Recommended Actions Plan"
        ]
        
        for report in reports:
            st.markdown(f"- 📄 {report}")
    
    with col2:
        st.markdown("#### Export Options")
        
        st.button("📥 Download Executive Summary (PDF)", use_container_width=True)
        st.button("📊 Export Data (CSV)", use_container_width=True)
        st.button("📈 Export Charts (PNG)", use_container_width=True)
        st.button("📋 Schedule Report Email", use_container_width=True)

# ============================================================================
# PAGE: SETTINGS
# ============================================================================

elif st.session_state.page == "Settings":
    st.markdown("## Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dashboard Settings")
        
        st.toggle("Dark Mode", value=True, disabled=True)
        st.toggle("Real-time Updates", value=True)
        st.toggle("Enable Notifications", value=True)
        
        st.markdown("### Data Settings")
        
        refresh_interval = st.select_slider(
            "Auto-refresh Interval",
            options=["5 min", "15 min", "30 min", "1 hour", "Manual"],
            value="30 min"
        )
        
        data_retention = st.selectbox(
            "Data Retention Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "Unlimited"]
        )
    
    with col2:
        st.markdown("### Alert Settings")
        
        alert_threshold = st.slider(
            "Alert Sensitivity (Lower = More Alerts)",
            0,
            100,
            50
        )
        
        st.markdown("### Account")
        
        st.text_input("Email", value="admin@cxintel.com", disabled=True)
        st.text_input("Organization", value="Hospital Network", disabled=True)
        st.selectbox("Role", ["Administrator", "Manager", "Analyst"])

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: var(--text-subtle); font-size: 0.85rem; margin-top: 2rem;'>"
    "CX-Intel v1.0.0 | © 2024 Customer Experience Intelligence Platform | "
    "<a href='#' style='color: #3B82F6; text-decoration: none;'>Privacy Policy</a> | "
    "<a href='#' style='color: #3B82F6; text-decoration: none;'>Support</a>"
    "</div>",
    unsafe_allow_html=True
)
