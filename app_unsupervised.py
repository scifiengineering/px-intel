"""
M4: Unsupervised-First Streamlit Dashboard
Interactive experience map with cluster auditing and causal insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import hashlib
import json
import os
from html import escape
from data_loader import DataLoader, LoaderStats
from unsupervised_clustering import UnsupervisedClusteringEngine
from cluster_audit import ClusterAuditEngine
from causal_reasoning import CausalReasoningEngine
from agent.action_intelligence import (
    CXActionIntelligenceAgent,
    insight_display_name,
    signal_reference,
)
from agent.ai_enhancement import AIEnhancementError, OpenAIInsightEnhancer
from pathlib import Path

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CX-Intel Discovery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_theme():
    """Apply Soft UI-inspired CX-Intel visual styling."""
    st.markdown(
        """
        <style>
        :root {
            --cx-ink: #172033;
            --cx-muted: #667085;
            --cx-subtle: #94a3b8;
            --cx-panel: rgba(255, 255, 255, 0.88);
            --cx-panel-solid: #ffffff;
            --cx-panel-soft: rgba(248, 250, 252, 0.82);
            --cx-border: rgba(94, 114, 228, 0.14);
            --cx-blue: #3b82f6;
            --cx-cyan: #06b6d4;
            --cx-green: #10b981;
            --cx-amber: #f59e0b;
            --cx-red: #ef4444;
            --cx-violet: #8b5cf6;
            --cx-shadow: 0 18px 45px rgba(20, 35, 70, 0.08);
            --cx-radius: 1rem;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(59, 130, 246, 0.16), transparent 26rem),
                radial-gradient(circle at 92% 0%, rgba(139, 92, 246, 0.12), transparent 24rem),
                linear-gradient(180deg, #f6f9ff 0%, #ffffff 44%, #f8fbff 100%);
            color: var(--cx-ink);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        [data-testid="stSidebar"] {
            background: var(--cx-panel-solid);
            border-right: 1px solid var(--cx-border);
            box-shadow: 12px 0 30px rgba(20, 35, 70, 0.04);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--cx-muted);
        }

        .cx-brand-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.35rem 0 1rem;
        }

        .cx-brand-mark {
            width: 38px;
            height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            color: #fff;
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.22);
            font-weight: 800;
        }

        .cx-brand-title {
            color: var(--cx-ink);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1;
        }

        .cx-brand-subtitle {
            color: var(--cx-muted);
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .cx-sidebar-card {
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel-soft);
            box-shadow: 0 10px 30px rgba(20, 35, 70, 0.06);
            margin: 0.8rem 0;
        }

        .cx-sidebar-status-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.44rem 0;
            color: var(--cx-muted);
            font-size: 0.82rem;
            font-weight: 720;
            border-bottom: 1px solid var(--cx-border);
        }

        .cx-sidebar-status-row:last-child {
            border-bottom: 0;
            padding-bottom: 0;
        }

        .cx-status-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.14);
            color: #059669;
            font-size: 0.72rem;
            font-weight: 850;
        }

        .cx-nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.72rem 0.82rem;
            border-radius: 0.8rem;
            color: var(--cx-muted);
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
        }

        .cx-nav-item.active {
            color: var(--cx-ink);
            background: linear-gradient(180deg, #ffffff, #eef5ff);
            box-shadow: 0 10px 24px rgba(20, 35, 70, 0.08);
        }

        .cx-nav-icon {
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 0.65rem;
            color: #fff;
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-cyan));
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.25rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.6rem 0.72rem;
            border-radius: 0.8rem;
            color: var(--cx-muted);
            font-weight: 750;
            transition: background 160ms ease, box-shadow 160ms ease, color 160ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: var(--cx-panel-soft);
            color: var(--cx-ink);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(180deg, #ffffff, #eef5ff);
            color: #172033;
            box-shadow: 0 10px 24px rgba(20, 35, 70, 0.08);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
            color: inherit !important;
        }

        .cx-hero {
            overflow: hidden;
            position: relative;
            padding: 1.55rem 1.75rem;
            margin-bottom: 1.15rem;
            border-radius: var(--cx-radius);
            background:
                radial-gradient(circle at 88% 18%, rgba(6, 182, 212, 0.25), transparent 28%),
                linear-gradient(135deg, #172033 0%, #263a7a 54%, #3b82f6 100%);
            box-shadow: var(--cx-shadow);
        }

        .cx-hero::after {
            content: "";
            position: absolute;
            inset: auto -8% -42% 46%;
            height: 230px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            transform: rotate(-10deg);
        }

        .cx-hero > * {
            position: relative;
            z-index: 1;
        }

        .cx-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
            gap: 1.25rem;
            align-items: center;
        }

        .cx-hero h1 {
            margin: 0 0 0.35rem 0;
            color: #ffffff;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 760;
            letter-spacing: 0;
        }

        .cx-hero p {
            margin: 0;
            color: rgba(255, 255, 255, 0.86);
            font-size: 1rem;
        }

        .cx-eyebrow {
            color: var(--cx-blue);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .cx-hero .cx-eyebrow {
            color: rgba(255, 255, 255, 0.72);
        }

        .cx-command {
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-status-dot {
            width: 10px;
            height: 10px;
            display: inline-block;
            border-radius: 999px;
        }

        .cx-dot-green { background: var(--cx-green); }
        .cx-dot-amber { background: var(--cx-amber); }
        .cx-dot-red { background: var(--cx-red); }

        .cx-kpi-card {
            min-height: 132px;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
            margin-bottom: 0.35rem;
        }

        .cx-kpi-label {
            margin: 0;
            color: var(--cx-muted);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .cx-kpi-value {
            margin: 0.45rem 0 0.2rem;
            color: var(--cx-ink);
            font-size: 1.75rem;
            line-height: 1;
            font-weight: 800;
        }

        .cx-kpi-trend {
            color: var(--cx-muted);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .cx-panel {
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-decision-card,
        .cx-action-card {
            min-height: 100%;
            padding: 1.1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-decision-card.featured {
            background:
                radial-gradient(circle at 90% 12%, rgba(59, 130, 246, 0.16), transparent 28%),
                var(--cx-panel);
        }

        .cx-stakeholder-panel {
            margin: 0.75rem 0 1.05rem;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background:
                radial-gradient(circle at 92% 10%, rgba(6, 182, 212, 0.12), transparent 28%),
                var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-stakeholder-panel h4 {
            margin: 0.45rem 0 0.45rem;
            color: var(--cx-ink);
            font-size: 1.05rem;
        }

        .cx-stakeholder-panel p {
            margin: 0;
            color: var(--cx-muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .cx-stakeholder-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .cx-stakeholder-meta span {
            display: inline-flex;
            padding: 0.32rem 0.55rem;
            border-radius: 999px;
            background: var(--cx-panel-soft);
            color: var(--cx-muted);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .cx-alert-band {
            margin: 1.25rem 0 1.55rem;
            padding: 1.1rem;
            border: 1px solid rgba(239, 68, 68, 0.24);
            border-radius: var(--cx-radius);
            background:
                radial-gradient(circle at 94% 16%, rgba(239, 68, 68, 0.14), transparent 30%),
                var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-alert-band h3 {
            margin: 0.25rem 0 0.35rem;
            color: var(--cx-ink);
            font-size: 1.08rem;
        }

        .cx-alert-band p {
            margin: 0;
            color: var(--cx-muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .cx-alert-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.85rem;
        }

        .cx-alert-grid.single {
            grid-template-columns: minmax(0, 1fr);
        }

        .cx-alert-card {
            min-height: 188px;
            padding: 0.9rem;
            border: 1px solid rgba(239, 68, 68, 0.18);
            border-radius: 0.9rem;
            background: var(--cx-panel-soft);
        }

        .cx-alert-grid.single .cx-alert-card {
            min-height: auto;
        }

        .cx-alert-card h4 {
            margin: 0.35rem 0 0.35rem;
            color: var(--cx-ink);
            font-size: 0.98rem;
        }

        .cx-alert-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.65rem;
        }

        .cx-alert-meta span {
            display: inline-flex;
            padding: 0.25rem 0.48rem;
            border-radius: 999px;
            background: rgba(239, 68, 68, 0.1);
            color: #dc2626;
            font-size: 0.7rem;
            font-weight: 850;
        }

        .cx-icon-block {
            width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 0.9rem;
            color: #fff;
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.18);
            font-size: 0.78rem;
            font-weight: 850;
        }

        .cx-card-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .cx-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.1);
            color: var(--cx-blue);
            font-size: 0.74rem;
            font-weight: 800;
        }

        .cx-chip.high {
            background: rgba(239, 68, 68, 0.12);
            color: #dc2626;
        }

        .cx-chip.medium {
            background: rgba(245, 158, 11, 0.14);
            color: #d97706;
        }

        .cx-chip.low {
            background: rgba(16, 185, 129, 0.13);
            color: #059669;
        }

        .cx-action-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.85rem 0 1.25rem;
        }

        .cx-action-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.85rem;
        }

        .cx-action-meta div {
            padding: 0.65rem;
            border-radius: 0.78rem;
            background: var(--cx-panel-soft);
        }

        .cx-action-meta span {
            display: block;
            color: var(--cx-muted);
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .cx-action-meta strong {
            display: block;
            color: var(--cx-ink);
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }

        .cx-intel-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.75rem 0 1rem;
        }

        .cx-segment-card,
        .cx-signal-panel {
            min-height: 100%;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-segment-card.featured {
            background:
                radial-gradient(circle at 94% 8%, rgba(16, 185, 129, 0.13), transparent 30%),
                var(--cx-panel);
        }

        .cx-segment-title {
            margin: 0 0 0.35rem;
            color: var(--cx-ink);
            font-size: 1rem;
            font-weight: 820;
        }

        .cx-quote {
            margin-top: 0.75rem;
            padding: 0.75rem;
            border-left: 3px solid var(--cx-blue);
            border-radius: 0.75rem;
            background: var(--cx-panel-soft);
            color: var(--cx-muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .cx-token-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.65rem;
        }

        .cx-token {
            display: inline-flex;
            padding: 0.24rem 0.5rem;
            border-radius: 999px;
            background: rgba(6, 182, 212, 0.1);
            color: var(--cx-cyan);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .cx-signal-item {
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--cx-border);
        }

        .cx-signal-item:last-child {
            border-bottom: 0;
            padding-bottom: 0;
        }

        .cx-signal-item h5 {
            margin: 0 0 0.25rem;
            color: var(--cx-ink);
            font-size: 0.92rem;
        }

        .cx-signal-item p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.42;
        }

        .cx-graph-detail {
            min-height: 100%;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-graph-type {
            display: inline-flex;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.1);
            color: var(--cx-blue);
            font-size: 0.72rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .cx-relationship-list {
            margin: 0.55rem 0 0;
            padding: 0;
            list-style: none;
        }

        .cx-relationship-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--cx-border);
            font-size: 0.82rem;
        }

        .cx-relationship-list li:last-child {
            border-bottom: 0;
        }

        .cx-graph-outline {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.8rem 0 1rem;
        }

        .cx-graph-step {
            min-height: 88px;
            padding: 0.75rem;
            border: 1px solid var(--cx-border);
            border-radius: 0.85rem;
            background: var(--cx-panel);
            box-shadow: 0 8px 22px rgba(20, 35, 70, 0.05);
        }

        .cx-graph-step span {
            display: inline-flex;
            width: 24px;
            height: 24px;
            align-items: center;
            justify-content: center;
            border-radius: 0.55rem;
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 850;
        }

        .cx-graph-step strong {
            display: block;
            margin-top: 0.45rem;
            color: var(--cx-ink);
            font-size: 0.84rem;
        }

        .cx-graph-step p {
            margin: 0.25rem 0 0;
            font-size: 0.72rem;
            line-height: 1.35;
        }

        .cx-graph-key {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.55rem 0 0.85rem;
        }

        .cx-graph-key-item {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.58rem;
            border: 1px solid var(--cx-border);
            border-radius: 999px;
            background: var(--cx-panel);
            color: var(--cx-muted);
            font-size: 0.76rem;
            font-weight: 800;
            box-shadow: 0 6px 16px rgba(20, 35, 70, 0.04);
        }

        .cx-key-dot {
            width: 0.65rem;
            height: 0.65rem;
            display: inline-block;
            border-radius: 999px;
        }

        .cx-readout-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem;
        }

        .cx-readout-card {
            min-height: 126px;
            padding: 0.9rem;
            border: 1px solid var(--cx-border);
            border-radius: 0.9rem;
            background: var(--cx-panel);
            box-shadow: 0 8px 22px rgba(20, 35, 70, 0.05);
        }

        .cx-readout-card h5 {
            margin: 0.35rem 0;
            color: var(--cx-ink);
            font-size: 0.95rem;
        }

        .cx-readout-card p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .cx-impact-card {
            min-height: 100%;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-impact-card h4 {
            margin: 0.35rem 0;
            color: var(--cx-ink);
            font-size: 1rem;
        }

        .cx-impact-card p {
            margin: 0;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .cx-impact-path {
            margin-top: 0.75rem;
            padding: 0.75rem;
            border-radius: 0.8rem;
            background: var(--cx-panel-soft);
            color: var(--cx-muted);
            font-size: 0.82rem;
        }

        .cx-command-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.95rem 0 1.45rem;
        }

        .cx-command-card {
            min-height: 164px;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-command-card h4 {
            margin: 0.45rem 0 0.35rem;
            color: var(--cx-ink);
            font-size: 1rem;
        }

        .cx-command-card p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.42;
        }

        .cx-priority-row {
            display: grid;
            grid-template-columns: 64px minmax(0, 1.1fr) minmax(0, 1.25fr) minmax(0, 1.15fr);
            gap: 0.85rem;
            align-items: stretch;
            padding: 0.9rem;
            margin-bottom: 0.7rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: 0 8px 22px rgba(20, 35, 70, 0.05);
        }

        .cx-rank-pill {
            display: inline-flex;
            width: 46px;
            height: 46px;
            align-items: center;
            justify-content: center;
            border-radius: 0.9rem;
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
            color: #ffffff;
            font-weight: 850;
        }

        .cx-priority-row h4 {
            margin: 0 0 0.3rem;
            color: var(--cx-ink);
            font-size: 1rem;
        }

        .cx-priority-row p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .cx-action-group {
            min-height: 100%;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-action-group h4 {
            margin: 0 0 0.6rem;
            color: var(--cx-ink);
        }

        .cx-action-group-item {
            padding: 0.65rem 0;
            border-bottom: 1px solid var(--cx-border);
        }

        .cx-action-group-item:last-child {
            border-bottom: 0;
        }

        .cx-action-group-item strong {
            display: block;
            color: var(--cx-ink);
            margin-bottom: 0.2rem;
        }

        .cx-section-heading {
            margin: 2rem 0 0.85rem;
            color: var(--cx-ink);
            font-weight: 800;
            font-size: 1.3rem;
            line-height: 1.22;
            letter-spacing: 0;
        }

        .cx-page-header {
            margin: 0.2rem 0 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--cx-border);
        }

        .cx-page-header-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .cx-page-header h2 {
            margin: 0;
            color: var(--cx-ink);
            font-size: 1.55rem;
            line-height: 1.18;
            font-weight: 840;
            letter-spacing: 0;
        }

        .cx-page-header p {
            max-width: 880px;
            margin: 0.42rem 0 0;
            color: var(--cx-muted);
            font-size: 0.93rem;
            line-height: 1.48;
        }

        .cx-slogan-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.1rem 0 1.15rem;
            padding: 0.95rem 1.05rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background:
                radial-gradient(circle at 88% 18%, rgba(6, 182, 212, 0.14), transparent 28%),
                linear-gradient(135deg, rgba(23, 32, 51, 0.96), rgba(38, 58, 122, 0.92));
            box-shadow: var(--cx-shadow);
        }

        .cx-slogan-strip h1 {
            margin: 0.15rem 0 0;
            color: #ffffff;
            font-size: 1.35rem;
            line-height: 1.2;
            font-weight: 820;
            letter-spacing: 0;
        }

        .cx-slogan-strip p {
            margin: 0.25rem 0 0;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.86rem;
            line-height: 1.4;
        }

        .cx-slogan-strip .cx-eyebrow {
            color: rgba(255, 255, 255, 0.68);
        }

        .cx-slogan-badge {
            flex: 0 0 auto;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.13);
            color: #ffffff;
            font-size: 0.74rem;
            font-weight: 850;
        }

        .cx-table-panel {
            margin: 1.2rem 0 0.55rem;
            padding: 0.95rem 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: 0 10px 28px rgba(20, 35, 70, 0.05);
        }

        .cx-table-panel h4 {
            margin: 0;
            color: var(--cx-ink);
            font-size: 1rem;
            font-weight: 820;
        }

        .cx-table-panel p {
            margin: 0.35rem 0 0;
            color: var(--cx-muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .cx-table-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.7rem;
        }

        .cx-table-meta span {
            display: inline-flex;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: var(--cx-panel-soft);
            color: var(--cx-muted);
            font-size: 0.72rem;
            font-weight: 800;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
            backdrop-filter: blur(12px);
        }

        div[data-testid="stMetric"] label {
            color: var(--cx-muted);
            font-weight: 750;
        }

        div[data-testid="stMetricValue"] {
            color: var(--cx-ink);
            font-weight: 760;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.35rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0.8rem;
            color: var(--cx-muted);
            font-weight: 750;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
            color: #ffffff;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: 0 8px 24px rgba(20, 35, 70, 0.06);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            overflow: hidden;
            box-shadow: var(--cx-shadow);
        }

        div[data-testid="stPlotlyChart"] {
            padding: 0.35rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-radius: 0.85rem;
            background: linear-gradient(180deg, #ffffff, #edf5ff);
            color: #172033 !important;
            font-weight: 800;
            box-shadow: 0 8px 18px rgba(20, 35, 70, 0.06);
        }

        .stButton > button *,
        .stDownloadButton > button * {
            color: #172033 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cx-blue);
            color: #174ea6 !important;
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.16);
        }

        .stButton > button:hover *,
        .stDownloadButton > button:hover * {
            color: #174ea6 !important;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background: var(--cx-panel);
            box-shadow: 0 8px 20px rgba(20, 35, 70, 0.06);
        }

        div[data-testid="stAlert"] {
            border-radius: var(--cx-radius);
            border: 1px solid var(--cx-border);
        }

        h2, h3 {
            color: var(--cx-ink);
            letter-spacing: 0;
        }

        p, li, label {
            color: var(--cx-muted);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --cx-ink: #f8fafc;
                --cx-muted: #cbd5e1;
                --cx-subtle: #94a3b8;
                --cx-panel: rgba(31, 42, 68, 0.86);
                --cx-panel-solid: #172033;
                --cx-panel-soft: rgba(31, 42, 68, 0.72);
                --cx-border: rgba(148, 163, 184, 0.18);
                --cx-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 8%, rgba(59, 130, 246, 0.14), transparent 26rem),
                    linear-gradient(180deg, #0f172a 0%, #111827 100%);
            }

            [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, var(--cx-blue), var(--cx-violet));
                color: #ffffff;
            }
        }

        @media (max-width: 900px) {
            .cx-hero-grid {
                grid-template-columns: 1fr;
            }

            .cx-action-grid,
            .cx-action-meta,
            .cx-alert-grid,
            .cx-intel-grid,
            .cx-graph-outline,
            .cx-readout-grid,
            .cx-command-grid,
            .cx-priority-row {
                grid-template-columns: 1fr;
            }

            .cx-slogan-strip {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_app_theme()


def parse_uploaded_feedback(upload_bytes, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            io.BytesIO(upload_bytes),
            engine="python",
            on_bad_lines="skip",
        )
    if suffix == ".txt":
        raw_text = upload_bytes.decode("utf-8", errors="replace")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return pd.DataFrame({"content": lines})
    raise ValueError("Upload a CSV or TXT file.")


def guess_feedback_text_column(columns):
    normalized = {str(column).lower().strip(): column for column in columns}
    for candidate in [
        "content",
        "feedback",
        "comment",
        "comments",
        "review",
        "text",
        "message",
        "description",
    ]:
        if candidate in normalized:
            return normalized[candidate]
    return columns[0] if columns else None


def normalize_feedback_dataframe(raw_df, text_column):
    df = raw_df.copy()
    if text_column not in df.columns:
        raise ValueError(f"Column `{text_column}` was not found in the uploaded file.")

    total_rows = len(df)
    df[text_column] = df[text_column].where(df[text_column].notna(), "")
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.strip() != ""].copy()
    empty_rows = total_rows - len(df)

    df["content"] = df[text_column]
    df["text_normalized"] = df["content"].apply(DataLoader._normalize_text)
    df = df[df["text_normalized"].str.strip() != ""].copy()

    text_lengths = df["content"].str.len()
    successful_rows = len(df)
    failed_rows = total_rows - successful_rows
    stats = LoaderStats(
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        empty_rows=empty_rows,
        success_rate=round(100 * successful_rows / total_rows, 2) if total_rows > 0 else 0,
        min_length=int(text_lengths.min()) if len(text_lengths) > 0 else 0,
        max_length=int(text_lengths.max()) if len(text_lengths) > 0 else 0,
        avg_length=float(text_lengths.mean()) if len(text_lengths) > 0 else 0,
    )
    return df, stats


def default_feedback_source():
    return {
        "id": "sample",
        "kind": "default",
        "label": "Sample data",
        "upload_bytes": None,
        "upload_name": None,
        "text_column": "content",
        "row_count": 108,
    }


def build_feedback_source(uploaded_file, selected_column=None):
    if uploaded_file is None:
        return default_feedback_source()

    upload_bytes = uploaded_file.getvalue()
    source_id = feedback_source_id(uploaded_file.name, upload_bytes, selected_column)
    return {
        "id": source_id,
        "kind": "upload",
        "label": uploaded_file.name,
        "upload_bytes": upload_bytes,
        "upload_name": uploaded_file.name,
        "text_column": selected_column,
        "row_count": None,
    }


def feedback_source_id(upload_name, upload_bytes, text_column):
    digest = hashlib.sha256(
        upload_bytes + str(text_column or "").encode("utf-8")
    ).hexdigest()[:12]
    safe_name = Path(str(upload_name or "upload")).stem[:24] or "upload"
    return f"upload-{safe_name}-{digest}"


def get_feedback_sources():
    upload_map = {}
    for source in load_saved_feedback_sources():
        upload_map[source["id"]] = source
    for source in st.session_state.get("uploaded_feedback_sources", []):
        upload_map[source["id"]] = source
    return [default_feedback_source()] + list(upload_map.values())


def remember_uploaded_feedback_source(source, row_count):
    stored_source = dict(source)
    stored_source["row_count"] = row_count
    uploads = [
        item
        for item in st.session_state.get("uploaded_feedback_sources", [])
        if item.get("id") != stored_source["id"]
    ]
    uploads.append(stored_source)
    st.session_state.uploaded_feedback_sources = uploads
    persist_feedback_source(stored_source, row_count)


def select_feedback_source_by_id(source_id):
    for source in get_feedback_sources():
        if source.get("id") == source_id:
            return source
    return default_feedback_source()


def feedback_source_label(source):
    row_text = (
        f" · {source['row_count']:,} rows"
        if source.get("row_count") is not None
        else ""
    )
    return f"{source['label']}{row_text}"


PX_INTEL_HOME = Path.home() / ".px_intel"
AI_SETTINGS_PATH = PX_INTEL_HOME / "ai_settings.json"
UPLOADS_DIR = PX_INTEL_HOME / "uploads"
UPLOADS_INDEX_PATH = UPLOADS_DIR / "uploads.json"


def safe_local_filename(value):
    """Return a filesystem-safe filename fragment."""
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value or "upload")
    )
    return cleaned.strip("._") or "upload"


def load_saved_feedback_sources():
    """Load persisted upload metadata and file bytes from local disk."""
    try:
        if not UPLOADS_INDEX_PATH.exists():
            return []
        index_data = json.loads(UPLOADS_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    sources = []
    for item in index_data if isinstance(index_data, list) else []:
        try:
            upload_path = Path(item.get("upload_path", ""))
            if not upload_path.exists():
                continue
            sources.append(
                {
                    "id": item["id"],
                    "kind": "upload",
                    "label": item.get("label", upload_path.name),
                    "upload_bytes": upload_path.read_bytes(),
                    "upload_name": item.get("upload_name", upload_path.name),
                    "text_column": item.get("text_column", "content"),
                    "row_count": item.get("row_count"),
                    "saved": True,
                    "upload_path": str(upload_path),
                }
            )
        except Exception:
            continue
    return sources


def write_saved_feedback_sources(metadata_items):
    """Write persisted upload metadata."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_INDEX_PATH.write_text(
        json.dumps(metadata_items, indent=2),
        encoding="utf-8",
    )


def persist_feedback_source(source, row_count):
    """Persist uploaded data locally so it survives app refreshes."""
    if source.get("kind") != "upload" or not source.get("upload_bytes"):
        return

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(source.get("upload_name") or "upload.csv")).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        suffix = ".csv"
    safe_id = safe_local_filename(source.get("id"))
    upload_path = UPLOADS_DIR / f"{safe_id}{suffix}"
    upload_path.write_bytes(source["upload_bytes"])

    metadata = {
        "id": source["id"],
        "kind": "upload",
        "label": source.get("label") or source.get("upload_name") or upload_path.name,
        "upload_name": source.get("upload_name") or upload_path.name,
        "text_column": source.get("text_column") or "content",
        "row_count": row_count,
        "upload_path": str(upload_path),
    }
    existing = []
    if UPLOADS_INDEX_PATH.exists():
        try:
            existing = json.loads(UPLOADS_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing = [
        item
        for item in existing
        if isinstance(item, dict) and item.get("id") != metadata["id"]
    ]
    existing.append(metadata)
    write_saved_feedback_sources(existing)


def clear_saved_feedback_sources():
    """Remove locally persisted uploads."""
    if UPLOADS_INDEX_PATH.exists():
        try:
            for item in json.loads(UPLOADS_INDEX_PATH.read_text(encoding="utf-8")):
                upload_path = Path(item.get("upload_path", ""))
                if upload_path.exists() and upload_path.parent == UPLOADS_DIR:
                    upload_path.unlink()
        except Exception:
            pass
    try:
        if UPLOADS_INDEX_PATH.exists():
            UPLOADS_INDEX_PATH.unlink()
    except Exception:
        pass


def load_saved_ai_settings():
    """Load locally saved AI settings from this Mac, if available."""
    try:
        if AI_SETTINGS_PATH.exists():
            data = json.loads(AI_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_ai_settings(api_key, model, generation_strength, enabled=True):
    """Persist AI settings locally so refreshes keep the configured mode."""
    AI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "api_key": str(api_key or "").strip(),
        "model": str(model or "gpt-4o-mini").strip(),
        "generation_strength": generation_strength or "Board-ready",
        "enabled": bool(enabled),
    }
    AI_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_saved_ai_settings():
    """Remove the local AI settings file."""
    try:
        if AI_SETTINGS_PATH.exists():
            AI_SETTINGS_PATH.unlink()
    except Exception:
        pass


def read_openai_key_from_config():
    """Read an OpenAI key from Streamlit secrets or the shell environment."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("OPENAI_API_KEY", "")).strip()


def read_openai_model_from_config():
    """Read the preferred AI model from config with a low-cost default."""
    try:
        secret_model = st.secrets.get("OPENAI_MODEL", "")
    except Exception:
        secret_model = ""
    return str(secret_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()


def build_ai_config(
    api_key,
    model,
    enabled,
    refresh_requested=False,
    generation_strength="Board-ready",
):
    """Return normalized optional AI enhancement settings."""
    clean_key = str(api_key or "").strip()
    return {
        "enabled": bool(enabled and clean_key),
        "api_key": clean_key,
        "model": str(model or "gpt-4o-mini").strip(),
        "refresh_requested": bool(refresh_requested),
        "generation_strength": generation_strength,
    }


def render_sidebar():
    """Render Soft UI-inspired PX-Intel navigation context."""
    with st.sidebar:
        st.markdown(
            """
            <div class="cx-brand-row">
                <div class="cx-brand-mark">PX</div>
                <div>
                    <div class="cx-brand-title">PX-Intel</div>
                    <div class="cx-brand-subtitle">Experience intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_labels = {
            "Overview": "01  Overview",
            "Customer Intelligence": "02  Customer Intelligence",
            "Cause & Effect Graph": "03  Cause & Effect Graph",
            "Cluster Analysis": "04  Signal Analysis",
            "Operational Impact": "05  Operational Impact",
            "Reports & Export": "06  Reports & Export",
        }
        selected_section = st.radio(
            "Workspace",
            options=list(nav_labels.keys()),
            format_func=lambda section: nav_labels[section],
            label_visibility="collapsed",
            key="cx_sidebar_section",
        )

        if "active_feedback_source_id" not in st.session_state:
            st.session_state.active_feedback_source_id = "sample"

        feedback_sources = get_feedback_sources()
        source_ids = [source["id"] for source in feedback_sources]
        if st.session_state.active_feedback_source_id not in source_ids:
            st.session_state.active_feedback_source_id = "sample"
        if (
            st.session_state.get("feedback_source_selector")
            != st.session_state.active_feedback_source_id
        ):
            st.session_state.feedback_source_selector = (
                st.session_state.active_feedback_source_id
            )

        feedback_source = select_feedback_source_by_id(
            st.session_state.active_feedback_source_id
        )
        with st.expander("Data source", expanded=True):
            selected_source_id = st.selectbox(
                "Active dataset",
                source_ids,
                index=source_ids.index(st.session_state.active_feedback_source_id),
                format_func=lambda source_id: feedback_source_label(
                    select_feedback_source_by_id(source_id)
                ),
                key="feedback_source_selector",
                help="Switch between the bundled sample and uploaded files kept in this session.",
            )
            if selected_source_id != st.session_state.active_feedback_source_id:
                st.session_state.active_feedback_source_id = selected_source_id
                feedback_source = select_feedback_source_by_id(selected_source_id)

            uploaded_file = st.file_uploader(
                "Upload customer feedback",
                type=["csv", "txt"],
                help="Upload a CSV with a feedback text column, or a TXT file with one comment per line.",
                key="customer_feedback_upload",
            )
            if uploaded_file is None:
                if feedback_source["kind"] == "default":
                    st.caption("Using bundled `text_data.csv` sample feedback.")
                elif feedback_source.get("saved"):
                    st.caption(
                        "Using a saved uploaded dataset. It is stored locally on this Mac."
                    )
                else:
                    st.caption(
                        "Using an uploaded dataset stored in this browser session."
                    )
            else:
                try:
                    upload_bytes = uploaded_file.getvalue()
                    preview_df = parse_uploaded_feedback(upload_bytes, uploaded_file.name)
                    if preview_df.empty:
                        raise ValueError("The uploaded file does not contain feedback rows.")
                    default_column = guess_feedback_text_column(list(preview_df.columns))
                    column_options = list(preview_df.columns)
                    default_index = (
                        column_options.index(default_column)
                        if default_column in column_options
                        else 0
                    )
                    selected_column = st.selectbox(
                        "Feedback text column",
                        column_options,
                        index=default_index,
                        key="customer_feedback_text_column",
                    )
                    feedback_source = build_feedback_source(
                        uploaded_file,
                        selected_column,
                    )
                    source_already_stored = any(
                        item.get("id") == feedback_source["id"]
                        for item in st.session_state.get(
                            "uploaded_feedback_sources", []
                        )
                    )
                    remember_uploaded_feedback_source(
                        feedback_source,
                        len(preview_df),
                    )
                    if not source_already_stored:
                        st.session_state.active_feedback_source_id = feedback_source["id"]
                        st.rerun()
                    if (
                        st.session_state.active_feedback_source_id
                        == feedback_source["id"]
                    ):
                        st.success(
                            f"Active upload saved locally: {len(preview_df):,} rows ready for PX-Intel."
                        )
                    else:
                        st.info(
                            f"Uploaded dataset saved locally: {len(preview_df):,} rows."
                        )
                        if st.button(
                            "Use this uploaded dataset",
                            key=f"use_uploaded_dataset_{feedback_source['id']}",
                            width="stretch",
                        ):
                            st.session_state.active_feedback_source_id = feedback_source["id"]
                            st.rerun()
                    st.caption(f"File: `{uploaded_file.name}`")
                except Exception as exc:
                    st.error(f"Upload could not be read: {exc}")
                    st.caption("PX-Intel will continue using the bundled sample data.")

            uploaded_count = max(len(get_feedback_sources()) - 1, 0)
            if uploaded_count:
                clear_uploads = st.button(
                    "Clear uploaded datasets",
                    key="clear_uploaded_feedback_sources",
                    help="Remove uploaded files from this session and local saved upload storage. The bundled sample data remains available.",
                    width="stretch",
                )
                if clear_uploads:
                    st.session_state.uploaded_feedback_sources = []
                    clear_saved_feedback_sources()
                    st.session_state.active_feedback_source_id = "sample"
                    st.rerun()
                st.caption(
                    f"{uploaded_count} uploaded dataset(s) available to PX-Intel. "
                    "Saved uploads are stored locally outside the project folder."
                )

        saved_ai_settings = load_saved_ai_settings()
        configured_key = read_openai_key_from_config() or saved_ai_settings.get(
            "api_key",
            "",
        )
        configured_model = saved_ai_settings.get(
            "model",
            read_openai_model_from_config(),
        )
        saved_strength = saved_ai_settings.get("generation_strength", "Board-ready")
        if saved_strength not in ["Board-ready", "Operational detail", "Brief"]:
            saved_strength = "Board-ready"
        if "openai_api_key_input" not in st.session_state and configured_key:
            st.session_state.openai_api_key_input = configured_key
        if "openai_model_input" not in st.session_state:
            st.session_state.openai_model_input = configured_model
        if "ai_generation_strength" not in st.session_state:
            st.session_state.ai_generation_strength = saved_strength
        with st.expander("AI enhancement", expanded=False):
            st.caption(
                "Add an OpenAI key once. PX-Intel will use it automatically for smarter signal language, reports, and agent answers."
            )
            entered_key = st.text_input(
                "OpenAI API key",
                type="password",
                placeholder="Uses secrets/env if left blank",
                key="openai_api_key_input",
            )
            ai_model = st.text_input(
                "AI model",
                key="openai_model_input",
            )
            generation_strength = st.selectbox(
                "Generation strength",
                ["Board-ready", "Operational detail", "Brief"],
                key="ai_generation_strength",
                help="Controls how much structure and detail AI-generated answers and reports should produce.",
            )
            ai_key = entered_key.strip() or configured_key
            ai_enabled = bool(ai_key)
            save_col, clear_col = st.columns(2)
            with save_col:
                save_ai = st.button(
                    "Save AI settings",
                    key="save_ai_settings",
                    help="Save the key and model locally on this Mac so refreshes keep AI mode enabled.",
                    width="stretch",
                )
            with clear_col:
                clear_ai = st.button(
                    "Clear saved AI",
                    key="clear_ai_settings",
                    help="Remove the locally saved AI settings file.",
                    width="stretch",
                )
            if save_ai:
                if ai_key:
                    save_ai_settings(
                        ai_key,
                        ai_model,
                        generation_strength,
                        True,
                    )
                    st.success("AI settings saved locally for future refreshes.")
                else:
                    st.warning("Enter an OpenAI API key before saving AI settings.")
            if clear_ai:
                clear_saved_ai_settings()
                st.success("Saved AI settings cleared. Refresh to start without saved AI settings.")
            refresh_requested = False
            if ai_enabled and ai_key:
                st.success("AI enhancement ready. PX-Intel will use the key automatically.")
                refresh_requested = st.button(
                    "Refresh AI language",
                    key="refresh_ai_language",
                    help="Regenerate enhanced signal names and recommendations for the current data.",
                )
            else:
                st.caption("PX-Intel will use the local intelligence layer until an OpenAI key is provided.")
            st.caption(
                "Saved settings are stored locally on this Mac, outside the git repository."
            )

        ai_config = build_ai_config(
            ai_key,
            ai_model,
            ai_enabled,
            refresh_requested,
            generation_strength,
        )

        st.markdown(
            """
            <div class="cx-sidebar-card">
                <div class="cx-card-topline" style="margin-bottom:0.55rem;">
                    <div>
                        <div class="cx-eyebrow">System status</div>
                        <h4 style="margin: 0.25rem 0 0;">M5 Action Agent online</h4>
                    </div>
                    <span class="cx-status-pill">Live</span>
                </div>
                <div class="cx-sidebar-status-row"><span class="cx-status-dot cx-dot-green"></span><span>Feedback ingestion ready</span></div>
                <div class="cx-sidebar-status-row"><span class="cx-status-dot cx-dot-amber"></span><span>Signal audit cache enabled</span></div>
                <div class="cx-sidebar-status-row"><span class="cx-status-dot cx-dot-red"></span><span>High-priority issues surfaced first</span></div>
                <p style="font-size: 0.8rem; margin:0.72rem 0 0;">M4 + M5 pipeline is preserved across every workspace.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return selected_section, feedback_source, ai_config


def render_hero():
    st.markdown(
        """
        <section class="cx-hero">
            <div class="cx-hero-grid">
                <div>
                    <p class="cx-eyebrow">PX-Intel AI business intelligence</p>
                    <h1>Turn feedback signals into faster service decisions.</h1>
                    <p>Service feedback discovery, customer intelligence, operational impact, and AI-assisted action support powered by the existing PX-Intel pipeline.</p>
                </div>
                <div>
                    <div class="cx-command">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                            <div>
                                <div class="cx-eyebrow">System status</div>
                                <h4 style="margin:0.15rem 0 0;">M5 Action Agent online</h4>
                            </div>
                            <span style="padding:0.32rem 0.6rem; border-radius:999px; background:rgba(16,185,129,0.14); color:#059669; font-weight:800; font-size:0.75rem;">Live</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.55rem;"><span class="cx-status-dot cx-dot-green"></span><span>Feedback ingestion ready</span></div>
                        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.55rem;"><span class="cx-status-dot cx-dot-amber"></span><span>Signal audit cache enabled</span></div>
                        <div style="display:flex; align-items:center; gap:0.55rem;"><span class="cx-status-dot cx-dot-red"></span><span>High-priority issues surfaced first</span></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_slogan_strip():
    st.markdown(
        """
        <section class="cx-slogan-strip">
            <div>
                <p class="cx-eyebrow">PX-Intel AI business intelligence</p>
                <h1>Turn feedback signals into faster service decisions.</h1>
                <p>Service feedback discovery, customer intelligence, operational impact, and AI-assisted action support powered by the existing PX-Intel pipeline.</p>
            </div>
            <span class="cx-slogan-badge">Decision support</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, trend, accent):
    st.markdown(
        f"""
        <div class="cx-kpi-card">
            <p class="cx-kpi-label">{label}</p>
            <div class="cx-kpi-value">{value}</div>
            <div class="cx-kpi-trend" style="color:{accent};">{trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle=None, chip=None):
    chip_html = f'<span class="cx-chip">{escape(chip)}</span>' if chip else ""
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="cx-page-header">
            <div class="cx-page-header-top">
                <h2>{escape(title)}</h2>
                {chip_html}
            </div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table_header(title, description, data_frame, accent_label="Report table"):
    row_count = 0 if data_frame is None else len(data_frame)
    column_count = 0 if data_frame is None else len(data_frame.columns)
    st.markdown(
        f"""
        <div class="cx-table-panel">
            <div class="cx-card-topline" style="margin-bottom:0;">
                <div>
                    <p class="cx-eyebrow" style="margin:0 0 0.25rem;">{escape(accent_label)}</p>
                    <h4>{escape(title)}</h4>
                </div>
                <span class="cx-chip">{row_count:,} rows</span>
            </div>
            <p>{escape(description)}</p>
            <div class="cx-table-meta">
                <span>{row_count:,} records</span>
                <span>{column_count:,} columns</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table_section(
    title,
    description,
    data_frame,
    empty_message,
    accent_label="Report table",
    expanded=False,
    height=360,
):
    render_table_header(title, description, data_frame, accent_label)
    if data_frame is None or data_frame.empty:
        st.info(empty_message)
        return

    with st.expander(f"Open {title.lower()}", expanded=expanded):
        st.dataframe(data_frame, width="stretch", hide_index=True, height=height)


def priority_class(priority_label):
    if priority_label.startswith("HIGH"):
        return "high"
    if priority_label.startswith("MEDIUM"):
        return "medium"
    return "low"


def render_decision_summary_card(top_insight, insight_count, high_count, medium_count):
    if top_insight is None:
        st.markdown(
            """
            <div class="cx-decision-card featured">
                <div class="cx-card-topline">
                    <span class="cx-icon-block">AI</span>
                    <span class="cx-chip">No signals</span>
                </div>
                <h3 style="margin:0 0 0.5rem;">No action insight is available yet.</h3>
                <p style="margin:0;">Run the PX-Intel pipeline with feedback data to populate decision support.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    signal_name = escape(insight_display_name(top_insight, include_reference=True))
    insight = escape(top_insight.key_insight)
    action = escape(top_insight.recommended_action)
    badge_class = priority_class(top_insight.priority_label)
    st.markdown(
        f"""
        <div class="cx-decision-card featured">
            <div class="cx-card-topline">
                <span class="cx-icon-block">AI</span>
                <span class="cx-chip {badge_class}">{escape(top_insight.priority_label)}</span>
            </div>
            <p class="cx-eyebrow" style="margin-bottom:0.35rem;">Recommended first move</p>
            <h3 style="margin:0 0 0.5rem;">{signal_name}</h3>
            <p style="margin:0 0 0.65rem;">{insight}</p>
            <div class="cx-panel-soft" style="padding:0.8rem; border-radius:0.85rem;">
                <strong style="color:var(--cx-ink);">Action:</strong>
                <span>{action}</span>
            </div>
            <div class="cx-action-meta">
                <div><span>Total insights</span><strong>{insight_count}</strong></div>
                <div><span>High priority</span><strong>{high_count}</strong></div>
                <div><span>Medium priority</span><strong>{medium_count}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_owner_for_insight(insight):
    theme = str(insight.issue_theme).lower()
    if "wait" in theme or "scheduling" in theme:
        return "Operations lead"
    if "staff" in theme or "communication" in theme:
        return "CX manager"
    if "clean" in theme or "facility" in theme:
        return "Facilities lead"
    if "billing" in theme or "payment" in theme:
        return "Revenue operations"
    return "Site manager"


def action_window_for_insight(insight):
    negative_rate = insight.metadata.get("negative_rate", 0)
    if insight.priority_label.startswith("HIGH") or negative_rate >= 0.75:
        return "Immediate"
    if insight.priority_label.startswith("MEDIUM") or negative_rate >= 0.4:
        return "Next 7 days"
    return "Monitor"


def expected_impact_for_insight(insight):
    negative_rate = insight.metadata.get("negative_rate", 0)
    if insight.priority_label.startswith("HIGH"):
        return "Reduce repeat friction"
    if negative_rate >= 0.4:
        return "Stabilize service quality"
    if customer_lens_for_insight(insight) == "Opportunity":
        return "Protect strength"
    return "Improve visibility"


def render_action_outline_card(insight, index):
    badge_class = priority_class(insight.priority_label)
    keywords = ", ".join(insight.keywords[:4]) if insight.keywords else "No keywords"
    signal_name = escape(insight_display_name(insight, include_reference=True))
    owner = escape(action_owner_for_insight(insight))
    action_window = escape(action_window_for_insight(insight))
    expected_impact = escape(expected_impact_for_insight(insight))
    st.markdown(
        f"""
        <div class="cx-action-card">
            <div class="cx-card-topline">
                <span class="cx-icon-block">P{index}</span>
                <span class="cx-chip {badge_class}">{escape(insight.priority_label)}</span>
            </div>
            <p class="cx-eyebrow" style="margin-bottom:0.35rem;">Customer experience action</p>
            <h4 style="margin:0 0 0.35rem;">{signal_name}</h4>
            <p style="margin:0 0 0.75rem;">{escape(insight.key_insight)}</p>
            <div class="cx-action-meta">
                <div><span>Owner</span><strong>{owner}</strong></div>
                <div><span>Timeline</span><strong>{action_window}</strong></div>
                <div><span>Impact</span><strong>{expected_impact}</strong></div>
            </div>
            <div style="margin-top:0.8rem;">
                <p style="margin:0 0 0.3rem;"><strong style="color:var(--cx-ink);">Evidence:</strong> {insight.metadata.get("cluster_size", 0):,} comments, {insight.metadata.get("negative_rate", 0):.0%} negative, score {insight.priority_score:.3f}</p>
                <p style="margin:0 0 0.3rem;"><strong style="color:var(--cx-ink);">Keywords:</strong> {escape(keywords)}</p>
                <p style="margin:0;"><strong style="color:var(--cx-ink);">Root cause:</strong> {escape(insight.root_cause)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def customer_lens_for_insight(insight):
    negative_rate = insight.metadata.get("negative_rate", 0)
    if insight.priority_label.startswith("HIGH") or negative_rate >= 0.4:
        return "At Risk"
    if insight.sentiment_label == "POSITIVE" or negative_rate <= 0.2:
        return "Opportunity"
    return "Mixed"


def render_customer_segment_card(insight, index):
    lens = customer_lens_for_insight(insight)
    lens_class = "high" if lens == "At Risk" else "low" if lens == "Opportunity" else "medium"
    keywords = insight.keywords[:4] if insight.keywords else ["general experience"]
    keyword_tokens = "".join(
        f'<span class="cx-token">{escape(keyword.title())}</span>'
        for keyword in keywords
    )
    negative_rate = insight.metadata.get("negative_rate", 0)
    cluster_share = insight.metadata.get("cluster_share", 0)
    example = escape(insight.example_feedback)
    signal_name = escape(insight_display_name(insight))

    st.markdown(
        f"""
        <div class="cx-segment-card {'featured' if index == 1 else ''}">
            <div class="cx-card-topline">
                <span class="cx-icon-block">CI{index}</span>
                <span class="cx-chip {lens_class}">{escape(lens)}</span>
            </div>
            <p class="cx-eyebrow" style="margin-bottom:0.35rem;">Customer segment</p>
            <h4 class="cx-segment-title">{signal_name}</h4>
            <p style="margin:0 0 0.7rem;">{escape(insight.key_insight)}</p>
            <div class="cx-action-meta">
                <div><span>Feedback</span><strong>{insight.metadata.get("cluster_size", 0):,}</strong></div>
                <div><span>Negative</span><strong>{negative_rate:.0%}</strong></div>
                <div><span>Share</span><strong>{cluster_share:.0%}</strong></div>
            </div>
            <div class="cx-token-row">{keyword_tokens}</div>
            <div class="cx-quote">"{example}"</div>
            <div style="margin-top:0.75rem;">
                <strong style="color:var(--cx-ink);">Suggested move:</strong>
                <span>{escape(insight.recommended_action)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_signal_panel(title, subtitle, insights):
    if not insights:
        body = '<div class="cx-signal-item"><p>No matching customer signals found yet.</p></div>'
    else:
        items = []
        for insight in insights[:4]:
            negative_rate = insight.metadata.get("negative_rate", 0)
            signal_name = escape(insight_display_name(insight, include_reference=True))
            items.append(
                '<div class="cx-signal-item">'
                f"<h5>{signal_name}</h5>"
                f"<p>{escape(insight.key_insight)}</p>"
                '<div class="cx-token-row">'
                f'<span class="cx-token">{escape(insight.priority_label)}</span>'
                f'<span class="cx-token">{negative_rate:.0%} negative</span>'
                "</div>"
                "</div>"
            )
        body = "".join(items)

    st.markdown(
        '<div class="cx-signal-panel">'
        f'<p class="cx-eyebrow" style="margin-bottom:0.35rem;">{escape(subtitle)}</p>'
        f'<h4 style="margin:0 0 0.55rem;">{escape(title)}</h4>'
        f"{body}"
        "</div>",
        unsafe_allow_html=True,
    )


def build_customer_voice_dataframe(insights):
    return pd.DataFrame(
        [
            {
                "Signal ID": signal_reference(insight.cluster_id),
                "Experience Signal": insight_display_name(insight),
                "Cluster": insight.cluster_id,
                "Customer Segment": insight_display_name(insight),
                "Signal Type": customer_lens_for_insight(insight),
                "Sentiment": insight.sentiment_label.title(),
                "Priority": insight.priority_label,
                "Negative Rate": f"{insight.metadata.get('negative_rate', 0):.0%}",
                "Representative Feedback": insight.example_feedback,
                "Recommended Action": insight.recommended_action,
            }
            for insight in insights
        ]
    )


def filter_customer_insights(insights, lens, query):
    filtered = [
        insight
        for insight in insights
        if lens == "All" or customer_lens_for_insight(insight) == lens
    ]

    normalized_query = query.strip().lower()
    if not normalized_query:
        return filtered

    return [
        insight
        for insight in filtered
        if normalized_query
        in " ".join(
            [
                insight.issue_theme,
                insight_display_name(insight),
                insight.key_insight,
                insight.example_feedback,
                insight.recommended_action,
                " ".join(insight.keywords),
            ]
        ).lower()
    ]


def render_customer_intelligence_graphics(insights):
    st.markdown(
        '<h4 class="cx-section-heading">At-A-Glance Signal Graphics</h4>',
        unsafe_allow_html=True,
    )

    if not insights:
        st.info("No customer intelligence graphics are available for the current filter.")
        return

    signal_order = ["At Risk", "Mixed", "Opportunity"]
    signal_colors = {
        "At Risk": "#ef4444",
        "Mixed": "#f59e0b",
        "Opportunity": "#10b981",
    }
    signal_counts = pd.Series(
        [customer_lens_for_insight(insight) for insight in insights]
    ).value_counts()
    signal_values = [int(signal_counts.get(signal, 0)) for signal in signal_order]

    chart_col1, chart_col2 = st.columns([1, 1.25])
    with chart_col1:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=signal_order,
                    values=signal_values,
                    hole=0.62,
                    marker=dict(
                        colors=[signal_colors[signal] for signal in signal_order],
                        line=dict(color="white", width=2),
                    ),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Segments: %{value}<extra></extra>",
                )
            ]
        )
        fig.update_layout(title="Customer Signal Mix", legend_title="Signal Type")
        apply_plotly_soft_ui(fig, height=340)
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        top_risk_rows = sorted(
            insights,
            key=lambda insight: (
                insight.metadata.get("negative_rate", 0),
                insight.priority_score,
            ),
            reverse=True,
        )[:6]
        bar_labels = [
            f"{signal_reference(insight.cluster_id)} · {insight_display_name(insight)}"
            for insight in top_risk_rows
        ]
        negative_rates = [
            insight.metadata.get("negative_rate", 0) for insight in top_risk_rows
        ]
        bar_colors = [
            signal_colors[customer_lens_for_insight(insight)]
            for insight in top_risk_rows
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=negative_rates,
                    y=bar_labels,
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{rate:.0%}" for rate in negative_rates],
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>Negative Rate: %{x:.0%}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Risk Intensity By Segment",
            xaxis_title="Negative feedback rate",
            yaxis_title=None,
            xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),
        )
        apply_plotly_soft_ui(fig, height=340, showlegend=False)
        st.plotly_chart(fig, width="stretch")


def render_customer_intelligence(
    audit_engine,
    action_insights,
    texts,
    ai_config=None,
    feedback_source=None,
    clustering_engine=None,
    causal_engine=None,
):
    render_page_header(
        "Customer Intelligence",
        "A business-facing view of customer segments, risk signals, opportunities, and source evidence from the existing PX-Intel pipeline.",
        "Customer view",
    )
    render_stakeholder_explanation(
        "Customer Intelligence",
        "customer segment risk, opportunity signals, and source evidence",
        action_insights,
        ai_config,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
    )

    red_count = len(audit_engine.get_red_zones())
    green_count = len(audit_engine.get_green_zones())
    neutral_count = len(audit_engine.get_neutral_zones())
    top_theme = (
        action_insights[0].issue_theme.title() if action_insights else "No theme yet"
    )
    avg_negative = (
        np.mean([item.metadata.get("negative_rate", 0) for item in action_insights])
        if action_insights
        else 0
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_kpi_card("Feedback Reviewed", f"{len(texts):,}", "Voice-of-customer base", "#3b82f6")
    with metric_col2:
        render_kpi_card("At-Risk Segments", red_count, "Negative-dominant signals", "#ef4444")
    with metric_col3:
        render_kpi_card("Opportunity Segments", green_count, "Positive-dominant signals", "#10b981")
    with metric_col4:
        render_kpi_card("Top Customer Theme", top_theme, f"{avg_negative:.0%} avg negative", "#8b5cf6")

    control_col1, control_col2, download_col = st.columns([1.1, 1.35, 0.95])
    with control_col1:
        customer_lens = st.segmented_control(
            "Customer signal lens",
            options=["All", "At Risk", "Opportunity", "Mixed"],
            default="All",
            key="customer_intelligence_lens",
        )
    with control_col2:
        customer_search = st.text_input(
            "Search customer themes or feedback",
            placeholder="Search issue, keyword, quote, or action...",
            key="customer_intelligence_search",
        )

    visible_insights = filter_customer_insights(
        action_insights,
        customer_lens,
        customer_search,
    )
    voice_df = build_customer_voice_dataframe(visible_insights)
    with download_col:
        st.download_button(
            "Download CSV",
            data=voice_df.to_csv(index=False),
            file_name="px_intel_customer_intelligence.csv",
            mime="text/csv",
            help="Download the currently filtered customer intelligence data.",
            width="stretch",
        )

    current_filter_state = (customer_lens, customer_search)
    if st.session_state.get("customer_intel_filter_state") != current_filter_state:
        st.session_state.customer_intel_filter_state = current_filter_state
        st.session_state.customer_intel_page = 0

    render_customer_intelligence_graphics(visible_insights)

    st.markdown(
        '<h4 class="cx-section-heading">Customer Segment Intelligence</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing {len(visible_insights)} customer segment signals from the current PX-Intel analysis."
    )

    if visible_insights:
        page_size = 4
        total_pages = max(1, int(np.ceil(len(visible_insights) / page_size)))
        current_page = min(
            st.session_state.get("customer_intel_page", 0),
            total_pages - 1,
        )
        st.session_state.customer_intel_page = current_page
        page_start = current_page * page_size
        page_end = page_start + page_size
        page_insights = visible_insights[page_start:page_end]

        nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
        with nav_left:
            if st.button(
                "Previous",
                key="customer_segments_previous",
                disabled=current_page == 0,
                width="stretch",
            ):
                st.session_state.customer_intel_page = max(current_page - 1, 0)
                st.rerun()
        with nav_mid:
            st.markdown(
                f"""
                <div class="cx-panel-soft" style="padding:0.65rem; border-radius:0.85rem; text-align:center;">
                    Showing segments {page_start + 1}-{min(page_end, len(visible_insights))} of {len(visible_insights)}
                    &nbsp;|&nbsp; Page {current_page + 1} of {total_pages}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with nav_right:
            if st.button(
                "Next",
                key="customer_segments_next",
                disabled=current_page >= total_pages - 1,
                width="stretch",
            ):
                st.session_state.customer_intel_page = min(
                    current_page + 1,
                    total_pages - 1,
                )
                st.rerun()

        for start in range(0, len(page_insights), 2):
            segment_cols = st.columns(2)
            for offset, segment_col in enumerate(segment_cols):
                segment_index = start + offset
                if segment_index >= len(page_insights):
                    continue
                with segment_col:
                    render_customer_segment_card(
                        page_insights[segment_index],
                        page_start + segment_index + 1,
                    )
    else:
        st.info("No customer segments match the current lens or search.")

    risk_insights = sorted(
        [
            insight
            for insight in action_insights
            if customer_lens_for_insight(insight) == "At Risk"
        ],
        key=lambda insight: (
            insight.metadata.get("negative_rate", 0),
            insight.priority_score,
        ),
        reverse=True,
    )
    opportunity_insights = sorted(
        [
            insight
            for insight in action_insights
            if customer_lens_for_insight(insight) == "Opportunity"
        ],
        key=lambda insight: (
            1 - insight.metadata.get("negative_rate", 0),
            insight.metadata.get("cluster_share", 0),
        ),
        reverse=True,
    )

    st.markdown(
        '<h4 class="cx-section-heading">Risk & Opportunity Signals</h4>',
        unsafe_allow_html=True,
    )
    risk_col, opportunity_col = st.columns(2)
    with risk_col:
        render_customer_signal_panel(
            "Customer Risk Signals",
            "Friction to reduce",
            risk_insights,
        )
    with opportunity_col:
        render_customer_signal_panel(
            "Customer Opportunity Signals",
            "Strengths to expand",
            opportunity_insights,
        )

    render_table_section(
        "Voice Of Customer Explorer",
        "Filtered source evidence behind the visible customer segments, risk signals, and recommended actions.",
        voice_df,
        "No source evidence is available for the current filter.",
        accent_label="Evidence table",
    )


def shorten_text(value, limit=72):
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def find_date_column(df):
    date_hints = ("date", "time", "created", "submitted", "timestamp")
    for column in df.columns:
        if any(hint in str(column).lower() for hint in date_hints):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return column
    return None


def filter_insights_by_time_period(
    insights, df, cluster_assignments, date_column, period, custom_range
):
    if period == "All Time":
        return insights, None

    if date_column is None:
        return (
            insights,
            "Current feedback data has no timestamp column, so the time filter is using all records.",
        )

    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.notna().sum() == 0:
        return (
            insights,
            f"The `{date_column}` column could not be parsed as dates, so the graph is using all records.",
        )

    latest_date = dates.max().normalize()
    if period == "Last 7 Days":
        start_date = latest_date - pd.Timedelta(days=7)
        end_date = latest_date
    elif period == "Last 30 Days":
        start_date = latest_date - pd.Timedelta(days=30)
        end_date = latest_date
    elif period == "Last 90 Days":
        start_date = latest_date - pd.Timedelta(days=90)
        end_date = latest_date
    else:
        if not custom_range or len(custom_range) != 2:
            return insights, "Choose a valid custom date range to filter the graph."
        start_date = pd.Timestamp(custom_range[0])
        end_date = pd.Timestamp(custom_range[1])

    mask = dates.between(start_date, end_date, inclusive="both")
    assignment_array = np.array(cluster_assignments)
    if len(assignment_array) != len(df):
        return (
            insights,
            "Cluster assignments and source rows are not aligned, so the time filter is using all records.",
        )

    visible_clusters = set(assignment_array[mask.fillna(False).to_numpy()].tolist())
    filtered = [insight for insight in insights if insight.cluster_id in visible_clusters]
    return (
        filtered,
        f"Time filter applied from {start_date.date()} to {end_date.date()} using `{date_column}`.",
    )


def filter_cause_effect_insights(
    action_insights,
    sentiment_filter,
    priority_filter,
    cluster_filter,
    theme_filter,
):
    filtered = list(action_insights)
    if sentiment_filter != "All":
        filtered = [
            insight
            for insight in filtered
            if str(insight.sentiment_label).title() == sentiment_filter
        ]
    if priority_filter != "All":
        filtered = [
            insight for insight in filtered if insight.priority_label == priority_filter
        ]
    if cluster_filter != "All":
        filtered = [
            insight
            for insight in filtered
            if signal_reference(insight.cluster_id) == cluster_filter
        ]
    if theme_filter != "All":
        filtered = [
            insight
            for insight in filtered
            if insight.issue_theme.title() == theme_filter
        ]
    return filtered


def apply_graph_density(insights, density, cluster_filter):
    if cluster_filter != "All" or density == "All Matching":
        return insights
    limit = 3 if density == "Top 3" else 5
    return sorted(insights, key=lambda insight: insight.priority_score, reverse=True)[
        :limit
    ]


def node_type_label(node_type):
    return node_type.replace("_", " ").title()


def graph_node_marker_label(node):
    label_map = {
        "feedback": "FB",
        "theme": "TH",
        "root_cause": "RC",
        "issue": "ISS",
        "customer_segment": "SEG",
        "sentiment": "SNT",
        "impact": "RISK",
        "action": "ACT",
    }
    return f"{signal_reference(node['cluster_id'])}<br>{label_map.get(node['type'], 'NODE')}"


def add_graph_node(
    nodes,
    node_id,
    label,
    node_type,
    cluster_id,
    x,
    y,
    color,
    size,
    description,
    evidence,
    recommended_action,
    priority,
    priority_score,
    sentiment,
    root_cause,
    issue_theme,
):
    nodes[node_id] = {
        "id": node_id,
        "label": label,
        "display": shorten_text(label, 28),
        "type": node_type,
        "cluster_id": cluster_id,
        "x": x,
        "y": y,
        "color": color,
        "size": size,
        "description": description,
        "evidence": evidence,
        "recommended_action": recommended_action,
        "priority": priority,
        "priority_score": priority_score,
        "sentiment": sentiment,
        "root_cause": root_cause,
        "issue_theme": issue_theme,
    }


def add_graph_edge(edges, source, target, relation, weight, description):
    edges.append(
        {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": weight,
            "description": description,
        }
    )


def build_cause_effect_graph(action_insights, causal_engine):
    nodes = {}
    edges = []
    visible_cluster_ids = {insight.cluster_id for insight in action_insights}
    sentiment_colors = {
        "NEGATIVE": "#dc2626",
        "NEUTRAL": "#d97706",
        "POSITIVE": "#059669",
    }
    priority_colors = {
        "HIGH": "#dc2626",
        "MEDIUM": "#d97706",
        "LOW": "#059669",
    }

    for row_index, insight in enumerate(action_insights):
        cluster_id = insight.cluster_id
        signal_name = insight_display_name(insight)
        signal_name_with_ref = insight_display_name(insight, include_reference=True)
        y_base = -row_index * 2.05
        priority_key = insight.priority_label.split()[0]
        sentiment_key = str(insight.sentiment_label).upper()
        impact_color = priority_colors.get(priority_key, "#64748b")
        sentiment_color = sentiment_colors.get(sentiment_key, "#64748b")
        node_size = 24 + min(insight.metadata.get("cluster_size", 0), 45) * 0.2
        common = {
            "cluster_id": cluster_id,
            "evidence": insight.example_feedback,
            "recommended_action": insight.recommended_action,
            "priority": insight.priority_label,
            "priority_score": insight.priority_score,
            "sentiment": insight.sentiment_label,
            "root_cause": insight.root_cause,
            "issue_theme": signal_name,
        }

        feedback_id = f"feedback-{cluster_id}"
        theme_id = f"theme-{cluster_id}"
        issue_id = f"issue-{cluster_id}"
        root_id = f"root-{cluster_id}"
        segment_id = f"segment-{cluster_id}"
        sentiment_id = f"sentiment-{cluster_id}"
        impact_id = f"impact-{cluster_id}"
        action_id = f"action-{cluster_id}"

        add_graph_node(
            nodes,
            feedback_id,
            f"Feedback: {signal_reference(cluster_id)}",
            "feedback",
            x=0,
            y=y_base,
            color="#475569",
            size=24,
            description="Representative customer feedback used as graph evidence.",
            **common,
        )
        add_graph_node(
            nodes,
            theme_id,
            f"Theme: {insight.issue_theme.title()}",
            "theme",
            x=1.25,
            y=y_base + 0.22,
            color="#0891b2",
            size=node_size + 2,
            description="Model-derived theme from cluster vocabulary.",
            **common,
        )
        add_graph_node(
            nodes,
            root_id,
            f"Root Cause: {shorten_text(insight.root_cause, 42)}",
            "root_cause",
            x=2.45,
            y=y_base - 0.28,
            color="#7c3aed",
            size=26,
            description="Probable cause inferred from M3 causal validation and keywords.",
            **common,
        )
        add_graph_node(
            nodes,
            issue_id,
            signal_name_with_ref,
            "issue",
            x=3.35,
            y=y_base + 0.18,
            color="#d97706",
            size=node_size + 3,
            description=insight.key_insight,
            **common,
        )
        add_graph_node(
            nodes,
            segment_id,
            f"Segment: {signal_name}",
            "customer_segment",
            x=4.75,
            y=y_base + 0.28,
            color="#2563eb",
            size=node_size + 1,
            description=f"Customers affected by {signal_name}.",
            **common,
        )
        add_graph_node(
            nodes,
            sentiment_id,
            f"Sentiment: {insight.sentiment_label.title()}",
            "sentiment",
            x=4.75,
            y=y_base - 0.36,
            color=sentiment_color,
            size=24,
            description=(
                f"{insight.metadata.get('negative_rate', 0):.0%} of this signal is negative."
            ),
            **common,
        )
        add_graph_node(
            nodes,
            impact_id,
            f"Impact: {insight.priority_label}",
            "impact",
            x=6.1,
            y=y_base + 0.2,
            color=impact_color,
            size=24 + insight.priority_score * 24,
            description=f"Priority score {insight.priority_score:.3f}.",
            **common,
        )
        add_graph_node(
            nodes,
            action_id,
            f"Action: {shorten_text(insight.recommended_action, 42)}",
            "action",
            x=7.25,
            y=y_base - 0.42,
            color="#059669",
            size=26,
            description="Recommended mitigation action from the PX-Intel action agent.",
            **common,
        )

        add_graph_edge(
            edges,
            feedback_id,
            theme_id,
            "feedback_mentions_theme",
            0.7,
            "Representative feedback mentions the discovered customer theme.",
        )
        add_graph_edge(
            edges,
            theme_id,
            issue_id,
            "theme_contributes_to_issue",
            0.85,
            "The theme contributes to the issue pattern surfaced by the cluster.",
        )
        add_graph_edge(
            edges,
            root_id,
            issue_id,
            "root_cause_drives_issue",
            0.75,
            "The probable root cause helps explain the issue pattern.",
        )
        add_graph_edge(
            edges,
            issue_id,
            segment_id,
            "issue_impacts_segment",
            insight.metadata.get("cluster_share", 0),
            "The issue impacts this customer segment.",
        )
        add_graph_edge(
            edges,
            issue_id,
            sentiment_id,
            "issue_has_sentiment",
            insight.metadata.get("negative_rate", 0),
            "The issue is associated with this sentiment profile.",
        )
        add_graph_edge(
            edges,
            issue_id,
            impact_id,
            "issue_escalates_to_risk",
            insight.priority_score,
            "The issue escalates into operational priority or risk.",
        )
        add_graph_edge(
            edges,
            issue_id,
            action_id,
            "action_mitigates_issue",
            0.9,
            "The recommended action is intended to mitigate the issue.",
        )

    for insight in action_insights:
        source_issue = f"issue-{insight.cluster_id}"
        for cascade in getattr(causal_engine, "cascade_predictions", {}).get(
            insight.cluster_id, []
        )[:3]:
            target_cluster = int(cascade.get("target_cluster", -1))
            target_segment = f"segment-{target_cluster}"
            if target_cluster not in visible_cluster_ids or target_segment not in nodes:
                continue
            add_graph_edge(
                edges,
                source_issue,
                target_segment,
                "issue_impacts_segment",
                cascade.get("cascade_likelihood", 0),
                (
                    f"Shared-pattern cascade: fixing {signal_reference(insight.cluster_id)} "
                    f"may affect {signal_reference(target_cluster)}."
                ),
            )

    return nodes, edges


def create_cause_effect_figure(nodes, edges):
    edge_styles = {
        "feedback_mentions_theme": ("#94a3b8", "Feedback mentions theme"),
        "theme_contributes_to_issue": ("#06b6d4", "Theme contributes to issue"),
        "root_cause_drives_issue": ("#8b5cf6", "Root cause drives issue"),
        "issue_impacts_segment": ("#3b82f6", "Issue impacts segment"),
        "issue_has_sentiment": ("#64748b", "Issue sentiment"),
        "issue_escalates_to_risk": ("#ef4444", "Issue escalates to risk"),
        "action_mitigates_issue": ("#10b981", "Action mitigates issue"),
    }
    node_type_order = [
        "feedback",
        "theme",
        "root_cause",
        "issue",
        "customer_segment",
        "sentiment",
        "impact",
        "action",
    ]
    node_symbols = {
        "feedback": "circle",
        "theme": "diamond",
        "root_cause": "hexagon",
        "issue": "square",
        "customer_segment": "circle",
        "sentiment": "triangle-up",
        "impact": "star",
        "action": "pentagon",
    }
    fig = go.Figure()

    if nodes:
        y_values = [node["y"] for node in nodes.values()]
        header_y = max(y_values) + 1.0
        footer_y = min(y_values) - 0.85
        for index, x_value in enumerate([0, 1.25, 2.45, 3.35, 4.75, 6.1, 7.25]):
            fig.add_shape(
                type="rect",
                x0=x_value - 0.48,
                x1=x_value + 0.48,
                y0=footer_y,
                y1=header_y + 0.28,
                line=dict(width=0),
                fillcolor="#f8fafc" if index % 2 == 0 else "#eef6ff",
                opacity=0.72,
                layer="below",
            )

    for relation, (color, label) in edge_styles.items():
        relation_edges = [edge for edge in edges if edge["relation"] == relation]
        if not relation_edges:
            continue
        x_values = []
        y_values = []
        for edge in relation_edges:
            source = nodes.get(edge["source"])
            target = nodes.get(edge["target"])
            if not source or not target:
                continue
            x_values.extend([source["x"], target["x"], None])
            y_values.extend([source["y"], target["y"], None])
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=color, width=2.0),
                hoverinfo="skip",
                name=label,
                opacity=0.34,
                showlegend=False,
            )
        )

    for node_type in node_type_order:
        group = [node for node in nodes.values() if node["type"] == node_type]
        if not group:
            continue
        fig.add_trace(
            go.Scatter(
                x=[node["x"] for node in group],
                y=[node["y"] for node in group],
                mode="markers+text",
                name=node_type_label(node_type),
                text=[graph_node_marker_label(node) for node in group],
                textposition="middle center",
                customdata=[node["id"] for node in group],
                hovertext=[
                    (
                        f"{node_type_label(node['type'])}<br>"
                        f"{signal_reference(node['cluster_id'])}<br>"
                        f"{escape(shorten_text(node['label'], 120))}<br>"
                        f"{escape(shorten_text(node['description'], 140))}"
                    )
                    for node in group
                ],
                hovertemplate="<b>%{text}</b><br>%{hovertext}<extra></extra>",
                marker=dict(
                    size=[node["size"] for node in group],
                    color=[node["color"] for node in group],
                    symbol=node_symbols.get(node_type, "circle"),
                    line=dict(width=1.2, color="#ffffff"),
                    opacity=0.92,
                ),
                textfont=dict(size=9, color="#ffffff"),
                showlegend=False,
            )
        )

    if nodes:
        for x_value, label in [
            (0, "Feedback"),
            (1.25, "Theme"),
            (2.45, "Root Cause"),
            (3.35, "Issue"),
            (4.75, "Segment + Sentiment"),
            (6.1, "Risk"),
            (7.25, "Action"),
        ]:
            fig.add_annotation(
                x=x_value,
                y=header_y,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=13, color="#172033"),
                align="center",
            )
        fig.update_yaxes(range=[footer_y, header_y + 0.35])

    fig.update_layout(
        title="Cause & Effect Flow Map",
        clickmode="event+select",
        dragmode="pan",
        showlegend=False,
    )
    dynamic_height = max(620, min(980, 280 + len({node["cluster_id"] for node in nodes.values()}) * 98))
    apply_plotly_soft_ui(fig, height=dynamic_height, showlegend=False)
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=12),
        title=dict(font=dict(color="#172033", size=17)),
        margin=dict(l=36, r=42, t=76, b=42),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(23, 32, 51, 0.16)",
            font=dict(color="#172033"),
        ),
    )
    fig.update_xaxes(range=[-0.45, 8.15], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    return fig


def extract_selected_node_id(plotly_state):
    try:
        points = plotly_state.selection.points
    except AttributeError:
        points = (
            plotly_state.get("selection", {}).get("points", [])
            if isinstance(plotly_state, dict)
            else []
        )
    if not points:
        return None
    point = points[0]
    custom_data = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(custom_data, (list, tuple)):
        return custom_data[0] if custom_data else None
    return custom_data


def render_graph_node_detail(selected_node_id, nodes, edges):
    node = nodes.get(selected_node_id)
    if not node:
        st.info("Click a graph node or choose one from the selector to inspect evidence.")
        return

    connected_edges = [
        edge
        for edge in edges
        if edge["source"] == selected_node_id or edge["target"] == selected_node_id
    ]
    relationship_items = "".join(
        "<li>"
        f"<strong>{escape(edge['relation'])}</strong><br>"
        f"{escape(shorten_text(edge['description'], 120))}"
        "</li>"
        for edge in connected_edges[:7]
    )
    if not relationship_items:
        relationship_items = "<li>No direct graph relationships found for this node.</li>"

    st.markdown(
        '<div class="cx-graph-detail">'
        f'<span class="cx-graph-type">{escape(node_type_label(node["type"]))}</span>'
        f'<h4 style="margin:0.7rem 0 0.35rem;">{escape(node["label"])}</h4>'
        f'<p style="margin:0 0 0.75rem;">{escape(node["description"])}</p>'
        '<div class="cx-action-meta">'
        f'<div><span>Signal</span><strong>{signal_reference(node["cluster_id"])}</strong></div>'
        f'<div><span>Priority</span><strong>{escape(str(node["priority"]))}</strong></div>'
        f'<div><span>Sentiment</span><strong>{escape(str(node["sentiment"]).title())}</strong></div>'
        "</div>"
        f'<div class="cx-quote">"{escape(shorten_text(node["evidence"], 260))}"</div>'
        '<div style="margin-top:0.8rem;">'
        '<strong style="color:var(--cx-ink);">Root cause:</strong> '
        f'<span>{escape(node["root_cause"])}</span>'
        "</div>"
        '<div style="margin-top:0.65rem;">'
        '<strong style="color:var(--cx-ink);">Recommended action:</strong> '
        f'<span>{escape(node["recommended_action"])}</span>'
        "</div>"
        '<h5 style="margin:0.9rem 0 0.2rem; color:var(--cx-ink);">Connected relationships</h5>'
        f'<ul class="cx-relationship-list">{relationship_items}</ul>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_cause_effect_readout(insights):
    if not insights:
        return

    ranked = sorted(insights, key=lambda item: item.priority_score, reverse=True)
    top = ranked[0]
    top_action = next((item for item in ranked if item.recommended_action), top)
    top_root = next((item for item in ranked if item.root_cause), top)
    cards = [
        (
            "Priority path",
            insight_display_name(top, include_reference=True),
            f"{top.priority_label} with {top.metadata.get('negative_rate', 0):.0%} negative feedback.",
        ),
        (
            "Likely driver",
            top_root.root_cause,
            "Treat this as evidence-weighted, not absolute causality.",
        ),
        (
            "Best next action",
            insight_display_name(top_action, include_reference=True),
            top_action.recommended_action,
        ),
    ]
    html = "".join(
        '<div class="cx-readout-card">'
        f'<span class="cx-graph-type">{escape(label)}</span>'
        f"<h5>{escape(shorten_text(title, 72))}</h5>"
        f"<p>{escape(shorten_text(body, 150))}</p>"
        "</div>"
        for label, title, body in cards
    )
    st.markdown(
        '<div class="cx-readout-grid">'
        f"{html}"
        "</div>",
        unsafe_allow_html=True,
    )


def graph_category_nodes(nodes, category):
    category_types = {
        "Issues": {"issue"},
        "Risks": {"impact"},
        "Root Causes": {"root_cause"},
        "Actions": {"action"},
        "Segments": {"customer_segment"},
        "Feedback": {"feedback"},
        "Sentiment": {"sentiment"},
    }
    allowed_types = category_types.get(category, {"issue"})
    category_nodes = [
        node for node in nodes.values() if node["type"] in allowed_types
    ]
    return sorted(
        category_nodes,
        key=lambda node: (
            -float(node.get("priority_score", 0)),
            node["cluster_id"],
            node["type"],
        ),
    )


def graph_category_button_label(node):
    node_type = node["type"]
    cluster = signal_reference(node["cluster_id"])
    if node_type == "issue":
        return f"{cluster} · {node['issue_theme']}"
    if node_type == "impact":
        return f"{cluster} · {node['priority']}"
    if node_type == "root_cause":
        return f"{cluster} · {shorten_text(node['root_cause'], 34)}"
    if node_type == "action":
        return f"{cluster} · {shorten_text(node['recommended_action'], 42)}"
    if node_type == "customer_segment":
        return f"{cluster} · {node['issue_theme']}"
    if node_type == "feedback":
        return f"{cluster} · {shorten_text(node['evidence'], 42)}"
    if node_type == "sentiment":
        return f"{cluster} · {str(node['sentiment']).title()} sentiment"
    return f"{cluster} · {node['label']}"


def render_graph_category_selector(nodes):
    st.markdown(
        '<h4 class="cx-section-heading">Selected Node Insight</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Start with a business category, then choose the experience signal you want to inspect."
    )
    category_options = [
        "Issues",
        "Risks",
        "Root Causes",
        "Actions",
        "Segments",
        "Feedback",
        "Sentiment",
    ]
    if st.session_state.get("cause_effect_insight_category") not in category_options:
        st.session_state.cause_effect_insight_category = "Issues"
    category = st.segmented_control(
        "Insight category",
        options=category_options,
        default="Issues",
        key="cause_effect_insight_category",
    )
    category_items = graph_category_nodes(nodes, category)
    if not category_items:
        return None

    if st.session_state.get("cause_effect_last_category") != category:
        st.session_state.cause_effect_last_category = category
        st.session_state.cause_effect_selected_node = category_items[0]["id"]

    selected_node = st.session_state.get(
        "cause_effect_selected_node",
        category_items[0]["id"],
    )
    if selected_node not in nodes:
        selected_node = category_items[0]["id"]

    card_cols = st.columns(3)
    for index, node in enumerate(category_items[:6]):
        with card_cols[index % 3]:
            if st.button(
                graph_category_button_label(node),
                key=f"cause_effect_pick_{category}_{node['id']}",
                width="stretch",
            ):
                selected_node = node["id"]
                st.session_state.cause_effect_selected_node = selected_node
                st.rerun()

    return selected_node


def build_relationship_dataframe(nodes, edges):
    rows = []
    for edge in edges:
        source = nodes.get(edge["source"], {})
        target = nodes.get(edge["target"], {})
        rows.append(
            {
                "Relationship": edge["relation"],
                "From": source.get("label", edge["source"]),
                "To": target.get("label", edge["target"]),
                "Weight": round(float(edge.get("weight", 0)), 3),
                "Meaning": edge["description"],
            }
        )
    return pd.DataFrame(rows)


def render_cause_effect_outline():
    steps = [
        ("1", "Feedback", "Source customer comment."),
        ("2", "Theme", "Repeated language pattern."),
        ("3", "Root Cause", "Probable driver behind the issue."),
        ("4", "Issue", "Operational problem to manage."),
        ("5", "Segment", "Customer group being affected."),
        ("6", "Risk / Action", "Priority plus recommended mitigation."),
    ]
    cards = "".join(
        '<div class="cx-graph-step">'
        f"<span>{number}</span>"
        f"<strong>{escape(title)}</strong>"
        f"<p>{escape(description)}</p>"
        "</div>"
        for number, title, description in steps
    )
    st.markdown(
        '<div class="cx-graph-outline">'
        f"{cards}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_cause_effect_key():
    key_items = [
        ("#475569", "Feedback"),
        ("#0891b2", "Theme"),
        ("#7c3aed", "Root cause"),
        ("#d97706", "Issue"),
        ("#dc2626", "Risk"),
        ("#059669", "Action"),
    ]
    chips = "".join(
        '<span class="cx-graph-key-item">'
        f'<span class="cx-key-dot" style="background:{color};"></span>'
        f"{escape(label)}"
        "</span>"
        for color, label in key_items
    )
    st.markdown(
        '<div class="cx-graph-key">'
        f"{chips}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_cause_effect_graph(
    audit_engine,
    causal_engine,
    action_insights,
    df,
    cluster_assignments,
    ai_config=None,
    feedback_source=None,
    texts=None,
    clustering_engine=None,
):
    render_page_header(
        "Cause & Effect Graph",
        "Explore how feedback, themes, root causes, issues, customer segments, sentiment, impact, and actions connect across PX-Intel outputs.",
        "Relationship map",
    )
    render_cause_effect_outline()
    render_stakeholder_explanation(
        "Cause & Effect Graph",
        "cause-and-effect paths from feedback themes to issues, root causes, risk, impact, and actions",
        action_insights,
        ai_config,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
    )

    date_column = find_date_column(df)
    sentiment_options = ["All"] + sorted(
        {str(insight.sentiment_label).title() for insight in action_insights}
    )
    priority_options = ["All"] + sorted(
        {insight.priority_label for insight in action_insights},
        key=lambda label: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(label.split()[0], 9),
    )
    signal_label_map = {
        signal_reference(insight.cluster_id): insight_display_name(insight)
        for insight in sorted(action_insights, key=lambda item: item.cluster_id)
    }
    cluster_options = ["All"] + list(signal_label_map.keys())
    theme_options = ["All"] + sorted(
        {insight.issue_theme.title() for insight in action_insights}
    )

    filter_cols = st.columns([0.95, 0.95, 0.95, 1.15, 1.05])
    with filter_cols[0]:
        sentiment_filter = st.selectbox(
            "Sentiment",
            sentiment_options,
            key="cause_effect_sentiment_filter",
        )
    with filter_cols[1]:
        priority_filter = st.selectbox(
            "Priority",
            priority_options,
            key="cause_effect_priority_filter",
        )
    with filter_cols[2]:
        cluster_filter = st.selectbox(
            "Signal",
            cluster_options,
            format_func=lambda value: "All" if value == "All" else f"{value} · {signal_label_map[value]}",
            key="cause_effect_cluster_filter",
        )
    with filter_cols[3]:
        theme_filter = st.selectbox(
            "Theme",
            theme_options,
            key="cause_effect_theme_filter",
        )
    with filter_cols[4]:
        time_filter = st.selectbox(
            "Time period",
            ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"],
            key="cause_effect_time_filter",
        )

    custom_range = None
    if date_column and time_filter == "Custom Range":
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not parsed_dates.empty:
            custom_range = st.date_input(
                "Custom graph date range",
                value=(parsed_dates.min().date(), parsed_dates.max().date()),
                key="cause_effect_custom_range",
            )

    filtered_insights = filter_cause_effect_insights(
        action_insights,
        sentiment_filter,
        priority_filter,
        cluster_filter,
        theme_filter,
    )
    filtered_insights, time_note = filter_insights_by_time_period(
        filtered_insights,
        df,
        cluster_assignments,
        date_column,
        time_filter,
        custom_range,
    )
    if time_note:
        st.caption(time_note)

    density_col, density_note_col = st.columns([1, 2])
    with density_col:
        graph_density = st.segmented_control(
            "Graph density",
            options=["Top 3", "Top 5", "All Matching"],
            default="Top 5",
            key="cause_effect_graph_density",
        )
    filtered_insights = apply_graph_density(
        filtered_insights,
        graph_density,
        cluster_filter,
    )
    with density_note_col:
        st.caption(
            "The graph defaults to the top priority signals so the flow stays readable. Use All Matching when you need the complete relationship map."
        )

    render_cause_effect_readout(filtered_insights)

    nodes, edges = build_cause_effect_graph(filtered_insights, causal_engine)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_kpi_card("Graph Nodes", len(nodes), "Feedback to actions", "#3b82f6")
    with metric_cols[1]:
        render_kpi_card("Relationships", len(edges), "Weighted evidence links", "#06b6d4")
    with metric_cols[2]:
        render_kpi_card("Signals Shown", len(filtered_insights), "After filters", "#8b5cf6")
    with metric_cols[3]:
        render_kpi_card(
            "Causal Links",
            sum(
                len(getattr(causal_engine, "cascade_predictions", {}).get(insight.cluster_id, []))
                for insight in filtered_insights
            ),
            "M3 cascade candidates",
            "#10b981",
        )

    if not nodes:
        st.info("No cause-and-effect graph matches the current filters.")
        return

    st.caption("Click a node in the graph to inspect evidence and recommended actions below.")
    render_cause_effect_key()
    graph_state = st.plotly_chart(
        create_cause_effect_figure(nodes, edges),
        width="stretch",
        key="cause_effect_graph_plot",
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": True},
    )
    selected_from_chart = extract_selected_node_id(graph_state)
    if selected_from_chart:
        st.session_state.cause_effect_selected_node = selected_from_chart

    selected_node = render_graph_category_selector(nodes)
    if selected_node:
        render_graph_node_detail(selected_node, nodes, edges)

    relationship_df = build_relationship_dataframe(nodes, edges)
    render_table_section(
        "Relationship Evidence Table",
        "Structured node and edge evidence for the relationships currently visible in the graph.",
        relationship_df,
        "No relationship evidence is available for the current graph filters.",
        accent_label="Graph evidence",
    )


def impact_type_for_insight(insight, cascade_count):
    negative_rate = insight.metadata.get("negative_rate", 0)
    if insight.priority_label.startswith("HIGH") or cascade_count >= 3:
        return "Systemic Risk"
    if negative_rate >= 0.4:
        return "Service Recovery"
    if str(insight.sentiment_label).upper() == "POSITIVE":
        return "Protect Strength"
    return "Monitor"


def action_window_for_impact(row):
    if row["impact_type"] == "Systemic Risk" or row["impact_score"] >= 0.62:
        return "Immediate"
    if row["impact_type"] == "Service Recovery" or row["negative_rate"] >= 0.3:
        return "Next 7 Days"
    return "Monitor"


def is_quick_win(row):
    return (
        row["impact_type"] in {"Service Recovery", "Monitor"}
        and row["cascade_count"] <= 1
        and row["impact_score"] >= 0.28
    )


def build_operational_impact_rows(action_insights, causal_engine):
    rows = []
    insight_by_cluster = {insight.cluster_id: insight for insight in action_insights}
    for insight in action_insights:
        cascades = getattr(causal_engine, "cascade_predictions", {}).get(
            insight.cluster_id, []
        )
        cascade_targets = [
            int(cascade.get("target_cluster"))
            for cascade in cascades
            if cascade.get("target_cluster") in insight_by_cluster
        ]
        top_cascade = max(
            [cascade.get("cascade_likelihood", 0) for cascade in cascades],
            default=0,
        )
        negative_rate = insight.metadata.get("negative_rate", 0)
        cluster_size = insight.metadata.get("cluster_size", 0)
        impact_score = (
            insight.priority_score * 0.62
            + negative_rate * 0.22
            + min(len(cascade_targets), 4) * 0.04
        )
        row = {
            "cluster_id": insight.cluster_id,
            "signal_id": signal_reference(insight.cluster_id),
            "signal_name": insight_display_name(insight),
            "theme": insight.issue_theme.title(),
            "priority": insight.priority_label,
            "priority_score": insight.priority_score,
            "impact_score": impact_score,
            "impact_type": impact_type_for_insight(insight, len(cascade_targets)),
            "negative_rate": negative_rate,
            "cluster_size": cluster_size,
            "cascade_count": len(cascade_targets),
            "top_cascade": top_cascade,
            "cascade_targets": cascade_targets,
            "root_cause": insight.root_cause,
            "recommended_action": insight.recommended_action,
            "example_feedback": insight.example_feedback,
            "key_insight": insight.key_insight,
        }
        row["action_window"] = action_window_for_impact(row)
        row["is_quick_win"] = is_quick_win(row)
        rows.append(row)
    return sorted(rows, key=lambda row: row["impact_score"], reverse=True)


def filter_operational_impact_rows(rows, focus_filter, priority_filter, cluster_filter, theme_filter):
    filtered = list(rows)
    if focus_filter == "Systemic Risks":
        filtered = [row for row in filtered if row["impact_type"] == "Systemic Risk"]
    elif focus_filter == "Service Recovery":
        filtered = [row for row in filtered if row["impact_type"] == "Service Recovery"]
    elif focus_filter == "Quick Wins":
        filtered = [row for row in filtered if row["is_quick_win"]]
    elif focus_filter == "Protect Strengths":
        filtered = [row for row in filtered if row["impact_type"] == "Protect Strength"]
    elif focus_filter == "Monitor":
        filtered = [row for row in filtered if row["impact_type"] == "Monitor"]
    if priority_filter != "All":
        filtered = [row for row in filtered if row["priority"] == priority_filter]
    if cluster_filter != "All":
        filtered = [
            row for row in filtered if row["signal_id"] == cluster_filter
        ]
    if theme_filter != "All":
        filtered = [row for row in filtered if row["theme"] == theme_filter]
    return filtered


def operational_impact_dataframe(rows):
    return pd.DataFrame(
        [
            {
                "Signal ID": row["signal_id"],
                "Experience Signal": row["signal_name"],
                "Cluster": row["cluster_id"],
                "Theme": row["theme"],
                "Impact Type": row["impact_type"],
                "Priority": row["priority"],
                "Impact Score": round(row["impact_score"], 3),
                "Action Window": row["action_window"],
                "Quick Win": "Yes" if row["is_quick_win"] else "No",
                "Negative Rate": f"{row['negative_rate']:.0%}",
                "Feedback Volume": row["cluster_size"],
                "Cascade Targets": ", ".join(
                    [signal_reference(target) for target in row["cascade_targets"]]
                )
                or "None",
                "Root Cause": row["root_cause"],
                "Recommended Action": row["recommended_action"],
            }
            for row in rows
        ]
    )


def render_operational_impact_card(row, index):
    badge_class = (
        "high"
        if row["impact_type"] == "Systemic Risk"
        else "medium"
        if row["impact_type"] == "Service Recovery"
        else "low"
    )
    cascade_text = (
        " -> ".join([signal_reference(target) for target in row["cascade_targets"][:4]])
        if row["cascade_targets"]
        else "No strong cascade target"
    )
    st.markdown(
        '<div class="cx-impact-card">'
        '<div class="cx-card-topline">'
        f'<span class="cx-icon-block">OP{index}</span>'
        f'<span class="cx-chip {badge_class}">{escape(row["impact_type"])}</span>'
        "</div>"
        f'<p class="cx-eyebrow" style="margin-bottom:0.35rem;">{escape(row["signal_id"])} · {escape(row["priority"])}</p>'
        f"<h4>{escape(row['signal_name'])}</h4>"
        f"<p>{escape(row['key_insight'])}</p>"
        '<div class="cx-action-meta">'
        f'<div><span>Impact</span><strong>{row["impact_score"]:.3f}</strong></div>'
        f'<div><span>Negative</span><strong>{row["negative_rate"]:.0%}</strong></div>'
        f'<div><span>Cascades</span><strong>{row["cascade_count"]}</strong></div>'
        "</div>"
        f'<div class="cx-impact-path"><strong style="color:var(--cx-ink);">Cascade path:</strong> {escape(cascade_text)}</div>'
        '<div style="margin-top:0.75rem;">'
        '<strong style="color:var(--cx-ink);">Action:</strong> '
        f'<span>{escape(row["recommended_action"])}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_operational_command_cards(rows):
    if not rows:
        return

    risk = next(
        (row for row in rows if row["impact_type"] == "Systemic Risk"),
        rows[0],
    )
    recovery = max(
        rows,
        key=lambda row: (
            row["negative_rate"],
            row["impact_score"],
        ),
    )
    connected = max(
        rows,
        key=lambda row: (
            row["cascade_count"],
            row["top_cascade"],
            row["impact_score"],
        ),
    )
    quick_win = next((row for row in rows if row["is_quick_win"]), None)

    cards = [
        (
            "Highest Risk",
            f"{risk['signal_id']}: {risk['signal_name']}",
            f"{risk['priority']} · {risk['negative_rate']:.0%} negative · impact {risk['impact_score']:.2f}",
            "#dc2626",
        ),
        (
            "Recovery Opportunity",
            f"{recovery['signal_id']}: {recovery['signal_name']}",
            f"Focus service recovery where negative feedback is most concentrated.",
            "#d97706",
        ),
        (
            "Most Connected",
            f"{connected['signal_id']}: {connected['signal_name']}",
            f"{connected['cascade_count']} connected signal target(s).",
            "#2563eb",
        ),
        (
            "Best Quick Win",
            (
                f"{quick_win['signal_id']}: {quick_win['signal_name']}"
                if quick_win
                else "No clear quick win"
            ),
            (
                quick_win["recommended_action"]
                if quick_win
                else "Current filters do not expose a low-complexity, action-ready issue."
            ),
            "#059669",
        ),
    ]
    html = "".join(
        '<div class="cx-command-card">'
        f'<span class="cx-graph-type" style="color:{accent}; background:rgba(59,130,246,0.08);">{escape(label)}</span>'
        f"<h4>{escape(shorten_text(title, 58))}</h4>"
        f"<p>{escape(shorten_text(body, 136))}</p>"
        "</div>"
        for label, title, body, accent in cards
    )
    st.markdown(
        '<div class="cx-command-grid">'
        f"{html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_operational_priority_queue(rows):
    st.markdown(
        '<h4 class="cx-section-heading">Operational Priority Queue</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ranked by impact score, customer pain, and cascade reach. Start at the top unless leadership has a different constraint."
    )
    if not rows:
        st.info("No priority queue rows match the current filters.")
        return

    for index, row in enumerate(rows[:7], 1):
        cascade_text = (
            ", ".join([signal_reference(target) for target in row["cascade_targets"][:4]])
            if row["cascade_targets"]
            else "No strong connected signal"
        )
        st.markdown(
            '<div class="cx-priority-row">'
            f'<div><span class="cx-rank-pill">{index}</span></div>'
            "<div>"
            f"<h4>{escape(row['signal_id'])}: {escape(row['signal_name'])}</h4>"
            f'<span class="cx-chip {priority_class(row["priority"])}">{escape(row["priority"])}</span>'
            f'<p style="margin-top:0.45rem;">{escape(row["impact_type"])} · {row["action_window"]}</p>'
            "</div>"
            "<div>"
            f"<p><strong style='color:var(--cx-ink);'>Why it matters:</strong> {escape(shorten_text(row['key_insight'], 160))}</p>"
            f"<p style='margin-top:0.45rem;'><strong style='color:var(--cx-ink);'>Ripple:</strong> {escape(cascade_text)}</p>"
            "</div>"
            "<div>"
            f"<p><strong style='color:var(--cx-ink);'>Action:</strong> {escape(shorten_text(row['recommended_action'], 170))}</p>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def apply_operational_chart_theme(fig, height=360, showlegend=False):
    apply_plotly_soft_ui(fig, height=height, showlegend=showlegend)
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=11),
        title=dict(font=dict(color="#172033", size=16)),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#172033")),
    )
    return fig


def render_operational_impact_charts(rows):
    if not rows:
        st.info("No operational impact graphics match the current filters.")
        return

    st.markdown(
        '<h4 class="cx-section-heading">Impact Matrix</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Quadrants translate model output into operational posture: fix first, recover service, monitor closely, or keep low priority."
    )
    scatter_fig = go.Figure()
    cascade_threshold = max(1, int(np.median([row["cascade_count"] for row in rows])))
    scatter_fig.add_trace(
        go.Scatter(
            x=[row["negative_rate"] for row in rows],
            y=[row["cascade_count"] for row in rows],
            mode="markers+text",
            text=[row["signal_id"] for row in rows],
            textposition="top center",
            marker=dict(
                size=[18 + row["impact_score"] * 28 for row in rows],
                color=[row["impact_score"] for row in rows],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Impact"),
                line=dict(color="#ffffff", width=1.2),
            ),
            hovertext=[
                f"{row['signal_name']}<br>{row['priority']}<br>{row['recommended_action']}"
                for row in rows
            ],
            hovertemplate=(
                "<b>%{text}</b><br>Negative rate: %{x:.0%}<br>"
                "Cascade count: %{y}<br>%{hovertext}<extra></extra>"
            ),
        )
    )
    scatter_fig.add_vrect(
        x0=0.4,
        x1=1,
        y0=cascade_threshold,
        y1=max([row["cascade_count"] for row in rows] + [cascade_threshold]) + 1,
        fillcolor="#fee2e2",
        opacity=0.45,
        line_width=0,
        annotation_text="Fix first",
        annotation_position="top left",
    )
    scatter_fig.add_vrect(
        x0=0.4,
        x1=1,
        y0=-0.25,
        y1=cascade_threshold,
        fillcolor="#ffedd5",
        opacity=0.4,
        line_width=0,
        annotation_text="Service recovery",
        annotation_position="bottom left",
    )
    scatter_fig.add_vrect(
        x0=0,
        x1=0.4,
        y0=cascade_threshold,
        y1=max([row["cascade_count"] for row in rows] + [cascade_threshold]) + 1,
        fillcolor="#dbeafe",
        opacity=0.38,
        line_width=0,
        annotation_text="Monitor closely",
        annotation_position="top right",
    )
    scatter_fig.add_vrect(
        x0=0,
        x1=0.4,
        y0=-0.25,
        y1=cascade_threshold,
        fillcolor="#dcfce7",
        opacity=0.34,
        line_width=0,
        annotation_text="Lower priority",
        annotation_position="bottom right",
    )
    scatter_fig.add_vline(x=0.4, line=dict(color="#172033", dash="dash", width=1))
    scatter_fig.add_hline(
        y=cascade_threshold,
        line=dict(color="#172033", dash="dash", width=1),
    )
    scatter_fig.update_layout(
        title="Impact Matrix: Customer Pain vs Ripple Reach",
        xaxis_title="Negative feedback rate",
        yaxis_title="Cascade targets",
        xaxis_tickformat=".0%",
    )
    apply_operational_chart_theme(scatter_fig, height=500)
    st.plotly_chart(scatter_fig, width="stretch")

    top_rows = rows[:8]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[row["impact_score"] for row in top_rows],
                y=[f"{row['signal_id']}: {row['signal_name']}" for row in top_rows],
                orientation="h",
                marker=dict(
                    color=[
                        "#dc2626"
                        if row["impact_type"] == "Systemic Risk"
                        else "#d97706"
                        if row["impact_type"] == "Service Recovery"
                        else "#059669"
                        if row["impact_type"] == "Protect Strength"
                        else "#64748b"
                        for row in top_rows
                    ]
                ),
                text=[f"{row['impact_score']:.2f}" for row in top_rows],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Impact score: %{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Impact Score Ranking",
        xaxis_title="Impact score",
        yaxis=dict(autorange="reversed"),
    )
    apply_operational_chart_theme(fig, height=360)
    st.plotly_chart(fig, width="stretch")


def render_ripple_summary(rows):
    st.markdown(
        '<h4 class="cx-section-heading">Ripple Summary</h4>',
        unsafe_allow_html=True,
    )
    if not rows:
        st.info("No ripple summaries match the current filters.")
        return

    connected_rows = [row for row in rows if row["cascade_targets"]]
    if not connected_rows:
        st.info("No strong ripple paths are visible for the current filters.")
        return

    for row in connected_rows[:4]:
        target_text = ", ".join([signal_reference(target) for target in row["cascade_targets"][:4]])
        st.markdown(
            '<div class="cx-impact-path">'
            f'<strong style="color:var(--cx-ink);">{escape(row["signal_id"])}: {escape(row["signal_name"])}</strong><br>'
            f"Fixing this issue may also affect {escape(target_text)}. "
            f"Recommended move: {escape(shorten_text(row['recommended_action'], 170))}"
            "</div>",
            unsafe_allow_html=True,
        )


def render_operational_action_plan(rows):
    st.markdown(
        '<h4 class="cx-section-heading">Action Plan</h4>',
        unsafe_allow_html=True,
    )
    groups = [
        ("Immediate", "Fix first", "#dc2626"),
        ("Next 7 Days", "Plan next", "#d97706"),
        ("Monitor", "Watch signals", "#64748b"),
    ]
    cols = st.columns(3)
    for col, (window, subtitle, accent) in zip(cols, groups):
        items = [row for row in rows if row["action_window"] == window][:4]
        with col:
            item_html = ""
            if items:
                for row in items:
                    item_html += (
                        '<div class="cx-action-group-item">'
                        f'<strong>{escape(row["signal_id"])}: {escape(row["signal_name"])}</strong>'
                        f'<p>{escape(shorten_text(row["recommended_action"], 150))}</p>'
                        "</div>"
                    )
            else:
                item_html = '<div class="cx-action-group-item"><p>No matching actions.</p></div>'
            st.markdown(
                '<div class="cx-action-group">'
                f'<span class="cx-graph-type" style="color:{accent};">{escape(subtitle)}</span>'
                f"<h4>{escape(window)}</h4>"
                f"{item_html}"
                "</div>",
                unsafe_allow_html=True,
            )


def build_intervention_projection(selected_cluster_id, action_insights, causal_engine, improvement_pct):
    insight_by_cluster = {insight.cluster_id: insight for insight in action_insights}
    selected_insight = insight_by_cluster.get(selected_cluster_id)
    if selected_insight is None:
        return pd.DataFrame(), None

    rows = []
    direct_negative = selected_insight.metadata.get("negative_rate", 0)
    direct_reduction = improvement_pct / 100
    rows.append(
        {
            "Signal": signal_reference(selected_insight.cluster_id),
            "Theme": insight_display_name(selected_insight),
            "Impact Path": "Direct issue",
            "Current Negative Rate": direct_negative,
            "Projected Negative Rate": max(0, direct_negative * (1 - direct_reduction)),
            "Projected Change": direct_negative * direct_reduction,
            "Likelihood": 1.0,
            "Why It Changes": "This is the selected problem being improved directly.",
            "Recommended Action": selected_insight.recommended_action,
        }
    )

    for cascade in getattr(causal_engine, "cascade_predictions", {}).get(
        selected_cluster_id, []
    )[:5]:
        target_cluster = int(cascade.get("target_cluster", -1))
        target_insight = insight_by_cluster.get(target_cluster)
        if target_insight is None:
            continue
        likelihood = float(cascade.get("cascade_likelihood", 0))
        current_negative = target_insight.metadata.get("negative_rate", 0)
        cascade_reduction = direct_reduction * likelihood * 0.55
        projected_negative = max(0, current_negative * (1 - cascade_reduction))
        rows.append(
            {
                "Signal": signal_reference(target_cluster),
                "Theme": insight_display_name(target_insight),
                "Impact Path": "Related cascade",
                "Current Negative Rate": current_negative,
                "Projected Negative Rate": projected_negative,
                "Projected Change": current_negative - projected_negative,
                "Likelihood": likelihood,
                "Why It Changes": cascade.get(
                    "cascade_interpretation",
                    f"{signal_reference(target_cluster)} shares service factors with {signal_reference(selected_cluster_id)}.",
                ),
                "Recommended Action": target_insight.recommended_action,
            }
        )

    projection_df = pd.DataFrame(rows)
    return projection_df, selected_insight


def render_change_impact_simulator(action_insights, causal_engine):
    st.markdown(
        '<h4 class="cx-section-heading">Change Impact Simulator</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Model a practical scenario: if one customer problem improves, which connected signals may benefit next?"
    )

    if not action_insights:
        st.info("No action insights are available for scenario modeling.")
        return

    scenario_col, strength_col = st.columns([1.45, 1])
    with scenario_col:
        selected_cluster_id = st.selectbox(
            "Problem to improve",
            [insight.cluster_id for insight in action_insights],
            format_func=lambda cluster_id: (
                f"{signal_reference(cluster_id)} · "
                f"{next(insight_display_name(item) for item in action_insights if item.cluster_id == cluster_id)}"
            ),
            key="operational_change_simulator_cluster",
        )
    with strength_col:
        improvement_pct = st.slider(
            "Expected direct improvement",
            min_value=10,
            max_value=80,
            value=35,
            step=5,
            format="%d%%",
            key="operational_change_simulator_improvement",
        )

    projection_df, selected_insight = build_intervention_projection(
        selected_cluster_id,
        action_insights,
        causal_engine,
        improvement_pct,
    )
    if projection_df.empty or selected_insight is None:
        st.info("No scenario projection is available for this problem.")
        return

    connected_count = max(len(projection_df) - 1, 0)
    projected_total_change = projection_df["Projected Change"].sum()
    highest_related = projection_df.iloc[1] if len(projection_df) > 1 else None

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_kpi_card(
            "Selected Problem",
            signal_reference(selected_cluster_id),
            insight_display_name(selected_insight),
            "#3b82f6",
        )
    with metric_cols[1]:
        render_kpi_card(
            "Connected Signals",
            connected_count,
            "Likely downstream effects",
            "#06b6d4",
        )
    with metric_cols[2]:
        render_kpi_card(
            "Total Negative Shift",
            f"{projected_total_change:.0%}",
            "Across modeled signals",
            "#10b981",
        )

    st.markdown(
        '<div class="cx-decision-card">'
        '<div class="cx-card-topline">'
        '<span class="cx-icon-block">IF</span>'
        f'<span class="cx-chip {priority_class(selected_insight.priority_label)}">{escape(selected_insight.priority_label)}</span>'
        "</div>"
        f"<h4>If {escape(insight_display_name(selected_insight, include_reference=True))} improves by {improvement_pct}%</h4>"
        f"<p>{escape(selected_insight.key_insight)}</p>"
        '<div class="cx-panel-soft" style="padding:0.8rem; border-radius:0.85rem; margin-top:0.75rem;">'
        f'<strong style="color:var(--cx-ink);">Manager move:</strong> {escape(selected_insight.recommended_action)}'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    chart_df = projection_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["Signal"],
            y=chart_df["Current Negative Rate"],
            name="Current",
            marker_color="#ef4444",
            hovertemplate="<b>%{x}</b><br>Current negative: %{y:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=chart_df["Signal"],
            y=chart_df["Projected Negative Rate"],
            name="Projected after change",
            marker_color="#10b981",
            hovertemplate="<b>%{x}</b><br>Projected negative: %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Before vs After Scenario",
        yaxis_title="Negative feedback rate",
        xaxis_title="Affected signal",
        yaxis_tickformat=".0%",
        barmode="group",
    )
    apply_operational_chart_theme(fig, height=360)
    st.plotly_chart(fig, width="stretch")

    if highest_related is not None:
        st.markdown(
            '<div class="cx-impact-path">'
            f'<strong style="color:var(--cx-ink);">Most likely related impact: {escape(highest_related["Signal"])} - {escape(highest_related["Theme"])}</strong><br>'
            f'{escape(highest_related["Why It Changes"])}'
            "</div>",
            unsafe_allow_html=True,
        )

    display_df = projection_df.copy()
    for column in ["Current Negative Rate", "Projected Negative Rate", "Projected Change", "Likelihood"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.0%}")
    render_table_section(
        "Scenario Impact Detail",
        "A readable estimate of direct and related improvements from the selected change.",
        display_df,
        "No scenario details are available.",
        accent_label="What-if table",
    )


def render_operational_impact(
    causal_engine,
    action_insights,
    ai_config=None,
    feedback_source=None,
    texts=None,
    clustering_engine=None,
):
    render_page_header(
        "Operational Impact",
        "A manager-facing view of which experience signals are most likely to affect operations, cascade into related signals, and require action.",
        "Operations",
    )
    render_stakeholder_explanation(
        "Operational Impact",
        "operational risk, cascade exposure, action timing, and change impact decisions",
        action_insights,
        ai_config,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
    )

    impact_rows = build_operational_impact_rows(action_insights, causal_engine)
    priority_options = ["All"] + sorted(
        {row["priority"] for row in impact_rows},
        key=lambda label: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(label.split()[0], 9),
    )
    signal_label_map = {
        row["signal_id"]: row["signal_name"]
        for row in sorted(impact_rows, key=lambda item: item["cluster_id"])
    }
    cluster_options = ["All"] + list(signal_label_map.keys())
    theme_options = ["All"] + sorted({row["theme"] for row in impact_rows})

    focus_col, download_col = st.columns([3, 0.85])
    with focus_col:
        focus_filter = st.segmented_control(
            "Operational focus",
            options=[
                "All",
                "Systemic Risks",
                "Service Recovery",
                "Quick Wins",
                "Protect Strengths",
                "Monitor",
            ],
            default="All",
            key="operational_impact_focus_filter",
        )

    with st.expander("Advanced filters"):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            priority_filter = st.selectbox(
                "Priority",
                priority_options,
                key="operational_impact_priority_filter",
            )
        with filter_col2:
            cluster_filter = st.selectbox(
                "Signal",
                cluster_options,
                format_func=lambda value: "All" if value == "All" else f"{value} · {signal_label_map[value]}",
                key="operational_impact_cluster_filter",
            )
        with filter_col3:
            theme_filter = st.selectbox(
                "Theme",
                theme_options,
                key="operational_impact_theme_filter",
            )

    visible_rows = filter_operational_impact_rows(
        impact_rows,
        focus_filter,
        priority_filter,
        cluster_filter,
        theme_filter,
    )
    export_df = operational_impact_dataframe(visible_rows)
    with download_col:
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False),
            file_name="px_intel_operational_impact.csv",
            mime="text/csv",
            width="stretch",
        )

    top_row = visible_rows[0] if visible_rows else None
    systemic_count = sum(1 for row in visible_rows if row["impact_type"] == "Systemic Risk")
    total_cascades = sum(row["cascade_count"] for row in visible_rows)
    avg_negative = (
        np.mean([row["negative_rate"] for row in visible_rows]) if visible_rows else 0
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_kpi_card("Impact Signals", len(visible_rows), "After filters", "#3b82f6")
    with metric_cols[1]:
        render_kpi_card("Systemic Risks", systemic_count, "Highest operational concern", "#dc2626")
    with metric_cols[2]:
        render_kpi_card("Cascade Targets", total_cascades, "Shared impact links", "#06b6d4")
    with metric_cols[3]:
        render_kpi_card("Avg Negative Rate", f"{avg_negative:.0%}", "Visible signals", "#d97706")

    render_operational_command_cards(visible_rows)

    if top_row:
        st.markdown(
            '<div class="cx-decision-card featured">'
            '<div class="cx-card-topline">'
            '<span class="cx-icon-block">OP</span>'
            f'<span class="cx-chip {priority_class(top_row["priority"])}">{escape(top_row["priority"])}</span>'
            "</div>"
            '<p class="cx-eyebrow" style="margin-bottom:0.35rem;">Highest operational impact</p>'
            f'<h3 style="margin:0 0 0.5rem;">{escape(top_row["signal_name"])} ({escape(top_row["signal_id"])})</h3>'
            f'<p style="margin:0 0 0.7rem;">{escape(top_row["key_insight"])}</p>'
            '<div class="cx-panel-soft" style="padding:0.8rem; border-radius:0.85rem;">'
            '<strong style="color:var(--cx-ink);">Action:</strong> '
            f'<span>{escape(top_row["recommended_action"])}</span>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    render_operational_priority_queue(visible_rows)
    render_operational_impact_charts(visible_rows)
    render_change_impact_simulator(action_insights, causal_engine)
    render_ripple_summary(visible_rows)
    render_operational_action_plan(visible_rows)

    render_table_section(
        "Operational Evidence Table",
        "Filtered operational evidence with impact type, action window, cascade targets, and recommended response.",
        export_df,
        "No operational impact table rows are available for the current filters.",
        accent_label="Operational evidence",
    )

    with st.expander("M3 causal model details"):
        st.markdown(causal_engine.get_summary())
        cluster_options = sorted(causal_engine.cluster_lda_features.keys())
        if cluster_options:
            selected_cluster = st.selectbox(
                "Review cluster causal details",
                cluster_options,
                format_func=lambda cluster_id: f"Cluster {cluster_id}",
                key="operational_impact_causal_detail_cluster",
            )
            st.markdown(causal_engine.get_cluster_summary(selected_cluster))
        causal_df = causal_engine.export_to_dataframe()
        render_table_header(
            "Causal Model Export",
            "Detailed M3 causal output for the selected operational context.",
            causal_df,
            accent_label="Model table",
        )
        st.dataframe(causal_df, width="stretch", hide_index=True, height=340)


def apply_plotly_soft_ui(fig, height=520, showlegend=True):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=11),
        height=height,
        margin=dict(l=48, r=24, t=52, b=52),
        showlegend=showlegend,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(94,114,228,0.18)", font=dict(color="#172033")),
    )
    fig.update_xaxes(gridcolor="rgba(23,32,51,0.08)", zerolinecolor="rgba(23,32,51,0.08)")
    fig.update_yaxes(gridcolor="rgba(23,32,51,0.08)", zerolinecolor="rgba(23,32,51,0.08)")
    return fig


def render_overview_coverage_cards(
    audit_engine,
    causal_engine,
    action_insights,
    clustering_engine,
    texts,
):
    """Render compact cross-app status cards for the Overview."""
    at_risk_count = sum(
        1 for insight in action_insights if customer_lens_for_insight(insight) == "At Risk"
    )
    opportunity_count = sum(
        1
        for insight in action_insights
        if customer_lens_for_insight(insight) == "Opportunity"
    )
    _, graph_edges = build_cause_effect_graph(action_insights, causal_engine)
    relationship_count = len(graph_edges)
    red_count = len(audit_engine.get_red_zones())
    export_rows = len(clustering_engine.export_to_dataframe(texts))

    cards = [
        (
            "CI",
            "Customer Signals",
            f"{at_risk_count} at risk",
            f"{opportunity_count} opportunity signals",
            "#3b82f6",
        ),
        (
            "CE",
            "Cause Links",
            f"{relationship_count} relationships",
            "Mapped from themes, issues, causes, and actions",
            "#06b6d4",
        ),
        (
            "CA",
            "Signal Health",
            f"{red_count} distress zones",
            f"{len(audit_engine.cluster_texts)} audited signals",
            "#f59e0b",
        ),
        (
            "RP",
            "Report Data",
            f"{export_rows:,} rows",
            "Ready for CSV export",
            "#10b981",
        ),
    ]

    st.markdown(
        '<h4 class="cx-section-heading">Intelligence Coverage</h4>',
        unsafe_allow_html=True,
    )
    cards_html = ""
    for icon, title, metric, detail, accent in cards:
        cards_html += (
            '<div class="cx-command-card">'
            '<div class="cx-card-topline">'
            f'<span class="cx-icon-block">{escape(icon)}</span>'
            f'<span class="cx-graph-type" style="color:{accent};">{escape(metric)}</span>'
            "</div>"
            f"<h4>{escape(title)}</h4>"
            f"<p>{escape(detail)}</p>"
            "</div>"
        )
    st.markdown(f'<div class="cx-command-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_immediate_action_alerts(action_insights, causal_engine):
    """Surface urgent signals that need immediate stakeholder attention."""
    impact_rows = build_operational_impact_rows(action_insights, causal_engine)
    alert_rows = [
        row
        for row in impact_rows
        if row["action_window"] == "Immediate"
        or row["impact_type"] == "Systemic Risk"
        or str(row["priority"]).startswith("HIGH")
    ]
    alert_rows = sorted(
        alert_rows,
        key=lambda row: (
            row["action_window"] == "Immediate",
            row["priority"].startswith("HIGH"),
            row["impact_score"],
        ),
        reverse=True,
    )
    shown_rows = alert_rows[:3]

    if shown_rows:
        intro = (
            f"{len(alert_rows)} signal(s) need immediate review. "
            "These alerts combine high priority, operational impact, negative concentration, and cascade exposure."
        )
    else:
        intro = (
            "No immediate action alerts are active. Continue monitoring medium-priority signals and protect positive patterns."
        )

    cards_html = ""
    if shown_rows:
        for index, row in enumerate(shown_rows, start=1):
            cards_html += (
                '<div class="cx-alert-card">'
                '<div class="cx-card-topline" style="margin-bottom:0.25rem;">'
                f'<span class="cx-icon-block">A{index}</span>'
                f'<span class="cx-chip {priority_class(row["priority"])}">{escape(row["priority"])}</span>'
                "</div>"
                f'<h4>{escape(row["signal_name"])} ({escape(row["signal_id"])})</h4>'
                f'<p>{escape(shorten_text(row["key_insight"], 150))}</p>'
                '<div class="cx-alert-meta">'
                f'<span>{escape(row["action_window"])}</span>'
                f'<span>{escape(row["impact_type"])}</span>'
                f'<span>{row["negative_rate"]:.0%} negative</span>'
                f'<span>{row["cascade_count"]} linked signals</span>'
                "</div>"
                '<div class="cx-panel-soft" style="padding:0.7rem; border-radius:0.75rem; margin-top:0.7rem;">'
                '<strong style="color:var(--cx-ink);">Next move:</strong> '
                f'<span>{escape(shorten_text(row["recommended_action"], 170))}</span>'
                "</div>"
                "</div>"
            )
    else:
        cards_html = (
            '<div class="cx-alert-card">'
            '<div class="cx-card-topline" style="margin-bottom:0.25rem;">'
            '<span class="cx-icon-block">OK</span>'
            '<span class="cx-chip low">Stable</span>'
            "</div>"
            "<h4>No immediate alerts</h4>"
            "<p>The current dataset has no high-priority immediate action signal. Keep the watchlist active and revisit after new feedback is uploaded.</p>"
            "</div>"
        )

    st.markdown(
        f"""
        <section class="cx-alert-band">
            <div class="cx-card-topline" style="margin-bottom:0.2rem;">
                <div>
                    <p class="cx-eyebrow" style="margin:0;">Immediate action alerts</p>
                    <h3>Signals that need attention now</h3>
                    <p>{escape(intro)}</p>
                </div>
                <span class="cx-chip high">{len(alert_rows)} active</span>
            </div>
            <div class="cx-alert-grid {'single' if len(shown_rows) <= 1 else ''}">{cards_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_agent_decision_support(
    action_agent,
    action_insights,
    top_agent_insight,
    high_priority_count,
    medium_priority_count,
):
    st.markdown(
        '<h3 class="cx-section-heading">Decision Needed</h3>',
        unsafe_allow_html=True,
    )

    decision_col, outline_col = st.columns([1.35, 1])
    with decision_col:
        render_decision_summary_card(
            top_agent_insight,
            len(action_insights),
            high_priority_count,
            medium_priority_count,
        )

    with outline_col:
        st.markdown(
            f"""
            <div class="cx-decision-card">
                <div class="cx-card-topline">
                    <span class="cx-icon-block">DS</span>
                    <span class="cx-chip">Decision view</span>
                </div>
                <p class="cx-eyebrow" style="margin-bottom:0.35rem;">Action intelligence</p>
                <h4 style="margin:0 0 0.5rem;">Manager-ready signal summary</h4>
                <p style="margin:0 0 0.5rem;">The strongest issues are scored, explained, and paired with the next operational move.</p>
                <div class="cx-action-meta">
                    <div><span>Insights</span><strong>{len(action_insights)}</strong></div>
                    <div><span>High</span><strong>{high_priority_count}</strong></div>
                    <div><span>Medium</span><strong>{medium_priority_count}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def infer_agent_response_mode(question):
    """Infer the best AI response format from the user's question."""
    text = str(question or "").lower()
    if any(
        phrase in text
        for phrase in (
            "30-day",
            "30 day",
            "action plan",
            "operational plan",
            "operating plan",
            "roadmap",
            "cadence",
        )
    ):
        return "Operational action plan"
    if any(phrase in text for phrase in ("root cause", "why is", "why are", "cause")):
        return "Root cause analysis"
    if any(
        phrase in text
        for phrase in ("evidence", "proof", "show me the data", "customer language")
    ):
        return "Evidence memo"
    if any(
        phrase in text
        for phrase in ("recover", "recovery", "apology", "customer recovery")
    ):
        return "Customer recovery plan"
    if any(phrase in text for phrase in ("report", "write up", "write-up", "section")):
        return "Report section"
    return "Decision brief"


def render_agent_chat(
    action_agent,
    action_insights,
    ai_config=None,
    ai_context=None,
):
    st.markdown(
        '<h4 class="cx-section-heading">Ask PX-Intel Agent</h4>',
        unsafe_allow_html=True,
    )
    if ai_config and ai_config.get("enabled"):
        st.caption(
            f"AI generation active: {ai_config.get('model')} · {ai_config.get('generation_strength')}. PX-Intel chooses the response format from your question."
        )
    else:
        st.caption(
            "Local intelligence is active. Add an OpenAI key in the sidebar for richer generated answers."
        )

    chat_signature = (
        "auto_intent_v1",
        "ai" if ai_config and ai_config.get("enabled") else "local",
        ai_config.get("model") if ai_config else "local",
        ai_config.get("generation_strength") if ai_config else "local",
        len(action_insights),
    )
    if st.session_state.get("cx_agent_messages_signature") != chat_signature:
        st.session_state.pop("cx_agent_messages", None)
        st.session_state.cx_agent_messages_signature = chat_signature

    if "cx_agent_messages" not in st.session_state:
        st.session_state.cx_agent_messages = [
            {
                "role": "assistant",
                "content": answer_with_optional_ai(
                    "summarize for leadership",
                    action_agent,
                    action_insights,
                    ai_config,
                    ai_context,
                    infer_agent_response_mode("summarize for leadership"),
                ),
            }
        ]

    prompt_cols = st.columns(4)
    suggested_prompts = [
        "Generate a decision brief.",
        "Build a 30-day action plan.",
        "Explain the root causes.",
        "Draft a report section.",
    ]
    for col, prompt in zip(prompt_cols, suggested_prompts):
        with col:
            if st.button(prompt, key=f"overview_agent_prompt_{prompt}"):
                st.session_state.cx_agent_messages.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.cx_agent_messages.append(
                    {
                        "role": "assistant",
                        "content": answer_with_optional_ai(
                            prompt,
                            action_agent,
                            action_insights,
                            ai_config,
                            ai_context,
                            infer_agent_response_mode(prompt),
                        ),
                    }
                )

    for message in st.session_state.cx_agent_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question_col, ask_col = st.columns([3.2, 0.8])
    with question_col:
        user_question = st.text_area(
            "Ask PX-Intel",
            placeholder="Ask about priorities, cascades, actions, or a specific signal...",
            key="cx_agent_question_input",
            label_visibility="collapsed",
            height=84,
        )
    with ask_col:
        ask_submitted = st.button(
            "Ask PX-Intel",
            key="cx_agent_question_submit",
            width="stretch",
        )
        st.caption("Answers use the active dataset and visible PX-Intel signals.")

    if ask_submitted and user_question.strip():
        cleaned_question = user_question.strip()
        st.session_state.cx_agent_messages.append(
            {"role": "user", "content": cleaned_question}
        )
        answer = answer_with_optional_ai(
            cleaned_question,
            action_agent,
            action_insights,
            ai_config,
            ai_context,
            infer_agent_response_mode(cleaned_question),
        )
        st.session_state.cx_agent_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()
    if ask_submitted and not user_question.strip():
        st.warning(
            "Type a question or use one of the suggested prompts above."
        )


def render_customer_action_dashboard(action_agent, action_insights):
    st.markdown(
        '<h3 class="cx-section-heading">Customer Experience Action Dashboard</h3>',
        unsafe_allow_html=True,
    )

    filter_col, download_col = st.columns([2.3, 1])
    with filter_col:
        priority_filter = st.segmented_control(
            "Priority filter",
            options=["All", "HIGH 🔥", "MEDIUM ⚠", "LOW ✅"],
            default="All",
            key="overview_action_priority_segment",
        )
    filtered_insights = (
        action_insights
        if priority_filter == "All"
        else [
            insight
            for insight in action_insights
            if insight.priority_label == priority_filter
        ]
    )

    if st.session_state.get("overview_action_priority_filter") != priority_filter:
        st.session_state.overview_action_priority_filter = priority_filter
        st.session_state.overview_action_dashboard_page = 0

    action_df = action_agent.build_dashboard_dataframe(filtered_insights)
    with download_col:
        st.download_button(
            label="Download CSV",
            data=action_df.to_csv(index=False),
            file_name="cx_intel_action_dashboard.csv",
            mime="text/csv",
            help="Download the currently selected action data.",
            width="stretch",
        )

    page_size = 4
    total_pages = max(1, int(np.ceil(len(filtered_insights) / page_size)))
    current_page = min(
        st.session_state.get("overview_action_dashboard_page", 0),
        total_pages - 1,
    )
    st.session_state.overview_action_dashboard_page = current_page
    page_start = current_page * page_size
    page_end = page_start + page_size
    visible_insights = filtered_insights[page_start:page_end]

    if not filtered_insights:
        st.info("No action insights match this priority filter.")
        return

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if st.button(
            "Previous",
            key="overview_action_dashboard_previous",
            disabled=current_page == 0,
            width="stretch",
        ):
            st.session_state.overview_action_dashboard_page = max(current_page - 1, 0)
            st.rerun()
    with nav_mid:
        st.markdown(
            f"""
            <div class="cx-panel-soft" style="padding:0.65rem; border-radius:0.85rem; text-align:center;">
                Showing actions {page_start + 1}-{min(page_end, len(filtered_insights))} of {len(filtered_insights)}
                &nbsp;|&nbsp; Page {current_page + 1} of {total_pages}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button(
            "Next",
            key="overview_action_dashboard_next",
            disabled=current_page >= total_pages - 1,
            width="stretch",
        ):
            st.session_state.overview_action_dashboard_page = min(
                current_page + 1,
                total_pages - 1,
            )
            st.rerun()

    for start in range(0, len(visible_insights), 2):
        card_cols = st.columns(2)
        for offset, card_col in enumerate(card_cols):
            insight_index = start + offset
            if insight_index >= len(visible_insights):
                continue
            with card_col:
                render_action_outline_card(
                    visible_insights[insight_index],
                    page_start + insight_index + 1,
                )


def render_overview(
    audit_engine,
    causal_engine,
    clustering_engine,
    texts,
    action_agent,
    action_insights,
    top_agent_insight,
    high_priority_count,
    medium_priority_count,
    ai_config=None,
    feedback_source=None,
):
    render_page_header(
        "PX-Intel Overview",
        "A focused command view for KPIs, action priorities, customer experience actions, and the PX-Intel agent.",
        "Overview",
    )

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_kpi_card(
            "Feedback Entries",
            f"{len(texts):,}",
            "Loaded from PX-Intel data",
            "#3b82f6",
        )
    with kpi_col2:
        render_kpi_card(
            "Experience Signals",
            clustering_engine.optimal_n_clusters,
            "Auto-selected by M1",
            "#06b6d4",
        )
    with kpi_col3:
        render_kpi_card(
            "High Priority",
            high_priority_count,
            "M5 action score",
            "#ef4444",
        )
    with kpi_col4:
        render_kpi_card(
            "Medium Priority",
            medium_priority_count,
            "Watchlist signals",
            "#d97706",
        )

    render_immediate_action_alerts(action_insights, causal_engine)
    render_agent_decision_support(
        action_agent,
        action_insights,
        top_agent_insight,
        high_priority_count,
        medium_priority_count,
    )
    render_customer_action_dashboard(action_agent, action_insights)
    render_overview_coverage_cards(
        audit_engine,
        causal_engine,
        action_insights,
        clustering_engine,
        texts,
    )
    agent_ai_context = build_report_ai_context(
        "Agent Decision Support",
        "Leadership",
        ai_config.get("generation_strength", "Board-ready") if ai_config else "Brief",
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
        action_insights,
    )
    render_agent_chat(action_agent, action_insights, ai_config, agent_ai_context)


def build_experience_map_figure(
    landscape_lens,
    clustering_engine,
    audit_engine,
    action_insights,
    cluster_assignments,
    texts,
):
    insight_by_cluster = {item.cluster_id: item for item in action_insights}
    fig = go.Figure()

    priority_color_map = {
        "HIGH 🔥": "#dc2626",
        "MEDIUM ⚠": "#d97706",
        "LOW ✅": "#059669",
    }
    zone_color_map = {
        "RED_ZONE": "#ef4444",
        "GREEN_ZONE": "#10b981",
        "NEUTRAL_ZONE": "#f59e0b",
    }
    cluster_palette = px.colors.qualitative.Safe

    for cluster_id in np.unique(cluster_assignments):
        mask = cluster_assignments == cluster_id
        sentiment_dist = audit_engine.cluster_sentiment_results[cluster_id][
            "sentiment_distribution"
        ]
        zone = audit_engine.cluster_zones[cluster_id]["zone_type"]
        insight = insight_by_cluster.get(int(cluster_id))

        if landscape_lens == "Priority Heatmap" and insight is not None:
            color = priority_color_map.get(insight.priority_label, "#64748b")
            trace_name = (
                f"{insight.priority_label} | {signal_reference(cluster_id)} | "
                f"{insight_display_name(insight)}"
            )
        elif landscape_lens == "Sentiment Health":
            color = zone_color_map.get(zone, "#64748b")
            readable_zone = zone.replace("_", " ").title()
            trace_name = f"{signal_reference(cluster_id)} | {readable_zone}"
        else:
            color = cluster_palette[int(cluster_id) % len(cluster_palette)]
            theme = insight_display_name(insight) if insight else "Theme"
            trace_name = f"{signal_reference(cluster_id)} | {theme}"

        hover_text = []
        for text in np.array(texts)[mask]:
            feedback = escape(str(text)[:180])
            if insight is not None:
                hover_text.append(
                    f"<b>{escape(insight_display_name(insight, include_reference=True))}</b><br>"
                    f"Theme: {escape(insight.issue_theme.title())}<br>"
                    f"Priority: {escape(insight.priority_label)} "
                    f"({insight.priority_score:.3f})<br>"
                    f"Negative Feedback: {sentiment_dist['NEGATIVE']:.1%}<br>"
                    f"Action: {escape(insight.recommended_action)}<br><br>"
                    f"Feedback: {feedback}"
                )
            else:
                hover_text.append(
                    f"<b>{signal_reference(cluster_id)}</b><br>"
                    f"Negative: {sentiment_dist['NEGATIVE']:.1%}<br>"
                    f"Feedback: {feedback}"
                )

        fig.add_trace(
            go.Scatter(
                x=clustering_engine.tsne_projection[mask, 0],
                y=clustering_engine.tsne_projection[mask, 1],
                mode="markers",
                name=trace_name,
                marker=dict(
                    size=(
                        10
                        if insight and insight.priority_label.startswith("HIGH")
                        else 7
                    ),
                    color=color,
                    opacity=0.78,
                    line=dict(width=0.6, color="white"),
                ),
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Experience Map - {landscape_lens}",
        xaxis_title="Experience similarity dimension 1",
        yaxis_title="Experience similarity dimension 2",
        hovermode="closest",
        legend_title="Map Legend",
    )
    apply_plotly_soft_ui(fig, height=560)
    return fig


def render_cluster_analysis(
    audit_engine,
    clustering_engine,
    texts,
    cluster_assignments,
    action_agent,
    action_insights,
    top_agent_insight,
    ai_config=None,
    feedback_source=None,
    causal_engine=None,
):
    insight_by_cluster = {item.cluster_id: item for item in action_insights}
    high_clusters = [
        item for item in action_insights if item.priority_label.startswith("HIGH")
    ]
    red_count = len(audit_engine.get_red_zones())
    green_count = len(audit_engine.get_green_zones())
    neutral_count = len(audit_engine.get_neutral_zones())

    render_page_header(
        "Signal Analysis",
        "Inspect feedback signals, sentiment zones, representative themes, and the audit output behind the PX-Intel intelligence layer.",
        "Model audit",
    )
    render_stakeholder_explanation(
        "Signal Analysis",
        "model audit, signal quality, sentiment zones, and source evidence inspection",
        action_insights,
        ai_config,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        render_kpi_card("Feedback Entries", f"{len(texts):,}", "Mapped comments", "#3b82f6")
    with summary_col2:
        render_kpi_card(
            "Experience Signals",
            clustering_engine.optimal_n_clusters,
            "M1 discovery output",
            "#06b6d4",
        )
    with summary_col3:
        render_kpi_card("High Priority", len(high_clusters), "Needs action", "#ef4444")
    with summary_col4:
        render_kpi_card(
            "Top Priority",
            (
                insight_display_name(top_agent_insight)
                if top_agent_insight is not None
                else "None"
            ),
            "Highest M5 score",
            "#d97706",
        )

    st.markdown(
        '<h4 class="cx-section-heading">Experience Map</h4>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        landscape_lens = st.segmented_control(
            "Map lens",
            ["Priority Heatmap", "Sentiment Health", "Theme Clusters"],
            default="Priority Heatmap",
            key="cluster_analysis_map_lens",
        )
        fig = build_experience_map_figure(
            landscape_lens,
            clustering_engine,
            audit_engine,
            action_insights,
            cluster_assignments,
            texts,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown(
            '<h4 class="cx-section-heading">Signal Inspector</h4>',
            unsafe_allow_html=True,
        )
        if action_insights:
            selected_map_cluster = st.selectbox(
                "Choose a signal",
                [item.cluster_id for item in action_insights],
                format_func=lambda cid: (
                    f"{signal_reference(cid)} · "
                    f"{insight_display_name(insight_by_cluster[cid])} "
                    f"({insight_by_cluster[cid].priority_label})"
                ),
                key="cluster_analysis_selected_cluster",
            )
            selected_insight = insight_by_cluster[selected_map_cluster]
            st.markdown(
                '<div class="cx-graph-detail">'
                '<span class="cx-graph-type">AI Interpretation</span>'
                f"<h4>{escape(insight_display_name(selected_insight, include_reference=True))}</h4>"
                f"<p>{escape(selected_insight.key_insight)}</p>"
                '<div class="cx-panel-soft" style="padding:0.75rem; border-radius:0.85rem; margin-top:0.75rem;">'
                f'<strong style="color:var(--cx-ink);">Action:</strong> {escape(selected_insight.recommended_action)}'
                "</div>"
                '<div class="cx-panel-soft" style="padding:0.75rem; border-radius:0.85rem; margin-top:0.75rem;">'
                f'<strong style="color:var(--cx-ink);">Root cause:</strong> {escape(selected_insight.root_cause)}'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No signal insights are available yet.")

        st.markdown(
            '<h4 class="cx-section-heading">Sentiment Health</h4>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="cx-decision-card">'
            '<div class="cx-action-meta">'
            f"<div><span>Distress</span><strong>{red_count}</strong></div>"
            f"<div><span>Positive</span><strong>{green_count}</strong></div>"
            f"<div><span>Mixed</span><strong>{neutral_count}</strong></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<h4 class="cx-section-heading">Signal Auditing Results</h4>', unsafe_allow_html=True)
    selected_cluster = st.selectbox(
        "Select signal",
        sorted(audit_engine.cluster_texts.keys()),
        format_func=lambda x: (
            f"{signal_reference(x)} · "
            f"{insight_display_name(insight_by_cluster[x]) if x in insight_by_cluster else 'Unlabeled Signal'} "
            f"({audit_engine.cluster_zones[x]['zone_type']})"
        ),
        key="cluster_analysis_audit_cluster",
    )

    if selected_cluster in audit_engine.cluster_audit_reports:
        st.markdown(audit_engine.cluster_audit_reports[selected_cluster])

    audit_df = audit_engine.export_to_dataframe()
    render_table_section(
        "All Signals Audit Summary",
        "Signal-level sentiment, zone, vocabulary, and audit fields for deeper review.",
        audit_df,
        "No cluster audit rows are available yet.",
        accent_label="Audit table",
    )


def build_written_report(
    report_type,
    audience,
    clustering_engine,
    texts,
    audit_engine,
    causal_engine,
    action_insights,
    report_depth="Board-ready",
    feedback_source=None,
):
    generated_at = pd.Timestamp.now().strftime("%B %d, %Y")
    high_priority = [
        insight for insight in action_insights if insight.priority_label.startswith("HIGH")
    ]
    medium_priority = [
        insight for insight in action_insights if insight.priority_label.startswith("MEDIUM")
    ]
    impact_rows = build_operational_impact_rows(action_insights, causal_engine)
    systemic_rows = [row for row in impact_rows if row["impact_type"] == "Systemic Risk"]
    opportunity_rows = [
        insight
        for insight in action_insights
        if customer_lens_for_insight(insight) == "Opportunity"
    ]
    risk_rows = [
        insight
        for insight in action_insights
        if customer_lens_for_insight(insight) == "At Risk"
    ]
    top_insight = action_insights[0] if action_insights else None
    data_source_label = (
        feedback_source.get("label", "Current dataset")
        if feedback_source
        else "Current dataset"
    )
    action_windows = {}
    for row in impact_rows:
        action_windows.setdefault(row["action_window"], []).append(row)

    lines = [
        f"# PX-Intel {report_type}",
        "",
        f"Generated: {generated_at}",
        f"Audience: {audience}",
        f"Report depth: {report_depth}",
        f"Data source: {data_source_label}",
        "",
        "## Executive Readout",
        "",
    ]

    if top_insight:
        lines.extend(
            [
                (
                    f"PX-Intel reviewed {len(texts):,} feedback entries and identified "
                    f"{clustering_engine.optimal_n_clusters} experience signals. "
                    f"The strongest current signal is {insight_display_name(top_insight, include_reference=True)}, "
                    f"marked {top_insight.priority_label}."
                ),
                "",
                f"Recommended first move: {top_insight.recommended_action}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "PX-Intel does not have enough action insight data to generate a full decision report.",
                "",
            ]
        )

    lines.extend(
        [
            "## Current Signal Summary",
            "",
            f"- Feedback entries reviewed: {len(texts):,}",
            f"- Experience signals: {clustering_engine.optimal_n_clusters}",
            f"- High-priority signals: {len(high_priority)}",
            f"- Medium-priority signals: {len(medium_priority)}",
            f"- At-risk customer signals: {len(risk_rows)}",
            f"- Opportunity signals: {len(opportunity_rows)}",
            f"- Systemic operational risks: {len(systemic_rows)}",
            "",
        ]
    )

    lines.extend(["## What PX-Intel Learned From This Dataset", ""])
    if action_insights:
        negative_weighted = np.mean(
            [insight.metadata.get("negative_rate", 0) for insight in action_insights]
        )
        top_themes = ", ".join(
            [
                insight_display_name(insight)
                for insight in action_insights[: min(3, len(action_insights))]
            ]
        )
        lines.extend(
            [
                (
                    f"The current dataset is organized around {len(action_insights)} primary experience signals. "
                    f"Average visible negative concentration is {negative_weighted:.0%}, with the strongest signals concentrated in: {top_themes}."
                ),
                (
                    "PX-Intel is not treating the uploaded file as a generic survey export; the report is generated from the active clustering, sentiment, causal, and action-intelligence outputs."
                ),
                "",
            ]
        )
    else:
        lines.extend(["No learned signal profile is available yet.", ""])

    lines.extend(["## Priority Issues", ""])
    for index, insight in enumerate(action_insights[:5], start=1):
        lines.extend(
            [
                f"{index}. {insight_display_name(insight, include_reference=True)} ({insight.priority_label})",
                f"   - Insight: {insight.key_insight}",
                f"   - Root cause: {insight.root_cause}",
                f"   - Action: {insight.recommended_action}",
            ]
        )
    if not action_insights:
        lines.append("No priority issues are available.")
    lines.append("")

    lines.extend(["## Evidence Highlights", ""])
    for insight in action_insights[:5]:
        lines.extend(
            [
                f"- {insight_display_name(insight, include_reference=True)}",
                (
                    f"  - Signal strength: {insight.metadata.get('cluster_size', 0):,} feedback entries, "
                    f"{insight.metadata.get('negative_rate', 0):.0%} negative, priority score {insight.priority_score:.3f}"
                ),
                f"  - Evidence: {shorten_text(insight.example_feedback, 220)}",
                f"  - Manager response: {insight.recommended_action}",
            ]
        )
    if not action_insights:
        lines.append("No evidence highlights are available.")
    lines.append("")

    if report_type == "Operational Action Report":
        lines.extend(["## Operational Action Plan", ""])
        for row in impact_rows[:5]:
            cascade_text = (
                ", ".join([signal_reference(target) for target in row["cascade_targets"]])
                if row["cascade_targets"]
                else "No strong related cascade"
            )
            lines.extend(
                [
                    f"- {row['signal_name']} ({row['signal_id']})",
                    f"  - Impact type: {row['impact_type']}",
                    f"  - Action window: {row['action_window']}",
                    f"  - Related signals: {cascade_text}",
                    f"  - Recommended action: {row['recommended_action']}",
                ]
            )
        lines.append("")
    elif report_type == "Customer Intelligence Report":
        lines.extend(["## Customer Intelligence", ""])
        for insight in risk_rows[:4]:
            lines.extend(
                [
                    f"- Risk signal {insight_display_name(insight, include_reference=True)}",
                    f"  - Negative rate: {insight.metadata.get('negative_rate', 0):.0%}",
                    f"  - Evidence: {shorten_text(insight.example_feedback, 180)}",
                ]
            )
        for insight in opportunity_rows[:3]:
            lines.extend(
                [
                    f"- Opportunity signal {insight_display_name(insight, include_reference=True)}",
                    f"  - Strength to protect: {insight.key_insight}",
                ]
            )
        lines.append("")
    else:
        lines.extend(["## Leadership Recommendations", ""])
        for insight in action_insights[:3]:
            lines.append(
                f"- Prioritize {insight_display_name(insight, include_reference=True)} by taking this action: {insight.recommended_action}"
            )
        lines.append("")

    lines.extend(["## Action Playbook By Timing", ""])
    for window in ["Immediate", "Next 7 Days", "Monitor"]:
        window_rows = action_windows.get(window, [])[:4]
        if not window_rows:
            continue
        lines.append(f"### {window}")
        for row in window_rows:
            lines.extend(
                [
                    f"- {row['signal_name']} ({row['signal_id']}): {row['recommended_action']}",
                    (
                        f"  - Why now: {row['impact_type']}; impact score {row['impact_score']:.3f}; "
                        f"{row['negative_rate']:.0%} negative feedback."
                    ),
                ]
            )
        lines.append("")
    if not any(action_windows.values()):
        lines.append("No timing-based action plan is available.")
        lines.append("")

    top_cascade = None
    for row in impact_rows:
        if row["cascade_targets"]:
            top_cascade = row
            break
    lines.extend(["## Cause And Effect Note", ""])
    if top_cascade:
        related = ", ".join([signal_reference(target) for target in top_cascade["cascade_targets"][:3]])
        lines.append(
            f"Changing {top_cascade['signal_name']} ({top_cascade['signal_id']}) may also influence {related}. "
            "Use the Operational Impact simulator to test different improvement levels before selecting a response plan."
        )
    else:
        lines.append(
            "No strong cascade path is currently visible, so recommendations should focus on direct cluster-level improvements."
        )
    lines.append("")

    lines.extend(
        [
            "## Next Steps",
            "",
            "1. Assign an owner to the highest-priority signal.",
            "2. Use the Change Impact Simulator to test how fixing one issue may affect related signals.",
            "3. Export the action, audit, and impact CSVs for deeper operational follow-up.",
            "4. Review the report after new feedback data is loaded.",
            "5. If AI enhancement is enabled, use the AI report writer for a board-ready narrative over this same evidence base.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report_ai_context(
    report_type,
    audience,
    report_depth,
    feedback_source,
    texts,
    clustering_engine,
    causal_engine,
    action_insights,
):
    impact_rows = build_operational_impact_rows(action_insights, causal_engine)
    risk_count = sum(
        1 for insight in action_insights if customer_lens_for_insight(insight) == "At Risk"
    )
    opportunity_count = sum(
        1
        for insight in action_insights
        if customer_lens_for_insight(insight) == "Opportunity"
    )
    systemic_count = sum(
        1 for row in impact_rows if row["impact_type"] == "Systemic Risk"
    )
    return {
        "report_type": report_type,
        "audience": audience,
        "report_depth": report_depth,
        "active_data_source": feedback_source_label(feedback_source)
        if feedback_source
        else "Current dataset",
        "feedback_entries": len(texts),
        "experience_signals": clustering_engine.optimal_n_clusters,
        "signal_counts": {
            "at_risk": risk_count,
            "opportunity": opportunity_count,
            "systemic_operational_risk": systemic_count,
        },
        "generation_contract": [
            "Ground every claim in the active PX-Intel data.",
            "Convert signals into decisions, owners, actions, and success metrics.",
            "Do not mention unavailable source columns or make up customer demographics.",
            "Use professional signal names and PX-S identifiers.",
        ],
        "top_signals": [
            {
                "signal_id": signal_reference(insight.cluster_id),
                "name": insight_display_name(insight),
                "priority": insight.priority_label,
                "score": insight.priority_score,
                "negative_rate": insight.metadata.get("negative_rate", 0),
                "volume": insight.metadata.get("cluster_size", 0),
                "evidence": insight.example_feedback,
                "root_cause": insight.root_cause,
                "recommended_action": insight.recommended_action,
                "success_metric": (
                    "Reduce negative feedback concentration"
                    if insight.metadata.get("negative_rate", 0) >= 0.3
                    else "Protect positive feedback pattern"
                ),
            }
            for insight in action_insights[:8]
        ],
        "operational_impact": [
            {
                "signal_id": row["signal_id"],
                "name": row["signal_name"],
                "impact_type": row["impact_type"],
                "action_window": row["action_window"],
                "impact_score": row["impact_score"],
                "cascade_targets": [
                    signal_reference(target) for target in row["cascade_targets"]
                ],
                "recommended_action": row["recommended_action"],
            }
            for row in impact_rows[:8]
        ],
    }


def build_stakeholder_page_context(
    page_name,
    page_focus,
    action_insights,
    feedback_source=None,
    texts=None,
    clustering_engine=None,
    causal_engine=None,
    extra_context=None,
):
    """Build compact page context for AI and local stakeholder explanations."""
    action_insights = list(action_insights or [])
    feedback_entries = len(texts) if texts is not None else sum(
        int(insight.metadata.get("cluster_size", 0) or 0)
        for insight in action_insights
    )
    experience_signals = int(
        getattr(clustering_engine, "optimal_n_clusters", len(action_insights))
        or len(action_insights)
    )
    impact_rows = (
        build_operational_impact_rows(action_insights, causal_engine)
        if causal_engine is not None
        else []
    )
    high_count = sum(
        1 for insight in action_insights if insight.priority_label.startswith("HIGH")
    )
    medium_count = sum(
        1 for insight in action_insights if insight.priority_label.startswith("MEDIUM")
    )
    risk_count = sum(
        1 for insight in action_insights if customer_lens_for_insight(insight) == "At Risk"
    )
    opportunity_count = sum(
        1
        for insight in action_insights
        if customer_lens_for_insight(insight) == "Opportunity"
    )
    systemic_count = sum(
        1 for row in impact_rows if row["impact_type"] == "Systemic Risk"
    )

    return {
        "page_name": page_name,
        "page_focus": page_focus,
        "active_data_source": feedback_source_label(feedback_source)
        if feedback_source
        else "Current dataset",
        "feedback_entries": int(feedback_entries),
        "experience_signals": experience_signals,
        "priority_counts": {
            "high": int(high_count),
            "medium": int(medium_count),
            "total": int(len(action_insights)),
        },
        "customer_signal_counts": {
            "at_risk": int(risk_count),
            "opportunity": int(opportunity_count),
            "systemic_operational_risk": int(systemic_count),
        },
        "top_signals": [
            {
                "signal_id": signal_reference(insight.cluster_id),
                "name": insight_display_name(insight),
                "theme": insight.issue_theme.title(),
                "priority": insight.priority_label,
                "priority_score": float(insight.priority_score),
                "negative_rate": float(insight.metadata.get("negative_rate", 0) or 0),
                "feedback_volume": int(insight.metadata.get("cluster_size", 0) or 0),
                "keywords": [str(keyword) for keyword in insight.keywords[:6]],
                "insight": insight.key_insight,
                "root_cause": insight.root_cause,
                "recommended_action": insight.recommended_action,
                "evidence": shorten_text(insight.example_feedback, 260),
            }
            for insight in action_insights[:6]
        ],
        "operational_impact": [
            {
                "signal_id": row["signal_id"],
                "name": row["signal_name"],
                "impact_type": row["impact_type"],
                "action_window": row["action_window"],
                "impact_score": round(float(row["impact_score"]), 3),
                "cascade_targets": [
                    signal_reference(target) for target in row["cascade_targets"]
                ],
                "recommended_action": row["recommended_action"],
            }
            for row in impact_rows[:6]
        ],
        "extra_context": extra_context or {},
    }


def build_stakeholder_summary(context):
    top_signals = context.get("top_signals", [])
    if not top_signals:
        return (
            f"{context['page_name']} is ready to explain the current PX-Intel view, "
            "but the active dataset has not produced enough action signals yet."
        )

    top_signal = top_signals[0]
    return (
        f"{context['page_name']} summarizes {context['feedback_entries']:,} feedback "
        f"entries from {context['active_data_source']} into "
        f"{context['experience_signals']} experience signals. The leading signal is "
        f"{top_signal['name']} ({top_signal['signal_id']}), marked "
        f"{top_signal['priority']} with {top_signal['negative_rate']:.0%} negative "
        "concentration, so stakeholders can connect evidence to the next decision."
    )


def build_local_stakeholder_brief(context):
    top_signals = context.get("top_signals", [])
    impact_rows = context.get("operational_impact", [])
    counts = context.get("priority_counts", {})
    customer_counts = context.get("customer_signal_counts", {})

    lines = [
        "#### What this view shows",
        "",
        (
            f"This page explains **{context['page_focus']}** using the active PX-Intel "
            f"dataset: **{context['active_data_source']}**. The current analysis covers "
            f"**{context['feedback_entries']:,} feedback entries**, "
            f"**{context['experience_signals']} experience signals**, "
            f"**{counts.get('high', 0)} high-priority signals**, and "
            f"**{counts.get('medium', 0)} medium-priority signals**."
        ),
        "",
        "#### What stakeholders should notice",
        "",
    ]

    if top_signals:
        for signal in top_signals[:3]:
            lines.append(
                (
                    f"- **{signal['name']} ({signal['signal_id']})** is "
                    f"{signal['priority']} with {signal['feedback_volume']:,} related "
                    f"comments and {signal['negative_rate']:.0%} negative concentration. "
                    f"Recommended move: {signal['recommended_action']}"
                )
            )
    else:
        lines.append("- No stakeholder-ready signals are available for this page yet.")

    lines.extend(
        [
            "",
            "#### Decisions this supports",
            "",
            (
                f"- Prioritize service recovery for the "
                f"**{customer_counts.get('at_risk', 0)} at-risk** customer signals."
            ),
            (
                f"- Protect and scale the "
                f"**{customer_counts.get('opportunity', 0)} opportunity** signals that "
                "show stronger customer experience patterns."
            ),
        ]
    )

    if impact_rows:
        lead_impact = impact_rows[0]
        lines.append(
            (
                f"- Treat **{lead_impact['name']} ({lead_impact['signal_id']})** as the "
                f"lead operational watch item because it is categorized as "
                f"**{lead_impact['impact_type']}** with an **{lead_impact['action_window']}** action window."
            )
        )

    lines.extend(
        [
            "",
            "#### Recommended next move",
            "",
            (
                "Use this page to move from feedback discovery into a documented decision: "
                "confirm the evidence, assign an owner, choose the first action, and define "
                "the success metric PX-Intel should monitor after the change."
            ),
        ]
    )
    return "\n".join(lines)


def render_stakeholder_explanation(
    page_name,
    page_focus,
    action_insights,
    ai_config=None,
    feedback_source=None,
    texts=None,
    clustering_engine=None,
    causal_engine=None,
    extra_context=None,
    expanded=True,
):
    """Render a stakeholder-ready page explanation with optional OpenAI output."""
    context = build_stakeholder_page_context(
        page_name,
        page_focus,
        action_insights,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
        extra_context,
    )
    local_summary = build_stakeholder_summary(context)
    local_brief = build_local_stakeholder_brief(context)
    signature_payload = {
        "page_name": page_name,
        "model": ai_config.get("model") if ai_config else "local",
        "generation_strength": ai_config.get("generation_strength") if ai_config else "local",
        "context": context,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    brief_cache = st.session_state.setdefault("stakeholder_ai_briefs", {})
    error_cache = st.session_state.setdefault("stakeholder_ai_errors", {})
    if ai_config and ai_config.get("refresh_requested"):
        brief_cache.pop(signature, None)
        error_cache.pop(signature, None)

    ai_brief = brief_cache.get(signature)
    ai_error = error_cache.get(signature)
    if ai_config and ai_config.get("enabled") and not ai_brief and not ai_error:
        try:
            enhancer = OpenAIInsightEnhancer(
                api_key=ai_config.get("api_key", ""),
                model=ai_config.get("model", "gpt-4o-mini"),
                generation_strength=ai_config.get(
                    "generation_strength",
                    "Board-ready",
                ),
            )
            with st.spinner(f"Writing {page_name} stakeholder explanation..."):
                ai_brief = enhancer.explain_page(page_name, page_focus, context)
            brief_cache[signature] = ai_brief
        except AIEnhancementError as exc:
            ai_error = str(exc)
            error_cache[signature] = ai_error

    source_label = "AI-generated stakeholder brief" if ai_brief else "Local PX-Intel stakeholder brief"
    model_label = (
        ai_config.get("model", "OpenAI") if ai_config and ai_config.get("enabled") else "Local pipeline"
    )
    st.markdown(
        f"""
        <div class="cx-stakeholder-panel">
            <div class="cx-card-topline" style="margin-bottom:0.35rem;">
                <span class="cx-icon-block">AI</span>
                <span class="cx-chip">Stakeholder ready</span>
            </div>
            <div class="cx-eyebrow">{escape(source_label)}</div>
            <h4>{escape(page_name)} Explanation</h4>
            <p>{escape(local_summary)}</p>
            <div class="cx-stakeholder-meta">
                <span>{escape(context['active_data_source'])}</span>
                <span>{context['feedback_entries']:,} feedback entries</span>
                <span>{context['experience_signals']} experience signals</span>
                <span>{escape(model_label)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ai_error:
        st.caption(
            "AI stakeholder explanation is unavailable for this page, so PX-Intel is showing the local grounded readout."
        )
        st.caption(ai_error[:320])
    elif ai_config and not ai_config.get("enabled"):
        st.caption(
            "Enable AI enhancement in the sidebar to have OpenAI write this page explanation from the active PX-Intel evidence."
        )

    with st.expander("Stakeholder readout", expanded=expanded):
        st.markdown(ai_brief or local_brief)


def render_written_report_generator(
    clustering_engine,
    texts,
    audit_engine,
    causal_engine,
    action_insights,
    ai_config=None,
    feedback_source=None,
):
    st.markdown(
        '<h4 class="cx-section-heading">Written Report Builder</h4>',
        unsafe_allow_html=True,
    )
    report_col, audience_col, depth_col = st.columns([1, 1, 1])
    with report_col:
        report_type = st.selectbox(
            "Report type",
            [
                "Executive Summary Report",
                "Operational Action Report",
                "Customer Intelligence Report",
            ],
            key="written_report_type",
        )
    with audience_col:
        audience = st.selectbox(
            "Audience",
            ["Leadership", "Operations", "Customer Experience", "Analyst Review"],
            key="written_report_audience",
        )
    with depth_col:
        report_depth = st.selectbox(
            "Depth",
            ["Board-ready", "Operational detail", "Brief"],
            key="written_report_depth",
        )

    base_report = build_written_report(
        report_type,
        audience,
        clustering_engine,
        texts,
        audit_engine,
        causal_engine,
        action_insights,
        report_depth,
        feedback_source,
    )
    report_text = base_report
    report_signature = hashlib.sha256(
        f"{report_type}|{audience}|{report_depth}|{base_report}|{ai_config.get('model') if ai_config else ''}".encode(
            "utf-8"
        )
    ).hexdigest()
    report_context = build_report_ai_context(
        report_type,
        audience,
        report_depth,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
        action_insights,
    )

    if ai_config and ai_config.get("enabled"):
        ai_col, note_col = st.columns([0.9, 2.1])
        with ai_col:
            write_ai_report = st.button(
                "Generate full AI report",
                key="write_ai_enhanced_report",
                width="stretch",
            )
        with note_col:
            st.caption(
                f"Generates a {ai_config.get('generation_strength', 'Board-ready').lower()} report from the active dataset, PX-Intel metrics, evidence, root causes, and operational impact."
            )

        if write_ai_report:
            try:
                enhancer = OpenAIInsightEnhancer(
                    api_key=ai_config.get("api_key", ""),
                    model=ai_config.get("model", "gpt-4o-mini"),
                    generation_strength=ai_config.get(
                        "generation_strength",
                        report_depth,
                    ),
                )
                with st.spinner("Writing AI-enhanced report..."):
                    report_text = enhancer.write_report(
                        report_type,
                        audience,
                        base_report,
                        action_insights,
                        report_context,
                    )
                st.session_state.ai_written_report_signature = report_signature
                st.session_state.ai_written_report_text = report_text
                st.session_state.written_report_signature = report_signature
                st.session_state.written_report_output_text = report_text
            except AIEnhancementError as exc:
                st.warning("AI report writing is unavailable, so the local report draft is shown.")
                st.caption(str(exc)[:320])

        if st.session_state.get("ai_written_report_signature") == report_signature:
            report_text = st.session_state.get("ai_written_report_text", base_report)
    elif ai_config and not ai_config.get("enabled"):
        st.caption(
            "Enable AI enhancement in the sidebar to generate a stronger report from this evidence base."
        )

    if st.session_state.get("written_report_signature") != report_signature:
        st.session_state.written_report_signature = report_signature
        st.session_state.written_report_output_text = report_text

    st.text_area(
        "Report draft",
        height=420,
        key="written_report_output_text",
    )
    final_report_text = st.session_state.get("written_report_output_text", report_text)
    st.download_button(
        "Download written report",
        data=final_report_text,
        file_name="px_intel_written_report.md",
        mime="text/markdown",
        width="stretch",
    )


def render_reports_export(
    clustering_engine,
    texts,
    df,
    audit_engine,
    causal_engine,
    action_agent,
    action_insights,
    ai_config=None,
    feedback_source=None,
):
    render_page_header(
        "Reports & Export",
        "Download PX-Intel outputs and review the key data products generated by clustering, auditing, causal reasoning, and action intelligence.",
        "Exports",
    )
    render_stakeholder_explanation(
        "Reports & Export",
        "report drafting, evidence review, data export, and stakeholder documentation",
        action_insights,
        ai_config,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
    )

    export_df = clustering_engine.export_to_dataframe(texts, df)
    action_df = action_agent.build_dashboard_dataframe(action_insights)
    audit_df = audit_engine.export_to_dataframe()
    causal_df = causal_engine.export_to_dataframe()
    impact_df = operational_impact_dataframe(
        build_operational_impact_rows(action_insights, causal_engine)
    )

    download_cols = st.columns(5)
    downloads = [
        ("Enriched CSV", export_df, "cx_intel_enriched.csv"),
        ("Action CSV", action_df, "cx_intel_action_dashboard.csv"),
        ("Audit CSV", audit_df, "cx_intel_cluster_audit.csv"),
        ("Causal CSV", causal_df, "cx_intel_causal_reasoning.csv"),
        ("Impact CSV", impact_df, "px_intel_operational_impact.csv"),
    ]
    for col, (label, data_frame, file_name) in zip(download_cols, downloads):
        with col:
            st.download_button(
                label,
                data=data_frame.to_csv(index=False),
                file_name=file_name,
                mime="text/csv",
                width="stretch",
            )

    pickle_path = Path("clustering_results.pkl")
    if pickle_path.exists():
        st.download_button(
            "Download clustering pickle",
            data=pickle_path.read_bytes(),
            file_name="clustering_results.pkl",
            mime="application/octet-stream",
            width="stretch",
        )

    render_written_report_generator(
        clustering_engine,
        texts,
        audit_engine,
        causal_engine,
        action_insights,
        ai_config,
        feedback_source,
    )

    preview_col, summary_col = st.columns([1.45, 1])
    with preview_col:
        preview_df = export_df.head(12)
        render_table_header(
            "Data Preview",
            "A quick sample of the enriched PX-Intel output before export.",
            preview_df,
            accent_label="Preview table",
        )
        st.dataframe(preview_df, width="stretch", hide_index=True, height=360)

    with summary_col:
        st.markdown(
            '<h4 class="cx-section-heading">Clustering Summary</h4>',
            unsafe_allow_html=True,
        )
        st.info(clustering_engine.get_cluster_summary())

    with st.expander("M3 causal model details"):
        st.markdown(causal_engine.get_summary())
        cluster_options = sorted(causal_engine.cluster_lda_features.keys())
        if cluster_options:
            selected_cluster = st.selectbox(
                "Review cluster causal details",
                cluster_options,
                format_func=lambda cluster_id: f"Cluster {cluster_id}",
                key="reports_causal_detail_cluster",
            )
            st.markdown(causal_engine.get_cluster_summary(selected_cluster))
        render_table_header(
            "Causal Relationship Export",
            "Detailed M3 causal reasoning rows available for download and review.",
            causal_df,
            accent_label="Model table",
        )
        st.dataframe(causal_df, width="stretch", hide_index=True, height=360)


# ============================================================================
# Session State & Caching
# ============================================================================


@st.cache_data
def load_data():
    """Load and normalize feedback data."""
    loader = DataLoader("text_data.csv")
    df, stats = loader.load()
    return df, stats


@st.cache_data
def load_uploaded_data(upload_bytes, upload_name, text_column):
    """Load and normalize uploaded customer feedback."""
    raw_df = parse_uploaded_feedback(upload_bytes, upload_name)
    df, stats = normalize_feedback_dataframe(raw_df, text_column)
    return df, stats


@st.cache_resource
def run_clustering(source_kind, upload_bytes=None, upload_name=None, text_column="content"):
    """Run unsupervised clustering pipeline."""
    if source_kind == "upload":
        df, stats = load_uploaded_data(upload_bytes, upload_name, text_column)
    else:
        df, stats = load_data()

    if df.empty:
        raise ValueError("No usable feedback rows were found in the selected data source.")
    if len(df) < 8:
        raise ValueError(
            "PX-Intel needs at least 8 usable feedback rows to run clustering, audit, causal reasoning, and action intelligence."
        )

    texts = df["text_normalized"].tolist()

    engine = UnsupervisedClusteringEngine(random_state=42)
    engine.fit(texts, auto_select=True)

    return engine, df, texts, stats


@st.cache_resource
def run_audit(texts, cluster_assignments):
    """Run cluster auditing pipeline."""
    audit_engine = ClusterAuditEngine(model_device=-1)
    audit_engine.audit(texts, cluster_assignments, n_keywords=10)
    return audit_engine


@st.cache_resource
def run_causal_reasoning(
    cluster_lda_features, cluster_vocabularies, cluster_sentiments
):
    """Run causal reasoning pipeline."""
    causal_engine = CausalReasoningEngine(model_device=-1)
    causal_engine.reason(cluster_lda_features, cluster_vocabularies, cluster_sentiments)
    return causal_engine


def build_ai_signature(action_insights, model):
    """Build a stable signature so AI enhancement is not called on every rerun."""
    payload = {
        "model": model,
        "signals": [
            {
                "cluster_id": insight.cluster_id,
                "theme": insight.issue_theme,
                "sentiment": insight.sentiment_label,
                "priority": insight.priority_label,
                "score": insight.priority_score,
                "insight": insight.key_insight,
                "action": insight.recommended_action,
                "evidence": insight.example_feedback[:180],
            }
            for insight in action_insights
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collect_ai_updates(action_insights):
    """Capture enhanced fields so they can be reused during Streamlit reruns."""
    return {
        str(insight.cluster_id): {
            "professional_name": insight.professional_name,
            "key_insight": insight.key_insight,
            "root_cause": insight.root_cause,
            "recommended_action": insight.recommended_action,
            "ai_rationale": insight.ai_rationale,
            "ai_enhanced": insight.ai_enhanced,
        }
        for insight in action_insights
        if insight.ai_enhanced
    }


def apply_ai_updates(action_insights, updates):
    """Apply cached AI language to freshly rebuilt action insights."""
    for insight in action_insights:
        update = updates.get(str(insight.cluster_id), {})
        if not update:
            continue
        insight.professional_name = update.get("professional_name", insight.professional_name)
        insight.key_insight = update.get("key_insight", insight.key_insight)
        insight.root_cause = update.get("root_cause", insight.root_cause)
        insight.recommended_action = update.get(
            "recommended_action", insight.recommended_action
        )
        insight.ai_rationale = update.get("ai_rationale", insight.ai_rationale)
        insight.ai_enhanced = bool(update.get("ai_enhanced", insight.ai_enhanced))
    return action_insights


def maybe_enhance_action_insights(action_insights, ai_config):
    """Optionally improve manager-facing language with an OpenAI model."""
    metadata = {"enabled": False, "summary": ""}
    if not ai_config.get("enabled"):
        return action_insights, metadata

    signature = build_ai_signature(
        action_insights,
        f"{ai_config.get('model')}|{ai_config.get('generation_strength')}",
    )
    if ai_config.get("refresh_requested"):
        st.session_state.pop("ai_enhancement_signature", None)
        st.session_state.pop("ai_enhancement_updates", None)
        st.session_state.pop("ai_enhancement_metadata", None)

    if (
        st.session_state.get("ai_enhancement_signature") == signature
        and st.session_state.get("ai_enhancement_updates")
    ):
        action_insights = apply_ai_updates(
            action_insights,
            st.session_state.ai_enhancement_updates,
        )
        return action_insights, st.session_state.get(
            "ai_enhancement_metadata",
            {"enabled": True, "summary": "AI-enhanced language active."},
        )

    try:
        enhancer = OpenAIInsightEnhancer(
            api_key=ai_config.get("api_key", ""),
            model=ai_config.get("model", "gpt-4o-mini"),
            generation_strength=ai_config.get("generation_strength", "Board-ready"),
        )
        with st.spinner("Enhancing PX-Intel language with AI..."):
            action_insights, metadata = enhancer.enhance_action_insights(action_insights)
    except AIEnhancementError as exc:
        st.warning("AI enhancement is unavailable, so PX-Intel is using local intelligence.")
        st.caption(str(exc)[:320])
        return action_insights, {"enabled": False, "summary": "", "error": str(exc)}

    st.session_state.ai_enhancement_signature = signature
    st.session_state.ai_enhancement_updates = collect_ai_updates(action_insights)
    st.session_state.ai_enhancement_metadata = metadata
    return action_insights, metadata


def answer_with_optional_ai(
    question,
    action_agent,
    action_insights,
    ai_config,
    ai_context=None,
    response_mode="Decision brief",
):
    """Use the AI enhancer for chat answers when available, otherwise fallback."""
    if ai_config and ai_config.get("enabled"):
        try:
            enhancer = OpenAIInsightEnhancer(
                api_key=ai_config.get("api_key", ""),
                model=ai_config.get("model", "gpt-4o-mini"),
                generation_strength=ai_config.get(
                    "generation_strength",
                    "Board-ready",
                ),
            )
            return enhancer.answer_question(
                question,
                action_insights,
                context=ai_context,
                response_mode=response_mode,
            )
        except AIEnhancementError:
            pass
    return action_agent.answer_question(question, action_insights)


# ============================================================================
# Main Dashboard
# ============================================================================


def main():
    """Main dashboard flow."""
    active_section, feedback_source, ai_config = render_sidebar()
    render_slogan_strip()

    # Load data and run clustering
    try:
        with st.spinner("Loading feedback and discovering clusters..."):
            clustering_engine, df, texts, loader_stats = run_clustering(
                feedback_source["kind"],
                feedback_source.get("upload_bytes"),
                feedback_source.get("upload_name"),
                feedback_source.get("text_column"),
            )
            cluster_assignments = clustering_engine.cluster_assignments
    except Exception as exc:
        st.error(f"PX-Intel could not process the selected feedback data: {exc}")
        st.stop()

    st.caption(
        f"Active feedback source: {feedback_source['label']} | "
        f"{loader_stats.successful_rows:,} usable rows processed"
    )

    # Run audit
    with st.spinner("Auditing clusters (sentiment + vocabulary)..."):
        audit_engine = run_audit(texts, cluster_assignments)

    # Prepare data for causal reasoning
    cluster_lda_dict = {}
    for cid in np.unique(cluster_assignments):
        mask = cluster_assignments == cid
        cluster_lda_dict[cid] = clustering_engine.lda_features[mask]

    # Run causal reasoning
    with st.spinner("Analyzing causal relationships..."):
        causal_engine = run_causal_reasoning(
            cluster_lda_dict,
            audit_engine.cluster_vocabularies,
            audit_engine.cluster_sentiment_results,
        )

    # ====================================================================
    # Tab 1: Experience Map Visualization
    # ====================================================================

    action_agent = CXActionIntelligenceAgent()
    action_insights = action_agent.build_action_insights(
        audit_engine=audit_engine,
        causal_engine=causal_engine,
        clustering_engine=clustering_engine,
    )
    action_insights, ai_metadata = maybe_enhance_action_insights(
        action_insights,
        ai_config,
    )
    if ai_metadata.get("enabled") and ai_metadata.get("summary"):
        st.caption(
            f"AI-enhanced language active ({ai_metadata.get('model', ai_config.get('model'))}): "
            f"{ai_metadata['summary']}"
        )

    high_priority_count = sum(
        1 for insight in action_insights if insight.priority_label.startswith("HIGH")
    )
    medium_priority_count = sum(
        1
        for insight in action_insights
        if insight.priority_label.startswith("MEDIUM")
    )
    top_agent_insight = action_insights[0] if action_insights else None

    if active_section == "Customer Intelligence":
        render_customer_intelligence(
            audit_engine,
            action_insights,
            texts,
            ai_config,
            feedback_source,
            clustering_engine,
            causal_engine,
        )
        return

    if active_section == "Cause & Effect Graph":
        render_cause_effect_graph(
            audit_engine,
            causal_engine,
            action_insights,
            df,
            cluster_assignments,
            ai_config,
            feedback_source,
            texts,
            clustering_engine,
        )
        return

    if active_section == "Operational Impact":
        render_operational_impact(
            causal_engine,
            action_insights,
            ai_config,
            feedback_source,
            texts,
            clustering_engine,
        )
        return

    if active_section == "Cluster Analysis":
        render_cluster_analysis(
            audit_engine,
            clustering_engine,
            texts,
            cluster_assignments,
            action_agent,
            action_insights,
            top_agent_insight,
            ai_config,
            feedback_source,
            causal_engine,
        )
        return

    if active_section == "Reports & Export":
        render_reports_export(
            clustering_engine,
            texts,
            df,
            audit_engine,
            causal_engine,
            action_agent,
            action_insights,
            ai_config,
            feedback_source,
        )
        return

    render_overview(
        audit_engine,
        causal_engine,
        clustering_engine,
        texts,
        action_agent,
        action_insights,
        top_agent_insight,
        high_priority_count,
        medium_priority_count,
        ai_config,
        feedback_source,
    )
    return

    st.markdown('<h3 class="cx-section-heading">PX-Intel Overview</h3>', unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_kpi_card("Feedback Entries", f"{len(texts):,}", "Loaded from PX-Intel data", "#3b82f6")
    with kpi_col2:
        render_kpi_card("Experience Clusters", clustering_engine.optimal_n_clusters, "Auto-selected by M1", "#06b6d4")
    with kpi_col3:
        render_kpi_card("High Priority", high_priority_count, "M5 action score", "#ef4444")
    with kpi_col4:
        render_kpi_card("Medium Priority", medium_priority_count, "Watchlist clusters", "#d97706")

    if active_section != "Overview":
        st.info(
            f"{active_section} is still available in the dashboard tabs below while the sidebar workspace is being expanded."
        )

    tab_action, tab_landscape, tab_audit, tab_data = st.tabs(
        [
            "🤖 AI Agent (M5)",
            "🗺️ Experience Map",
            "📊 Cluster Audit",
            "📋 Data Export",
        ]
    )

    # ====================================================================
    # Tab 0: Customer Experience Action Dashboard (M5)
    # ====================================================================

    with tab_action:
        st.markdown(
            '<h3 class="cx-section-heading">AI Agent Decision Support</h3>',
            unsafe_allow_html=True,
        )
        st.caption(
            "M5 translates PX-Intel model outputs into a manager-facing decision workspace."
        )

        decision_col, outline_col = st.columns([1.35, 1])
        with decision_col:
            render_decision_summary_card(
                top_agent_insight,
                len(action_insights),
                high_priority_count,
                medium_priority_count,
            )

        with outline_col:
            st.markdown(
                f"""
                <div class="cx-decision-card">
                    <div class="cx-card-topline">
                        <span class="cx-icon-block">DS</span>
                        <span class="cx-chip">Outline</span>
                    </div>
                    <p class="cx-eyebrow" style="margin-bottom:0.35rem;">Decision support layout</p>
                    <h4 style="margin:0 0 0.5rem;">What this area should help answer</h4>
                    <p style="margin:0 0 0.5rem;">1. What needs attention first?</p>
                    <p style="margin:0 0 0.5rem;">2. What evidence supports the recommendation?</p>
                    <p style="margin:0 0 0.5rem;">3. What operational action should a manager take?</p>
                    <p style="margin:0;">4. Which clusters should be monitored next?</p>
                    <div class="cx-action-meta">
                        <div><span>Insights</span><strong>{len(action_insights)}</strong></div>
                        <div><span>High</span><strong>{high_priority_count}</strong></div>
                        <div><span>Medium</span><strong>{medium_priority_count}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "cx_agent_messages" not in st.session_state:
            st.session_state.cx_agent_messages = [
                {
                    "role": "assistant",
                    "content": action_agent.answer_question(
                        "summarize for leadership", action_insights
                    ),
                }
            ]

        st.markdown(
            '<h4 class="cx-section-heading">Ask PX-Intel Agent</h4>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Use the prompts to inspect the existing M5 action insights without changing the underlying pipeline."
        )

        prompt_cols = st.columns(4)
        suggested_prompts = [
            "What needs attention first?",
            "Show the evidence.",
            "Draft manager actions.",
            "Which clusters should we monitor?",
        ]
        for col, prompt in zip(prompt_cols, suggested_prompts):
            with col:
                if st.button(prompt, key=f"agent_prompt_{prompt}"):
                    st.session_state.cx_agent_messages.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.cx_agent_messages.append(
                        {
                            "role": "assistant",
                            "content": action_agent.answer_question(
                                prompt, action_insights
                            ),
                        }
                    )

        for message in st.session_state.cx_agent_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input(
            "Ask about priorities, cascades, actions, or a specific cluster..."
        )
        if user_question:
            st.session_state.cx_agent_messages.append(
                {"role": "user", "content": user_question}
            )
            answer = action_agent.answer_question(user_question, action_insights)
            st.session_state.cx_agent_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

        st.markdown(
            '<h3 class="cx-section-heading">Customer Experience Action Dashboard</h3>',
            unsafe_allow_html=True,
        )
        st.caption(
            "The action board below is a cleaner visual outline over the existing PX-Intel action dataframe."
        )

        filter_col, download_col = st.columns([2.3, 1])
        with filter_col:
            priority_filter = st.segmented_control(
                "Priority filter",
                options=["All", "HIGH 🔥", "MEDIUM ⚠", "LOW ✅"],
                default="All",
            )
        filtered_insights = (
            action_insights
            if priority_filter == "All"
            else [
                insight
                for insight in action_insights
                if insight.priority_label == priority_filter
            ]
        )

        if st.session_state.get("action_priority_filter") != priority_filter:
            st.session_state.action_priority_filter = priority_filter
            st.session_state.action_dashboard_page = 0

        action_df = action_agent.build_dashboard_dataframe(filtered_insights)
        with download_col:
            st.download_button(
                label="Download action data (CSV)",
                data=action_df.to_csv(index=False),
                file_name="cx_intel_action_dashboard.csv",
                mime="text/csv",
                help="Download the currently selected action data.",
                width="stretch",
            )

        page_size = 4
        total_pages = max(1, int(np.ceil(len(filtered_insights) / page_size)))
        current_page = min(
            st.session_state.get("action_dashboard_page", 0),
            total_pages - 1,
        )
        st.session_state.action_dashboard_page = current_page
        page_start = current_page * page_size
        page_end = page_start + page_size
        visible_insights = filtered_insights[page_start:page_end]

        if filtered_insights:
            nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
            with nav_left:
                if st.button(
                    "Previous",
                    key="action_dashboard_previous",
                    disabled=current_page == 0,
                    width="stretch",
                ):
                    st.session_state.action_dashboard_page = max(current_page - 1, 0)
                    st.rerun()
            with nav_mid:
                st.markdown(
                    f"""
                    <div class="cx-panel-soft" style="padding:0.65rem; border-radius:0.85rem; text-align:center;">
                        Showing actions {page_start + 1}-{min(page_end, len(filtered_insights))} of {len(filtered_insights)}
                        &nbsp;|&nbsp; Page {current_page + 1} of {total_pages}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with nav_right:
                if st.button(
                    "Next",
                    key="action_dashboard_next",
                    disabled=current_page >= total_pages - 1,
                    width="stretch",
                ):
                    st.session_state.action_dashboard_page = min(
                        current_page + 1,
                        total_pages - 1,
                    )
                    st.rerun()

            for start in range(0, len(visible_insights), 2):
                card_cols = st.columns(2)
                for offset, card_col in enumerate(card_cols):
                    insight_index = start + offset
                    if insight_index >= len(visible_insights):
                        continue
                    with card_col:
                        render_action_outline_card(
                            visible_insights[insight_index],
                            page_start + insight_index + 1,
                        )
        else:
            st.info("No action insights match this priority filter.")

        st.markdown(
            '<h4 class="cx-section-heading">Cluster Action Details</h4>',
            unsafe_allow_html=True,
        )
        for insight in filtered_insights:
            with st.expander(
                f"AI Agent Insight | Cluster {insight.cluster_id}: {insight.issue_theme.title()} "
                f"| {insight.priority_label}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cluster Size", insight.metadata["cluster_size"])
                with col2:
                    st.metric(
                        "Negative Rate",
                        f"{insight.metadata['negative_rate']:.0%}",
                    )
                with col3:
                    st.metric("Priority Score", f"{insight.priority_score:.3f}")

                st.markdown("**Keywords**")
                st.write(", ".join(insight.keywords) if insight.keywords else "None")

                st.markdown("**Root Cause**")
                st.write(insight.root_cause)

                st.markdown("**Example Feedback**")
                st.info(insight.example_feedback)

                st.markdown("**Recommended Action**")
                st.success(insight.recommended_action)

    with tab_landscape:
        insight_by_cluster = {item.cluster_id: item for item in action_insights}
        high_clusters = [
            item for item in action_insights if item.priority_label.startswith("HIGH")
        ]

        st.markdown("### Agent Summary")
        st.info(action_agent.answer_question("summarize for leadership", action_insights))

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Feedback Entries", len(texts))
        with summary_col2:
            st.metric("Experience Clusters", clustering_engine.optimal_n_clusters)
        with summary_col3:
            st.metric("High-Priority Clusters", len(high_clusters))
        with summary_col4:
            st.metric(
                "Top Priority",
                (
                    f"Cluster {top_agent_insight.cluster_id}"
                    if top_agent_insight is not None
                    else "None"
                ),
            )

        st.markdown("---")
        st.markdown("### Experience Map")
        st.caption(
            "This map groups similar feedback near each other. Use the lens selector "
            "to switch between action priority, sentiment health, and discovered themes."
        )

        with st.expander("How to read this map"):
            st.markdown(
                """
                - Each dot is one feedback entry.
                - Dots close together discuss similar experience patterns.
                - Color changes based on the selected lens.
                - Larger dots mark high-priority clusters when using the priority lens.
                - Hover over a dot to see the AI Agent's theme, priority, action, and feedback snippet.
                """
            )

        col1, col2 = st.columns([3, 1])

        with col1:
            landscape_lens = st.radio(
                "Map lens",
                ["Priority Heatmap", "Sentiment Health", "Theme Clusters"],
                horizontal=True,
            )

            # Create the 2D experience map from the t-SNE projection.
            fig = go.Figure()

            priority_color_map = {
                "HIGH 🔥": "#D64545",
                "MEDIUM ⚠": "#F2A93B",
                "LOW ✅": "#2FA66A",
            }
            zone_color_map = {
                "RED_ZONE": "#FF6B6B",
                "GREEN_ZONE": "#51CF66",
                "NEUTRAL_ZONE": "#FFD93D",
            }
            cluster_palette = px.colors.qualitative.Safe

            for cluster_id in np.unique(cluster_assignments):
                mask = cluster_assignments == cluster_id
                sentiment_dist = audit_engine.cluster_sentiment_results[cluster_id][
                    "sentiment_distribution"
                ]
                zone = audit_engine.cluster_zones[cluster_id]["zone_type"]
                insight = insight_by_cluster.get(int(cluster_id))

                if landscape_lens == "Priority Heatmap" and insight is not None:
                    color = priority_color_map.get(insight.priority_label, "#999999")
                    trace_name = (
                        f"{insight.priority_label} | C{cluster_id} | "
                        f"{insight.issue_theme.title()}"
                    )
                elif landscape_lens == "Sentiment Health":
                    color = zone_color_map.get(zone, "#999999")
                    readable_zone = zone.replace("_", " ").title()
                    trace_name = f"C{cluster_id} | {readable_zone}"
                else:
                    color = cluster_palette[int(cluster_id) % len(cluster_palette)]
                    theme = insight.issue_theme.title() if insight else "Theme"
                    trace_name = f"C{cluster_id} | {theme}"

                hover_text = []
                for text in np.array(texts)[mask]:
                    if insight is not None:
                        hover_text.append(
                            f"<b>Cluster {cluster_id}</b><br>"
                            f"Theme: {insight.issue_theme.title()}<br>"
                            f"Priority: {insight.priority_label} "
                            f"({insight.priority_score:.3f})<br>"
                            f"Negative Feedback: {sentiment_dist['NEGATIVE']:.1%}<br>"
                            f"Action: {insight.recommended_action}<br><br>"
                            f"Feedback: {str(text)[:180]}"
                        )
                    else:
                        hover_text.append(
                            f"<b>Cluster {cluster_id}</b><br>"
                            f"Negative: {sentiment_dist['NEGATIVE']:.1%}<br>"
                            f"Feedback: {str(text)[:180]}"
                        )

                fig.add_trace(
                    go.Scatter(
                        x=clustering_engine.tsne_projection[mask, 0],
                        y=clustering_engine.tsne_projection[mask, 1],
                        mode="markers",
                        name=trace_name,
                        marker=dict(
                            size=(
                                9
                                if insight
                                and insight.priority_label.startswith("HIGH")
                                else 6
                            ),
                            color=color,
                            opacity=0.76,
                            line=dict(width=0.5, color="white"),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title=f"Experience Map - {landscape_lens}",
                xaxis_title="Experience Similarity Dimension 1",
                yaxis_title="Experience Similarity Dimension 2",
                hovermode="closest",
                width=800,
                legend_title="Map Legend",
            )
            apply_plotly_soft_ui(fig, height=560)

            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("### Cluster Inspector")
            selected_map_cluster = st.selectbox(
                "Choose a cluster",
                [item.cluster_id for item in action_insights],
                format_func=lambda cid: (
                    f"Cluster {cid}: "
                    f"{insight_by_cluster[cid].issue_theme.title()} "
                    f"({insight_by_cluster[cid].priority_label})"
                ),
                key="landscape_agent_cluster",
            )
            selected_insight = insight_by_cluster[selected_map_cluster]

            st.markdown("**AI Agent Interpretation**")
            st.info(selected_insight.key_insight)

            st.markdown("**Recommended Action**")
            st.success(selected_insight.recommended_action)

            st.markdown("**Root Cause**")
            st.write(selected_insight.root_cause)

            if st.button(
                f"Ask Agent about Cluster {selected_map_cluster}",
                key="landscape_ask_agent",
            ):
                prompt = f"Tell me about cluster {selected_map_cluster}"
                st.session_state.cx_agent_messages.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.cx_agent_messages.append(
                    {
                        "role": "assistant",
                        "content": action_agent.answer_question(
                            prompt, action_insights
                        ),
                    }
                )
                st.toast(
                    "Added the cluster question to the AI Agent chat tab.",
                    icon="🤖",
                )

            st.markdown("### Sentiment Health")
            red_count = len(audit_engine.get_red_zones())
            green_count = len(audit_engine.get_green_zones())
            neutral_count = len(audit_engine.get_neutral_zones())

            st.markdown(f"""
            🔴 **Distress clusters**: {red_count}  
            🟢 **Positive clusters**: {green_count}  
            🟡 **Mixed clusters**: {neutral_count}
            """)

    # ====================================================================
    # Tab 2: Cluster Auditing
    # ====================================================================

    with tab_audit:
        st.markdown("### Cluster Auditing Results")

        # Cluster selector
        selected_cluster = st.selectbox(
            "Select Cluster",
            sorted(audit_engine.cluster_texts.keys()),
            format_func=lambda x: f"Cluster {x} ({audit_engine.cluster_zones[x]['zone_type']})",
        )

        # Display audit report
        if selected_cluster in audit_engine.cluster_audit_reports:
            st.markdown(audit_engine.cluster_audit_reports[selected_cluster])

        # Display audit DataFrame
        st.markdown("### All Clusters Summary")
        audit_df = audit_engine.export_to_dataframe()
        st.dataframe(audit_df, width="stretch")

    # ====================================================================
    # Tab 3: Data Export
    # ====================================================================

    with tab_data:
        st.markdown("### Export Enriched Data")

        # Combine all results into enriched dataframe
        export_df = clustering_engine.export_to_dataframe(texts, df)

        # Add audit results
        sentiment_data = pd.DataFrame(
            [
                {
                    "cluster_id": cid,
                    "sentiment_dominant": audit_engine.cluster_sentiment_results[cid][
                        "dominant_sentiment"
                    ],
                    "sentiment_density": audit_engine.cluster_sentiment_results[cid][
                        "sentiment_density"
                    ],
                    "zone_type": audit_engine.cluster_zones[cid]["zone_type"],
                }
                for cid in sorted(audit_engine.cluster_sentiment_results.keys())
            ]
        )

        # Create display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Download Options")
            st.download_button(
                label="📥 Download Enriched CSV",
                data=export_df.to_csv(index=False),
                file_name="cx_intel_enriched.csv",
                mime="text/csv",
            )

            st.download_button(
                label="📥 Download Clustering Results (Pickle)",
                data=(
                    open("clustering_results.pkl", "rb").read()
                    if Path("clustering_results.pkl").exists()
                    else b""
                ),
                file_name="clustering_results.pkl",
                mime="application/octet-stream",
            )

        with col2:
            st.markdown("#### Data Preview")
            st.dataframe(export_df.head(10), width="stretch")

        st.markdown("#### Clustering Summary")
        st.info(clustering_engine.get_cluster_summary())


if __name__ == "__main__":
    main()
