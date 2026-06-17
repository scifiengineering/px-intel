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
import re
import zipfile
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

        .cx-cause-lens-panel {
            margin: 0.75rem 0 1rem;
            padding: 1rem;
            border: 1px solid var(--cx-border);
            border-radius: var(--cx-radius);
            background:
                radial-gradient(circle at 96% 10%, rgba(59, 130, 246, 0.12), transparent 30%),
                var(--cx-panel);
            box-shadow: var(--cx-shadow);
        }

        .cx-cause-lens-panel h4 {
            margin: 0.3rem 0 0.35rem;
            color: var(--cx-ink);
            font-size: 1rem;
        }

        .cx-cause-lens-panel p {
            margin: 0;
            color: var(--cx-muted);
            font-size: 0.86rem;
            line-height: 1.48;
        }

        .cx-cause-path {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.85rem 0 1.1rem;
        }

        .cx-cause-path-card {
            min-height: 142px;
            padding: 0.95rem;
            border: 1px solid var(--cx-border);
            border-radius: 0.9rem;
            background: var(--cx-panel);
            box-shadow: 0 8px 22px rgba(20, 35, 70, 0.05);
        }

        .cx-cause-path-card h5 {
            margin: 0.42rem 0 0.32rem;
            color: var(--cx-ink);
            font-size: 0.96rem;
        }

        .cx-cause-path-card p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.43;
        }

        .cx-node-meaning {
            margin-top: 0.8rem;
            padding: 0.85rem;
            border-radius: 0.85rem;
            background: var(--cx-panel-soft);
            color: var(--cx-muted);
            font-size: 0.86rem;
            line-height: 1.45;
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

        .cx-relationship-list strong {
            color: var(--cx-ink);
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
            .cx-cause-path,
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


def reset_dataset_dependent_state():
    """Clear generated UI state that should follow the active feedback source."""
    for key in [
        "cx_agent_messages",
        "cx_agent_messages_signature",
        "written_report_signature",
        "written_report_output_text",
        "ai_written_report_signature",
        "ai_written_report_text",
        "written_report_generated_signature",
        "written_report_generated_text",
        "written_report_generated_source",
        "written_report_generated_at",
        "ai_enhancement_signature",
        "ai_enhancement_updates",
        "ai_enhancement_metadata",
    ]:
        st.session_state.pop(key, None)

    st.session_state.pop("stakeholder_ai_briefs", None)
    st.session_state.pop("stakeholder_ai_errors", None)


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


def infer_agent_response_mode(question):
    """Infer the best AI response format from the user's question."""
    text = str(question or "").lower()
    if any(
        phrase in text
        for phrase in (
            "nli",
            "natural language inference",
            "what does support mean",
            "weak support",
            "causal support",
            "issue signals supported",
            "supported issue signals",
            "entailment",
        )
    ):
        return "Terminology explanation"
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
            reset_dataset_dependent_state()
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
                st.session_state.feedback_source_selector = selected_source_id
                reset_dataset_dependent_state()
                st.rerun()

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
                    remember_uploaded_feedback_source(
                        feedback_source,
                        len(preview_df),
                    )
                    if (
                        st.session_state.active_feedback_source_id
                        != feedback_source["id"]
                    ):
                        st.session_state.active_feedback_source_id = feedback_source["id"]
                        st.session_state.feedback_source_selector = feedback_source["id"]
                        reset_dataset_dependent_state()
                        st.rerun()
                    st.success(
                        f"Active upload saved locally: {len(preview_df):,} rows ready for PX-Intel."
                    )
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
                    st.session_state.feedback_source_selector = "sample"
                    reset_dataset_dependent_state()
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
    total_signals = max(len(insights), 1)
    highest_priority = max(
        insights,
        key=lambda insight: insight.priority_score,
    )
    highest_negative = max(
        insights,
        key=lambda insight: insight.metadata.get("negative_rate", 0),
    )
    largest_segment = max(
        insights,
        key=lambda insight: insight.metadata.get("cluster_size", 0),
    )
    avg_negative = np.mean(
        [insight.metadata.get("negative_rate", 0) for insight in insights]
    )
    summary_cards = [
        (
            "At risk",
            signal_counts.get("At Risk", 0),
            f"{signal_counts.get('At Risk', 0) / total_signals:.0%} of visible signals",
            "#ef4444",
        ),
        (
            "Opportunities",
            signal_counts.get("Opportunity", 0),
            f"{signal_counts.get('Opportunity', 0) / total_signals:.0%} of visible signals",
            "#10b981",
        ),
        (
            "Avg negative",
            f"{avg_negative:.0%}",
            "Across visible segments",
            "#f59e0b",
        ),
        (
            "Largest segment",
            signal_reference(largest_segment.cluster_id),
            f"{largest_segment.metadata.get('cluster_size', 0):,} comments",
            "#3b82f6",
        ),
    ]
    summary_html = "".join(
        '<div class="cx-command-card">'
        '<div class="cx-card-topline">'
        f'<span class="cx-graph-type" style="color:{accent};">{escape(label)}</span>'
        f'<span class="cx-key-dot" style="background:{accent};"></span>'
        "</div>"
        f"<h4>{escape(str(value))}</h4>"
        f"<p>{escape(detail)}</p>"
        "</div>"
        for label, value, detail, accent in summary_cards
    )
    st.markdown(
        f'<div class="cx-command-grid" style="margin-bottom:1rem;">{summary_html}</div>',
        unsafe_allow_html=True,
    )

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
        signal_rows = sorted(
            insights,
            key=lambda insight: insight.priority_score,
            reverse=True,
        )
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[
                        insight.metadata.get("negative_rate", 0)
                        for insight in signal_rows
                    ],
                    y=[insight.priority_score for insight in signal_rows],
                    mode="markers+text",
                    text=[signal_reference(insight.cluster_id) for insight in signal_rows],
                    textposition="top center",
                    textfont=dict(color="#172033", size=12, family="Inter, Arial, sans-serif"),
                    marker=dict(
                        size=[
                            max(18, min(52, insight.metadata.get("cluster_size", 0) * 1.35))
                            for insight in signal_rows
                        ],
                        color=[
                            signal_colors[customer_lens_for_insight(insight)]
                            for insight in signal_rows
                        ],
                        opacity=0.82,
                        line=dict(color="#ffffff", width=2.4),
                    ),
                    cliponaxis=False,
                    customdata=[
                        [
                            insight_display_name(insight),
                            customer_lens_for_insight(insight),
                            insight.metadata.get("cluster_size", 0),
                            insight.recommended_action,
                        ]
                        for insight in signal_rows
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Lens: %{customdata[1]}<br>"
                        "Negative Rate: %{x:.0%}<br>"
                        "Priority Score: %{y:.3f}<br>"
                        "Feedback: %{customdata[2]:,} comments<br>"
                        "Action: %{customdata[3]}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title="Signal Priority Landscape",
            xaxis_title="Negative feedback rate",
            yaxis_title="Priority score",
            xaxis_tickformat=".0%",
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[0, max(0.85, highest_priority.priority_score + 0.1)]),
            annotations=[
                dict(
                    x=highest_negative.metadata.get("negative_rate", 0),
                    y=highest_negative.priority_score,
                    text="Highest friction",
                    showarrow=True,
                    arrowhead=2,
                    ax=38,
                    ay=-38,
                    font=dict(size=11),
                )
            ],
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


def shorten_text(value, limit=72):
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def filter_report_insights(
    action_insights,
    priority_scope,
    customer_lens,
    theme_filter,
    signal_limit,
):
    """Return the report-ready insight slice selected by the builder controls."""
    filtered = list(action_insights or [])

    if priority_scope == "High priority only":
        filtered = [
            insight
            for insight in filtered
            if insight.priority_label.startswith("HIGH")
        ]
    elif priority_scope == "High + medium priority":
        filtered = [
            insight
            for insight in filtered
            if insight.priority_label.startswith("HIGH")
            or insight.priority_label.startswith("MEDIUM")
        ]
    elif priority_scope == "Medium watchlist":
        filtered = [
            insight
            for insight in filtered
            if insight.priority_label.startswith("MEDIUM")
        ]
    elif priority_scope == "Low / opportunity signals":
        filtered = [
            insight
            for insight in filtered
            if insight.priority_label.startswith("LOW")
            or customer_lens_for_insight(insight) == "Opportunity"
        ]

    if customer_lens != "All customer lenses":
        filtered = [
            insight
            for insight in filtered
            if customer_lens_for_insight(insight) == customer_lens
        ]

    if theme_filter != "All themes":
        filtered = [
            insight
            for insight in filtered
            if insight.issue_theme.title() == theme_filter
        ]

    limit_map = {
        "Top 3 signals": 3,
        "Top 5 signals": 5,
        "Top 8 signals": 8,
    }
    limit = limit_map.get(signal_limit)
    if limit:
        filtered = filtered[:limit]

    return filtered


def build_report_filter_summary(report_filters):
    parts = [
        report_filters.get("priority_scope", "All priorities"),
        report_filters.get("customer_lens", "All customer lenses"),
        report_filters.get("theme_filter", "All themes"),
        report_filters.get("signal_limit", "All matching signals"),
    ]
    return " | ".join(str(part) for part in parts if part)


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


def apply_cause_effect_lens(insights, lens, cluster_filter):
    """Apply business-oriented graph lenses before density limiting."""
    if cluster_filter != "All" or lens == "Full relationship map":
        return insights
    if lens == "Recovery risks":
        return [
            insight
            for insight in insights
            if insight.priority_label.startswith(("HIGH", "MEDIUM"))
            or insight.metadata.get("negative_rate", 0) >= 0.35
        ]
    if lens == "Root-cause diagnosis":
        return [
            insight
            for insight in insights
            if insight.root_cause and insight.root_cause.lower() != "unknown"
        ]
    if lens == "Action impact":
        return [
            insight
            for insight in insights
            if insight.recommended_action and insight.priority_score >= 0.2
        ]
    if lens == "Strengths to scale":
        return [
            insight
            for insight in insights
            if customer_lens_for_insight(insight) == "Opportunity"
            or str(insight.sentiment_label).upper() == "POSITIVE"
            or insight.priority_label.startswith("LOW")
        ]
    return insights


def cause_effect_lens_description(lens):
    descriptions = {
        "Recovery risks": "Shows the issue paths most likely to require service recovery, escalation review, or immediate owner assignment.",
        "Root-cause diagnosis": "Keeps the map centered on likely drivers so teams can understand why the customer issue is appearing.",
        "Action impact": "Emphasizes the link between current issues and mitigation actions, useful for operating reviews.",
        "Strengths to scale": "Highlights positive or lower-risk patterns that can be protected, repeated, or used as service-quality references.",
        "Full relationship map": "Shows every matching signal after the filters, which is useful for audit review but can be denser.",
    }
    return descriptions.get(lens, descriptions["Recovery risks"])


def node_type_label(node_type):
    return node_type.replace("_", " ").title()


def graph_relation_label(relation):
    labels = {
        "feedback_mentions_theme": "Feedback mentions theme",
        "theme_contributes_to_issue": "Theme contributes to issue",
        "root_cause_drives_issue": "Root cause drives issue",
        "issue_impacts_segment": "Issue affects customer segment",
        "issue_has_sentiment": "Issue carries sentiment",
        "issue_escalates_to_risk": "Issue escalates to risk",
        "action_mitigates_issue": "Action mitigates issue",
    }
    return labels.get(relation, node_type_label(relation))


def graph_node_marker_label(node):
    label_map = {
        "feedback": "Feedback",
        "theme": "Theme",
        "root_cause": "Cause",
        "issue": "Issue",
        "customer_segment": "Segment",
        "sentiment": "Mood",
        "impact": "Risk",
        "action": "Action",
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
        node_size = 34 + min(insight.metadata.get("cluster_size", 0), 45) * 0.22
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
            size=34,
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
            size=node_size,
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
            size=34,
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
            size=36 + insight.priority_score * 28,
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
            size=node_size,
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
                x0=x_value - 0.56,
                x1=x_value + 0.56,
                y0=footer_y,
                y1=header_y + 0.28,
                line=dict(width=1, color="rgba(148, 163, 184, 0.18)"),
                fillcolor="#ffffff" if index % 2 == 0 else "#f3f8ff",
                opacity=0.92,
                layer="below",
            )
        issue_nodes = sorted(
            [node for node in nodes.values() if node["type"] == "issue"],
            key=lambda node: node["y"],
            reverse=True,
        )
        for issue_node in issue_nodes:
            fig.add_shape(
                type="line",
                x0=-0.32,
                x1=7.8,
                y0=issue_node["y"] - 0.82,
                y1=issue_node["y"] - 0.82,
                line=dict(color="rgba(148, 163, 184, 0.2)", width=1),
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
                line=dict(color=color, width=2.35),
                hoverinfo="skip",
                name=label,
                opacity=0.46,
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
                    size=[max(node["size"], 34) for node in group],
                    color=[node["color"] for node in group],
                    symbol=node_symbols.get(node_type, "circle"),
                    line=dict(width=2.4, color="#ffffff"),
                    opacity=0.96,
                ),
                selected=dict(marker=dict(opacity=1.0, size=46)),
                unselected=dict(marker=dict(opacity=0.45)),
                textfont=dict(size=9.5, color="#ffffff", family="Inter, Arial, sans-serif"),
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
                font=dict(size=12, color="#172033"),
                align="center",
                bgcolor="#ffffff",
                bordercolor="rgba(148, 163, 184, 0.22)",
                borderpad=4,
            )
        for issue_node in [node for node in nodes.values() if node["type"] == "issue"]:
            fig.add_annotation(
                x=issue_node["x"],
                y=issue_node["y"] + 0.66,
                text=(
                    f"<b>{signal_reference(issue_node['cluster_id'])}</b> "
                    f"{escape(shorten_text(issue_node['issue_theme'], 32))}"
                ),
                showarrow=False,
                font=dict(size=10, color="#172033"),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor="rgba(148, 163, 184, 0.22)",
                borderpad=3,
            )
        fig.update_yaxes(range=[footer_y, header_y + 0.35])

    fig.update_layout(
        title="Cause & Effect Flow Map",
        clickmode="event+select",
        dragmode="pan",
        showlegend=False,
    )
    dynamic_height = max(680, min(1080, 340 + len({node["cluster_id"] for node in nodes.values()}) * 112))
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


def graph_node_plain_meaning(node):
    node_type = node["type"]
    if node_type == "feedback":
        return "This is the representative customer evidence that begins the relationship path."
    if node_type == "theme":
        return "This is the repeated language pattern PX-Intel found across similar feedback."
    if node_type == "root_cause":
        return "This is the likely operating driver behind the issue. Treat it as evidence-weighted diagnosis, not final proof."
    if node_type == "issue":
        return "This is the decision item: the service problem or strength that should be managed."
    if node_type == "customer_segment":
        return "This is the customer group or experience segment affected by the issue."
    if node_type == "sentiment":
        return "This shows the emotional direction of the issue and how much negative feedback is concentrated in this path."
    if node_type == "impact":
        return "This shows how the issue converts into operational risk, priority, or monitoring need."
    if node_type == "action":
        return "This is the mitigation PX-Intel recommends for reducing the issue or protecting the strength."
    return "This node is part of the current PX-Intel relationship path."


def graph_node_decision_prompt(node):
    node_type = node["type"]
    if node_type in {"feedback", "theme"}:
        return "Use this to confirm whether the issue is showing up repeatedly in the customer language."
    if node_type == "root_cause":
        return "Validate this driver with the team closest to the workflow, then test the smallest operational fix."
    if node_type == "issue":
        return "Assign an owner, define the recovery metric, and decide whether this signal needs immediate action or monitoring."
    if node_type == "customer_segment":
        return "Review whether this affected segment needs a tailored recovery action or a broader service change."
    if node_type == "sentiment":
        return "Use the negative concentration to decide whether the issue needs escalation or routine monitoring."
    if node_type == "impact":
        return "Use this as the risk readout for priority setting and cross-team accountability."
    if node_type == "action":
        return "Turn this into an owner, timeline, and success measure before the next feedback cycle."
    return "Use this node to trace the relationship path before deciding the next action."


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
        f"<strong>{escape(graph_relation_label(edge['relation']))}</strong><br>"
        f"{escape(shorten_text(edge['description'], 132))}"
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
        f'<div class="cx-node-meaning"><strong style="color:var(--cx-ink);">What this means:</strong> {escape(graph_node_plain_meaning(node))}</div>'
        f'<div class="cx-quote">"{escape(shorten_text(node["evidence"], 260))}"</div>'
        '<div style="margin-top:0.8rem;">'
        '<strong style="color:var(--cx-ink);">Root cause:</strong> '
        f'<span>{escape(node["root_cause"])}</span>'
        "</div>"
        '<div style="margin-top:0.65rem;">'
        '<strong style="color:var(--cx-ink);">Recommended action:</strong> '
        f'<span>{escape(node["recommended_action"])}</span>'
        "</div>"
        '<div style="margin-top:0.65rem;">'
        '<strong style="color:var(--cx-ink);">Decision prompt:</strong> '
        f'<span>{escape(graph_node_decision_prompt(node))}</span>'
        "</div>"
        '<h5 style="margin:0.9rem 0 0.2rem; color:var(--cx-ink);">Connected relationships</h5>'
        f'<ul class="cx-relationship-list">{relationship_items}</ul>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_cause_effect_readout(insights, causal_engine=None, lens="Recovery risks"):
    if not insights:
        return

    ranked = sorted(insights, key=lambda item: item.priority_score, reverse=True)
    top = ranked[0]
    top_action = next((item for item in ranked if item.recommended_action), top)
    top_root = next((item for item in ranked if item.root_cause), top)
    cascade_predictions = getattr(causal_engine, "cascade_predictions", {}) if causal_engine else {}
    top_cascades = cascade_predictions.get(top.cluster_id, [])
    cascade_text = (
        ", ".join(
            signal_reference(int(item.get("target_cluster", -1)))
            for item in top_cascades[:3]
            if int(item.get("target_cluster", -1)) >= 0
        )
        or "No strong cascade path in the current view"
    )
    affected_volume = sum(
        int(insight.metadata.get("cluster_size", 0) or 0)
        for insight in ranked
    )
    cards = [
        (
            "Priority path",
            insight_display_name(top, include_reference=True),
            (
                f"Under the {lens.lower()} lens: {top.priority_label}; "
                f"{top.metadata.get('negative_rate', 0):.0%} negative across "
                f"{top.metadata.get('cluster_size', 0):,} comments."
            ),
        ),
        (
            "Likely driver",
            top_root.root_cause,
            "Treat this as evidence-weighted, not absolute causality.",
        ),
        (
            "Downstream effect",
            cascade_text,
            f"{affected_volume:,} feedback comments are represented by the visible relationship paths.",
        ),
        (
            "Recommended move",
            insight_display_name(top_action, include_reference=True),
            top_action.recommended_action,
        ),
    ]
    html = "".join(
        '<div class="cx-cause-path-card">'
        f'<span class="cx-graph-type">{escape(label)}</span>'
        f"<h5>{escape(shorten_text(title, 72))}</h5>"
        f"<p>{escape(shorten_text(body, 150))}</p>"
        "</div>"
        for label, title, body in cards
    )
    st.markdown(
        '<div class="cx-cause-path">'
        f"{html}"
        "</div>",
        unsafe_allow_html=True,
    )


def graph_category_nodes(nodes, category):
    category_types = {
        "What should we fix?": {"issue"},
        "Why is it happening?": {"root_cause"},
        "Who is affected?": {"customer_segment"},
        "What risk does it create?": {"impact", "sentiment"},
        "What action should we take?": {"action"},
        "What evidence supports it?": {"feedback", "theme"},
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
        '<h4 class="cx-section-heading">Inspect Relationship Path</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Choose the question you want the graph to answer, then select the signal path to inspect."
    )
    category_options = [
        "What should we fix?",
        "Why is it happening?",
        "Who is affected?",
        "What risk does it create?",
        "What action should we take?",
        "What evidence supports it?",
    ]
    if st.session_state.get("cause_effect_insight_category") not in category_options:
        st.session_state.cause_effect_insight_category = "What should we fix?"
    category = st.segmented_control(
        "Business question",
        options=category_options,
        default="What should we fix?",
        key="cause_effect_insight_category",
    )
    category = category or "What should we fix?"
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

    lens_options = [
        "Recovery risks",
        "Root-cause diagnosis",
        "Action impact",
        "Strengths to scale",
        "Full relationship map",
    ]
    if st.session_state.get("cause_effect_relationship_lens") not in lens_options:
        st.session_state.cause_effect_relationship_lens = "Recovery risks"
    relationship_lens = st.segmented_control(
        "Relationship lens",
        options=lens_options,
        default="Recovery risks",
        key="cause_effect_relationship_lens",
    )
    relationship_lens = relationship_lens or "Recovery risks"
    st.markdown(
        f"""
        <div class="cx-cause-lens-panel">
            <span class="cx-graph-type">Current lens</span>
            <h4>{escape(relationship_lens)}</h4>
            <p>{escape(cause_effect_lens_description(relationship_lens))}</p>
        </div>
        """,
        unsafe_allow_html=True,
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

    custom_range = None
    with st.expander("Refine relationship map", expanded=False):
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
    filtered_insights = apply_cause_effect_lens(
        filtered_insights,
        relationship_lens,
        cluster_filter,
    )

    density_col, density_note_col = st.columns([1, 2])
    with density_col:
        graph_density = st.segmented_control(
            "Graph density",
            options=["Top 3", "Top 5", "All Matching"],
            default="Top 5",
            key="cause_effect_graph_density",
        )
    graph_density = graph_density or "Top 5"
    filtered_insights = apply_graph_density(
        filtered_insights,
        graph_density,
        cluster_filter,
    )
    with density_note_col:
        st.caption(
            "The graph defaults to the top priority signals so the flow stays readable. Use All Matching when you need the complete relationship map."
        )

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
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=12),
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
    top_row = rows[0]
    most_connected = max(
        rows,
        key=lambda row: (
            row["cascade_count"],
            row["impact_score"],
        ),
    )
    matrix_cards = [
        (
            "Fix-first zone",
            f"{top_row['signal_id']}",
            f"{top_row['signal_name']} has impact {top_row['impact_score']:.2f}",
            "#dc2626",
        ),
        (
            "Most connected",
            f"{most_connected['signal_id']}",
            f"{most_connected['cascade_count']} cascade target(s)",
            "#2563eb",
        ),
        (
            "Bubble size",
            "Volume + impact",
            "Larger bubbles indicate bigger operational exposure",
            "#7c3aed",
        ),
    ]
    matrix_html = "".join(
        '<div class="cx-command-card">'
        '<div class="cx-card-topline">'
        f'<span class="cx-graph-type" style="color:{accent};">{escape(label)}</span>'
        f'<span class="cx-key-dot" style="background:{accent};"></span>'
        "</div>"
        f"<h4>{escape(str(value))}</h4>"
        f"<p>{escape(detail)}</p>"
        "</div>"
        for label, value, detail, accent in matrix_cards
    )
    st.markdown(
        f'<div class="cx-command-grid" style="margin-bottom:1rem;">{matrix_html}</div>',
        unsafe_allow_html=True,
    )

    scatter_fig = go.Figure()
    cascade_threshold = max(1, int(np.median([row["cascade_count"] for row in rows])))
    max_cascade = max([row["cascade_count"] for row in rows] + [cascade_threshold])
    y_max = max_cascade + 0.85
    quadrant_shapes = [
        (0.4, 1.05, cascade_threshold, y_max, "#fee2e2"),
        (0.4, 1.05, -0.35, cascade_threshold, "#ffedd5"),
        (0, 0.4, cascade_threshold, y_max, "#dbeafe"),
        (0, 0.4, -0.35, cascade_threshold, "#dcfce7"),
    ]
    for x0, x1, y0, y1, fill in quadrant_shapes:
        scatter_fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=fill,
            opacity=0.38,
            line_width=0,
            layer="below",
        )
    for label, x, y, accent in [
        ("Fix first", 0.76, min(y_max - 0.18, cascade_threshold + 0.55), "#dc2626"),
        ("Service recovery", 0.72, max(-0.08, cascade_threshold - 0.45), "#d97706"),
        ("Monitor closely", 0.18, min(y_max - 0.18, cascade_threshold + 0.55), "#2563eb"),
        ("Lower priority", 0.18, max(-0.08, cascade_threshold - 0.45), "#059669"),
    ]:
        scatter_fig.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(71,85,105,0.18)",
            borderwidth=1,
            font=dict(color=accent, size=12),
        )
    impact_colors = {
        "Systemic Risk": "#dc2626",
        "Service Recovery": "#d97706",
        "Protect Strength": "#059669",
        "Monitor": "#64748b",
    }
    for impact_type in sorted({row["impact_type"] for row in rows}):
        group_rows = [row for row in rows if row["impact_type"] == impact_type]
        scatter_fig.add_trace(
            go.Scatter(
                x=[row["negative_rate"] for row in group_rows],
                y=[row["cascade_count"] for row in group_rows],
                mode="markers+text",
                name=impact_type,
                text=[row["signal_id"] for row in group_rows],
                textposition="top center",
                textfont=dict(color="#172033", size=12, family="Inter, Arial, sans-serif"),
                marker=dict(
                    size=[
                        max(26, min(58, 22 + row["impact_score"] * 38))
                        for row in group_rows
                    ],
                    color=impact_colors.get(impact_type, "#64748b"),
                    opacity=0.86,
                    line=dict(color="#ffffff", width=2.4),
                ),
                cliponaxis=False,
                customdata=[
                    [
                        row["signal_name"],
                        row["priority"],
                        row["impact_score"],
                        row["recommended_action"],
                    ]
                    for row in group_rows
                ],
                hovertemplate=(
                    "<b>%{text}: %{customdata[0]}</b><br>"
                    "Priority: %{customdata[1]}<br>"
                    "Impact score: %{customdata[2]:.3f}<br>"
                    "Negative rate: %{x:.0%}<br>"
                    "Cascade targets: %{y}<br>"
                    "Action: %{customdata[3]}<extra></extra>"
                ),
            )
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
    scatter_fig.update_xaxes(range=[-0.06, 1.05])
    scatter_fig.update_yaxes(range=[-0.35, y_max], dtick=1)
    apply_operational_chart_theme(scatter_fig, height=520, showlegend=True)
    st.plotly_chart(scatter_fig, width="stretch")

    top_rows = rows[:8]
    max_impact = max([row["impact_score"] for row in top_rows] + [0.1])
    fig = go.Figure(
        data=[
            go.Bar(
                x=[row["impact_score"] for row in top_rows],
                y=[
                    f"{row['signal_id']}: {shorten_text(row['signal_name'], 34)}"
                    for row in top_rows
                ],
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
                text=[
                    f"{row['impact_score']:.2f} · {row['action_window']}"
                    for row in top_rows
                ],
                textposition="outside",
                textfont=dict(color="#172033", size=12),
                cliponaxis=False,
                customdata=[
                    [
                        row["impact_type"],
                        row["negative_rate"],
                        row["cascade_count"],
                        row["recommended_action"],
                    ]
                    for row in top_rows
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Impact score: %{x:.3f}<br>"
                    "Impact type: %{customdata[0]}<br>"
                    "Negative rate: %{customdata[1]:.0%}<br>"
                    "Cascade targets: %{customdata[2]}<br>"
                    "Action: %{customdata[3]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Impact Score Ranking",
        xaxis_title="Impact score",
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(range=[0, max_impact + 0.2])
    apply_operational_chart_theme(fig, height=360)
    fig.update_layout(margin=dict(l=210, r=140, t=58, b=58))
    st.plotly_chart(fig, width="stretch")


def render_operational_action_plan(rows):
    st.markdown(
        '<h4 class="cx-section-heading">Action Plan Recommendations</h4>',
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
            text=chart_df["Current Negative Rate"].map(lambda value: f"{value:.0%}"),
            textposition="outside",
            textfont=dict(color="#172033", size=12),
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Current negative: %{y:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=chart_df["Signal"],
            y=chart_df["Projected Negative Rate"],
            name="Projected after change",
            marker_color="#10b981",
            text=chart_df["Projected Negative Rate"].map(lambda value: f"{value:.0%}"),
            textposition="outside",
            textfont=dict(color="#172033", size=12),
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Projected negative: %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Before vs After Scenario",
        yaxis_title="Negative feedback rate",
        xaxis_title="Affected signal",
        yaxis_tickformat=".0%",
        barmode="group",
        legend_title="Scenario",
    )
    fig.update_yaxes(range=[0, 1.12])
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

    render_operational_action_plan(visible_rows)
    render_operational_priority_queue(visible_rows)
    render_operational_impact_charts(visible_rows)
    render_change_impact_simulator(action_insights, causal_engine)


def apply_plotly_soft_ui(fig, height=520, showlegend=True):
    chart_ink = "#172033"
    chart_muted = "#475569"
    chart_grid = "rgba(71,85,105,0.16)"
    chart_surface = "#ffffff"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=chart_surface,
        plot_bgcolor=chart_surface,
        font=dict(family="Inter, Arial, sans-serif", color=chart_ink, size=12),
        title=dict(font=dict(color=chart_ink, size=16)),
        height=height,
        margin=dict(l=64, r=32, t=58, b=58),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(94,114,228,0.18)",
            font=dict(color=chart_ink),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(94,114,228,0.16)",
            borderwidth=1,
            font=dict(color=chart_ink),
        ),
    )
    fig.update_xaxes(
        gridcolor=chart_grid,
        zerolinecolor=chart_grid,
        linecolor="rgba(71,85,105,0.22)",
        tickfont=dict(color=chart_muted),
        title_font=dict(color=chart_ink),
    )
    fig.update_yaxes(
        gridcolor=chart_grid,
        zerolinecolor=chart_grid,
        linecolor="rgba(71,85,105,0.22)",
        tickfont=dict(color=chart_muted),
        title_font=dict(color=chart_ink),
    )
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

    chat_signal_signature = hashlib.sha256(
        json.dumps(
            [
                {
                    "id": insight.cluster_id,
                    "name": insight_display_name(insight),
                    "priority": insight.priority_label,
                    "score": round(float(insight.priority_score), 4),
                    "evidence": shorten_text(insight.example_feedback, 120),
                }
                for insight in action_insights
            ],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    chat_signature = (
        "auto_intent_v2_glossary",
        "ai" if ai_config and ai_config.get("enabled") else "local",
        ai_config.get("model") if ai_config else "local",
        ai_config.get("generation_strength") if ai_config else "local",
        ai_context.get("active_data_source") if ai_context else "Current dataset",
        chat_signal_signature,
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

    with st.form("cx_agent_question_form", clear_on_submit=True):
        question_col, ask_col = st.columns([3.2, 0.8])
        with question_col:
            user_question = st.text_area(
                "Ask PX-Intel",
                placeholder="Ask about priorities, root causes, current actions, evidence, risks, strengths, or a specific signal...",
                key="cx_agent_question_input",
                label_visibility="collapsed",
                height=84,
            )
        with ask_col:
            ask_submitted = st.form_submit_button(
                "Ask PX-Intel",
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


def render_experience_map_readout(action_insights, landscape_lens):
    """Render a plain-language guide for the signal map."""
    if not action_insights:
        st.info("No experience signals are available for the current dataset.")
        return

    top_priority = max(action_insights, key=lambda insight: insight.priority_score)
    highest_negative = max(
        action_insights,
        key=lambda insight: insight.metadata.get("negative_rate", 0),
    )
    largest_segment = max(
        action_insights,
        key=lambda insight: insight.metadata.get("cluster_size", 0),
    )
    high_count = sum(
        1 for insight in action_insights if insight.priority_label.startswith("HIGH")
    )

    lens_notes = {
        "Priority Heatmap": "Read this as urgency: farther right means more negative feedback, higher up means higher priority.",
        "Sentiment Health": "Read this as balance: farther right is more negative, higher up is more positive.",
        "Theme Clusters": "Read this as scale and theme: farther right means a larger share of feedback, higher up means more action pressure.",
    }
    cards = [
        (
            "How to read",
            landscape_lens,
            lens_notes.get(landscape_lens, "Each bubble is one PX-Intel signal."),
            "#3b82f6",
        ),
        (
            "Top priority",
            signal_reference(top_priority.cluster_id),
            f"{insight_display_name(top_priority)} · score {top_priority.priority_score:.3f}",
            "#ef4444",
        ),
        (
            "Highest friction",
            signal_reference(highest_negative.cluster_id),
            f"{highest_negative.metadata.get('negative_rate', 0):.0%} negative feedback",
            "#d97706",
        ),
        (
            "Signals to act on",
            high_count,
            "High-priority items visible in this dataset",
            "#10b981",
        ),
    ]
    card_html = "".join(
        '<div class="cx-command-card">'
        '<div class="cx-card-topline">'
        f'<span class="cx-graph-type" style="color:{accent};">{escape(label)}</span>'
        f'<span class="cx-key-dot" style="background:{accent};"></span>'
        "</div>"
        f"<h4>{escape(str(value))}</h4>"
        f"<p>{escape(detail)}</p>"
        "</div>"
        for label, value, detail, accent in cards
    )
    st.markdown(
        f'<div class="cx-command-grid" style="margin-bottom:1rem;">{card_html}</div>',
        unsafe_allow_html=True,
    )


def build_experience_map_figure(
    landscape_lens,
    clustering_engine,
    audit_engine,
    action_insights,
    cluster_assignments,
    texts,
):
    fig = go.Figure()
    if not action_insights:
        apply_plotly_soft_ui(fig, height=520)
        return fig

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
    theme_palette = px.colors.qualitative.Safe
    theme_colors = {
        theme: theme_palette[index % len(theme_palette)]
        for index, theme in enumerate(
            sorted({insight.issue_theme.title() for insight in action_insights})
        )
    }

    rows = []
    for insight in action_insights:
        sentiment_dist = audit_engine.cluster_sentiment_results.get(
            insight.cluster_id,
            {},
        ).get("sentiment_distribution", {})
        zone = audit_engine.cluster_zones.get(
            insight.cluster_id,
            {},
        ).get("zone_type", "NEUTRAL_ZONE")
        negative_rate = insight.metadata.get(
            "negative_rate",
            sentiment_dist.get("NEGATIVE", 0),
        )
        positive_rate = sentiment_dist.get("POSITIVE", max(0, 1 - negative_rate))
        cluster_size = insight.metadata.get("cluster_size", 0)
        cluster_share = insight.metadata.get("cluster_share", 0)

        if landscape_lens == "Sentiment Health":
            x_value = negative_rate
            y_value = positive_rate
            group = zone.replace("_", " ").title()
            color = zone_color_map.get(zone, "#64748b")
        elif landscape_lens == "Theme Clusters":
            x_value = cluster_share
            y_value = insight.priority_score
            group = insight.issue_theme.title()
            color = theme_colors.get(group, "#64748b")
        else:
            x_value = negative_rate
            y_value = insight.priority_score
            group = insight.priority_label
            color = priority_color_map.get(insight.priority_label, "#64748b")

        rows.append(
            {
                "cluster_id": insight.cluster_id,
                "signal_id": signal_reference(insight.cluster_id),
                "name": insight_display_name(insight),
                "theme": insight.issue_theme.title(),
                "priority": insight.priority_label,
                "score": insight.priority_score,
                "negative_rate": negative_rate,
                "positive_rate": positive_rate,
                "cluster_size": cluster_size,
                "cluster_share": cluster_share,
                "root_cause": insight.root_cause,
                "recommended_action": insight.recommended_action,
                "example_feedback": insight.example_feedback,
                "x": x_value,
                "y": y_value,
                "group": group,
                "color": color,
            }
        )

    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        fig.add_trace(
            go.Scatter(
                x=[row["x"] for row in group_rows],
                y=[row["y"] for row in group_rows],
                mode="markers+text",
                name=group,
                text=[row["signal_id"] for row in group_rows],
                textposition="top center",
                textfont=dict(color="#172033", size=12, family="Inter, Arial, sans-serif"),
                marker=dict(
                    size=[
                        max(22, min(58, 18 + row["cluster_size"] * 0.9))
                        for row in group_rows
                    ],
                    color=[row["color"] for row in group_rows],
                    opacity=0.84,
                    line=dict(width=2.4, color="#ffffff"),
                ),
                cliponaxis=False,
                customdata=[
                    [
                        row["name"],
                        row["theme"],
                        row["priority"],
                        row["cluster_size"],
                        row["negative_rate"],
                        row["recommended_action"],
                        shorten_text(row["example_feedback"], 140),
                    ]
                    for row in group_rows
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Theme: %{customdata[1]}<br>"
                    "Priority: %{customdata[2]}<br>"
                    "Feedback: %{customdata[3]:,} comments<br>"
                    "Negative Rate: %{customdata[4]:.0%}<br>"
                    "Action: %{customdata[5]}<br>"
                    "Example: %{customdata[6]}<extra></extra>"
                ),
            )
        )

    if landscape_lens == "Sentiment Health":
        title = "Experience Map - Sentiment Balance"
        x_title = "Negative feedback rate"
        y_title = "Positive feedback rate"
        x_tickformat = ".0%"
        y_tickformat = ".0%"
        x_range = [-0.05, 1.05]
        y_range = [0, 1.05]
        shapes = [
            dict(type="line", x0=0.4, x1=0.4, y0=0, y1=1, line=dict(color="rgba(239,68,68,0.25)", dash="dash")),
            dict(type="line", x0=0, x1=1, y0=0.4, y1=0.4, line=dict(color="rgba(16,185,129,0.22)", dash="dash")),
        ]
        annotations = []
    elif landscape_lens == "Theme Clusters":
        title = "Experience Map - Theme Scale"
        x_title = "Share of feedback"
        y_title = "Priority score"
        x_tickformat = ".0%"
        y_tickformat = None
        x_range = [-0.02, max(0.42, max(row["x"] for row in rows) + 0.08)]
        y_range = [0, max(0.85, max(row["y"] for row in rows) + 0.1)]
        shapes = []
        annotations = []
    else:
        title = "Experience Map - Decision Priority"
        x_title = "Negative feedback rate"
        y_title = "Priority score"
        x_tickformat = ".0%"
        y_tickformat = None
        x_range = [-0.05, 1.05]
        y_range = [0, max(0.85, max(row["y"] for row in rows) + 0.1)]
        shapes = [
            dict(type="line", x0=0.4, x1=0.4, y0=0, y1=1, line=dict(color="rgba(239,68,68,0.24)", dash="dash")),
            dict(type="line", x0=0, x1=1, y0=0.55, y1=0.55, line=dict(color="rgba(239,68,68,0.24)", dash="dash")),
        ]
        annotations = [
            dict(
                x=0.82,
                y=max(0.68, min(y_range[1] - 0.05, 0.78)),
                text="Act first",
                showarrow=False,
                font=dict(size=12, color="#dc2626"),
            ),
            dict(
                x=0.18,
                y=0.18,
                text="Monitor / protect",
                showarrow=False,
                font=dict(size=12, color="#059669"),
            ),
        ]

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="closest",
        legend_title="Signal group",
        shapes=shapes,
        annotations=annotations,
    )
    fig.update_xaxes(range=x_range, tickformat=x_tickformat)
    if y_tickformat:
        fig.update_yaxes(range=y_range, tickformat=y_tickformat)
    else:
        fig.update_yaxes(range=y_range)
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
        render_experience_map_readout(action_insights, landscape_lens)
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
                '<span class="cx-graph-type">Selected signal</span>'
                f"<h4>{escape(insight_display_name(selected_insight, include_reference=True))}</h4>"
                f"<p>{escape(selected_insight.key_insight)}</p>"
                '<div class="cx-action-meta" style="margin-top:0.75rem;">'
                f'<div><span>Feedback</span><strong>{selected_insight.metadata.get("cluster_size", 0):,}</strong></div>'
                f'<div><span>Negative</span><strong>{selected_insight.metadata.get("negative_rate", 0):.0%}</strong></div>'
                f'<div><span>Score</span><strong>{selected_insight.priority_score:.3f}</strong></div>'
                "</div>"
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
    report_filters=None,
):
    report_filters = report_filters or {}
    generated_at = pd.Timestamp.now().strftime("%B %d, %Y")
    impact_rows = build_operational_impact_rows(action_insights, causal_engine)
    data_source_label = (
        feedback_source.get("label", "Current dataset")
        if feedback_source
        else "Current dataset"
    )
    available_signal_count = int(
        report_filters.get("available_signal_count", len(action_insights))
        or len(action_insights)
    )
    report_focus = build_report_filter_summary(report_filters)
    selected_volume = sum(
        int(insight.metadata.get("cluster_size", 0) or 0)
        for insight in action_insights
    )
    selected_share = selected_volume / max(len(texts), 1)
    high_priority = [
        insight for insight in action_insights if insight.priority_label.startswith("HIGH")
    ]
    medium_priority = [
        insight for insight in action_insights if insight.priority_label.startswith("MEDIUM")
    ]
    risk_rows = [
        insight
        for insight in action_insights
        if customer_lens_for_insight(insight) == "At Risk"
    ]
    opportunity_rows = [
        insight
        for insight in action_insights
        if customer_lens_for_insight(insight) == "Opportunity"
    ]
    systemic_rows = [row for row in impact_rows if row["impact_type"] == "Systemic Risk"]
    top_insight = action_insights[0] if action_insights else None
    top_impact = impact_rows[0] if impact_rows else None
    top_cascade = next((row for row in impact_rows if row["cascade_targets"]), None)
    detail_limit = {
        "Brief": 3,
        "Board-ready": 5,
        "Operational detail": 8,
    }.get(report_depth, 5)

    if action_insights:
        weights = [
            max(int(insight.metadata.get("cluster_size", 0) or 0), 1)
            for insight in action_insights
        ]
        avg_negative = float(
            np.average(
                [insight.metadata.get("negative_rate", 0) for insight in action_insights],
                weights=weights,
            )
        )
        theme_counts = pd.Series(
            [insight.issue_theme.title() for insight in action_insights]
        ).value_counts()
        top_theme_text = ", ".join(
            f"{theme} ({count})" for theme, count in theme_counts.head(3).items()
        )
    else:
        avg_negative = 0
        top_theme_text = "No matching themes"

    audience_goal = {
        "Leadership": "make a decision, assign accountability, and understand operational risk",
        "Operations": "sequence work, assign owners, and reduce repeat friction",
        "Customer Experience": "explain customer pain, recovery needs, and strengths to protect",
        "Analyst Review": "trace evidence, model outputs, and assumptions behind the recommendations",
    }.get(audience, "turn feedback evidence into a decision")

    report_intent = {
        "Executive Summary Report": "decision narrative for leadership review",
        "Operational Action Report": "execution plan for service and operations teams",
        "Customer Intelligence Report": "customer-risk and opportunity readout",
    }.get(report_type, "PX-Intel decision report")

    priority_scope = report_filters.get("priority_scope", "All priorities")
    customer_lens = report_filters.get("customer_lens", "All customer lenses")
    theme_filter = report_filters.get("theme_filter", "All themes")
    signal_limit = report_filters.get("signal_limit", "All matching signals")
    filter_interpretation = []
    if priority_scope == "High priority only":
        filter_interpretation.append(
            "This is an urgent recovery brief focused only on the highest-scoring issues."
        )
    elif priority_scope == "High + medium priority":
        filter_interpretation.append(
            "This is an active watchlist report covering urgent and near-term issues."
        )
    elif priority_scope == "Medium watchlist":
        filter_interpretation.append(
            "This is a watchlist report for issues that may escalate if left unmanaged."
        )
    elif priority_scope == "Low / opportunity signals":
        filter_interpretation.append(
            "This is a strength-protection report focused on positive or lower-risk patterns."
        )
    else:
        filter_interpretation.append(
            "This report covers the selected visible signal set across priority levels."
        )

    if customer_lens == "At Risk":
        filter_interpretation.append(
            "The customer lens is narrowed to at-risk segments, so the language emphasizes recovery and trust repair."
        )
    elif customer_lens == "Opportunity":
        filter_interpretation.append(
            "The customer lens is narrowed to opportunity signals, so the language emphasizes repeatable strengths."
        )
    elif customer_lens == "Mixed":
        filter_interpretation.append(
            "The customer lens is narrowed to mixed signals, so the language emphasizes diagnosis before action."
        )
    if theme_filter != "All themes":
        filter_interpretation.append(
            f"The theme filter is narrowed to {theme_filter}, so recommendations stay inside that operating area."
        )
    if signal_limit != "All matching signals":
        filter_interpretation.append(
            f"The report is intentionally limited to {signal_limit.lower()} to keep the readout focused."
        )

    def success_metric_for_report(insight):
        negative_rate = insight.metadata.get("negative_rate", 0)
        if negative_rate >= 0.4 or insight.priority_label.startswith("HIGH"):
            return "Reduce negative feedback concentration and repeat complaints in the next feedback cycle."
        if customer_lens_for_insight(insight) == "Opportunity":
            return "Maintain positive concentration while documenting the behavior or workflow to replicate."
        return "Increase monitoring visibility and reduce movement into higher-priority status."

    def decision_implication(insight):
        lens = customer_lens_for_insight(insight)
        if lens == "At Risk":
            return "Treat this as a recovery signal: the next decision should reduce friction for affected customers."
        if lens == "Opportunity":
            return "Treat this as a strength signal: protect the behavior and reuse it as a service-quality reference."
        return "Treat this as a diagnostic signal: confirm the operating cause before scaling a response."

    lines = [
        f"# PX-Intel {report_type}",
        "",
        f"Generated: {generated_at}",
        f"Audience: {audience}",
        f"Report depth: {report_depth}",
        f"Data source: {data_source_label}",
        f"Report focus: {report_focus}",
        f"Signals included: {len(action_insights)} of {available_signal_count}",
        "",
        "## Executive Readout",
        "",
    ]

    if not action_insights:
        lines.extend(
            [
                (
                    "PX-Intel processed the active dataset, but the current report filters "
                    "do not match any visible action signals. Broaden the filters or switch "
                    "the active dataset to generate a decision-ready report."
                ),
                "",
                "## Filter Interpretation",
                "",
                *[f"- {item}" for item in filter_interpretation],
                "",
                "## Next Move",
                "",
                "1. Expand the priority, customer lens, or theme filter.",
                "2. Confirm that the active dataset contains enough usable feedback rows.",
                "3. Regenerate the report after the signal set is visible.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            (
                f"This {report_intent} uses **{len(action_insights)} in-scope signals** "
                f"from **{selected_volume:,} feedback comments** "
                f"({selected_share:.0%} of the active feedback base). "
                f"It is written for {audience.lower()} stakeholders who need to {audience_goal}."
            ),
            "",
            (
                f"The selected signal set averages **{avg_negative:.0%} negative concentration**. "
                f"The strongest themes in scope are **{top_theme_text}**."
            ),
            "",
            (
                f"The lead decision item is **{insight_display_name(top_insight, include_reference=True)}** "
                f"because it is marked **{top_insight.priority_label}**, has a priority score of "
                f"**{top_insight.priority_score:.3f}**, and represents "
                f"**{top_insight.metadata.get('cluster_size', 0):,} comments**."
            ),
            "",
            "## Filter Interpretation",
            "",
            *[f"- {item}" for item in filter_interpretation],
            "",
            "## Decision Needed",
            "",
            (
                f"**Recommended decision:** assign **{action_owner_for_insight(top_insight)}** "
                f"to lead the first response to **{insight_display_name(top_insight, include_reference=True)}** "
                f"within the **{action_window_for_insight(top_insight)}** window."
            ),
            "",
            f"**First move:** {top_insight.recommended_action}",
            "",
            f"**Success measure:** {success_metric_for_report(top_insight)}",
            "",
        ]
    )

    lines.extend(
        [
            "## Signal Portfolio Snapshot",
            "",
            f"- Feedback entries reviewed: {len(texts):,}",
            f"- Signals included in this report: {len(action_insights)} of {available_signal_count}",
            f"- Selected feedback volume: {selected_volume:,} comments",
            f"- High-priority signals in scope: {len(high_priority)}",
            f"- Medium-priority signals in scope: {len(medium_priority)}",
            f"- At-risk customer signals in scope: {len(risk_rows)}",
            f"- Opportunity signals in scope: {len(opportunity_rows)}",
            f"- Systemic operational risks in scope: {len(systemic_rows)}",
            "",
        ]
    )

    if report_type == "Executive Summary Report":
        lines.extend(["## Leadership Implications", ""])
        leadership_moves = [
            (
                "Approve immediate recovery work",
                top_insight,
                "Use this when the priority is reducing customer pain quickly.",
            ),
            (
                "Protect positive patterns",
                opportunity_rows[0] if opportunity_rows else action_insights[-1],
                "Use this when leadership wants repeatable service strengths, not only issue remediation.",
            ),
            (
                "Monitor escalation risk",
                medium_priority[0] if medium_priority else top_insight,
                "Use this when the team needs a watchlist for the next feedback cycle.",
            ),
        ]
        for title, insight, implication in leadership_moves:
            lines.extend(
                [
                    f"### {title}",
                    (
                        f"- **Signal:** {insight_display_name(insight, include_reference=True)} "
                        f"({insight.priority_label}, {insight.metadata.get('negative_rate', 0):.0%} negative)"
                    ),
                    f"- **Why it matters:** {implication}",
                    f"- **Move:** {insight.recommended_action}",
                    "",
                ]
            )

    elif report_type == "Operational Action Report":
        lines.extend(["## Operational Playbook", ""])
        for window in ["Immediate", "Next 7 Days", "Monitor"]:
            window_rows = [
                row for row in impact_rows if row["action_window"] == window
            ][:detail_limit]
            if not window_rows:
                continue
            lines.append(f"### {window}")
            for row in window_rows:
                cascade_text = (
                    ", ".join([signal_reference(target) for target in row["cascade_targets"][:3]])
                    if row["cascade_targets"]
                    else "No strong related cascade"
                )
                lines.extend(
                    [
                        f"- **{row['signal_name']} ({row['signal_id']})**",
                        f"  - Owner: {action_owner_for_insight(next(item for item in action_insights if item.cluster_id == row['cluster_id']))}",
                        f"  - Impact type: {row['impact_type']}",
                        f"  - Impact score: {row['impact_score']:.3f}",
                        f"  - Related signals: {cascade_text}",
                        f"  - Action: {row['recommended_action']}",
                    ]
                )
            lines.append("")

    elif report_type == "Customer Intelligence Report":
        lines.extend(["## Customer Intelligence Readout", ""])
        if risk_rows:
            lines.append("### Risk Signals To Reduce")
            for insight in risk_rows[:detail_limit]:
                lines.extend(
                    [
                        (
                            f"- **{insight_display_name(insight, include_reference=True)}**: "
                            f"{insight.metadata.get('cluster_size', 0):,} comments, "
                            f"{insight.metadata.get('negative_rate', 0):.0%} negative concentration."
                        ),
                        f"  - Customer signal: {insight.key_insight}",
                        f"  - Recovery move: {insight.recommended_action}",
                    ]
                )
            lines.append("")
        if opportunity_rows:
            lines.append("### Strengths To Expand")
            for insight in opportunity_rows[:detail_limit]:
                lines.extend(
                    [
                        (
                            f"- **{insight_display_name(insight, include_reference=True)}**: "
                            f"{insight.metadata.get('cluster_size', 0):,} comments, "
                            f"{insight.metadata.get('negative_rate', 0):.0%} negative concentration."
                        ),
                        f"  - What to protect: {insight.key_insight}",
                        f"  - Replication move: {insight.recommended_action}",
                    ]
                )
            lines.append("")

    lines.extend(["## Evidence-Backed Signal Detail", ""])
    for index, insight in enumerate(action_insights[:detail_limit], start=1):
        lines.extend(
            [
                f"### {index}. {insight_display_name(insight, include_reference=True)}",
                f"- Priority: {insight.priority_label} | Score: {insight.priority_score:.3f}",
                (
                    f"- Evidence base: {insight.metadata.get('cluster_size', 0):,} comments, "
                    f"{insight.metadata.get('negative_rate', 0):.0%} negative concentration, "
                    f"{insight.metadata.get('cluster_share', 0):.0%} of processed feedback."
                ),
                f"- Decision implication: {decision_implication(insight)}",
                f"- Root cause: {insight.root_cause}",
                f"- Representative feedback: {shorten_text(insight.example_feedback, 240)}",
                f"- Recommended owner: {action_owner_for_insight(insight)}",
                f"- Timing: {action_window_for_insight(insight)}",
                f"- Recommended action: {insight.recommended_action}",
                f"- Success metric: {success_metric_for_report(insight)}",
                "",
            ]
        )

    lines.extend(["## Cause And Effect Implications", ""])
    if top_cascade:
        related = ", ".join(
            [signal_reference(target) for target in top_cascade["cascade_targets"][:3]]
        )
        lines.extend(
            [
                (
                    f"Changing **{top_cascade['signal_name']} ({top_cascade['signal_id']})** "
                    f"may also influence **{related}**. This means the first fix should be "
                    "tracked not only against the selected signal, but also against related feedback patterns."
                ),
                "",
                f"Recommended operating test: run the change through the Operational Impact simulator and monitor whether negative concentration drops in the related signals during the next feedback cycle.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "PX-Intel does not show a strong cascade path inside the current filter set. "
                    "The report should therefore focus on direct signal-level improvement and monitoring."
                ),
                "",
            ]
        )

    lines.extend(["## Action Plan By Owner", ""])
    for owner in sorted({action_owner_for_insight(insight) for insight in action_insights}):
        owner_items = [
            insight
            for insight in action_insights
            if action_owner_for_insight(insight) == owner
        ][:detail_limit]
        lines.append(f"### {owner}")
        for insight in owner_items:
            lines.append(
                (
                    f"- **{insight_display_name(insight, include_reference=True)}** "
                    f"({action_window_for_insight(insight)}): {insight.recommended_action}"
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Recommended Next Decisions",
            "",
            f"1. Confirm the owner and action window for {insight_display_name(top_insight, include_reference=True)}.",
            "2. Define the metric that PX-Intel should compare after the next feedback cycle.",
            "3. Review whether the selected filters should become a standing leadership view or a one-time investigation.",
            "4. Use the Operational Impact simulator when the team needs to understand how changing one issue may affect related signals.",
            "5. Regenerate this report after new feedback is uploaded or after changing the report filters.",
            "",
        ]
    )
    return "\n".join(lines)
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
        f"Report focus: {report_focus}",
        f"Signals included: {len(action_insights)} of {available_signal_count}",
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
                (
                    "PX-Intel has processed the active dataset, but the current report "
                    "filters do not match any visible action signals. Broaden the filters "
                    "or switch the active dataset to generate a full decision report."
                ),
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
                    f"The selected report slice includes {len(action_insights)} of {available_signal_count} primary experience signals. "
                    f"Average visible negative concentration is {negative_weighted:.0%}, with the strongest signals concentrated in: {top_themes}."
                ),
                (
                    "PX-Intel is not treating the selected feedback as a generic survey export; the report is generated from the active clustering, sentiment, causal, and action-intelligence outputs."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "No learned signal profile matches the current report filters. "
                    "The underlying pipeline remains available; adjust the filter controls to expand the report slice."
                ),
                "",
            ]
        )

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
            "4. Review the report after new feedback data is loaded or report filters are changed.",
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
    report_filters=None,
):
    report_filters = report_filters or {}
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
        "included_signals": len(action_insights),
        "available_signals": int(
            report_filters.get("available_signal_count", len(action_insights))
            or len(action_insights)
        ),
        "report_filters": report_filters,
        "report_focus": build_report_filter_summary(report_filters),
        "signal_counts": {
            "at_risk": risk_count,
            "opportunity": opportunity_count,
            "systemic_operational_risk": systemic_count,
        },
        "generation_contract": [
            "Ground every claim in the active PX-Intel data.",
            "Use the selected report filters as the scope of the report.",
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


def markdown_to_docx_bytes(markdown_text, title="PX-Intel Report"):
    """Create a lightweight Word document from Markdown using stdlib only."""

    def xml_text(value):
        return escape(str(value), quote=False)

    def runs_from_inline_markdown(text):
        parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", str(text))
        runs = []
        for part in parts:
            if not part:
                continue
            is_bold = part.startswith("**") and part.endswith("**")
            is_code = part.startswith("`") and part.endswith("`")
            clean = part[2:-2] if is_bold else part[1:-1] if is_code else part
            run_props = ""
            if is_bold:
                run_props = "<w:rPr><w:b/></w:rPr>"
            elif is_code:
                run_props = '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/></w:rPr>'
            runs.append(
                f"<w:r>{run_props}<w:t xml:space=\"preserve\">{xml_text(clean)}</w:t></w:r>"
            )
        return "".join(runs) or "<w:r><w:t></w:t></w:r>"

    def paragraph(text="", style=None):
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{style_xml}{runs_from_inline_markdown(text)}</w:p>"

    body_parts = []
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            body_parts.append("<w:p/>")
            continue
        if stripped.startswith("# "):
            body_parts.append(paragraph(stripped[2:].strip(), "Title"))
        elif stripped.startswith("## "):
            body_parts.append(paragraph(stripped[3:].strip(), "Heading1"))
        elif stripped.startswith("### "):
            body_parts.append(paragraph(stripped[4:].strip(), "Heading2"))
        elif stripped.startswith("#### "):
            body_parts.append(paragraph(stripped[5:].strip(), "Heading3"))
        elif stripped.startswith("- "):
            body_parts.append(paragraph(f"• {stripped[2:].strip()}", "BodyText"))
        elif re.match(r"^\d+\.\s+", stripped):
            body_parts.append(paragraph(stripped, "BodyText"))
        else:
            body_parts.append(paragraph(stripped, "BodyText"))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body_parts)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="BodyText">
    <w:name w:val="Body Text"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="150" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:spacing w:after="260"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:color w:val="172033"/><w:sz w:val="38"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="260" w:after="140"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:color w:val="1F4E79"/><w:sz w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="210" w:after="100"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:color w:val="2563EB"/><w:sz w:val="25"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="160" w:after="80"/><w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:b/><w:color w:val="475569"/><w:sz w:val="23"/></w:rPr>
  </w:style>
</w:styles>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>
"""
    package_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>
"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xml_text(title)}</dc:title>
  <dc:creator>PX-Intel</dc:creator>
  <cp:lastModifiedBy>PX-Intel</cp:lastModifiedBy>
</cp:coreProperties>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml)
        docx.writestr("_rels/.rels", package_rels_xml)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("docProps/core.xml", core_xml)
    return buffer.getvalue()


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

    theme_options = ["All themes"] + sorted(
        {insight.issue_theme.title() for insight in action_insights}
    )
    if st.session_state.get("written_report_theme_filter") not in theme_options:
        st.session_state.written_report_theme_filter = "All themes"
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
    with filter_col1:
        priority_scope = st.selectbox(
            "Priority scope",
            [
                "All priorities",
                "High priority only",
                "High + medium priority",
                "Medium watchlist",
                "Low / opportunity signals",
            ],
            key="written_report_priority_scope",
        )
    with filter_col2:
        customer_lens = st.selectbox(
            "Customer lens",
            ["All customer lenses", "At Risk", "Opportunity", "Mixed"],
            key="written_report_customer_lens",
        )
    with filter_col3:
        theme_filter = st.selectbox(
            "Theme",
            theme_options,
            key="written_report_theme_filter",
        )
    with filter_col4:
        signal_limit = st.selectbox(
            "Signals",
            ["Top 5 signals", "Top 3 signals", "Top 8 signals", "All matching signals"],
            key="written_report_signal_limit",
        )

    filtered_action_insights = filter_report_insights(
        action_insights,
        priority_scope,
        customer_lens,
        theme_filter,
        signal_limit,
    )
    report_filters = {
        "priority_scope": priority_scope,
        "customer_lens": customer_lens,
        "theme_filter": theme_filter,
        "signal_limit": signal_limit,
        "included_signal_count": len(filtered_action_insights),
        "available_signal_count": len(action_insights),
    }
    st.caption(
        f"Report scope: {len(filtered_action_insights)} of {len(action_insights)} signals "
        f"using {build_report_filter_summary(report_filters)}."
    )
    if action_insights and not filtered_action_insights:
        st.warning(
            "No signals match these report filters yet. Broaden the priority, lens, or theme controls to generate a fuller report."
        )

    base_report = build_written_report(
        report_type,
        audience,
        clustering_engine,
        texts,
        audit_engine,
        causal_engine,
        filtered_action_insights,
        report_depth,
        feedback_source,
        report_filters,
    )
    report_text = base_report
    report_context = build_report_ai_context(
        report_type,
        audience,
        report_depth,
        feedback_source,
        texts,
        clustering_engine,
        causal_engine,
        filtered_action_insights,
        report_filters,
    )
    report_signature_payload = {
        "report_type": report_type,
        "audience": audience,
        "report_depth": report_depth,
        "feedback_source_id": feedback_source.get("id") if feedback_source else "current",
        "ai_model": ai_config.get("model") if ai_config else "local",
        "ai_strength": ai_config.get("generation_strength") if ai_config else "local",
        "filters": report_filters,
        "base_report": base_report,
    }
    report_signature = hashlib.sha256(
        json.dumps(
            report_signature_payload,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    ai_enabled = bool(ai_config and ai_config.get("enabled"))
    generation_label = (
        "Generate AI-enriched report preview"
        if ai_enabled
        else "Generate report preview"
    )
    generation_note = (
        f"AI enrichment is enabled with {ai_config.get('model', 'the selected model')} "
        f"and {ai_config.get('generation_strength', report_depth).lower()} depth."
        if ai_enabled
        else "AI enhancement is off, so PX-Intel will generate a local evidence-based report."
    )

    st.markdown(
        f"""
        <div class="cx-stakeholder-panel">
            <h4>Ready to generate</h4>
            <p>Select the report settings and filters above, then generate a preview. The report body, action plan, evidence, and decision framing will be rebuilt from the current PX-Intel data slice.</p>
            <div class="cx-stakeholder-meta">
                <span>{escape(report_type)}</span>
                <span>{escape(audience)}</span>
                <span>{escape(report_depth)}</span>
                <span>{len(filtered_action_insights)} signals selected</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    generate_col, note_col = st.columns([0.9, 2.1])
    with generate_col:
        generate_report = st.button(
            generation_label,
            key="write_report_preview",
            width="stretch",
            type="primary",
        )
    with note_col:
        st.caption(generation_note)
        st.caption(
            f"Current scope: {build_report_filter_summary(report_filters)}."
        )

    if generate_report:
        report_text = base_report
        report_source = "Local dynamic report"
        if ai_enabled:
            try:
                enhancer = OpenAIInsightEnhancer(
                    api_key=ai_config.get("api_key", ""),
                    model=ai_config.get("model", "gpt-4o-mini"),
                    generation_strength=ai_config.get(
                        "generation_strength",
                        report_depth,
                    ),
                )
                with st.spinner("Writing AI-enriched report preview..."):
                    report_text = enhancer.write_report(
                        report_type,
                        audience,
                        base_report,
                        filtered_action_insights,
                        report_context,
                    )
                report_source = "AI-enriched report"
                st.session_state.ai_written_report_signature = report_signature
                st.session_state.ai_written_report_text = report_text
            except AIEnhancementError as exc:
                st.warning(
                    "AI report writing is unavailable, so PX-Intel generated a local evidence-based report."
                )
                st.caption(str(exc)[:320])

        st.session_state.written_report_signature = report_signature
        st.session_state.written_report_output_text = report_text
        st.session_state.written_report_generated_signature = report_signature
        st.session_state.written_report_generated_text = report_text
        st.session_state.written_report_generated_source = report_source
        st.session_state.written_report_generated_at = pd.Timestamp.now().strftime(
            "%B %d, %Y %I:%M %p"
        )

    generated_is_current = (
        st.session_state.get("written_report_generated_signature")
        == report_signature
    )
    generated_text = st.session_state.get(
        "written_report_generated_text",
        base_report,
    )

    if generated_is_current:
        final_report_text = st.session_state.get(
            "written_report_output_text",
            generated_text,
        )
        generated_source = st.session_state.get(
            "written_report_generated_source",
            "Local dynamic report",
        )
        generated_at = st.session_state.get("written_report_generated_at", "Current run")
        st.markdown(
            f"""
            <div class="cx-stakeholder-panel">
                <h4>Report preview generated</h4>
                <p>The preview and downloads below are tied to the current report controls. Regenerate after changing report type, audience, depth, filters, AI model, or active data source.</p>
                <div class="cx-stakeholder-meta">
                    <span>{escape(generated_source)}</span>
                    <span>{len(filtered_action_insights)} signals included</span>
                    <span>{escape(build_report_filter_summary(report_filters))}</span>
                    <span>{escape(generated_at)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview_tab, edit_tab = st.tabs(["Preview", "Editable source"])
        with preview_tab:
            st.markdown(final_report_text)
        with edit_tab:
            st.text_area(
                "Editable report source",
                height=420,
                key="written_report_output_text",
                help="Edit this Markdown source before downloading the report.",
            )
            final_report_text = st.session_state.get(
                "written_report_output_text",
                final_report_text,
            )

        safe_report_name = re.sub(r"[^a-z0-9]+", "_", report_type.lower()).strip("_")
        download_col1, download_col2 = st.columns([1, 1])
        with download_col1:
            st.download_button(
                "Download Markdown",
                data=final_report_text,
                file_name=f"px_intel_{safe_report_name}.md",
                mime="text/markdown",
                width="stretch",
            )
        with download_col2:
            st.download_button(
                "Download Word (.docx)",
                data=markdown_to_docx_bytes(final_report_text, report_type),
                file_name=f"px_intel_{safe_report_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width="stretch",
            )
    else:
        if st.session_state.get("written_report_generated_signature"):
            st.warning(
                "Report settings changed. Generate a new preview to update the report body and downloads."
            )
        else:
            st.info(
                "Select report settings and filters, then click Generate report preview to create the report."
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


def build_ai_signature(action_insights, model, data_source_id="current"):
    """Build a stable signature so AI enhancement is not called on every rerun."""
    payload = {
        "model": model,
        "data_source_id": data_source_id,
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


def maybe_enhance_action_insights(action_insights, ai_config, feedback_source=None):
    """Optionally improve manager-facing language with an OpenAI model."""
    metadata = {"enabled": False, "summary": ""}
    if not ai_config.get("enabled"):
        return action_insights, metadata

    signature = build_ai_signature(
        action_insights,
        f"{ai_config.get('model')}|{ai_config.get('generation_strength')}",
        feedback_source.get("id") if feedback_source else "current",
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
        feedback_source,
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
