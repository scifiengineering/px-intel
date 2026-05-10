"""
M4: Unsupervised-First Streamlit Dashboard
Interactive t-SNE landscape with cluster auditing and causal insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_loader import DataLoader
from unsupervised_clustering import UnsupervisedClusteringEngine
from cluster_audit import ClusterAuditEngine
from causal_reasoning import CausalReasoningEngine
from pathlib import Path

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="PX-Intel Unsupervised",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 PX-Intel: Unsupervised Discovery")
st.markdown(
    "**Landscape Mapping of Hospital Feedback** | LDA + K-Means + RoBERTa Audit"
)


# ============================================================================
# Session State & Caching
# ============================================================================


@st.cache_resource
def load_data():
    """Load and normalize feedback data."""
    loader = DataLoader("text_data.csv")
    df, stats = loader.load()
    return df, stats


@st.cache_resource
def run_clustering():
    """Run unsupervised clustering pipeline."""
    df, stats = load_data()
    texts = df["text_normalized"].tolist()

    engine = UnsupervisedClusteringEngine(random_state=42)
    engine.fit(texts, auto_select=True)

    return engine, df, texts


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


# ============================================================================
# Main Dashboard
# ============================================================================


def main():
    """Main dashboard flow."""

    # Load data and run clustering
    with st.spinner("Loading data and discovering clusters..."):
        clustering_engine, df, texts = run_clustering()
        cluster_assignments = clustering_engine.cluster_assignments

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
    # Tab 1: Landscape Visualization
    # ====================================================================

    tab_landscape, tab_audit, tab_causal, tab_data = st.tabs(
        ["🗺️ Landscape", "📊 Cluster Audit", "⚡ Operational Impact", "📋 Data Export"]
    )

    with tab_landscape:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### t-SNE Landscape")
            st.markdown(
                "Each point is a feedback entry. Colors indicate clusters. Hover for details."
            )

            # Create t-SNE scatter plot
            fig = go.Figure()

            for cluster_id in np.unique(cluster_assignments):
                mask = cluster_assignments == cluster_id
                sentiment_dist = audit_engine.cluster_sentiment_results[cluster_id][
                    "sentiment_distribution"
                ]
                zone = audit_engine.cluster_zones[cluster_id]["zone_type"]

                # Color by zone
                color_map = {
                    "RED_ZONE": "#FF6B6B",
                    "GREEN_ZONE": "#51CF66",
                    "NEUTRAL_ZONE": "#FFD93D",
                }
                color = color_map.get(zone, "#999999")

                fig.add_trace(
                    go.Scatter(
                        x=clustering_engine.tsne_projection[mask, 0],
                        y=clustering_engine.tsne_projection[mask, 1],
                        mode="markers",
                        name=f"Cluster {cluster_id} ({zone})",
                        marker=dict(
                            size=6, color=color, opacity=0.7, line=dict(width=0)
                        ),
                        text=[
                            f"Cluster {cluster_id}<br>Sentiment: {sentiment_dist['NEGATIVE']:.1%} negative"
                            for _ in range(np.sum(mask))
                        ],
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title="Hospital Feedback Landscape (t-SNE Projection)",
                xaxis_title="t-SNE Dimension 1",
                yaxis_title="t-SNE Dimension 2",
                hovermode="closest",
                height=500,
                width=800,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Summary")
            st.metric("Total Entries", len(texts))
            st.metric("Clusters Discovered", clustering_engine.optimal_n_clusters)
            st.metric("LDA Topics", clustering_engine.optimal_n_topics)

            st.markdown("### Zone Distribution")
            red_count = len(audit_engine.get_red_zones())
            green_count = len(audit_engine.get_green_zones())
            neutral_count = len(audit_engine.get_neutral_zones())

            st.markdown(f"""
            🚨 **Red Zones** (Distress): {red_count}  
            ✨ **Green Zones** (Satisfaction): {green_count}  
            ⚖️ **Neutral**: {neutral_count}
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
        st.dataframe(audit_df, use_container_width=True)

    # ====================================================================
    # Tab 3: Causal Analysis
    # ====================================================================

    with tab_causal:
        st.markdown("### Operational Impact Results")
        st.markdown(causal_engine.get_summary())

        cluster_options = sorted(causal_engine.cluster_lda_features.keys())
        if cluster_options:
            selected_causal_cluster = st.selectbox(
                "Select Cluster for Causal Details",
                cluster_options,
                format_func=lambda x: f"Cluster {x} ({audit_engine.cluster_zones.get(x, {}).get('zone_type', 'UNKNOWN')})",
            )

            st.markdown(causal_engine.get_cluster_summary(selected_causal_cluster))

            cascades = causal_engine.cascade_predictions.get(
                selected_causal_cluster, []
            )
            st.markdown(
                f"#### Cascade Predictions for Cluster {selected_causal_cluster}"
            )

            if cascades:
                for cascade in cascades:
                    st.info(f"""
                    **→ Cluster {cascade['target_cluster']}**  
                    Similarity: {cascade['similarity']:.1%}  
                    Cascade Likelihood: {cascade['cascade_likelihood']:.0%}  
                    {cascade['cascade_interpretation']}
                    """)
            else:
                st.info("No significant cascades detected for this cluster.")
        else:
            st.info("No causal clusters are available yet.")

        # Causal DataFrame
        st.markdown("### Operational Impact Summary")
        causal_df = causal_engine.export_to_dataframe()
        st.dataframe(causal_df, use_container_width=True)

    # ====================================================================
    # Tab 4: Data Export
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
                file_name="px_intel_enriched.csv",
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
            st.dataframe(export_df.head(10), use_container_width=True)

        st.markdown("#### Clustering Summary")
        st.info(clustering_engine.get_cluster_summary())


if __name__ == "__main__":
    main()
