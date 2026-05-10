"""
M1: Unsupervised Clustering Engine
Landscape Discovery using LDA + K-Means + Jensen-Shannon Divergence
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class UnsupervisedClusteringEngine:
    """
    Unsupervised clustering engine for discovering service gaps in feedback.
    
    Pipeline:
    1. TF-IDF vectorization for text representation
    2. LDA topic modeling to discover latent themes
    3. K-Means clustering on LDA topic distributions
    4. Jensen-Shannon Divergence to compute Paradox Score
    5. t-SNE for 2D visualization
    
    Attributes:
        optimal_n_clusters (int): Auto-detected optimal number of clusters
        optimal_n_topics (int): Auto-detected optimal number of LDA topics
        cluster_assignments (np.array): Cluster ID for each document
        paradox_scores (dict): JS divergence between each cluster pair
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize clustering engine."""
        self.random_state = random_state
        self.vectorizer = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.lda_model = None
        self.lda_features = None
        self.kmeans_model = None
        self.cluster_assignments = None
        self.tsne_projection = None
        self.paradox_scores = {}
        self.optimal_n_clusters = None
        self.optimal_n_topics = None
        self.cluster_statistics = {}
        
    # ========================================================================
    # Step 1: Text Vectorization
    # ========================================================================
    
    def _fit_tfidf(self, texts: List[str], max_features: int = 5000) -> None:
        """
        Fit TF-IDF vectorizer on normalized texts.
        
        Args:
            texts: List of normalized text documents
            max_features: Maximum vocabulary size
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,           # Ignore terms appearing in <5 docs
            max_df=0.7,         # Ignore terms in >70% of docs
            ngram_range=(1, 2), # Unigrams + bigrams
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def _fit_count_matrix(self, texts: List[str], max_features: int = 5000) -> None:
        """
        Fit count vectorizer for LDA (requires non-negative integer counts).
        
        Args:
            texts: List of normalized text documents
            max_features: Maximum vocabulary size
        """
        count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 1),  # Unigrams only for LDA
            stop_words='english'
        )
        self.count_matrix = count_vectorizer.fit_transform(texts)
    
    # ========================================================================
    # Step 2: LDA Topic Modeling
    # ========================================================================
    
    def _fit_lda(self, n_topics: int = 10, max_iter: int = 20) -> None:
        """
        Fit LDA model on count matrix.
        
        Args:
            n_topics: Number of latent topics
            max_iter: Maximum number of LDA iterations
        """
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=max_iter,
            learning_method='batch',
            verbose=0
        )
        self.lda_features = self.lda_model.fit_transform(self.count_matrix)
    
    def _compute_topic_coherence(self, n_topics: int) -> float:
        """
        Approximate topic coherence score (higher is better).
        Uses proportion of high-probability terms per topic.
        
        Args:
            n_topics: Number of topics to evaluate
            
        Returns:
            Coherence score (0-1 scale, approximate)
        """
        self._fit_lda(n_topics=n_topics, max_iter=10)
        
        # Get top words per topic (high components = important words)
        top_word_importance = []
        for topic_idx in range(n_topics):
            weights = self.lda_model.components_[topic_idx]
            top_indices = np.argsort(weights)[-5:]  # Top 5 words
            importance = weights[top_indices].sum() / weights.sum()
            top_word_importance.append(importance)
        
        return np.mean(top_word_importance)
    
    def _select_optimal_n_topics(self, max_topics: int = 20) -> int:
        """
        Auto-select optimal number of topics via coherence search.
        
        Args:
            max_topics: Maximum topics to try
            
        Returns:
            Optimal number of topics
        """
        print(f"Searching optimal topics (2-{max_topics})...")
        coherence_scores = []
        
        for n_topics in range(2, min(max_topics + 1, 15)):  # Limit search
            score = self._compute_topic_coherence(n_topics)
            coherence_scores.append(score)
            print(f"  Topics={n_topics}: coherence={score:.3f}")
        
        optimal_idx = np.argmax(coherence_scores)
        optimal_topics = int(optimal_idx + 2)  # Offset for range(2, ...) + convert to Python int
        print(f"Selected {optimal_topics} topics (coherence={coherence_scores[optimal_idx]:.3f})")
        
        return optimal_topics
    
    # ========================================================================
    # Step 3: K-Means Clustering
    # ========================================================================
    
    def _fit_kmeans(self, n_clusters: int) -> None:
        """
        Fit K-Means on LDA topic distributions.
        
        Args:
            n_clusters: Number of clusters
        """
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.cluster_assignments = self.kmeans_model.fit_predict(self.lda_features)
    
    def _compute_silhouette_scores(self, n_clusters: int) -> float:
        """
        Compute silhouette score for cluster quality (higher is better).
        
        Args:
            n_clusters: Number of clusters to evaluate
            
        Returns:
            Average silhouette score
        """
        from sklearn.metrics import silhouette_score
        self._fit_kmeans(n_clusters)
        score = silhouette_score(self.lda_features, self.cluster_assignments)
        return score
    
    def _select_optimal_n_clusters(self, max_clusters: int = 10) -> int:
        """
        Auto-select optimal number of clusters via silhouette search.
        
        Args:
            max_clusters: Maximum clusters to try
            
        Returns:
            Optimal number of clusters
        """
        print(f"Searching optimal clusters (2-{max_clusters})...")
        silhouette_scores = []
        
        for n_clusters in range(2, min(max_clusters + 1, 8)):
            score = self._compute_silhouette_scores(n_clusters)
            silhouette_scores.append(score)
            print(f"  Clusters={n_clusters}: silhouette={score:.3f}")
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = int(optimal_idx + 2)  # Offset for range(2, ...) + convert to Python int
        print(f"Selected {optimal_clusters} clusters (silhouette={silhouette_scores[optimal_idx]:.3f})")
        
        return optimal_clusters
    
    # ========================================================================
    # Step 4: Jensen-Shannon Divergence & Paradox Score
    # ========================================================================
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two probability distributions.
        
        Measures "distance" between two clusters' topic distributions.
        Range: [0, 1] where 0 = identical, 1 = completely different
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence score
        """
        # Normalize to probabilities if needed
        p = np.array(p) / np.sum(p)
        q = np.array(q) / np.sum(q)
        
        # Compute Jensen-Shannon divergence
        m = 0.5 * (p + q)
        divergence = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
        
        return float(divergence)
    
    def _compute_paradox_scores(self) -> None:
        """
        Compute Paradox Score for all cluster pairs.
        
        Paradox Score = Jensen-Shannon divergence between cluster topic distributions
        High score = clusters are different (service gaps exist)
        Low score = clusters are similar (no gaps)
        """
        n_clusters = len(np.unique(self.cluster_assignments))
        self.paradox_scores = {}
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Get mean topic distribution for each cluster
                cluster_i_mask = self.cluster_assignments == i
                cluster_j_mask = self.cluster_assignments == j
                
                p_i = np.mean(self.lda_features[cluster_i_mask], axis=0)
                p_j = np.mean(self.lda_features[cluster_j_mask], axis=0)
                
                # Compute JS divergence
                js_div = self._jensen_shannon_divergence(p_i, p_j)
                self.paradox_scores[(i, j)] = js_div
    
    # ========================================================================
    # Step 5: t-SNE Visualization
    # ========================================================================
    
    def _fit_tsne(self, perplexity: int = 30, max_iter: int = 1000) -> None:
        """
        Fit t-SNE to project LDA features to 2D for visualization.
        
        Args:
            perplexity: t-SNE perplexity parameter (typically 5-50)
            max_iter: Number of t-SNE iterations (must be >= 250)
        """
        print("Computing t-SNE projection...")
        # Ensure max_iter is at least 250 (sklearn requirement)
        max_iter = max(250, max_iter)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(self.lda_features) - 1),
            random_state=self.random_state,
            max_iter=max_iter,
            verbose=1
        )
        self.tsne_projection = tsne.fit_transform(self.lda_features)
    
    # ========================================================================
    # Step 6: Cluster Statistics
    # ========================================================================
    
    def _compute_cluster_statistics(self, texts: List[str]) -> None:
        """
        Compute statistics for each cluster.
        
        Args:
            texts: Original text documents
        """
        self.cluster_statistics = {}
        
        for cluster_id in np.unique(self.cluster_assignments):
            mask = self.cluster_assignments == cluster_id
            cluster_texts = np.array(texts)[mask]
            text_lengths = np.array([len(t) for t in cluster_texts])
            
            self.cluster_statistics[int(cluster_id)] = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(self.cluster_assignments) * 100),
                'avg_length': float(np.mean(text_lengths)),
                'min_length': int(np.min(text_lengths)),
                'max_length': int(np.max(text_lengths)),
                'std_length': float(np.std(text_lengths)),
            }
    
    # ========================================================================
    # Main Pipeline
    # ========================================================================
    
    def fit(self, texts: List[str], auto_select: bool = True) -> None:
        """
        Complete unsupervised clustering pipeline.
        
        Args:
            texts: List of normalized text documents
            auto_select: If True, auto-select optimal topics/clusters
        """
        print(f"Starting unsupervised clustering on {len(texts)} documents...")
        
        # Step 1: Vectorization
        print("\n[1/6] TF-IDF vectorization...")
        self._fit_tfidf(texts)
        self._fit_count_matrix(texts)
        print(f"  Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        # Step 2: Topic modeling
        print("\n[2/6] LDA topic modeling...")
        if auto_select:
            self.optimal_n_topics = self._select_optimal_n_topics(max_topics=15)
        else:
            self.optimal_n_topics = 10  # Default
        self._fit_lda(n_topics=self.optimal_n_topics)
        
        # Step 3: Clustering
        print("\n[3/6] K-Means clustering...")
        if auto_select:
            self.optimal_n_clusters = self._select_optimal_n_clusters(max_clusters=8)
        else:
            self.optimal_n_clusters = 5  # Default
        self._fit_kmeans(self.optimal_n_clusters)
        print(f"  Assigned {len(np.unique(self.cluster_assignments))} clusters")
        
        # Step 4: Paradox Score
        print("\n[4/6] Computing Paradox Scores (JS divergence)...")
        self._compute_paradox_scores()
        avg_paradox = np.mean(list(self.paradox_scores.values()))
        print(f"  Average Paradox Score: {avg_paradox:.3f}")
        
        # Step 5: t-SNE
        print("\n[5/6] t-SNE projection...")
        self._fit_tsne()
        
        # Step 6: Statistics
        print("\n[6/6] Computing cluster statistics...")
        self._compute_cluster_statistics(texts)
        
        print("\n✓ Clustering complete!")
    
    # ========================================================================
    # Persistence & Export
    # ========================================================================
    
    def save(self, filepath: str) -> None:
        """
        Save clustering results to pickle file.
        
        Args:
            filepath: Path to save pickle file
        """
        data = {
            'cluster_assignments': self.cluster_assignments,
            'lda_features': self.lda_features,
            'tsne_projection': self.tsne_projection,
            'paradox_scores': self.paradox_scores,
            'optimal_n_clusters': self.optimal_n_clusters,
            'optimal_n_topics': self.optimal_n_topics,
            'cluster_statistics': self.cluster_statistics,
            'kmeans_centers': self.kmeans_model.cluster_centers_ if self.kmeans_model else None,
            'lda_components': self.lda_model.components_ if self.lda_model else None,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Clustering results saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load clustering results from pickle file.
        
        Args:
            filepath: Path to pickle file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.cluster_assignments = data['cluster_assignments']
        self.lda_features = data['lda_features']
        self.tsne_projection = data['tsne_projection']
        self.paradox_scores = data['paradox_scores']
        self.optimal_n_clusters = data['optimal_n_clusters']
        self.optimal_n_topics = data['optimal_n_topics']
        self.cluster_statistics = data['cluster_statistics']
        
        print(f"Clustering results loaded from {filepath}")
    
    def export_to_dataframe(self, texts: List[str], original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Export clustering results as enriched DataFrame.
        
        Args:
            texts: Original normalized texts
            original_df: Original DataFrame to preserve columns
            
        Returns:
            DataFrame with cluster assignments + metadata
        """
        if original_df is not None:
            result_df = original_df.copy()
        else:
            result_df = pd.DataFrame({'text': texts})
        
        result_df['cluster_id'] = self.cluster_assignments
        result_df['tsne_x'] = self.tsne_projection[:, 0]
        result_df['tsne_y'] = self.tsne_projection[:, 1]
        
        # Add LDA topic features
        for i in range(self.lda_features.shape[1]):
            result_df[f'lda_topic_{i}'] = self.lda_features[:, i]
        
        return result_df
    
    def get_cluster_summary(self) -> str:
        """
        Generate human-readable summary of clustering results.
        
        Returns:
            Summary string
        """
        summary = f"""
╔════════════════════════════════════════════════════════════╗
║         Unsupervised Clustering Summary (M1)              ║
╚════════════════════════════════════════════════════════════╝

LDA Topics: {self.optimal_n_topics}
K-Means Clusters: {self.optimal_n_clusters}

Cluster Sizes:
"""
        for cluster_id in sorted(self.cluster_statistics.keys()):
            stats = self.cluster_statistics[cluster_id]
            summary += f"  Cluster {cluster_id}: {stats['size']:5d} docs ({stats['percentage']:5.1f}%) | "
            summary += f"Avg length: {stats['avg_length']:6.0f} chars\n"
        
        summary += f"\nParadox Scores (Top 5 divergences):\n"
        sorted_paradox = sorted(self.paradox_scores.items(), key=lambda x: x[1], reverse=True)
        for (c1, c2), score in sorted_paradox[:5]:
            summary += f"  Cluster {c1} ↔ Cluster {c2}: {score:.3f}\n"
        
        return summary
