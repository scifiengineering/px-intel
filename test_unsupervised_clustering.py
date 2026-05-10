"""
Unit tests for M1: Unsupervised Clustering Engine
Tests LDA, K-Means, Jensen-Shannon Divergence, t-SNE, and cluster statistics
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unsupervised_clustering import UnsupervisedClusteringEngine


@pytest.fixture
def sample_texts():
    """Create sample hospital feedback texts for testing."""
    return [
        "long wait times in emergency department during peak hours",
        "staff was rude and unprofessional to patients",
        "facility is clean and well maintained throughout",
        "doctor waited too long to provide diagnosis",
        "nurses were helpful and caring during my stay",
        "waiting room temperature was uncomfortable and cold",
        "staff communication was excellent and clear",
        "emergency room had severe delays and confusion",
        "hospital rooms are modern and comfortable",
        "wait to see specialist exceeded three hours",
        "staff attitude was dismissive and unhelpful",
        "facility cleanliness needs improvement in bathrooms",
    ] * 50  # Repeat to have ~600 documents


@pytest.fixture
def engine():
    """Create fresh clustering engine instance."""
    return UnsupervisedClusteringEngine(random_state=42)


class TestTextVectorization:
    """Tests for TF-IDF and count matrix vectorization."""
    
    def test_tfidf_vectorization(self, engine, sample_texts):
        """Test TF-IDF matrix creation."""
        engine._fit_tfidf(sample_texts)
        assert engine.vectorizer is not None
        assert engine.tfidf_matrix is not None
        assert engine.tfidf_matrix.shape[0] == len(sample_texts)
        assert engine.tfidf_matrix.shape[1] > 0
    
    def test_count_matrix_creation(self, engine, sample_texts):
        """Test count matrix for LDA."""
        engine._fit_count_matrix(sample_texts)
        assert engine.count_matrix is not None
        assert engine.count_matrix.shape[0] == len(sample_texts)
        # Count matrix should have non-negative integer values
        assert np.all(engine.count_matrix.data >= 0)
    
    def test_vocabulary_size(self, engine, sample_texts):
        """Test vocabulary size is reasonable."""
        engine._fit_tfidf(sample_texts)
        vocab_size = len(engine.vectorizer.get_feature_names_out())
        assert vocab_size > 0
        assert vocab_size < len(sample_texts) * 2  # Vocabulary shouldn't explode


class TestLDATopicModeling:
    """Tests for LDA topic modeling."""
    
    def test_lda_initialization(self, engine, sample_texts):
        """Test LDA model creation with fixed topics."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=3, max_iter=5)
        
        assert engine.lda_model is not None
        assert engine.lda_features is not None
        assert engine.lda_features.shape == (len(sample_texts), 3)
    
    def test_lda_features_sum_to_one(self, engine, sample_texts):
        """Test LDA features are valid probability distributions."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=5, max_iter=5)
        
        # Each row should sum to approximately 1 (probability distribution)
        row_sums = np.sum(engine.lda_features, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)
    
    def test_topic_coherence_computation(self, engine, sample_texts):
        """Test topic coherence metric."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        
        coherence = engine._compute_topic_coherence(n_topics=4)
        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1
    
    def test_optimal_topics_selection(self, engine, sample_texts):
        """Test automatic optimal topic selection."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        
        optimal = engine._select_optimal_n_topics(max_topics=8)
        assert isinstance(optimal, int)
        assert 2 <= optimal <= 8


class TestKMeansClustering:
    """Tests for K-Means clustering."""
    
    def test_kmeans_initialization(self, engine, sample_texts):
        """Test K-Means model creation."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=5, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        
        assert engine.kmeans_model is not None
        assert engine.cluster_assignments is not None
        assert len(engine.cluster_assignments) == len(sample_texts)
    
    def test_cluster_assignments_valid(self, engine, sample_texts):
        """Test cluster assignments are valid integers."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        
        clusters = np.unique(engine.cluster_assignments)
        assert len(clusters) == 3
        assert np.all(clusters >= 0)
        assert np.all(clusters < 3)
    
    def test_silhouette_score_computation(self, engine, sample_texts):
        """Test silhouette score calculation."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        
        score = engine._compute_silhouette_scores(n_clusters=3)
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    def test_optimal_clusters_selection(self, engine, sample_texts):
        """Test automatic optimal cluster selection."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=5, max_iter=5)
        
        optimal = engine._select_optimal_n_clusters(max_clusters=5)
        assert isinstance(optimal, int)
        assert 2 <= optimal <= 5


class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon divergence and Paradox Score."""
    
    def test_js_divergence_identical_distributions(self, engine):
        """Test JS divergence between identical distributions is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        divergence = engine._jensen_shannon_divergence(p, q)
        assert np.isclose(divergence, 0.0, atol=1e-6)
    
    def test_js_divergence_different_distributions(self, engine):
        """Test JS divergence between different distributions is > 0."""
        p = np.array([0.9, 0.05, 0.03, 0.02])
        q = np.array([0.02, 0.03, 0.05, 0.9])
        
        divergence = engine._jensen_shannon_divergence(p, q)
        assert divergence > 0
        assert divergence <= 1.0  # JS divergence is bounded
    
    def test_js_divergence_symmetry(self, engine):
        """Test JS divergence is symmetric."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = np.array([0.4, 0.3, 0.2, 0.1])
        
        div_pq = engine._jensen_shannon_divergence(p, q)
        div_qp = engine._jensen_shannon_divergence(q, p)
        assert np.isclose(div_pq, div_qp)
    
    def test_paradox_scores_computation(self, engine, sample_texts):
        """Test Paradox Score computation."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        engine._compute_paradox_scores()
        
        assert len(engine.paradox_scores) > 0
        # For 3 clusters, should have 3 choose 2 = 3 pairs
        assert len(engine.paradox_scores) == 3
        
        # All scores should be between 0 and 1
        for score in engine.paradox_scores.values():
            assert 0 <= score <= 1.0


class TestTSNEProjection:
    """Tests for t-SNE visualization."""
    
    def test_tsne_projection_shape(self, engine, sample_texts):
        """Test t-SNE projection dimensions."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        engine._fit_tsne(perplexity=20, max_iter=300)
        
        assert engine.tsne_projection is not None
        assert engine.tsne_projection.shape == (len(sample_texts), 2)
    
    def test_tsne_projection_values_numeric(self, engine, sample_texts):
        """Test t-SNE projection contains valid numbers."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        engine._fit_tsne(perplexity=20, max_iter=300)
        
        assert np.all(np.isfinite(engine.tsne_projection))
        assert not np.any(np.isnan(engine.tsne_projection))


class TestClusterStatistics:
    """Tests for cluster statistics computation."""
    
    def test_cluster_statistics_computation(self, engine, sample_texts):
        """Test cluster statistics are computed."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        engine._compute_cluster_statistics(sample_texts)
        
        assert len(engine.cluster_statistics) == 3
        
        # Each cluster should have expected keys
        for cluster_id, stats in engine.cluster_statistics.items():
            assert 'size' in stats
            assert 'percentage' in stats
            assert 'avg_length' in stats
            assert 'min_length' in stats
            assert 'max_length' in stats
            assert 'std_length' in stats
    
    def test_cluster_statistics_values_valid(self, engine, sample_texts):
        """Test cluster statistics have valid values."""
        engine._fit_tfidf(sample_texts)
        engine._fit_count_matrix(sample_texts)
        engine._fit_lda(n_topics=4, max_iter=5)
        engine._fit_kmeans(n_clusters=3)
        engine._compute_cluster_statistics(sample_texts)
        
        total_size = 0
        for cluster_id, stats in engine.cluster_statistics.items():
            assert stats['size'] > 0
            assert 0 < stats['percentage'] <= 100
            assert stats['avg_length'] > 0
            assert stats['min_length'] >= 0
            assert stats['max_length'] >= stats['min_length']
            total_size += stats['size']
        
        # Total size should match number of documents
        assert total_size == len(sample_texts)


class TestCompletePipeline:
    """Tests for the complete fit pipeline."""
    
    def test_full_pipeline_execution(self, engine, sample_texts):
        """Test complete unsupervised clustering pipeline."""
        engine.fit(sample_texts, auto_select=False)
        
        # Verify all components are initialized
        assert engine.cluster_assignments is not None
        assert engine.lda_features is not None
        assert engine.tsne_projection is not None
        assert engine.paradox_scores is not None
        assert engine.cluster_statistics is not None
    
    def test_pipeline_auto_select(self, engine, sample_texts):
        """Test pipeline with auto topic/cluster selection."""
        engine.fit(sample_texts, auto_select=True)
        
        assert engine.optimal_n_topics is not None
        assert engine.optimal_n_clusters is not None
        assert engine.optimal_n_topics >= 2
        assert engine.optimal_n_clusters >= 2


class TestExportAndPersistence:
    """Tests for data export and persistence."""
    
    def test_export_to_dataframe(self, engine, sample_texts):
        """Test exporting results to DataFrame."""
        engine.fit(sample_texts, auto_select=False)
        df = engine.export_to_dataframe(sample_texts)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_texts)
        assert 'cluster_id' in df.columns
        assert 'tsne_x' in df.columns
        assert 'tsne_y' in df.columns
    
    def test_export_with_original_dataframe(self, engine, sample_texts):
        """Test exporting with preserving original DataFrame columns."""
        original_df = pd.DataFrame({
            'text': sample_texts,
            'id': range(len(sample_texts))
        })
        
        engine.fit(sample_texts, auto_select=False)
        df = engine.export_to_dataframe(sample_texts, original_df=original_df)
        
        assert 'id' in df.columns
        assert 'cluster_id' in df.columns
    
    def test_save_and_load(self, engine, sample_texts):
        """Test saving and loading clustering results."""
        engine.fit(sample_texts, auto_select=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'clustering.pkl'
            
            # Save
            engine.save(str(filepath))
            assert filepath.exists()
            
            # Load into new engine
            engine2 = UnsupervisedClusteringEngine()
            engine2.load(str(filepath))
            
            # Verify loaded data matches
            assert np.array_equal(engine.cluster_assignments, engine2.cluster_assignments)
            assert np.array_equal(engine.tsne_projection, engine2.tsne_projection)
            assert engine.paradox_scores == engine2.paradox_scores


class TestClusterSummary:
    """Tests for cluster summary generation."""
    
    def test_get_cluster_summary(self, engine, sample_texts):
        """Test cluster summary string generation."""
        engine.fit(sample_texts, auto_select=False)
        summary = engine.get_cluster_summary()
        
        assert isinstance(summary, str)
        assert 'LDA Topics' in summary
        assert 'K-Means Clusters' in summary
        assert 'Paradox Scores' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
