"""
Unit tests for M2: Cluster Auditing Engine
Tests sentiment analysis, zone identification, vocabulary extraction, and reports
"""

import pytest
import numpy as np
import pandas as pd
from cluster_audit import ClusterAuditEngine


@pytest.fixture
def sample_data():
    """Create sample cluster data for testing."""
    texts = [
        # Red zone texts (negative)
        "long wait times very disappointing",
        "staff was rude and unprofessional",
        "terrible experience waste of time",
        "waited hours for appointment",
        "staff attitude poor and dismissive",
        # Green zone texts (positive)
        "excellent care very professional",
        "staff was wonderful and caring",
        "best hospital experience ever",
        "highly satisfied with service",
        "nurses were amazing helpful",
        # Neutral zone texts (balanced)
        "average experience nothing special",
        "some good some bad overall okay",
        "treatment was standard procedure",
        "mixed feelings about visit",
        "facility adequate nothing remarkable",
    ] * 10  # Repeat to have ~150 documents
    
    # Cluster assignments
    cluster_assignments = np.array([0]*50 + [1]*50 + [2]*50)
    
    return texts, cluster_assignments


@pytest.fixture
def engine():
    """Create fresh audit engine instance."""
    return ClusterAuditEngine(model_device=-1)


class TestClusterPreparation:
    """Tests for cluster preparation."""
    
    def test_prepare_clusters(self, engine, sample_data):
        """Test cluster text organization."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        
        assert len(engine.cluster_texts) == 3
        assert 0 in engine.cluster_texts
        assert 1 in engine.cluster_texts
        assert 2 in engine.cluster_texts
    
    def test_cluster_sizes(self, engine, sample_data):
        """Test cluster sizes are correct."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        
        for cluster_id in range(3):
            assert len(engine.cluster_texts[cluster_id]) == 50


class TestModelLoading:
    """Tests for model loading."""
    
    def test_load_sentiment_model(self, engine):
        """Test sentiment model loading."""
        engine.load_sentiment_model()
        assert engine.sentiment_pipeline is not None
    
    def test_load_keybert_model(self, engine):
        """Test KeyBERT model loading."""
        engine.load_keybert_model()
        assert engine.keybert_model is not None


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""
    
    def test_analyze_single_cluster_sentiment(self, engine, sample_data):
        """Test sentiment analysis for single cluster."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        result = engine.analyze_cluster_sentiment(0, batch_size=8)
        
        assert result is not None
        assert 'cluster_id' in result
        assert 'sentiment_distribution' in result
        assert 'sentiment_density' in result
        assert 'dominant_sentiment' in result
    
    def test_sentiment_distribution_sums_to_one(self, engine, sample_data):
        """Test sentiment probabilities sum to 1."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        result = engine.analyze_cluster_sentiment(0)
        distribution = result['sentiment_distribution']
        
        total = sum(distribution.values())
        assert np.isclose(total, 1.0, atol=0.01)
    
    def test_analyze_all_clusters_sentiment(self, engine, sample_data):
        """Test sentiment analysis for all clusters."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        engine.analyze_all_cluster_sentiments()
        
        assert len(engine.cluster_sentiment_results) == 3
        for cluster_id in range(3):
            assert cluster_id in engine.cluster_sentiment_results


class TestZoneIdentification:
    """Tests for zone identification."""
    
    def test_identify_zones(self, engine, sample_data):
        """Test zone classification."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        
        assert len(engine.cluster_zones) == 3
        
        for cluster_id, zone in engine.cluster_zones.items():
            assert 'zone_type' in zone
            assert zone['zone_type'] in ['RED_ZONE', 'GREEN_ZONE', 'NEUTRAL_ZONE']
            assert 'intensity_score' in zone
            assert 'distress_level' in zone
    
    def test_red_zones_filter(self, engine, sample_data):
        """Test red zones are correctly identified."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        
        red_zones = engine.get_red_zones()
        assert isinstance(red_zones, list)
        assert all(isinstance(cid, int) for cid in red_zones)
    
    def test_green_zones_filter(self, engine, sample_data):
        """Test green zones are correctly identified."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        
        green_zones = engine.get_green_zones()
        assert isinstance(green_zones, list)
    
    def test_neutral_zones_filter(self, engine, sample_data):
        """Test neutral zones are correctly identified."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        
        neutral_zones = engine.get_neutral_zones()
        assert isinstance(neutral_zones, list)


class TestVocabularyExtraction:
    """Tests for vocabulary extraction."""
    
    def test_extract_single_vocabulary(self, engine, sample_data):
        """Test vocabulary extraction for single cluster."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_keybert_model()
        
        keywords = engine.extract_cluster_vocabulary(0, n_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Each keyword is a tuple of (keyword, score)
        for keyword, score in keywords:
            assert isinstance(keyword, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1.0
    
    def test_extract_all_vocabularies(self, engine, sample_data):
        """Test vocabulary extraction for all clusters."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_keybert_model()
        
        engine.extract_all_vocabularies(n_keywords=5)
        
        assert len(engine.cluster_vocabularies) == 3


class TestAuditReportGeneration:
    """Tests for audit report generation."""
    
    def test_generate_single_audit_report(self, engine, sample_data):
        """Test single cluster audit report generation."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        engine.load_keybert_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        engine.extract_all_vocabularies()
        
        report = engine.generate_cluster_audit_report(0)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Cluster 0' in report
    
    def test_generate_all_audit_reports(self, engine, sample_data):
        """Test audit report generation for all clusters."""
        texts, assignments = sample_data
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        engine.load_keybert_model()
        
        engine.analyze_all_cluster_sentiments()
        engine.identify_cluster_zones()
        engine.extract_all_vocabularies()
        engine.generate_all_audit_reports()
        
        assert len(engine.cluster_audit_reports) == 3


class TestCompletePipeline:
    """Tests for complete audit pipeline."""
    
    def test_full_audit_pipeline(self, engine, sample_data):
        """Test complete audit pipeline."""
        texts, assignments = sample_data
        
        engine.audit(texts, assignments, n_keywords=5)
        
        # Verify all outputs are present
        assert len(engine.cluster_texts) == 3
        assert len(engine.cluster_sentiment_results) == 3
        assert len(engine.cluster_zones) == 3
        assert len(engine.cluster_vocabularies) == 3
        assert len(engine.cluster_audit_reports) == 3


class TestExportAndSummary:
    """Tests for export and summary functionality."""
    
    def test_export_to_dataframe(self, engine, sample_data):
        """Test exporting audit results to DataFrame."""
        texts, assignments = sample_data
        engine.audit(texts, assignments)
        
        df = engine.export_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'cluster_id' in df.columns
        assert 'dominant_sentiment' in df.columns
        assert 'zone_type' in df.columns
        assert 'top_keywords' in df.columns
    
    def test_get_summary(self, engine, sample_data):
        """Test summary generation."""
        texts, assignments = sample_data
        engine.audit(texts, assignments)
        
        summary = engine.get_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Red Zones' in summary or 'Green Zones' in summary


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_cluster_handling(self, engine):
        """Test handling of empty clusters."""
        texts = ["test"] * 5
        assignments = np.array([0] * 5)
        
        engine.prepare_clusters(texts, assignments)
        
        # Should not crash on valid input
        assert len(engine.cluster_texts) == 1
    
    def test_single_text_cluster(self, engine):
        """Test cluster with single text."""
        texts = ["single text document"]
        assignments = np.array([0])
        
        engine.prepare_clusters(texts, assignments)
        engine.load_sentiment_model()
        
        result = engine.analyze_cluster_sentiment(0)
        assert result['num_texts'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
