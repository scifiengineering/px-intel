"""
Unit tests for data_loader.py
Tests CSV loading, text normalization, and error handling
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
from data_loader import DataLoader, LoaderStats


class TestDataLoader:
    """Test suite for DataLoader class"""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("id,content,category\n")
            f.write("1,This is a test feedback,general\n")
            f.write("2,Another feedback entry,issue\n")
            f.write("3,UPPERCASE TEXT,issue\n")
            f.write("4,Text with   extra   spaces,general\n")
            f.write("5,,empty\n")
            f.write("6,Special chars: éèê ñ,test\n")
            return f.name
    
    def test_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader("dummy.csv")
        assert loader.filepath.name == "dummy.csv"
        assert loader.encoding == "utf-8"
        assert loader.data is None
    
    def test_normalize_text_lowercase(self):
        """Test text normalization converts to lowercase"""
        result = DataLoader._normalize_text("HELLO WORLD")
        assert result == "hello world"
    
    def test_normalize_text_whitespace(self):
        """Test text normalization collapses whitespace"""
        result = DataLoader._normalize_text("hello    world")
        assert result == "hello world"
    
    def test_normalize_text_unicode(self):
        """Test text normalization handles Unicode"""
        result = DataLoader._normalize_text("Café NAÏVE")
        assert "e" in result or "é" in result
    
    def test_normalize_text_empty(self):
        """Test text normalization handles empty strings"""
        result = DataLoader._normalize_text("")
        assert result == ""
    
    def test_normalize_text_none(self):
        """Test text normalization handles None"""
        result = DataLoader._normalize_text(None)
        assert result == ""
    
    def test_normalize_text_whitespace_only(self):
        """Test text normalization handles whitespace-only strings"""
        result = DataLoader._normalize_text("   \t\n  ")
        assert result == ""
    
    def test_normalize_text_combined(self):
        """Test text normalization with all transformations"""
        result = DataLoader._normalize_text("  HELLO   World  ")
        assert result == "hello world"
    
    def test_load_csv(self, sample_csv):
        """Test loading CSV file"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(stats, LoaderStats)
        assert stats.total_rows > 0
    
    def test_stats_calculation(self, sample_csv):
        """Test statistics calculation"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        assert stats.successful_rows > 0
        assert stats.success_rate > 0
        assert stats.min_length >= 0
        assert stats.max_length >= stats.min_length
    
    def test_empty_row_detection(self, sample_csv):
        """Test detection of empty rows"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        # Should filter out empty row (id=5)
        assert stats.failed_rows > 0
    
    def test_text_normalization_in_load(self, sample_csv):
        """Test that loaded data includes normalized text"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        assert "text_normalized" in df.columns
        # Check that UPPERCASE text was normalized to lowercase
        uppercase_row = df[df['id'] == 3]
        if len(uppercase_row) > 0:
            assert uppercase_row['text_normalized'].values[0] == "uppercase text"
    
    def test_stats_success_rate(self, sample_csv):
        """Test success rate calculation"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        expected_success_rate = (stats.successful_rows / stats.total_rows) * 100
        assert abs(stats.success_rate - expected_success_rate) < 0.01
    
    def test_loader_stores_data(self, sample_csv):
        """Test that loader stores data after loading"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        assert loader.data is not None
        assert isinstance(loader.data, pd.DataFrame)
    
    def test_loader_stores_stats(self, sample_csv):
        """Test that loader stores stats after loading"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        assert loader.stats is not None
        assert isinstance(loader.stats, LoaderStats)
    
    def test_print_stats_no_data(self, capsys):
        """Test print_stats handles no data gracefully"""
        loader = DataLoader("dummy.csv")
        loader.print_stats()
        
        captured = capsys.readouterr()
        assert "No statistics available" in captured.out
    
    def test_print_stats_with_data(self, sample_csv, capsys):
        """Test print_stats outputs correctly"""
        loader = DataLoader(sample_csv)
        loader.load(text_column="content")
        loader.print_stats()
        
        captured = capsys.readouterr()
        assert "Data Loading Statistics" in captured.out
        assert "Text Length Statistics" in captured.out
    
    def test_special_characters_handling(self, sample_csv):
        """Test handling of special characters"""
        loader = DataLoader(sample_csv)
        df, stats = loader.load(text_column="content")
        
        # Should have successfully loaded row with special chars
        assert stats.successful_rows >= 5
    
    def test_multiple_loads(self, sample_csv):
        """Test loading multiple times"""
        loader = DataLoader(sample_csv)
        df1, stats1 = loader.load(text_column="content")
        df2, stats2 = loader.load(text_column="content")
        
        # Results should be identical
        assert stats1.successful_rows == stats2.successful_rows
        assert len(df1) == len(df2)
    
    def test_stats_dataclass(self):
        """Test LoaderStats dataclass"""
        stats = LoaderStats(
            total_rows=100,
            successful_rows=95,
            failed_rows=5,
            empty_rows=0,
            success_rate=95.0,
            min_length=10,
            max_length=500,
            avg_length=250.5
        )
        
        assert stats.total_rows == 100
        assert stats.success_rate == 95.0
        assert stats.min_length < stats.max_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
