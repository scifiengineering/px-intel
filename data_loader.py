"""
Robust CSV Data Loader for Hospital Feedback Analysis
Handles mixed line terminators, UTF-8 encoding, and large text entries
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LoaderStats:
    """Statistics about the data loading process"""
    total_rows: int
    successful_rows: int
    failed_rows: int
    empty_rows: int
    success_rate: float
    min_length: int
    max_length: int
    avg_length: float


class DataLoader:
    """
    Robust data loader for hospital feedback CSV files.
    
    Handles:
    - Mixed line terminators (CRLF, LF, CR)
    - UTF-8 encoding with fallback
    - Unicode normalization (NFKD)
    - Text cleaning (lowercase, whitespace normalization)
    - Error handling with detailed reporting
    """
    
    def __init__(self, filepath: str, encoding: str = "utf-8"):
        """
        Initialize DataLoader
        
        Args:
            filepath: Path to CSV file
            encoding: Text encoding (default: utf-8)
        """
        self.filepath = Path(filepath)
        self.encoding = encoding
        self.data = None
        self.stats = None
        self.errors = []
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text: lowercase, Unicode NFKD, whitespace collapse
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Unicode normalization (NFKD form)
        text = unicodedata.normalize("NFKD", text)
        
        # Collapse whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def load(self, text_column: str = "content") -> Tuple[pd.DataFrame, LoaderStats]:
        """
        Load and process CSV file with robust error handling
        
        Args:
            text_column: Name of column containing feedback text
            
        Returns:
            Tuple of (processed_dataframe, statistics)
        """
        try:
            # Try loading with python engine (handles mixed line terminators)
            df = pd.read_csv(
                self.filepath,
                engine="python",
                encoding=self.encoding,
                on_bad_lines="skip",
                dtype={text_column: str}
            )
        except (UnicodeDecodeError, LookupError):
            # Fallback: use latin-1 encoding
            df = pd.read_csv(
                self.filepath,
                engine="python",
                encoding="latin-1",
                on_bad_lines="skip",
                dtype={text_column: str},
                errors="replace"
            )
            self.encoding = "latin-1"
        
        total_rows = len(df)
        
        # Filter empty rows
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != ""]
        empty_rows = initial_count - len(df)
        
        # Normalize text
        if text_column in df.columns:
            df["text_normalized"] = df[text_column].apply(self._normalize_text)
        
        # Calculate statistics
        text_lengths = df[text_column].str.len()
        normalized_lengths = df["text_normalized"].str.len()
        
        successful_rows = len(df)
        failed_rows = total_rows - successful_rows
        
        self.stats = LoaderStats(
            total_rows=total_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            empty_rows=empty_rows,
            success_rate=round(100 * successful_rows / total_rows, 2) if total_rows > 0 else 0,
            min_length=int(text_lengths.min()) if len(text_lengths) > 0 else 0,
            max_length=int(text_lengths.max()) if len(text_lengths) > 0 else 0,
            avg_length=float(text_lengths.mean()) if len(text_lengths) > 0 else 0
        )
        
        self.data = df
        return df, self.stats
    
    def save_processed_data(self, output_dir: str = "data") -> dict:
        """
        Save processed data in multiple formats
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        files_saved = {}
        
        # Save as pickle (fast loading)
        pkl_path = output_path / "processed_feedback.pkl"
        self.data.to_pickle(pkl_path)
        files_saved["pickle"] = str(pkl_path)
        
        # Save as CSV
        csv_path = output_path / "processed_feedback.csv"
        self.data.to_csv(csv_path, index=False)
        files_saved["csv"] = str(csv_path)
        
        # Save as JSON
        json_path = output_path / "processed_feedback.json"
        self.data.to_json(json_path, orient="records", indent=2)
        files_saved["json"] = str(json_path)
        
        return files_saved
    
    def print_stats(self):
        """Print loading statistics"""
        if self.stats is None:
            print("No statistics available. Call load() first.")
            return
        
        print(f"{'='*60}")
        print(f"Data Loading Statistics")
        print(f"{'='*60}")
        print(f"Total rows in file:       {self.stats.total_rows:,}")
        print(f"Successfully loaded:      {self.stats.successful_rows:,}")
        print(f"Failed/empty rows:        {self.stats.failed_rows:,}")
        print(f"Success rate:             {self.stats.success_rate}%")
        print(f"{'='*60}")
        print(f"Text Length Statistics")
        print(f"{'='*60}")
        print(f"Minimum length:           {self.stats.min_length} chars")
        print(f"Maximum length:           {self.stats.max_length} chars")
        print(f"Average length:           {self.stats.avg_length:.1f} chars")
        print(f"{'='*60}")


def main():
    """Example usage of DataLoader"""
    loader = DataLoader("text_data.csv")
    df, stats = loader.load(text_column="content")
    loader.print_stats()
    
    # Display sample
    print("\nFirst 5 entries:")
    print(df.head())
    
    # Save processed data
    saved_files = loader.save_processed_data()
    print(f"\nProcessed data saved to:")
    for format_type, path in saved_files.items():
        print(f"  {format_type}: {path}")


if __name__ == "__main__":
    main()
