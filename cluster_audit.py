"""
M2: Cluster Auditing Functions
Per-cluster sentiment analysis, zone identification (Red/Green), and vocabulary extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
from keybert import KeyBERT
import warnings

warnings.filterwarnings('ignore')


class ClusterAuditEngine:
    """
    Audit clusters using sentiment analysis, zone identification, and KeyBERT extraction.
    
    Methods:
    1. Load cluster assignments and text data
    2. Compute sentiment density per cluster (RoBERTa)
    3. Identify Red/Green/Neutral zones based on sentiment
    4. Extract top keywords per cluster (KeyBERT)
    5. Generate per-cluster audit reports
    
    Attributes:
        cluster_texts (dict): List of texts per cluster
        cluster_sentiment_results (dict): Sentiment analysis per cluster
        cluster_zones (dict): Zone type (RED/GREEN/NEUTRAL) per cluster
        cluster_vocabularies (dict): Top keywords per cluster
        cluster_audit_reports (dict): Human-readable audit reports
    """
    
    def __init__(self, model_device: int = -1):
        """
        Initialize cluster audit engine.
        
        Args:
            model_device: Device for models (-1=CPU, 0+=GPU)
        """
        self.model_device = model_device
        self.sentiment_pipeline = None
        self.keybert_model = None
        self.cluster_texts = {}
        self.cluster_sentiment_results = {}
        self.cluster_zones = {}
        self.cluster_vocabularies = {}
        self.cluster_audit_reports = {}
        self.cluster_statistics = {}
    
    # ========================================================================
    # Model Loading (Cached in Streamlit context)
    # ========================================================================
    
    def load_sentiment_model(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Load RoBERTa sentiment analysis model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        if self.sentiment_pipeline is None:
            print(f"Loading sentiment model: {model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=self.model_device
            )
    
    def load_keybert_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load KeyBERT model for vocabulary extraction.
        
        Args:
            model_name: HuggingFace model identifier
        """
        if self.keybert_model is None:
            print(f"Loading KeyBERT model: {model_name}")
            self.keybert_model = KeyBERT(model=model_name, device=self.model_device)
    
    # ========================================================================
    # Cluster Preparation
    # ========================================================================
    
    def prepare_clusters(self, texts: List[str], cluster_assignments: np.ndarray) -> None:
        """
        Organize texts by cluster.
        
        Args:
            texts: List of text documents
            cluster_assignments: Cluster ID per document
        """
        self.cluster_texts = {}
        
        for cluster_id in np.unique(cluster_assignments):
            mask = cluster_assignments == cluster_id
            cluster_text_list = np.array(texts)[mask].tolist()
            self.cluster_texts[int(cluster_id)] = cluster_text_list
        
        print(f"Organized {len(texts)} texts into {len(self.cluster_texts)} clusters")
    
    # ========================================================================
    # Sentiment Analysis per Cluster
    # ========================================================================
    
    def analyze_cluster_sentiment(self, cluster_id: int, batch_size: int = 32) -> Dict:
        """
        Analyze sentiment for all texts in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            batch_size: Batch size for sentiment analysis
            
        Returns:
            Dictionary with sentiment distribution and statistics
        """
        if cluster_id not in self.cluster_texts:
            return {}
        
        texts = self.cluster_texts[cluster_id]
        if not texts:
            return {}
        
        # Truncate texts to 512 chars (RoBERTa max)
        texts_truncated = [t[:512] for t in texts]
        
        try:
            # Batch sentiment analysis
            sentiments = self.sentiment_pipeline(texts_truncated, batch_size=batch_size)
            
            # Parse results
            sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
            sentiment_scores = []
            
            for result in sentiments:
                label = result['label'].upper()
                score = result['score']
                sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
                sentiment_scores.append(score)
            
            # Calculate statistics
            total_texts = len(texts_truncated)
            sentiment_distribution = {
                label: count / total_texts for label, count in sentiment_counts.items()
            }
            
            # Sentiment density (concentration of most common sentiment)
            max_proportion = max(sentiment_distribution.values())
            sentiment_density = max_proportion  # Higher = more concentrated
            
            # Average confidence score
            avg_confidence = np.mean(sentiment_scores)
            
            result = {
                'cluster_id': int(cluster_id),
                'num_texts': total_texts,
                'sentiment_distribution': sentiment_distribution,
                'sentiment_density': float(sentiment_density),
                'avg_confidence': float(avg_confidence),
                'dominant_sentiment': max(sentiment_distribution, key=sentiment_distribution.get)
            }
            
            self.cluster_sentiment_results[cluster_id] = result
            return result
            
        except Exception as e:
            print(f"Error analyzing cluster {cluster_id}: {e}")
            return {}
    
    def analyze_all_cluster_sentiments(self, batch_size: int = 32) -> None:
        """
        Analyze sentiment for all clusters.
        
        Args:
            batch_size: Batch size for sentiment analysis
        """
        print(f"Analyzing sentiment for {len(self.cluster_texts)} clusters...")
        
        for cluster_id in self.cluster_texts.keys():
            self.analyze_cluster_sentiment(cluster_id, batch_size=batch_size)
    
    # ========================================================================
    # Zone Identification (Red/Green/Neutral)
    # ========================================================================
    
    def identify_cluster_zones(self) -> None:
        """
        Classify clusters as Red/Green/Neutral zones based on sentiment.
        
        Zone Rules:
        - RED ZONE: Dominant sentiment is NEGATIVE OR sentiment_density >= 0.60
        - GREEN ZONE: Dominant sentiment is POSITIVE OR sentiment_density >= 0.60
        - NEUTRAL ZONE: Dominant sentiment is NEUTRAL OR sentiment_density < 0.50
        """
        self.cluster_zones = {}
        
        for cluster_id, result in self.cluster_sentiment_results.items():
            dominant = result['dominant_sentiment']
            density = result['sentiment_density']
            
            if dominant == 'NEGATIVE' or (dominant != 'NEUTRAL' and density >= 0.60):
                zone_type = 'RED_ZONE'
            elif dominant == 'POSITIVE' or (dominant != 'NEUTRAL' and density >= 0.60):
                zone_type = 'GREEN_ZONE'
            else:
                zone_type = 'NEUTRAL_ZONE'
            
            self.cluster_zones[cluster_id] = {
                'zone_type': zone_type,
                'intensity_score': float(density),
                'distress_level': 'High' if zone_type == 'RED_ZONE' else 'Low'
            }
    
    def get_red_zones(self) -> List[int]:
        """
        Get list of Red Zone cluster IDs (high distress).
        
        Returns:
            List of cluster IDs with high negative sentiment
        """
        return [cid for cid, zone in self.cluster_zones.items() if zone['zone_type'] == 'RED_ZONE']
    
    def get_green_zones(self) -> List[int]:
        """
        Get list of Green Zone cluster IDs (high satisfaction).
        
        Returns:
            List of cluster IDs with high positive sentiment
        """
        return [cid for cid, zone in self.cluster_zones.items() if zone['zone_type'] == 'GREEN_ZONE']
    
    def get_neutral_zones(self) -> List[int]:
        """
        Get list of Neutral Zone cluster IDs.
        
        Returns:
            List of cluster IDs with balanced sentiment
        """
        return [cid for cid, zone in self.cluster_zones.items() if zone['zone_type'] == 'NEUTRAL_ZONE']
    
    # ========================================================================
    # Vocabulary Extraction (KeyBERT)
    # ========================================================================
    
    def extract_cluster_vocabulary(self, cluster_id: int, n_keywords: int = 10, 
                                  top_n_texts: int = 100) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a cluster using KeyBERT.
        
        Args:
            cluster_id: Cluster identifier
            n_keywords: Number of keywords to extract
            top_n_texts: Use top N longest texts for extraction (balances speed & quality)
            
        Returns:
            List of (keyword, score) tuples
        """
        if cluster_id not in self.cluster_texts:
            return []
        
        texts = self.cluster_texts[cluster_id]
        if not texts:
            return []
        
        # Use top N longest texts (usually more informative)
        texts_sorted = sorted(texts, key=len, reverse=True)[:top_n_texts]
        cluster_text_combined = ' '.join(texts_sorted)
        
        try:
            # Extract keywords using KeyBERT
            keywords = self.keybert_model.extract_keywords(
                cluster_text_combined,
                language="english",
                stop_words="english",
                top_n=n_keywords,
                use_maxsum=True,      # Diversity boost
                keyphrase_ngram_range=(1, 3)  # 1-3 word phrases
            )
            
            self.cluster_vocabularies[cluster_id] = keywords
            return keywords
            
        except Exception as e:
            print(f"Error extracting vocabulary for cluster {cluster_id}: {e}")
            return []
    
    def extract_all_vocabularies(self, n_keywords: int = 10) -> None:
        """
        Extract vocabulary for all clusters.
        
        Args:
            n_keywords: Number of keywords per cluster
        """
        print(f"Extracting vocabularies for {len(self.cluster_texts)} clusters...")
        
        for cluster_id in self.cluster_texts.keys():
            self.extract_cluster_vocabulary(cluster_id, n_keywords=n_keywords)
    
    # ========================================================================
    # Audit Report Generation
    # ========================================================================
    
    def generate_cluster_audit_report(self, cluster_id: int) -> str:
        """
        Generate human-readable audit report for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Formatted audit report string
        """
        if cluster_id not in self.cluster_sentiment_results:
            return ""
        
        sentiment_result = self.cluster_sentiment_results[cluster_id]
        zone_info = self.cluster_zones.get(cluster_id, {})
        vocabulary = self.cluster_vocabularies.get(cluster_id, [])
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                Cluster {cluster_id} Audit Report                ║
╚══════════════════════════════════════════════════════════════╝

📊 SENTIMENT ANALYSIS
  Dominant Sentiment: {sentiment_result['dominant_sentiment']}
  Sentiment Distribution:
    - Positive: {sentiment_result['sentiment_distribution']['POSITIVE']:.1%}
    - Neutral:  {sentiment_result['sentiment_distribution']['NEUTRAL']:.1%}
    - Negative: {sentiment_result['sentiment_distribution']['NEGATIVE']:.1%}
  Sentiment Density: {sentiment_result['sentiment_density']:.3f} (concentration)
  Average Confidence: {sentiment_result['avg_confidence']:.3f}
  Total Texts: {sentiment_result['num_texts']}

🎯 ZONE CLASSIFICATION
  Zone Type: {zone_info.get('zone_type', 'UNKNOWN')}
  Intensity Score: {zone_info.get('intensity_score', 0):.3f}
  Distress Level: {zone_info.get('distress_level', 'Unknown')}

�� TOP KEYWORDS
"""
        
        if vocabulary:
            for i, (keyword, score) in enumerate(vocabulary[:10], 1):
                report += f"  {i:2d}. {keyword:<30} ({score:.3f})\n"
        else:
            report += "  (No keywords extracted)\n"
        
        report += "\n📝 SUMMARY\n"
        if zone_info.get('zone_type') == 'RED_ZONE':
            report += f"  🚨 High-distress cluster with {sentiment_result['sentiment_distribution']['NEGATIVE']:.1%} "
            report += "negative sentiment.\n"
            report += "  Priority: Immediate action needed to address issues.\n"
        elif zone_info.get('zone_type') == 'GREEN_ZONE':
            report += f"  ✨ High-satisfaction cluster with {sentiment_result['sentiment_distribution']['POSITIVE']:.1%} "
            report += "positive sentiment.\n"
            report += "  Priority: Maintain current practices and potentially replicate.\n"
        else:
            report += "  ⚖️  Balanced sentiment distribution across feedback.\n"
            report += "  Priority: Monitor for changes; investigate dominant themes.\n"
        
        self.cluster_audit_reports[cluster_id] = report
        return report
    
    def generate_all_audit_reports(self) -> None:
        """Generate audit reports for all clusters."""
        print(f"Generating audit reports for {len(self.cluster_texts)} clusters...")
        
        for cluster_id in self.cluster_texts.keys():
            self.generate_cluster_audit_report(cluster_id)
    
    # ========================================================================
    # Complete Audit Pipeline
    # ========================================================================
    
    def audit(self, texts: List[str], cluster_assignments: np.ndarray, 
              n_keywords: int = 10) -> None:
        """
        Complete audit pipeline: sentiment → zones → vocabulary → reports.
        
        Args:
            texts: List of text documents
            cluster_assignments: Cluster ID per document
            n_keywords: Number of keywords per cluster
        """
        print("\n[M2] Starting Cluster Audit Pipeline...")
        
        # Load models
        print("\n[1/5] Loading models...")
        self.load_sentiment_model()
        self.load_keybert_model()
        
        # Prepare clusters
        print("\n[2/5] Preparing clusters...")
        self.prepare_clusters(texts, cluster_assignments)
        
        # Sentiment analysis
        print("\n[3/5] Sentiment analysis...")
        self.analyze_all_cluster_sentiments()
        self.identify_cluster_zones()
        
        # Vocabulary extraction
        print("\n[4/5] Vocabulary extraction...")
        self.extract_all_vocabularies(n_keywords=n_keywords)
        
        # Audit reports
        print("\n[5/5] Generating audit reports...")
        self.generate_all_audit_reports()
        
        print("\n✓ Cluster audit complete!")
    
    # ========================================================================
    # Export & Summary
    # ========================================================================
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export audit results as DataFrame.
        
        Returns:
            DataFrame with per-cluster audit results
        """
        rows = []
        
        for cluster_id in sorted(self.cluster_sentiment_results.keys()):
            sentiment = self.cluster_sentiment_results[cluster_id]
            zone = self.cluster_zones.get(cluster_id, {})
            vocab = self.cluster_vocabularies.get(cluster_id, [])
            
            top_keywords = ', '.join([kw for kw, _ in vocab[:5]]) if vocab else ''
            
            row = {
                'cluster_id': cluster_id,
                'num_texts': sentiment['num_texts'],
                'dominant_sentiment': sentiment['dominant_sentiment'],
                'sentiment_density': sentiment['sentiment_density'],
                'avg_confidence': sentiment['avg_confidence'],
                'zone_type': zone.get('zone_type'),
                'intensity_score': zone.get('intensity_score'),
                'top_keywords': top_keywords
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> str:
        """
        Generate summary of all cluster audits.
        
        Returns:
            Summary string
        """
        red_zones = self.get_red_zones()
        green_zones = self.get_green_zones()
        neutral_zones = self.get_neutral_zones()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           Cluster Audit Summary (M2)                        ║
╚══════════════════════════════════════════════════════════════╝

📊 ZONE DISTRIBUTION
  Red Zones (High Distress):   {len(red_zones)} clusters → {red_zones}
  Green Zones (Satisfaction):  {len(green_zones)} clusters → {green_zones}
  Neutral Zones:               {len(neutral_zones)} clusters → {neutral_zones}

🚨 RED ZONES (Immediate Action Needed)
"""
        
        for cluster_id in red_zones:
            result = self.cluster_sentiment_results[cluster_id]
            vocab = self.cluster_vocabularies.get(cluster_id, [])
            top_kw = vocab[0][0] if vocab else 'N/A'
            summary += f"  Cluster {cluster_id}: {result['num_texts']} texts | "
            summary += f"{result['sentiment_distribution']['NEGATIVE']:.1%} negative | "
            summary += f"Top issue: {top_kw}\n"
        
        summary += "\n✨ GREEN ZONES (Maintain Current Practices)\n"
        for cluster_id in green_zones:
            result = self.cluster_sentiment_results[cluster_id]
            vocab = self.cluster_vocabularies.get(cluster_id, [])
            top_kw = vocab[0][0] if vocab else 'N/A'
            summary += f"  Cluster {cluster_id}: {result['num_texts']} texts | "
            summary += f"{result['sentiment_distribution']['POSITIVE']:.1%} positive | "
            summary += f"Top feature: {top_kw}\n"
        
        return summary
