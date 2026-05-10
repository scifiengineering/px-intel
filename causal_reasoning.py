"""
M3: Causal Reasoning & Predictions
Causal linkage validation, cross-cluster similarity, and cascade effect prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class CausalReasoningEngine:
    """
    Causal reasoning and prediction engine for discovered clusters.
    
    Methods:
    1. Causal linkage validation (keyword → sentiment entailment via NLI)
    2. Cross-cluster similarity analysis (semantic similarity between clusters)
    3. Cascade effect prediction (if Cluster A is fixed, predict impact on B)
    4. Generate "If-Then" predictive statements
    
    Attributes:
        deberta_model: Zero-shot NLI model (DeBERTa)
        cluster_lda_features: LDA topic distributions per cluster
        cluster_vocabularies: Top keywords per cluster
        causal_validations: Keyword entailment results
        cluster_similarities: Pairwise cluster similarities
        cascade_predictions: Cross-cluster impact predictions
    """
    
    def __init__(self, model_device: int = -1):
        """
        Initialize causal reasoning engine.
        
        Args:
            model_device: Device for models (-1=CPU, 0+=GPU)
        """
        self.model_device = model_device
        self.deberta_model = None
        self.cluster_lda_features = {}
        self.cluster_vocabularies = {}
        self.cluster_sentiment_results = {}
        self.causal_validations = {}
        self.cluster_similarities = {}
        self.cascade_predictions = {}
    
    # ========================================================================
    # Model Loading
    # ========================================================================
    
    def load_nli_model(self, model_name: str = "microsoft/deberta-v3-large"):
        """
        Load DeBERTa zero-shot NLI model for causal validation.
        
        Args:
            model_name: HuggingFace model identifier
        """
        if self.deberta_model is None:
            print(f"Loading NLI model: {model_name}")
            self.deberta_model = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.model_device
            )
    
    # ========================================================================
    # Causal Linkage Validation
    # ========================================================================
    
    def validate_causal_linkage(self, keyword: str, sentiment_label: str = "NEGATIVE",
                               confidence_threshold: float = 0.50) -> Dict:
        """
        Validate if a keyword entails a sentiment using NLI.
        
        Uses hypothesis: "[Keyword] causes [sentiment]" entailment
        If entailment > threshold, the link is confirmed causal.
        
        Args:
            keyword: Extracted keyword/issue from cluster
            sentiment_label: Sentiment to validate (e.g., "NEGATIVE")
            confidence_threshold: Minimum confidence (0-1)
            
        Returns:
            Dictionary with validation result
        """
        if not self.deberta_model:
            return {}
        
        # Create NLI text and hypothesis
        text = f"Patients reported: {keyword}"
        hypothesis = f"This leads to {sentiment_label.lower()} sentiment"
        
        try:
            result = self.deberta_model(
                text,
                [hypothesis],
                multi_class=False,
                truncation=True,
                max_length=512
            )
            
            # Extract entailment score
            label = result['labels'][0]
            score = result['scores'][0]
            
            entailment_confirmed = label == 'ENTAILMENT' and score >= confidence_threshold
            
            validation_result = {
                'keyword': keyword,
                'sentiment': sentiment_label,
                'hypothesis': hypothesis,
                'label': label,
                'confidence': float(score),
                'causal_confirmed': entailment_confirmed,
                'interpretation': f"'{keyword}' {'ENTAILS' if entailment_confirmed else 'does not entail'} "
                                 f"{sentiment_label} ({score:.2%})"
            }
            
            return validation_result
            
        except Exception as e:
            print(f"Error validating causal linkage for '{keyword}': {e}")
            return {}
    
    def validate_cluster_causal_links(self, cluster_id: int, 
                                     cluster_vocabulary: List[Tuple[str, float]],
                                     sentiment: str = "NEGATIVE",
                                     top_n: int = 5) -> List[Dict]:
        """
        Validate causal links for top keywords in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            cluster_vocabulary: List of (keyword, score) tuples
            sentiment: Sentiment type to validate
            top_n: Number of top keywords to validate
            
        Returns:
            List of validation results for top keywords
        """
        validations = []
        
        for keyword, keyword_score in cluster_vocabulary[:top_n]:
            validation = self.validate_causal_linkage(keyword, sentiment_label=sentiment)
            if validation:
                validation['keyword_score'] = float(keyword_score)
                validations.append(validation)
        
        self.causal_validations[cluster_id] = validations
        return validations
    
    # ========================================================================
    # Cross-Cluster Similarity
    # ========================================================================
    
    def compute_cluster_similarities(self, cluster_lda_features: Dict[int, np.ndarray]) -> None:
        """
        Compute semantic similarity between all cluster pairs.
        
        Uses cosine similarity on LDA topic distributions.
        High similarity = clusters share similar issues/themes
        
        Args:
            cluster_lda_features: LDA features per cluster
                                Format: {cluster_id: np.ndarray of shape (n_docs, n_topics)}
        """
        self.cluster_lda_features = cluster_lda_features
        cluster_ids = sorted(cluster_lda_features.keys())
        n_clusters = len(cluster_ids)
        
        # Compute mean LDA vector per cluster
        cluster_vectors = {}
        for cid in cluster_ids:
            features = cluster_lda_features[cid]
            # Average LDA features for the cluster
            mean_vector = np.mean(features, axis=0)
            cluster_vectors[cid] = mean_vector.reshape(1, -1)
        
        # Compute pairwise similarities
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cid1 = cluster_ids[i]
                cid2 = cluster_ids[j]
                
                vec1 = cluster_vectors[cid1]
                vec2 = cluster_vectors[cid2]
                
                # Cosine similarity
                similarity = cosine_similarity(vec1, vec2)[0, 0]
                
                self.cluster_similarities[(cid1, cid2)] = float(similarity)
    
    def get_most_similar_clusters(self, cluster_id: int, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Get most similar clusters to a given cluster.
        
        Args:
            cluster_id: Reference cluster ID
            top_n: Number of similar clusters to return
            
        Returns:
            List of (similar_cluster_id, similarity_score) tuples
        """
        similar = []
        
        for (c1, c2), similarity in self.cluster_similarities.items():
            if c1 == cluster_id:
                similar.append((c2, similarity))
            elif c2 == cluster_id:
                similar.append((c1, similarity))
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:top_n]
    
    # ========================================================================
    # Cascade Effect Prediction
    # ========================================================================
    
    def predict_cascade_effects(self, cluster_id: int,
                               cluster_sentiments: Dict[int, Dict],
                               similarity_threshold: float = 0.60) -> List[Dict]:
        """
        Predict cascading effects if cluster issues are resolved.
        
        Logic:
        - Find similar clusters (similarity > threshold)
        - Predict sentiment improvement based on shared factors
        - Generate "If-Then" statements
        
        Args:
            cluster_id: Source cluster to fix
            cluster_sentiments: Sentiment results per cluster
            similarity_threshold: Min similarity for cascade prediction
            
        Returns:
            List of cascade predictions
        """
        predictions = []
        
        # Find similar clusters
        similar_clusters = self.get_most_similar_clusters(cluster_id, top_n=10)
        
        source_sentiment = cluster_sentiments.get(cluster_id, {}).get('dominant_sentiment', 'UNKNOWN')
        
        for similar_cid, similarity in similar_clusters:
            if similarity < similarity_threshold:
                continue
            
            target_sentiment = cluster_sentiments.get(similar_cid, {}).get('dominant_sentiment', 'UNKNOWN')
            
            # Predict improvement
            improvement_likelihood = similarity  # Higher similarity = higher cascade likelihood
            
            prediction = {
                'source_cluster': int(cluster_id),
                'target_cluster': int(similar_cid),
                'similarity': float(similarity),
                'source_sentiment': source_sentiment,
                'target_sentiment': target_sentiment,
                'cascade_likelihood': float(improvement_likelihood),
                'cascade_interpretation': f"Fixing Cluster {cluster_id} ({source_sentiment}) "
                                         f"has ~{improvement_likelihood:.0%} chance to improve "
                                         f"Cluster {similar_cid} ({target_sentiment})",
            }
            
            predictions.append(prediction)
        
        self.cascade_predictions[cluster_id] = predictions
        return predictions
    
    # ========================================================================
    # If-Then Statement Generation
    # ========================================================================
    
    def generate_if_then_statements(self, cluster_id: int,
                                   cluster_vocabulary: List[Tuple[str, float]],
                                   cluster_sentiments: Dict[int, Dict],
                                   sentiment: str = "NEGATIVE") -> List[str]:
        """
        Generate interpretable "If-Then" causal statements.
        
        Format:
        "If [keyword] is addressed, then [cluster] sentiment likely improves
         because it shares [similarity]% factors with [similar_clusters]"
        
        Args:
            cluster_id: Cluster ID
            cluster_vocabulary: Keywords for cluster
            cluster_sentiments: Sentiment data per cluster
            sentiment: Sentiment type
            
        Returns:
            List of If-Then statements
        """
        statements = []
        
        # Get top keywords
        top_keywords = [kw for kw, _ in cluster_vocabulary[:3]]
        
        # Get similar clusters and cascade predictions
        cascades = self.cascade_predictions.get(cluster_id, [])
        
        if not cascades:
            # Simple If-Then without cascades
            for keyword in top_keywords:
                stmt = f"If '{keyword}' issues in Cluster {cluster_id} are addressed, " \
                       f"then sentiment improvements should follow."
                statements.append(stmt)
        else:
            # If-Then with cascade predictions
            for keyword in top_keywords:
                cascade_targets = [f"Cluster {p['target_cluster']}" for p in cascades[:2]]
                cascade_text = ' and '.join(cascade_targets)
                
                stmt = f"If '{keyword}' issues in Cluster {cluster_id} are resolved, " \
                       f"then direct sentiment improves, and {cascade_text} may also benefit."
                statements.append(stmt)
        
        return statements
    
    # ========================================================================
    # Complete Reasoning Pipeline
    # ========================================================================
    
    def reason(self, cluster_lda_features: Dict[int, np.ndarray],
               cluster_vocabularies: Dict[int, List[Tuple[str, float]]],
               cluster_sentiments: Dict[int, Dict]) -> None:
        """
        Complete causal reasoning pipeline.
        
        Args:
            cluster_lda_features: LDA features per cluster
            cluster_vocabularies: Vocabularies per cluster
            cluster_sentiments: Sentiment results per cluster
        """
        print("\n[M3] Starting Causal Reasoning Pipeline...")
        
        # Load model
        print("\n[1/4] Loading NLI model...")
        self.load_nli_model()
        
        # Compute cluster similarities
        print("\n[2/4] Computing cluster similarities...")
        self.compute_cluster_similarities(cluster_lda_features)
        
        # Validate causal links
        print("\n[3/4] Validating causal linkages...")
        for cluster_id, vocabulary in cluster_vocabularies.items():
            self.validate_cluster_causal_links(cluster_id, vocabulary, top_n=5)
        
        # Predict cascades
        print("\n[4/4] Predicting cascade effects...")
        self.cascade_predictions = {}
        for cluster_id in cluster_vocabularies.keys():
            self.predict_cascade_effects(cluster_id, cluster_sentiments)
        
        print("\n✓ Causal reasoning complete!")
    
    # ========================================================================
    # Export & Summary
    # ========================================================================
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export causal reasoning results as DataFrame.
        
        Returns:
            DataFrame with per-cluster reasoning results
        """
        rows = []
        
        for cluster_id, validations in self.causal_validations.items():
            cascades = self.cascade_predictions.get(cluster_id, [])
            
            # Get causal keywords
            causal_keywords = [v['keyword'] for v in validations if v.get('causal_confirmed')]
            causal_text = ', '.join(causal_keywords) if causal_keywords else 'None confirmed'
            
            # Get cascade targets
            cascade_targets = [f"C{p['target_cluster']}" for p in cascades[:3]]
            cascade_text = ', '.join(cascade_targets) if cascade_targets else 'None'
            
            row = {
                'cluster_id': cluster_id,
                'num_keywords_validated': len(validations),
                'num_causal_confirmed': len(causal_keywords),
                'causal_keywords': causal_text,
                'cascade_targets': cascade_text,
                'num_cascades': len(cascades),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> str:
        """
        Generate summary of causal reasoning results.
        
        Returns:
            Summary string
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║            Causal Reasoning Summary (M3)                    ║
╚══════════════════════════════════════════════════════════════╝

🔗 CAUSAL LINKAGE VALIDATION
  Total Clusters Analyzed: {len(self.causal_validations)}
"""
        
        total_validations = sum(len(v) for v in self.causal_validations.values())
        total_causal = sum(len([x for x in v if x.get('causal_confirmed')]) 
                          for v in self.causal_validations.values())
        
        summary += f"  Total Keywords Validated: {total_validations}\n"
        summary += f"  Causal Links Confirmed: {total_causal} ({total_causal/total_validations*100:.1f}%)\n"
        
        summary += "\n🌉 INTER-CLUSTER SIMILARITIES (Cascades)\n"
        
        for cluster_id, cascades in self.cascade_predictions.items():
            if cascades:
                top_cascade = cascades[0]
                summary += f"  Cluster {cluster_id} → Cluster {top_cascade['target_cluster']}: "
                summary += f"{top_cascade['cascade_likelihood']:.0%} likelihood\n"
        
        summary += f"\n⚡ TOP CASCADE OPPORTUNITIES\n"
        
        # Rank all cascades by likelihood
        all_cascades = []
        for cid, cascades in self.cascade_predictions.items():
            all_cascades.extend(cascades)
        
        all_cascades.sort(key=lambda x: x['cascade_likelihood'], reverse=True)
        
        for cascade in all_cascades[:5]:
            summary += f"  Cluster {cascade['source_cluster']} → {cascade['target_cluster']}: "
            summary += f"{cascade['cascade_likelihood']:.0%}\n"
        
        return summary
