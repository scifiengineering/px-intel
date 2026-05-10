# Development Plan: PX-Intel Unsupervised Audit & Predictive System

## 1. Project Vision

The system uses an **Unsupervised-First** approach to map the feedback landscape. We reject pre-defined categories; instead, we allow the data to form its own clusters (via LDA/K-Means). Once the "landscape" is mapped, we deploy the **BERT Family** as high-precision diagnostic tools to explain the causes, predict consequences, and monitor thresholds.

## 2. Technical Architecture & BERT Implementation

### Phase 1: Landscape Discovery (The Foundation)

* **Method:** Latent Dirichlet Allocation (LDA) + K-Means Clustering.
* **Goal:** Discover the administrative and operational "Service Gaps" automatically.
* **Validation:** Use **t-SNE** to visualize the distance between service groups.
* **Divergence Metric:** Dynamically calculate the **Paradox Score** by measuring the Jensen-Shannon Divergence between cluster topic distributions (proving the "18.50" gap is a measured reality, not a fixed target).

### Phase 2: Cluster Auditing (The BERT "Senses")

Once clusters are defined by the unsupervised engine, we apply the following:

* **RoBERTa (Sentiment Intensity):** Calculate the sentiment density within each discovered cluster to identify "Red Zones" (high-intensity distress) vs. "Green Zones" (satisfaction).
* **KeyBERT (Issue Distillation):** Extract the specific vocabularies that define each cluster (e.g., distinguishing between "Staff Attitude" in one cluster and "Waiting Room Temperature" in another).

### Phase 3: Causal Reasoning & "If-Then" Predictions

* **DeBERTa-v3 (Causal Linkage):** Use Natural Language Inference (NLI) to verify the "Why."
* *Task:* Test if [Extracted Keyword] logically entails the [Negative Sentiment] found in that cluster.


* **Secondary Issue Prediction:** Analyze the relationship between clusters. If Cluster A (Staffing) and Cluster B (Safety) share high semantic similarity, the system predicts that resolving issues in A will have a cascading effect on B.

### Phase 4: Adaptive Scheduling & Alert System

* **Computational Scheduling:** Implement a retraining logic triggered by incoming data volume or significant changes in the Paradox Score baseline.
* **Weighting & Thresholds:** Prioritize feedback that falls into high-distress clusters. Trigger **Real-time Alerts** if the sentiment density of a specific administrative cluster drops below the established baseline.

## 3. Implementation Roadmap

| Milestone | Objective | Deliverable |
| --- | --- | --- |
| **M1: Discovery** | Port `Master_Colab` logic to Python scripts. | Unsupervised clustering engine. |
| **M2: Audit** | Integrate RoBERTa and KeyBERT. | Per-cluster sentiment and issue reports. |
| **M3: Reasoning** | Implement DeBERTa causal validation. | "If-Then" predictive analysis logic. |
| **M4: Deployment** | Build Streamlit Dashboard. | Interactive t-SNE landscape with "Deep Dive" audits. |

## 4. Interaction Strategy for AI CLI

* **Standard:** "Load `text_data.csv` and replicate the LDA/K-Means logic. Do not use pre-defined labels."
* **Diagnostic:** "Now that clusters are formed, use **RoBERTa** to audit the sentiment of Cluster 1 only. Tell me the intensity score."
* **Causal:** "Using the keywords found by **KeyBERT**, use **DeBERTa** to verify if 'medication delays' is the primary cause of the distress in this group."

---
