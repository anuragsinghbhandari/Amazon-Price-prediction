
---

#  **Amazon items price predictions**

---

## **1. Executive Summary**

Our solution predicts product prices using a multimodal learning approach that integrates textual and numeric signals from catalog data.
We combined **TF-IDF** and **SentenceTransformer embeddings** with **unit-normalized numeric features**, training a **LightGBM model** optimized using a **custom SMAPE-based objective**.
This approach balances semantic understanding with numerical precision for improved generalization.

---

## **2. Methodology Overview**

### **2.1 Problem Analysis**

We framed the challenge as a **supervised regression problem**: predicting product prices from product metadata and textual descriptions.

**Key Observations:**

* Product prices exhibit a **right-skewed distribution**, so `log1p(price)` helps stabilize training.
* Text data (item names, bullet points) reveal strong category and quantity signals.
* Extracting and normalizing “Value” and “Unit” fields provides significant numeric cues.
* Missing or ambiguous units (e.g., “pack”, “count”) required categorization into `weight`, `volume`, or `count`.

---

### **2.2 Solution Strategy**

**Approach Type:** Hybrid (Text + Numeric)
**Core Innovation:**
Fusion of **semantic text embeddings (BERT)**, **statistical text representation (TF-IDF)**, and **normalized numeric-unit features**, optimized using a **custom SMAPE objective** that better aligns model learning with the evaluation metric.

---

## **3. Model Architecture**

### **3.1 Architecture Overview**

**Data Flow Diagram:**

```
┌──────────────────────┐
│  catalog_content     │
└──────────┬───────────┘
           │
    ┌──────▼───────┐
    │ TF-IDF (10k) │───┐
    └──────────────┘   │
                       │   ┌─────────────┐
                       ├──▶│ SVD (256D)  │
                       │   └─────────────┘
┌─────────────┐        │
│ BERT (MiniLM)│───────┤
└─────────────┘        │
                       ▼
              [TF-IDF + BERT + Numeric Features]
                       │
                       ▼
              ┌────────────────────┐
              │ LightGBM Regressor │
              └────────────────────┘
                       │
                       ▼
                 Predicted Price
```

---

### **3.2 Model Components**

#### **Text Processing Pipeline**

* **Preprocessing:**
  Lowercasing, stopword removal, punctuation cleanup
* **Model Type:**
  `TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)`
* **Dimensionality Reduction:**
  `TruncatedSVD(n_components=256)`
* **Semantic Embeddings:**
  `SentenceTransformer('all-MiniLM-L6-v2')`
* **Scaling:**
  `StandardScaler` for both TF-IDF and BERT embeddings

#### **Numeric Feature Pipeline**

* **Extracted Features:**

  * `Value` (e.g., 12.0)
  * `Unit` (e.g., "Ounce")
* **Unit Normalization:**

  * Weight → grams
  * Volume → milliliters
  * Count-based → count
* **Derived Columns:**
  `standardized_value`, `standardized_unit`, `unit_type` (encoded via LabelEncoder)

---

### **Feature Fusion**

Final combined feature vector:

```
[TF-IDF reduced (256D)] + [BERT embeddings (384D)] + [standardized_value, unit_type]
→ total ≈ 642 features
```

---

## **4. Model Performance**

### **4.1 Validation Results**

| Metric                 | Score         |
| :--------------------- | :------------ |
| **SMAPE (Validation)** |  53.8%*       |



### **4.2 Training Configuration**

* **Objective:** Custom refined SMAPE objective
* **Metric:** MAE (for monitoring)
* **Optimizer:** LightGBM (5,000 boosting rounds)
* **Learning Rate:** 0.05
* **Subsample:** 0.7
* **Loss Transformation:** log1p(price)

---

## **5. Conclusion**

This hybrid TF-IDF + BERT + numeric LightGBM approach effectively combines text semantics and quantitative reasoning for product price prediction.
Key achievements include unit normalization, embedding fusion, and SMAPE-optimized training.
Future extensions may include visual embeddings (CLIP/Vision Transformer) to leverage product imagery.

---

## **Appendix**

### **A. Code Artefacts**

Files saved:

* `lgbm_model.txt`
* `tfidf_vectorizer.pkl`
* `svd_transformer.pkl`
* `scaler_tfidf.pkl`
* `scaler_bert.pkl`
* `labelencoder.pkl`
* `text_embeddings.npy`

---

### **B. Additional Results**

* Feature importance chart (LightGBM)
* SMAPE vs. iterations plot
* TF-IDF top weighted terms visualization

---

###  **Summary Table**

| Component                | Technique                  | Purpose                              |
| ------------------------ | -------------------------- | ------------------------------------ |
| Text Encoding            | TF-IDF + BERT              | Statistical + Semantic understanding |
| Dimensionality Reduction | TruncatedSVD               | Reduce sparsity                      |
| Numeric Extraction       | Regex + Unit normalization | Add quantitative context             |
| Model                    | LightGBM                   | Efficient gradient boosting          |
| Objective                | Refined SMAPE              | Align with competition metric        |

---

