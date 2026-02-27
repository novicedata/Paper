# Grove

## Contents

### 📌 Overview

### 📌 Problem Definition

### 📌 Experimental Design

### 📌 Experiments and Results

### 📌 Conclusion

---

### 📌 Overview

- Conventional CAT primarily relies on IRT, which requires strong assumptions and limits real-world applicability.
- Tree-based CAT addresses some issues but suffers from overfitting and item exposure problems.
- Boosting models achieve high predictive accuracy but lack interpretability and direct compatibility with CAT.
- This study proposes the Grove algorithm, which decomposes boosting models into structured subsets of trees.

- **Key Activities**
  
  - Understanding the internal structure of boosting algorithms (Gradient Boosting, CatBoost) and the fundamental mechanism of CAT.
  - Developing the Grove algorithm by decomposing boosting models into tree subsets suitable for CAT.
  - Implementing Grove using CatBoost: model training → tree decomposition → Grove construction → CAT simulation.
  - Empirical validation:
    - Verifying Grove's CAT mechanics using cognitive assessment data (VIQ).
    - Comparing Grove with DT-based CAT and CatBoost using psychological (TMAS) and medical licensing (KMLE) datasets.
    - Visualization and analysis, including Grove path visualization, ROC-AUC and accuracy comparison, item exposure and individualized path simulations, and overfitting analysis.
- **Key Learnings**
  
  - In-depth understanding of ML (especially boosting) algorithm structures and their implementation.
  - Skills in designing model architectures and implementing them in code.
  - Improved ability to design model structures according to application purposes.
  - Enhanced Python programming skills.

---

### 📌 Problem Definition

- **Computerized Adaptive Testing (CAT)**
  
  - CAT is essential for high measurement precision with fewer items in education, psychological assessment, and licensing exams. Traditionally based on IRT, CAT selects items sequentially according to estimated latent traits.
- **Limitations of IRT-based CAT**
  
  - Assumptions of **unidimensionality and local independence** are often unmet in real data.
  - Complex parameter estimation reduces flexibility, leading to the exploration of Decision Tree-based CAT.
- **Why is this a problem?**
  
  - 1️️ **Overfitting:** Trees may overfit noise in training data, reducing generalizability.
  - 2️️ **Item exposure and security:** Fixed root nodes cause repetitive initial item exposure.
  - 3️️ **Limited adaptability:** DT-CAT lacks dynamic ability re-estimation after responses, limiting adaptability.
- **What needs to be addressed?**
  
  - Maintain high predictive performance and adaptability.
  - Strengthen item exposure control and test security.
  - Mitigate overfitting and enhance generalizability.
  - Enable provision of multiple alternate-form tests.
- **Grove Model Proposal**
  
  - Boosting models are highly predictive but complex, making CAT application difficult.
    
  - **Grove leverages the additive structure of boosting models, decomposing them into individual trees, forming subsets for CAT use.**
    
    - **Resolution Strategy:**
      - ✅ **Overfitting mitigation:** Using deeper tree subsets for high accuracy with fewer items.
      - ✅ **Item exposure control:** Different subsets diversify test paths and reduce exposure.
      - ✅ **Adaptability:** Provides individualized paths based on responses.
      - ✅ **Alternate-form tests:** Multiple Grove configurations enable secure, equivalent test forms.

![image](https://github.com/user-attachments/assets/d15cc304-f86e-47c4-9c5a-2260907b38ca)

---

### 📌 Experimental Design

- 1️️ **Feasibility Test**
  
  - Validate Grove's ability to deliver adaptive item selection.
  - Simulate CAT using VIQ data (45 items, 11,502 participants).
  - Confirm individualized paths and omission of repeated items.
- 2️️ **Performance Comparison**
  
  - Compare Grove, DT-based CAT, and CatBoost on ROC-AUC, accuracy, efficiency, and overfitting.
  - Use TMAS (50 items, multiclass) and KMLE (360 items, binary) datasets.
  - Construct Groves with 3-4 tree combinations and assess test form flexibility and performance.
- **Key Comparison Metrics:**
  
  - ✅ Adaptive item selection
  - ✅ Predictive performance (ROC-AUC, accuracy)
  - ✅ Reduction in administered items and enhanced security
  - ✅ Overfitting mitigation and generalizability
  - ✅ Alternate-form test provision

---

### 📌 Experiments and Results

- **Method**
  
  - Evaluate Grove, DT-based CAT, and CatBoost on identical datasets.
  - Validate under binary and multiclass scenarios using VIQ, TMAS, and KMLE.
  - Train CatBoost, decompose trees, form Groves with 3-4 trees, and simulate CAT.
- **Datasets and Evaluation:**
  
  - ✅ VIQ: Feasibility verification.
  - ✅ TMAS, KMLE: Performance and efficiency testing.
  - ✅ Metrics: ROC-AUC, accuracy, item count, overfitting, alternate-form viability.

![image](https://github.com/user-attachments/assets/66053c00-a0e7-48ad-845f-3e83925faf0b)
![image](https://github.com/user-attachments/assets/a62e2f81-1710-471a-8c8e-778a5a2e1b7f)

- **Key Findings:**
  - ✅ Grove replicates adaptive delivery with individualized paths and omits redundant items.
  - ✅ Grove achieves higher ROC-AUC and comparable accuracy to DT-CAT, close to CatBoost.
  - ✅ Performance improves with 3-4 trees, enabling multiple secure test forms.
  - ✅ Retains predictive power while reducing overfitting, diversifying item exposure.

---


### 📌 Conclusion

This study confirms **Grove's viability as a CAT method, offering strong predictive performance, adaptability, and efficiency with enhanced interpretability and security**.

- ✅ Replicates adaptive delivery, reducing redundant item administration.
- ✅ Achieves high ROC-AUC and accuracy, matching CatBoost performance.
- ✅ Enables secure alternate-form tests with diverse item exposure.
- ✅ Mitigates overfitting while maintaining predictive accuracy.

**Limitations and Future Work:**

- 🔹 Requires further hyperparameter tuning and tree configuration exploration.
- 🔹 Expansion to various test types and real-world deployment.
- 🔹 Future development of automated pipelines, visualization tools, and exposure control for live Grove-based CAT systems.
