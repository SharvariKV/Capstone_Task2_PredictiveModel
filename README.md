# Titanic Predictive Modeling Pipeline (Task 2)

## 1. Problem Framing & Dataset
-Business Question: Can we accurately predict passenger survival on the Titanic to evaluate key maritime survival factors?
-Problem Type: Binary Classification (Target: `Survived`).
-Feature Matrix (X): `Pclass` (Ordinal), `Sex` (Nominal), `Embarked` (Nominal), and `Fare` (Continuous Numerical).

---

## 2. Preprocessing & Validation Highlights
-Data Leakage Prevention: Data was split using `train_test_split` before fitting any transformer, encoder, or scaler.
-Encoding Choices:`Pclass` was ordinally encoded to preserve class rank. `Sex` and `Embarked` were one-hot encoded as nominal categories.
-Feature Scaling: `Fare` was scaled using `StandardScaler` to normalize continuous numerical values.
-Class Rebalancing: SMOTE was applied strictly to the training set to address target class imbalance (~38.34% positive class).

---

## 3. Model Comparison & Metrics
Models were evaluated using 5-Fold Stratified Cross-Validation, with **Binary F1-Score** selected as the primary comparison metric:

*1st Place — Random Forest Classifier (Tuned & Recommended)
  - Binary F1-Score: 0.7914
  - Accuracy: 0.8212
  - Precision: 0.7857
  - Recall: 0.7971

*2nd Place — Decision Tree Classifier
  - Binary F1-Score: 0.7429
  - Accuracy: 0.7821
  - Precision: 0.7324
  - Recall: 0.7536

*3rd Place — Logistic Regression
  - Binary F1-Score: 0.7273
  - Accuracy: 0.7709
  - Precision: 0.7027
  - Recall: 0.7536

---

## 4. Final Recommendation
The Tuned Random Forest Classifier is recommended for deployment. It achieved the highest Binary F1-Score (0.7914) while maintaining an optimal balance between precision and recall across all cross-validation folds. The complete pipeline and trained estimator are serialized and saved at `models/trained_model.pkl`.

## 5. How to run
-Execute Modeling pipeline:
   In Terminal -> 
       python src/model_training.py
