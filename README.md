# 🎙️ Human Voice Gender Classification — Project Report

---

## 🧠 Problem Statement

Develop a machine learning-based system that can classify and cluster human voice samples by gender using extracted audio features.

---

## 📊 Business Use Cases

1. **Call Center Analytics** – Segment calls by gender for insights.
2. **Speaker Identification** – Part of multi-modal biometric systems.
3. **Speech Analytics** – Use gender trends for behavioral analysis.
4. **Accessibility** – Improve assistive technology interfaces.

---

## 🗃️ Dataset Overview

* **File:** `vocal_gender_features_new.csv`
* **Rows:** \~16,000+
* **Columns:** 44 (audio features + label)
* **Label:** `0 = Female`, `1 = Male`

### Features include:

* Spectral properties (`centroid`, `bandwidth`, `contrast`)
* Pitch statistics (`mean`, `std`, `min`, `max`)
* MFCCs (`mfcc_1_mean` to `mfcc_13_std`)
* Energy & zero-crossing rate

---

## 🧹 Data Preprocessing

* Checked for missing/null values (✅ none found)
* Normalized all features using `StandardScaler`
* Separated features and label column
* Used `train_test_split` (80/20 ratio)

---

## 🧠 Model Development

* **Model Used:** `RandomForestClassifier` (from scikit-learn)

* **Why Random Forest?**

  * Handles feature noise well
  * Robust to overfitting
  * Fast and accurate

* **Accuracy:** `99.4%` on test set

---

## 🔍 Key Model Findings

* ✅ **High Accuracy**: Random Forest achieved **\~99.4% accuracy**, showing features are highly predictive.
* ✅ **Balanced Classification**: Predictions showed a nearly equal split of male and female voices.
* ✅ **Clean Dataset**: No missing values and fully numeric features.
* ✅ **Likely Important Features**: Pitch, MFCCs, and spectral attributes contribute significantly.
* ✅ **Model is Robust**: No overfitting observed, and no hyperparameter tuning needed.
* ✅ **Functional App**: Streamlit UI gives real-time predictions and visualization.

---

## 💾 Model Persistence

* Saved trained model as `voice_model.pkl`
* Saved feature scaler as `scaler.pkl`

---

## 🌐 Streamlit Web App

### Key Features:

* Upload CSV file with audio features (no label)
* Get predicted gender for each voice sample
* Display:

  * Predictions table
  * Summary count
  * Pie chart visualization

### Tools:

* `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`

---

## 📈 Final Results

* Predictions: 0 (Female), 1 (Male)
* Visualization: Pie chart showing distribution
* UI: Clean interface to drag/drop CSV and get results instantly




