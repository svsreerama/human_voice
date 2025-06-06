# 🎙️ Human Voice Gender Classifier

A machine learning web application that classifies human voices as **Male** or **Female** using extracted audio features. Built using Python, scikit-learn, and deployed with Streamlit.

---

## 📌 Project Overview

This project processes pre-extracted audio features and uses a trained Random Forest model to classify gender.

### 🔍 Use Cases
- Call center analytics
- Voice-based authentication
- Media and accessibility tools

---

## 🧠 Features

- Reads voice features from CSV
- Predicts gender (0 = Female, 1 = Male)
- Displays:
  - Prediction table
  - Gender distribution summary
  - Pie chart visualization

---

## 📂 File Structure
```bash
human-voice-gender/
│
├── app.py                   # Streamlit app
├── step1_load_data.py       # Data loading
├── step2_clean_data.py      # Data inspection
├── step3_normalize_data.py  # Scaling features
├── step4_split_data.py      # Train-test split
├── step5_train_model.py     # Train classifier
├── step6_save_model.py      # Save model and scaler
├── make_input.py            # Create input_features.csv (no label)
├── voice_model.pkl          # Trained model
├── scaler.pkl               # Saved scaler
├── vocal_gender_features_new.csv  # Full dataset
└── input_features.csv       # For predictions
