# Team-TT
# 🐾 Predict Animal Disease Before They Happen Using AI

---

## 🌍 Overview

Animals behave differently depending on **region, climate, species, and environment**.  
A single global model often fails across regions.  

This project implements an **AI-based hybrid model** with **region-awareness**:

- 🌐 **XGBoost** → Structured/tabular data.  
- 🔁 **LSTM** → Sequential data (symptom history, previous diseases).  
- 🗺️ **Region Feature** → **Central component**: Ensures predictions are **adapted to local climate, temperature, and environmental conditions**.  
- 🔄 **Continuous Learning** → Automated retraining, monitoring, and performance tracking.

**Goal:** Predict pet diseases **before they occur**, enabling preventive care based on **region-specific conditions**.

---

## ⚠️ Problem Statement

This project predicts **potential diseases in pets before they occur** using **machine learning** by analyzing:

- Pet features: species, breed, age, sex, weight
- Lifestyle: activity level, diet, daily calories
- Health history: previous diseases, current symptoms
- Environment: region, temperature, humidity, rainfall, air quality, local outbreak risk

---

## 🏗️ Features

- Input pet info, environment, and symptom history.  
- **Hybrid AI prediction with XGBoost + LSTM**.  
- **Region-aware predictions** for higher accuracy (**region is a key feature in both models**).  
- Real-time predictions via **React frontend + FastAPI/Flask backend**.  
- Handles missing values, encoding, and scaling.  
- Automated retraining and monitoring via **MLflow + Airflow + Grafana**.

---

## ⚙️ Tech Stack (Region-Focused)

| Component          | Technology           | Purpose |
|------------------|--------------------|---------|
| Frontend          | React.js, Tailwind CSS | Input pet & region data, display predictions |
| Backend / API     | FastAPI / Flask     | Serve region-aware ML models via /predict endpoint |
| ML Model          | XGBoost, TensorFlow/Keras LSTM/GRU ,RandomForest | Region-aware hybrid model (tabular + sequential data) |
| Data Handling     | pandas, numpy       | Preprocessing, missing value handling, scaling |
| Region Embedding  | Keras Embedding     | Encode region into ML models for better adaptation |
| Model Serialization | joblib / h5py      | Save/load trained models & scalers |
| Automation        | Airflow             | Retraining scheduler |
| Tracking          | MLflow              | Model versioning & metrics |
| Monitoring        | Grafana             | Performance visualization |

---

## 🧠 Project Workflow (Region Highlighted)

```
START
  |
  v
+------------------------------+
| 1️⃣ Collect Animal Data       |
| - Multiple regions           |
| - Each record tagged by region|
+------------------------------+
  |
  v
+------------------------------+
| 2️⃣ Preprocess Data           |
| - Handle missing values      |
| - Encode categorical & region|
| - Scale numeric & sequential |
+------------------------------+
  |
  v
+------------------------------+
| 3️⃣ Train XGBoost Model       |
| - Structured/tabular features|
| - Region included as key input|
+------------------------------+
  |
  v
+------------------------------+
| 4️⃣ Train LSTM Model          |
| - Sequential data (symptoms history)|
| - Region info concatenated    |
+------------------------------+
  |
  v
+------------------------------+
| 5️⃣ Combine Hybrid Outputs    |
| - XGBoost + LSTM             |
| - Region-awareness improves prediction accuracy|
+------------------------------+
  |
  v
+------------------------------+
| 6️⃣ Region-Aware Fine-Tuning |
| - Optional per region model  |
| - Adjust for local conditions|
+------------------------------+
  |
  v
+------------------------------+
| 7️⃣ Backend API (FastAPI/Flask)|
| - /predict endpoint          |
| - Selects correct region-aware model|
+------------------------------+
  |
  v
+------------------------------+
| 8️⃣ Frontend (React)          |
| - Input pet & environment + region|
| - Display predicted disease  |
+------------------------------+
  |
  v
+------------------------------+
| 9️⃣ Monitoring & Retraining   |
| - MLflow tracking            |
| - Airflow automated retraining|
| - Grafana visualization       |
+------------------------------+

  |
  v
 END
```
---

## 👥 Team Members

**Team Name:** 🏆 Team-TT  

**Members:**  
- 👩‍💻 L. Amusha  
- 👩‍💻 P. Kavyasri  
- 👩‍💻 K. Deepika  
- 👩‍💻 K. Lavanya

---


