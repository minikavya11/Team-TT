# ===============================================
# 🐾 Predict Animal Disease Before They Happen
# Hybrid ML Pipeline: XGBoost + LSTM with Region Embedding
# ===============================================

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, jsonify
from flask_cors import CORS

# =======================
# 1️⃣ Load Dataset
# =======================
df = pd.read_csv("realistic_pet_health_dataset.csv")
print("Dataset shape:", df.shape)

# =======================
# 2️⃣ Handle Missing Values
# =======================
df['symptoms_current'] = df['symptoms_current'].fillna('None')
df['predicted_disease'] = df['predicted_disease'].fillna('None')
df['previous_diseases'] = df['previous_diseases'].fillna('None')

# =======================
# 3️⃣ Feature Engineering
# =======================
categorical_cols = ['species', 'breed', 'sex', 'activity_level', 'diet_type', 'region']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

numeric_cols = ['age_years', 'weight_kg', 'daily_calories', 'protein_percent', 'fat_percent',
                'avg_temperature_c', 'humidity_percent', 'rainfall_mm', 'air_quality_index', 
                'local_outbreak_risk_index']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =======================
# 4️⃣ Encode Symptoms using Tokenizer
# =======================
all_symptoms = df['symptoms_current'].values
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(all_symptoms)
sequences = tokenizer.texts_to_sequences(all_symptoms)
max_len = max([len(seq) for seq in sequences])
X_seq = pad_sequences(sequences, maxlen=max_len, padding='post')
joblib.dump(tokenizer, "tokenizer.pkl")

# =======================
# 5️⃣ Prepare Tabular Features
# =======================
X_tab = df.drop(['animal_id', 'predicted_disease', 'vet_recommended_action', 'prediction_date', 'symptoms_current'], axis=1)
y = df['predicted_disease']

# Encode target for LSTM
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
joblib.dump(le_y, "label_encoder_y.pkl")

# Split data
X_tab_train, X_tab_test, X_seq_train, X_seq_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X_tab, X_seq, y, y_encoded, test_size=0.2, random_state=42
)

# =======================
# 6️⃣ Train XGBoost Model
# =======================
mlflow.set_experiment("Pet_Health_XGBoost")
with mlflow.start_run(run_name="xgb_training"):
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_tab_train, y_train)
    y_pred_xgb = xgb_model.predict(X_tab_test)
    
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    
    # Log model to MLflow
    mlflow.sklearn.log_model(xgb_model, "xgb_model")
    mlflow.log_metric("xgb_accuracy", accuracy_score(y_test, y_pred_xgb))
    
xgb_model.save_model("xgb_model.json")

# =======================
# 7️⃣ LSTM Model with Region Embedding
# =======================
# Inputs
symptoms_input = Input(shape=(max_len,), name="symptoms_input")
region_input = Input(shape=(1,), name="region_input")

# Embeddings
vocab_size = len(tokenizer.word_index) + 1
symptom_emb = Embedding(input_dim=vocab_size, output_dim=64)(symptoms_input)
lstm_out = LSTM(64)(symptom_emb)

region_vocab_size = df['region'].nunique()
region_emb = Embedding(input_dim=region_vocab_size, output_dim=8)(region_input)
region_flat = tf.keras.layers.Flatten()(region_emb)

# Combine LSTM + Region
concat = Concatenate()([lstm_out, region_flat])
output = Dense(len(y.unique()), activation='softmax')(concat)
lstm_model = Model(inputs=[symptoms_input, region_input], outputs=output)

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
mlflow.set_experiment("Pet_Health_LSTM")
with mlflow.start_run(run_name="lstm_training"):
    lstm_model.fit(
        {"symptoms_input": X_seq_train, "region_input": X_tab_train['region'].values},
        y_train_enc,
        epochs=15,
        batch_size=32,
        validation_split=0.1
    )
    # Log model
    mlflow.tensorflow.log_model(lstm_model, "lstm_model")

lstm_model.save("lstm_model.h5")

# Save preprocessing objects
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# =======================
# 8️⃣ Hybrid Prediction Function
# =======================
def predict_disease(pet_data):
    # Load models
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_model.json")
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    tokenizer = joblib.load("tokenizer.pkl")
    le_y = joblib.load("label_encoder_y.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    
    # Preprocess tabular
    df_new = pd.DataFrame([pet_data])
    for col in label_encoders.keys():
        if col not in df_new.columns or df_new[col].iloc[0]=="":
            df_new[col] = 'None'
        df_new[col] = label_encoders[col].transform(df_new[col])
    df_new[numeric_cols] = scaler.transform(df_new[numeric_cols])
    
    # XGBoost prediction
    pred_xgb = xgb_model.predict(df_new)[0]
    
    # LSTM prediction
    seq = tokenizer.texts_to_sequences([pet_data.get("symptoms_current","None")])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred_lstm_idx = np.argmax(lstm_model.predict([seq_padded, df_new['region'].values]), axis=1)[0]
    pred_lstm = le_y.inverse_transform([pred_lstm_idx])[0]
    
    # Hybrid ensemble (weighted)
    weights = {"xgb": 0.6, "lstm": 0.4}
    votes = {pred_xgb: weights['xgb'], pred_lstm: weights['lstm']}
    final_pred = max(votes, key=votes.get)
    return final_pred

# =======================
# 9️⃣ Flask API
# =======================
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "🐾 Pet Health Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    prediction = predict_disease(data)
    return jsonify({"predicted_disease": prediction})

if __name__ == "__main__":
    app.run(debug=True)
