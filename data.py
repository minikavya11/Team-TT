# =====================================================
# 🐾 REALISTIC PET HEALTH DATASET GENERATOR
# =====================================================

import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

# ------------------------------
# 1️⃣ Base Values
# ------------------------------
species_list = ["Dog", "Cat", "Bird", "Hamster", "Rabbit"]
dog_breeds = ["Labrador", "German Shepherd", "Pug", "Beagle", "Golden Retriever"]
cat_breeds = ["Persian", "Siamese", "Maine Coon", "British Shorthair", "Bengal"]
bird_breeds = ["Cockatiel", "Canary", "Parakeet"]
hamster_breeds = ["Syrian", "Roborovski"]
rabbit_breeds = ["Netherland Dwarf", "Holland Lop"]

sex_list = ["M", "F"]
activity_level = ["Low", "Medium", "High"]
diet_type = ["Commercial Kibble", "Dry + Wet Mix", "Raw + Veg Mix", "Homemade Cooked", "Mixed Diet"]
regions = ["Kerala, India", "Maharashtra, India", "Tamil Nadu, India", "Delhi, India", "Gujarat, India"]
symptoms = ["None", "Sneezing", "Lethargy", "Vomiting", "Loss of Appetite", "Coughing", "Limping", "Watery Eyes"]
diseases = ["None", "Parvovirus", "Arthritis", "Feline Upper Resp Infection", "Ear Infection", "Skin Allergy", "Respiratory Infection"]

region_env = {
    "Kerala, India": {"temp": (25, 32), "humidity": (70, 90), "rainfall": (150, 400), "air_quality": (30, 60)},
    "Maharashtra, India": {"temp": (22, 34), "humidity": (40, 70), "rainfall": (50, 250), "air_quality": (50, 120)},
    "Tamil Nadu, India": {"temp": (24, 35), "humidity": (60, 80), "rainfall": (80, 200), "air_quality": (40, 90)},
    "Delhi, India": {"temp": (15, 40), "humidity": (20, 60), "rainfall": (10, 100), "air_quality": (150, 350)},
    "Gujarat, India": {"temp": (22, 38), "humidity": (30, 60), "rainfall": (10, 150), "air_quality": (70, 150)}
}

num_rows = 20000
data = []

# ------------------------------
# 2️⃣ Generate Data
# ------------------------------
for i in range(1, num_rows + 1):
    sp = random.choice(species_list)
    if sp == "Dog": breed = random.choice(dog_breeds)
    elif sp == "Cat": breed = random.choice(cat_breeds)
    elif sp == "Bird": breed = random.choice(bird_breeds)
    elif sp == "Hamster": breed = random.choice(hamster_breeds)
    else: breed = random.choice(rabbit_breeds)

    age = round(random.uniform(0.5, 15), 1)
    weight = round(random.uniform(3, 40) if sp == "Dog" else random.uniform(2, 12), 1)
    sex = random.choice(sex_list)
    act = random.choice(activity_level)
    diet = random.choice(diet_type)
    daily_calories = round(random.uniform(300, 2200), 0)
    protein = round(random.uniform(20, 35), 1)
    fat = round(random.uniform(10, 20), 1)
    prev_disease = random.choices(diseases, weights=[50,10,10,10,10,5,5])[0]

    region = random.choice(regions)
    env = region_env[region]

    avg_temp = round(random.uniform(*env["temp"]),1)
    humidity = round(random.uniform(*env["humidity"]),1)
    rainfall = round(random.uniform(*env["rainfall"]),1)
    air_quality = round(random.uniform(*env["air_quality"]),1)

    # Outbreak risk
    outbreak_risk = round(random.uniform(0.05, 0.3) + (air_quality / 1000) + (humidity / 1000),2)

    # Symptoms
    symptom = random.choices(symptoms, weights=[50,10,10,10,5,5,5,5])[0]

    # Disease probability mapping
    disease_prob = {
        "None": 0.7,
        "Parvovirus": 0.05 + 0.1*(humidity>80),
        "Arthritis": 0.05 + 0.05*(age>7),
        "Feline Upper Resp Infection": 0.05 + 0.05*(humidity>70),
        "Ear Infection": 0.05 + 0.05*(humidity>60),
        "Skin Allergy": 0.05 + 0.05*(air_quality>100),
        "Respiratory Infection": 0.05 + 0.05*(air_quality>150)
    }

    # Assign disease
    if symptom == "None":
        pred_disease = "None"
    else:
        weights = [disease_prob[d] for d in diseases]
        total = sum(weights)
        weights = [w/total for w in weights]
        pred_disease = random.choices(diseases, weights=weights)[0]

    # Recommended action
    if pred_disease == "None":
        action = "Routine Checkup"
    else:
        action = random.choice([
            "Vet exam and vaccination update",
            "Adjust diet: increase protein",
            "Provide joint supplements",
            "Administer antibiotics",
            "Hydration therapy"
        ])

    prediction_date = datetime.now().date() - timedelta(days=random.randint(0,30))

    data.append([
        i, sp, breed, age, sex, weight, act, diet, daily_calories,
        protein, fat, prev_disease, region, avg_temp, humidity, rainfall,
        air_quality, outbreak_risk, symptom, pred_disease, prediction_date, action
    ])

# ------------------------------
# 3️⃣ Create DataFrame
# ------------------------------
columns = [
    "animal_id","species","breed","age_years","sex","weight_kg",
    "activity_level","diet_type","daily_calories","protein_percent",
    "fat_percent","previous_diseases","region","avg_temperature_c",
    "humidity_percent","rainfall_mm","air_quality_index","local_outbreak_risk_index",
    "symptoms_current","predicted_disease","prediction_date","vet_recommended_action"
]

df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv("realistic_pet_health_dataset.csv", index=False)
print("✅ Realistic dataset created: realistic_pet_health_dataset.csv")
print(df.head())
