import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle, json

# Load data
df = pd.read_csv("bengaluru_house_prices.csv")

# Clean data
df.dropna(subset=["location", "size", "bath", "price"], inplace=True)
df["size"] = df["size"].str.extract(r"(\d+)").astype(float)

# Handle total_sqft ranges like "1000-1200" → average
def parse_sqft(val):
    try:
        if "-" in str(val):
            parts = str(val).split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(val)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(parse_sqft)
df.dropna(subset=["total_sqft"], inplace=True)
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(0)

# Remove outliers
df = df[df["price"] < 500]
df = df[df["total_sqft"] < 10000]

# Keep top 30 locations, group rest as "Other"
top_locations = df["location"].value_counts().head(30).index
df["location"] = df["location"].apply(lambda x: x if x in top_locations else "Other")

# Encode location
le = LabelEncoder()
df["location_enc"] = le.fit_transform(df["location"])

# Features and target
features = ["total_sqft", "bath", "balcony", "size", "location_enc"]
X = df[features]
y = df["price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.2f}")

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save location list for dropdown
locations = sorted(list(le.classes_))
with open("locations.json", "w") as f:
    json.dump(locations, f)

print("Model saved! Locations saved:", len(locations))
