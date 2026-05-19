# 🏠 Bengaluru House Price Predictor

A simple ML web app built with Python + Flask for a school project.

---

## 📁 Project Structure
```
bengaluru_house_price/
├── bengaluru_house_prices.csv   ← dataset
├── train_model.py               ← trains and saves the model
├── app.py                       ← Flask web server
├── templates/
│   └── index.html               ← website UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (in VS Code Terminal)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (only needed once)
```bash
python train_model.py
```
This creates: `model.pkl`, `encoder.pkl`, `locations.json`

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
Go to: **http://127.0.0.1:5000**

---

## 🤖 How it works
- **Model**: Linear Regression (scikit-learn)
- **Features used**: Location, Total Sq.ft, BHK, Bathrooms, Balconies
- **Dataset**: 13,320 Bengaluru house listings
- **R² Score**: ~0.56 (56% accuracy)
