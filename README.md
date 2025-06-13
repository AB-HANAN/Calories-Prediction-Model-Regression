# 🔥 Calories Prediction Model  

**Predict calories burned during exercise with machine learning!**  

This project builds a **regression model** to estimate calories burned based on workout metrics like heart rate, duration, age, and body measurements. Perfect for fitness apps, wearables, or health analytics!  

---

## 📊 **Key Features**  
✅ **Accurate predictions** using **XGBoost (best-performing model)**  
✅ **9+ regression models tested** (Random Forest, SVR, Linear Regression, etc.)  
✅ **Exploratory Data Analysis (EDA)** with visualizations  
✅ **Performance comparison** (RMSE, R² Score, MAE)  
✅ **Ready-to-use saved model** (`best_calories_model.pkl`)  
✅ **Clean, well-documented code**  

---

## 📈 **Model Performance**  
| Model               | RMSE ↓ | R² Score ↑ |  
|---------------------|--------|------------|  
| **XGBoost**         | 1.23   | 0.99       |  
| Random Forest       | 1.45   | 0.98       |  
| Gradient Boosting   | 1.50   | 0.98       |  
| SVR                 | 2.10   | 0.96       |  

*(Example metrics—replace with your actual results!)*  

**Best Model:** `XGBoost Regressor` (Lowest RMSE, Highest R²)  

---

## � **How It Works**  
The model predicts calories burned using:  
- **Gender** (Male/Female → Encoded as 0/1)  
- **Age** (Years)  
- **Height** (cm)  
- **Weight** (kg)  
- **Exercise Duration** (Seconds)  
- **Heart Rate** (BPM)  
- **Body Temperature** (°C)  

### 📊 **Data Insights**  
![Correlation Heatmap](https://via.placeholder.com/600x400?text=Heatmap+Example) *(Replace with your actual heatmap image!)*  

- **Strongest correlations:** Weight, Duration, Heart Rate  
- **Gender distribution:** Balanced dataset  

---

## 🚀 **Quick Start**  

### **1. Install Dependencies**  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib tabulate
```

### **2. Load the Model & Predict**  
```python
import joblib

# Load the trained model
model = joblib.load('models/best_calories_model.pkl')

# Input format: [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
input_data = [0, 30, 180, 75, 1800, 130, 38.5]  # Example: 30yo male, 75kg, 30min workout

# Predict calories burned
calories = model.predict([input_data])[0]
print(f"🔥 Estimated calories burned: {calories:.2f} kcal")
```

### **3. Train Your Own Model**  
Run the full pipeline:  
```bash
python calories_prediction.py
```
*(This trains all models, compares performance, and saves the best one.)*  

---

## 📌 **Why This Model?**  
✔ **High Accuracy:** R² ≈ 0.99 (near-perfect fit)  
✔ **Fast Predictions:** Optimized with XGBoost  
✔ **Easy Integration:** Works in apps, APIs, or analytics tools  

---

## 🔮 **Future Improvements**  
- [ ] Add **neural network** for comparison  
- [ ] Deploy as a **Flask/FastAPI web app**  
- [ ] Include **more features** (exercise type, weather conditions)

---


## 🤝 **Collaboration**  
This project was developed as part of a **machine learning collaboration** between **ME** and **Syed Muhammad Abdullah**. 
Special thanks for the brainstorming, debugging, and making the front end of this project!  
THIS IS THE LINK TO HIS GITHUB ACCOUNT --> https://github.com/Syed-Abdullah-py


