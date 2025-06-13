# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def prediction_model(data):
    input_np = np.array(data)
    input_np = input_np.reshape(1, -1)
    prediction = best_model.predict(input_np)
    return prediction[0]

# loading the datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
calories = pd.read_csv(os.path.join(script_dir, "calories.csv"))
exerciseData = pd.read_csv(os.path.join(script_dir, "exercise.csv"))

# First few rows of the datasets
print("Calories.csv")
print(calories.head(), "\n")

print("Exercise.csv")
print(exerciseData.head(), "\n")

# Combining the two datasets
calories_data = pd.concat([exerciseData, calories['Calories']], axis=1)

# displaying the shape and few rows of the updated dataset
print("Combined Dataset:")
print(calories_data.head())
print("Shape: ", calories_data.shape, "\n")

# Checking for missing data
print("Column \tMissing Values in Column")
print(calories_data.isnull().sum(), "\n")

# displaying information about the data
calories_data.info()

# DATA ANALYSIS PART
# Numerical Information regarding the dataset
calories_data.describe()

# DATA VISUALIZATION
sns.set_theme()

# Plotting the Gender Column with custom colors
gender_colors = {"male": "#3498db", "female": "#e74c3c"}
sns.countplot(x='Gender', hue='Gender', data=calories_data, palette=gender_colors, legend=False)
plt.title("Gender Distribution")
plt.show()

# Distribution of Age
sns.displot(calories_data['Age'], color='#2ecc71')
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# Distribution of Height
sns.displot(calories_data['Height'], color='#9b59b6')
plt.title("Height Distribution")
plt.tight_layout()
plt.show()

# Distribution of Weight
sns.displot(calories_data['Weight'], color='#f1c40f')
plt.title("Weight Distribution")
plt.tight_layout()
plt.show()

# Distribution of Duration
sns.displot(calories_data['Duration'], color='#1abc9c')
plt.title("Duration Distribution")
plt.tight_layout()
plt.show()

# Distribution of Heart_Rate
sns.displot(calories_data['Heart_Rate'], color='#d35400')
plt.title("Heart_Rate Distribution")
plt.tight_layout()
plt.show()

# Distribution of Body_Temp
sns.displot(calories_data['Body_Temp'], color='#34495e')
plt.title("Body_Temperature Distribution")
plt.tight_layout()
plt.show()

# FINDING CORRELATION AND CONSTRUCTING HEATMAP
# Replacing Male with 1 and Female with 0 and converting Duration from minutes to seconds
calories_data['Gender'] = calories_data['Gender'].map({'male': 0, 'female': 1})
calories_data['Duration'] = calories_data['Duration'] * 60

# Finding the correlation
correlation = calories_data.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Separating Feature Space and Label Space
X = calories_data.drop(columns=['User_ID', 'Calories'])
Y = calories_data['Calories']

# Splitting the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
print("Training Data Shape: ", X_train.shape, Y_train.shape)
print("Testing Data Shape: ", X_test.shape, Y_test.shape, "\n")

# MODEL TRAINING
# Define multiple regression models
models = {
    'XGBoost': XGBRegressor(random_state=40),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=40),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=40),
    'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10))]),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=40)
}

# Custom color palette for the plots
plot_colors = {
    'XGBoost': ('#3498db', '#2ecc71'),  # (dots color, line color)
    'Random Forest': ('#e74c3c', '#34495e'),
    'Gradient Boosting': ('#9b59b6', '#f1c40f'),
    'SVR': ('#1abc9c', '#d35400'),
    'Linear Regression': ('#27ae60', '#8e44ad'),
    'Ridge Regression': ('#16a085', '#c0392b'),
    'Lasso Regression': ('#2980b9', '#f39c12'),
    'K-Neighbors': ('#d35400', '#2c3e50'),
    'Decision Tree': ('#7f8c8d', '#e74c3c')
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, Y_train)
    test_predict = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test, test_predict)
    mse = mean_squared_error(Y_test, test_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, test_predict)
    
    # Store results
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'Model Object': model
    })
    
    # Plot predictions vs actual with custom colors
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test, test_predict, alpha=0.7, color=plot_colors[name][0], 
                edgecolor='w', s=80, label='Predictions')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 
             linestyle='--', lw=2, color=plot_colors[name][1], label='Perfect Prediction')
    plt.xlabel('Actual Calories', fontsize=12)
    plt.ylabel('Predicted Calories', fontsize=12)
    plt.title(f'Actual vs Predicted: {name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('model_plots', exist_ok=True)
    plt.savefig(f'model_plots/actual_vs_predicted_{name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

    # Print sample predictions for XGBoost (original model)
    if name == 'XGBoost':
        print(f"{'Predicted':>12} {'Actual':>12} {'Difference':>12}")
        for pred, actual in zip(test_predict[:5], Y_test[:5].values):
            print(f"{pred:12.2f} {actual:12.2f} {pred-actual:12.2f}")

# Convert results to DataFrame for easier manipulation
results_df = pd.DataFrame(results)

# Find best model based on RMSE (you can change this to other metrics)
best_model_info = results_df.loc[results_df['RMSE'].idxmin()]
best_model = best_model_info['Model Object']
best_model_name = best_model_info['Model']

print("\nMODEL COMPARISON:")
print(tabulate(results_df.drop(columns=['Model Object']), 
      headers='keys', tablefmt='grid', floatfmt=".2f"))

print(f"\nBest Model: {best_model_name}")
print(f"Best RMSE: {best_model_info['RMSE']:.2f}")
print(f"Best R2 Score: {best_model_info['R2 Score']:.4f}")

# Save the best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_calories_model.pkl")
print("\nBest model saved as 'models/best_calories_model.pkl'")

# Additional visualization of model performance
plt.figure(figsize=(12, 8))
models_sorted = results_df.sort_values('RMSE')
plt.barh(models_sorted['Model'], models_sorted['RMSE'], color='#3498db')
plt.xlabel('RMSE (Lower is better)')
plt.title('Model Performance Comparison (RMSE)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_plots/model_performance_comparison.png', dpi=300)
plt.show()