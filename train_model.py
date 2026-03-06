import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

print("Loading data...")
# Load data
health = pd.read_csv('health.csv', parse_dates=['date'])
rainfall = pd.read_csv('rainfall.csv', parse_dates=['date'])
water = pd.read_csv('water_quality.csv', parse_dates=['date'])

# Calculate total cases across all diseases
health['total_cases'] = (health['cholera_cases'] + health['typhoid_cases'] + 
                         health['dysentery_cases'] + health['hepatitis_cases'])

# Create target: 1 if total cases > threshold (top 30% as high risk)
threshold = health['total_cases'].quantile(0.7)
health['high_risk'] = (health['total_cases'] > threshold).astype(int)

print(f"High risk threshold: {threshold:.2f} cases/week")
print(f"High risk events: {health['high_risk'].sum()} ({health['high_risk'].mean()*100:.1f}%)")

# Feature engineering function
def get_weekly_rainfall(health_date, ward):
    start = health_date - pd.Timedelta(days=7)
    end = health_date
    mask = (rainfall['date'] >= start) & (rainfall['date'] < end) & (rainfall['ward'] == ward)
    return rainfall.loc[mask, 'rainfall_mm'].sum()

print("Building features...")
# Build features
features = []
for idx, row in health.iterrows():
    ward = row['ward']
    date = row['date']
    
    # Rainfall features
    rain_7d = get_weekly_rainfall(date, ward)
    rain_prev_week = get_weekly_rainfall(date - pd.Timedelta(days=7), ward)
    rain_14d = rain_7d + rain_prev_week
    
    # Water quality
    water_vals = water[(water['ward']==ward) & (water['date'] <= date)].sort_values('date', ascending=False)
    if not water_vals.empty:
        chlorine = water_vals.iloc[0]['chlorine_mgL']
        turbidity = water_vals.iloc[0]['turbidity_NTU']
        # Also get trend (change from previous)
        if len(water_vals) > 1:
            chlorine_trend = chlorine - water_vals.iloc[1]['chlorine_mgL']
            turbidity_trend = turbidity - water_vals.iloc[1]['turbidity_NTU']
        else:
            chlorine_trend = 0
            turbidity_trend = 0
    else:
        chlorine = 0.5
        turbidity = 2.0
        chlorine_trend = 0
        turbidity_trend = 0
    
    # Previous week cases (for each disease)
    prev_health = health[(health['ward']==ward) & (health['date'] == date - pd.Timedelta(days=7))]
    if not prev_health.empty:
        prev_total = prev_health.iloc[0]['total_cases']
        prev_cholera = prev_health.iloc[0]['cholera_cases']
        prev_typhoid = prev_health.iloc[0]['typhoid_cases']
        prev_dysentery = prev_health.iloc[0]['dysentery_cases']
        prev_hepatitis = prev_health.iloc[0]['hepatitis_cases']
    else:
        prev_total = 0
        prev_cholera = prev_typhoid = prev_dysentery = prev_hepatitis = 0
    
    # Month (for seasonality)
    month = date.month
    
    features.append({
        'ward': ward,
        'date': date,
        'month': month,
        'rain_7d': rain_7d,
        'rain_prev_week': rain_prev_week,
        'rain_14d': rain_14d,
        'chlorine': chlorine,
        'chlorine_trend': chlorine_trend,
        'turbidity': turbidity,
        'turbidity_trend': turbidity_trend,
        'prev_total_cases': prev_total,
        'prev_cholera': prev_cholera,
        'prev_typhoid': prev_typhoid,
        'prev_dysentery': prev_dysentery,
        'prev_hepatitis': prev_hepatitis,
        'target': row['high_risk']
    })

feature_df = pd.DataFrame(features)
feature_df = feature_df.dropna()

# One-hot encode ward
feature_df = pd.get_dummies(feature_df, columns=['ward'], prefix='ward')

# Train/test split (time-based)
feature_df = feature_df.sort_values('date')
train_size = int(0.8 * len(feature_df))
train = feature_df.iloc[:train_size]
test = feature_df.iloc[train_size:]

# Define features (exclude date and target)
X_cols = [c for c in feature_df.columns if c not in ['date', 'target']]
X_train = train[X_cols]
y_train = train['target']
X_test = test[X_cols]
y_test = test['target']

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Features: {len(X_cols)}")

# Train XGBoost
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'feature': X_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Save model and feature columns
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(X_cols, 'feature_columns.pkl')
joblib.dump(threshold, 'risk_threshold.pkl')

print("\nModel saved to xgb_model.pkl")
print("Feature columns saved to feature_columns.pkl")
print("Risk threshold saved to risk_threshold.pkl")
