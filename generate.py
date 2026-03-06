import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("Starting data generation for Coimbatore wards...")

# Real Coimbatore ward names
wards = [
    "Gandhipuram", "RS Puram", "Saibaba Colony", "Peelamedu", "Singanallur",
    "Saravanampatti", "Race Course", "Ukkadam", "Town Hall", "Kuniyamuthur",
    "Sundarapuram", "Podanur", "Thudiyalur", "Vadakovai", "Ganapathy",
    "Kalapatti", "Kovaipudur", "Selvapuram", "Kurichi", "Vadavalli"
]

print(f"Loaded {len(wards)} Coimbatore wards")

# Parameters
num_wards = len(wards)
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start_date, end_date, freq='D')

# Ward population (realistic estimates for Coimbatore)
ward_population = {
    "Gandhipuram": 45000, "RS Puram": 38000, "Saibaba Colony": 42000,
    "Peelamedu": 35000, "Singanallur": 40000, "Saravanampatti": 55000,
    "Race Course": 32000, "Ukkadam": 48000, "Town Hall": 28000,
    "Kuniyamuthur": 36000, "Sundarapuram": 33000, "Podanur": 29000,
    "Thudiyalur": 52000, "Vadakovai": 31000, "Ganapathy": 44000,
    "Kalapatti": 58000, "Kovaipudur": 39000, "Selvapuram": 34000,
    "Kurichi": 37000, "Vadavalli": 41000
}

# Ward infrastructure quality (affects baseline water quality)
ward_infrastructure = {
    ward: np.random.choice(['poor', 'average', 'good'], p=[0.3, 0.5, 0.2]) 
    for ward in wards
}

print("Generating rainfall data...")
# Generate rainfall data (daily, mm)
rainfall_data = []
for date in dates:
    for ward in wards:
        # Simulate monsoon: June-September higher rainfall
        month = date.month
        if 6 <= month <= 9:
            # Monsoon months
            rain = np.random.gamma(3, 12)  # higher rainfall
        elif month in [10, 11]:  # Post-monsoon
            rain = np.random.gamma(2, 8)
        else:  # Dry season
            rain = np.random.gamma(1, 3)
        
        rainfall_data.append([date.date(), ward, round(max(0, rain), 1)])

rainfall_df = pd.DataFrame(rainfall_data, columns=['date', 'ward', 'rainfall_mm'])
rainfall_df.to_csv('rainfall.csv', index=False)
print(f"Rainfall data saved: {len(rainfall_df)} records")

print("Generating water quality data...")
# Generate water quality data (weekly)
water_quality_data = []
current_date = start_date
while current_date <= end_date:
    for ward in wards:
        # Base chlorine depends on ward infrastructure
        infra = ward_infrastructure[ward]
        if infra == 'good':
            base_chlorine = np.random.uniform(0.5, 0.9)
        elif infra == 'average':
            base_chlorine = np.random.uniform(0.3, 0.6)
        else:  # poor
            base_chlorine = np.random.uniform(0.1, 0.4)
        
        # Chlorine decreases after heavy rain (lag effect)
        week_start = current_date - timedelta(days=7)
        week_rain = rainfall_df[(rainfall_df['ward']==ward) & 
                                (rainfall_df['date'] >= week_start.date()) &
                                (rainfall_df['date'] <= current_date.date())]['rainfall_mm'].sum()
        
        chlorine = max(0.05, base_chlorine - week_rain * 0.015 + np.random.normal(0, 0.08))
        turbidity = 1 + week_rain * 0.6 + np.random.gamma(2, 1.5)
        
        water_quality_data.append([current_date.date(), ward, 
                                   round(chlorine, 2), 
                                   round(turbidity, 1)])
    current_date += timedelta(days=7)

water_df = pd.DataFrame(water_quality_data, columns=['date', 'ward', 'chlorine_mgL', 'turbidity_NTU'])
water_df.to_csv('water_quality.csv', index=False)
print(f"Water quality data saved: {len(water_df)} records")

print("Generating health data with multiple diseases...")
# Generate health data (weekly cases for all diseases)
health_data = []
current_date = start_date + timedelta(days=7)

# Disease parameters (relative prevalence and seasonality)
disease_params = {
    'cholera': {'base_rate': 0.04, 'seasonal_peak': [6,7,8,9], 'severity': 1.2},
    'typhoid': {'base_rate': 0.06, 'seasonal_peak': [7,8,9,10], 'severity': 1.0},
    'dysentery': {'base_rate': 0.05, 'seasonal_peak': [5,6,7,8], 'severity': 0.9},
    'hepatitis': {'base_rate': 0.03, 'seasonal_peak': [8,9,10,11], 'severity': 1.1}
}

while current_date <= end_date:
    for ward in wards:
        # Get previous week total cases
        prev_cases = 0
        prev_health = [h for h in health_data if h[1]==ward and h[0] == (current_date - timedelta(days=7)).date()]
        if prev_health:
            prev_cases = sum(prev_health[0][2:6])  # Sum all diseases
        
        # Get recent rainfall (last 14 days)
        rain_start = current_date - timedelta(days=14)
        rain_end = current_date - timedelta(days=1)
        rain_period = rainfall_df[(rainfall_df['ward']==ward) & 
                                 (rainfall_df['date'] >= rain_start.date()) &
                                 (rainfall_df['date'] <= rain_end.date())]
        
        last_14d_rain = rain_period['rainfall_mm'].mean() if len(rain_period) > 0 else 0

        # Get latest water quality
        latest_water = water_df[(water_df['ward']==ward) & 
                               (water_df['date'] <= current_date.date())].sort_values('date', ascending=False)
        if not latest_water.empty:
            chlorine = latest_water.iloc[0]['chlorine_mgL']
            turbidity = latest_water.iloc[0]['turbidity_NTU']
        else:
            chlorine = 0.5
            turbidity = 2.0

        # Calculate risk multiplier (common for all diseases)
        risk_mult = 1.0
        risk_mult += 0.08 * min(prev_cases, 15)
        risk_mult += 0.15 * min(last_14d_rain / 25, 6)
        risk_mult += max(0, (0.4 - chlorine)) * 2.5
        risk_mult += min(turbidity / 8, 2.5)
        risk_mult = max(0.5, min(risk_mult, 12))

        # Generate cases for each disease
        disease_cases = []
        month = current_date.month
        
        for disease, params in disease_params.items():
            # Base rate per 1000 population
            base_rate = params['base_rate']
            
            # Seasonal adjustment
            seasonal_factor = 1.5 if month in params['seasonal_peak'] else 1.0
            
            # Infrastructure adjustment
            infra = ward_infrastructure[ward]
            if infra == 'poor':
                infra_factor = 1.4
            elif infra == 'average':
                infra_factor = 1.0
            else:  # good
                infra_factor = 0.7
            
            # Calculate expected cases
            population_in_thousands = ward_population[ward] / 1000
            expected = (base_rate * population_in_thousands * 
                       risk_mult * seasonal_factor * infra_factor * params['severity'])
            
            # Ensure minimum for Poisson
            expected = max(0.1, min(expected, 50))
            
            # Generate cases
            try:
                cases = int(np.random.poisson(expected))
            except:
                cases = int(expected)
            
            disease_cases.append(cases)
        
        # Add to health data [date, ward, cholera, typhoid, dysentery, hepatitis]
        health_data.append([current_date.date(), ward] + disease_cases)
    
    current_date += timedelta(days=7)
    
    # Print progress
    if current_date.month == 1 and current_date.day <= 7:
        print(f"Progress: Generated up to {current_date.date()}")

# Create health dataframe
health_df = pd.DataFrame(health_data, columns=['date', 'ward', 
                                                'cholera_cases', 'typhoid_cases', 
                                                'dysentery_cases', 'hepatitis_cases'])
health_df.to_csv('health.csv', index=False)
print(f"Health data saved: {len(health_df)} records")

# Summary statistics
print("\n" + "="*60)
print("DATA GENERATION COMPLETE - COIMBATORE DISTRICT")
print("="*60)
print(f"Total wards: {len(wards)}")
print(f"Wards: {', '.join(wards[:5])}... (and {len(wards)-5} more)")
print(f"Date range: {start_date.date()} to {end_date.date()}")

print("\nDisease Statistics (Average Weekly Cases):")
for disease in ['cholera_cases', 'typhoid_cases', 'dysentery_cases', 'hepatitis_cases']:
    avg = health_df[disease].mean()
    max_val = health_df[disease].max()
    print(f"  {disease:15s}: Avg {avg:.2f}, Max {max_val}")

print("\nSample health data (first 5 records):")
print(health_df.head())

# Save ward information
ward_info = pd.DataFrame({
    'ward': wards,
    'population': [ward_population[w] for w in wards],
    'infrastructure': [ward_infrastructure[w] for w in wards],
    'latitude': [11.0168 + np.random.uniform(-0.08, 0.08) for _ in wards],
    'longitude': [76.9558 + np.random.uniform(-0.08, 0.08) for _ in wards]
})
ward_info.to_csv('ward_info.csv', index=False)
print("\nWard information saved to ward_info.csv")
