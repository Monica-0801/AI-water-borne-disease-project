import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import shap
from datetime import datetime, timedelta
import calendar

# Page config
st.set_page_config(layout="wide", page_title="Coimbatore Disease EWS")
st.markdown("""
<style>
    /* Risk Legend Styling - Direct targeting */
    div.risk-legend {
        background-color: #1e293b !important;  /* Dark blue background */
        padding: 20px !important;
        border-radius: 10px !important;
        border: 2px solid #334155 !important;
       
    }
    
    div.risk-legend p {
        color: white !important;
        font-size: 16px !important;
        margin: 10px 0 !important;
    }
    
    div.risk-legend span {
        font-size: 20px !important;
    }
    
    div.risk-legend b {
        color: white !important;
    }
    
    /* Metric cards styling */
    div.custom-metric {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 10px 0;
    }
    
    div.custom-metric p.label {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }
    
    div.custom-metric p.value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 700;
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)
# Title
st.title("🌊 AI-Based Early Warning System for Water-Borne Diseases")
st.markdown("### Coimbatore District - Multi-Disease Surveillance")

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb_model.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        threshold = joblib.load('risk_threshold.pkl')
        return model, feature_cols, threshold
    except:
        st.warning("Model files not found. Please run train_model.py first.")
        return None, None, None

model, feature_cols, risk_threshold = load_model()

@st.cache_data
def load_data():
    try:
        health = pd.read_csv('health.csv', parse_dates=['date'])
        rainfall = pd.read_csv('rainfall.csv', parse_dates=['date'])
        water = pd.read_csv('water_quality.csv', parse_dates=['date'])
        ward_info = pd.read_csv('ward_info.csv')
        return health, rainfall, water, ward_info
    except:
        st.error("Data files not found. Please run generate.py first.")
        return None, None, None, None

health, rainfall, water, ward_info = load_data()

if health is None:
    st.stop()

# Ensure dates are datetime with mixed format handling
health['date'] = pd.to_datetime(health['date'], format='mixed', dayfirst=True, errors='coerce')
rainfall['date'] = pd.to_datetime(rainfall['date'], format='mixed', dayfirst=True, errors='coerce')
water['date'] = pd.to_datetime(water['date'], format='mixed', dayfirst=True, errors='coerce')

# Drop any rows with invalid dates
health = health.dropna(subset=['date'])
rainfall = rainfall.dropna(subset=['date'])
water = water.dropna(subset=['date'])

# Real Coimbatore ward names with coordinates from ward_info
wards = ward_info['ward'].tolist()
ward_coordinates = {row['ward']: (row['latitude'], row['longitude']) 
                   for _, row in ward_info.iterrows()}
ward_population = {row['ward']: row['population'] for _, row in ward_info.iterrows()}
ward_infrastructure = {row['ward']: row['infrastructure'] for _, row in ward_info.iterrows()}

# Disease configuration
diseases = ['cholera', 'typhoid', 'dysentery', 'hepatitis']
disease_colors = {'cholera': '#1f77b4', 'typhoid': '#ff7f0e', 
                  'dysentery': '#2ca02c', 'hepatitis': '#d62728'}
disease_names = {'cholera': 'Cholera', 'typhoid': 'Typhoid', 
                 'dysentery': 'Dysentery', 'hepatitis': 'Hepatitis'}

# ============================================================================
# EARLY WARNING SYSTEM
# ============================================================================
class EarlyWarningSystem:
    def __init__(self):
        self.alerts = []
        self.alert_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def generate_alerts(self, results_df):
        """Generate alerts based on risk scores"""
        alerts = []
        
        for _, row in results_df.iterrows():
            ward = row['ward']
            risk_score = row['risk_score']
            
            if risk_score >= self.alert_thresholds['high']:
                alert_level = "🔴 CRITICAL"
                message = f"CRITICAL: {ward} at {risk_score:.1%} risk - IMMEDIATE ACTION REQUIRED"
                actions = [
                    "Deploy emergency medical team",
                    "Issue boil water advisory",
                    "Conduct emergency chlorination",
                    "Activate emergency response protocol"
                ]
            elif risk_score >= self.alert_thresholds['medium']:
                alert_level = "🟠 HIGH"
                message = f"HIGH: {ward} at {risk_score:.1%} risk - Prepare for intervention"
                actions = [
                    "Increase water quality monitoring",
                    "Stockpile medical supplies",
                    "Alert local health workers",
                    "Prepare public awareness campaign"
                ]
            elif risk_score >= self.alert_thresholds['low']:
                alert_level = "🟡 MEDIUM"
                message = f"MEDIUM: {ward} at {risk_score:.1%} risk - Monitor closely"
                actions = [
                    "Increase surveillance",
                    "Review water quality data",
                    "Alert community health volunteers"
                ]
            else:
                continue  # No alert for low risk
            
            # Get contributing factors
            contributing_factors = []
            if row['chlorine'] < 0.3:
                contributing_factors.append("Low chlorine levels")
            if row['turbidity'] > 5:
                contributing_factors.append("High turbidity")
            if row['rain_7d'] > 50:
                contributing_factors.append("Heavy rainfall")
            if row['prev_cases'] > 5:
                contributing_factors.append(f"Recent cases ({int(row['prev_cases'])})")
            
            alerts.append({
                'ward': ward,
                'risk_score': risk_score,
                'level': alert_level,
                'message': message,
                'actions': actions,
                'factors': contributing_factors,
                'timestamp': datetime.now()
            })
        
        return sorted(alerts, key=lambda x: x['risk_score'], reverse=True)
    
    def display_alerts(self, alerts):
        """Display alerts in Streamlit"""
        if not alerts:
            st.success("✅ No active alerts. All wards at low risk.")
            return
        
        st.subheader("🚨 Active Early Warnings")
        
        for alert in alerts:
            if "CRITICAL" in alert['level']:
                with st.expander(f"{alert['level']} - {alert['ward']}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Risk Score:** {alert['risk_score']:.1%}")
                        st.markdown(f"**Message:** {alert['message']}")
                        st.markdown("**Contributing Factors:**")
                        for factor in alert['factors']:
                            st.markdown(f"- {factor}")
                    
                    with col2:
                        st.markdown("**Required Actions:**")
                        for action in alert['actions']:
                            st.markdown(f"- {action}")
                        
                        # Add acknowledge button
                        if st.button(f"Acknowledge Alert - {alert['ward']}", key=f"ack_{alert['ward']}"):
                            st.success(f"Alert acknowledged for {alert['ward']}")
            
            elif "HIGH" in alert['level']:
                with st.expander(f"{alert['level']} - {alert['ward']}"):
                    st.markdown(f"**Risk Score:** {alert['risk_score']:.1%}")
                    st.markdown(f"**Message:** {alert['message']}")
                    st.markdown("**Recommended Actions:**")
                    for action in alert['actions'][:3]:  # Show top 3
                        st.markdown(f"- {action}")
            
            else:  # MEDIUM
                st.warning(f"{alert['level']} - {alert['ward']}: {alert['message']}")

# Initialize warning system
ews = EarlyWarningSystem()

# ============================================================================
# CURRENT RISK ASSESSMENT
# ============================================================================
def prepare_features_for_date(target_date):
    """Prepare features for prediction"""
    # Ensure target_date is datetime
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date, format='mixed', dayfirst=True)
    
    rows = []
    valid_wards = []
    
    for ward in wards:
        # Rainfall features
        start = target_date - pd.Timedelta(days=7)
        end = target_date
        rain_7d = rainfall[(rainfall['date'] >= start) & 
                          (rainfall['date'] < end) & 
                          (rainfall['ward']==ward)]['rainfall_mm'].sum()
        
        start_prev = target_date - pd.Timedelta(days=14)
        end_prev = target_date - pd.Timedelta(days=7)
        rain_prev = rainfall[(rainfall['date'] >= start_prev) & 
                            (rainfall['date'] < end_prev) & 
                            (rainfall['ward']==ward)]['rainfall_mm'].sum()
        
        rain_14d = rain_7d + rain_prev
        
        # Water quality
        wq = water[(water['ward']==ward) & (water['date'] <= target_date)].sort_values('date', ascending=False)
        if not wq.empty:
            chlorine = wq.iloc[0]['chlorine_mgL']
            turbidity = wq.iloc[0]['turbidity_NTU']
            if len(wq) > 1:
                chlorine_trend = chlorine - wq.iloc[1]['chlorine_mgL']
                turbidity_trend = turbidity - wq.iloc[1]['turbidity_NTU']
            else:
                chlorine_trend = 0
                turbidity_trend = 0
        else:
            chlorine = 0.5
            turbidity = 2.0
            chlorine_trend = 0
            turbidity_trend = 0
        
        # Previous week cases
        prev_health = health[(health['ward']==ward) & 
                            (health['date'] == target_date - pd.Timedelta(days=7))]
        if not prev_health.empty:
            prev_total = (prev_health.iloc[0]['cholera_cases'] + 
                         prev_health.iloc[0]['typhoid_cases'] +
                         prev_health.iloc[0]['dysentery_cases'] + 
                         prev_health.iloc[0]['hepatitis_cases'])
            prev_cholera = prev_health.iloc[0]['cholera_cases']
            prev_typhoid = prev_health.iloc[0]['typhoid_cases']
            prev_dysentery = prev_health.iloc[0]['dysentery_cases']
            prev_hepatitis = prev_health.iloc[0]['hepatitis_cases']
        else:
            prev_total = 0
            prev_cholera = prev_typhoid = prev_dysentery = prev_hepatitis = 0
        
        month = target_date.month
        
        # Create feature row
        row = {
            'month': month,
            'rain_7d': rain_7d,
            'rain_prev_week': rain_prev,
            'rain_14d': rain_14d,
            'chlorine': chlorine,
            'chlorine_trend': chlorine_trend,
            'turbidity': turbidity,
            'turbidity_trend': turbidity_trend,
            'prev_total_cases': prev_total,
            'prev_cholera': prev_cholera,
            'prev_typhoid': prev_typhoid,
            'prev_dysentery': prev_dysentery,
            'prev_hepatitis': prev_hepatitis
        }
        
        # Add ward dummies
        for col in feature_cols:
            if col.startswith('ward_'):
                ward_name = col.replace('ward_', '')
                row[col] = 1 if ward_name == ward else 0
        
        rows.append(row)
        valid_wards.append(ward)
    
    df = pd.DataFrame(rows)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_cols]
    return df, valid_wards

# Get latest date (ensure it's datetime)
latest_date = health['date'].max()
if isinstance(latest_date, str):
    latest_date = pd.to_datetime(latest_date)

st.info(f"📊 Latest data date: {latest_date.strftime('%Y-%m-%d')}")

# Prepare features and predict
if model is not None:
    X_latest, valid_wards = prepare_features_for_date(latest_date)
    
    # Predict risk
    risk_probs = model.predict_proba(X_latest)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'ward': valid_wards,
        'risk_score': risk_probs,
        'chlorine': X_latest['chlorine'].values,
        'turbidity': X_latest['turbidity'].values,
        'rain_7d': X_latest['rain_7d'].values,
        'prev_cases': X_latest['prev_total_cases'].values
    })
    
    # Add disease-specific data for latest week
    latest_health = health[health['date'] == latest_date]
    if not latest_health.empty:
        for disease in ['cholera_cases', 'typhoid_cases', 'dysentery_cases', 'hepatitis_cases']:
            disease_map = dict(zip(latest_health['ward'], latest_health[disease]))
            results[disease] = results['ward'].map(disease_map).fillna(0).astype(int)
    else:
        # If no data for latest date, use most recent available
        latest_health = health.sort_values('date').groupby('ward').last().reset_index()
        for disease in ['cholera_cases', 'typhoid_cases', 'dysentery_cases', 'hepatitis_cases']:
            disease_map = dict(zip(latest_health['ward'], latest_health[disease]))
            results[disease] = results['ward'].map(disease_map).fillna(0).astype(int)
    
    # Risk classification
    def risk_category(score):
        if score < 0.3:
            return 'Low'
        elif score < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    results['risk_level'] = results['risk_score'].apply(risk_category)
    results['color'] = results['risk_level'].map({'Low': 'green', 'Medium': 'orange', 'High': 'red'})
    
    # ============================================================================
    # DISPLAY EARLY WARNINGS
    # ============================================================================
    alerts = ews.generate_alerts(results)
    ews.display_alerts(alerts)
    
    # ============================================================================
    # TABS
    # ============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Risk Map", "📊 Disease Analysis", "📈 Forecast", "🏥 Ward Details"])
    
    # ============================================================================
    # TAB 1: RISK MAP
    # ============================================================================
    with tab1:
        st.subheader(f"Current Risk Assessment - {latest_date.strftime('%B %d, %Y')}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            m = folium.Map(location=[11.0168, 76.9558], zoom_start=12)
            
            for _, row in results.iterrows():
                if row['ward'] in ward_coordinates:
                    lat, lon = ward_coordinates[row['ward']]
                    
                    # Create detailed popup
                    popup_html = f"""
                    <div style="font-family: Arial; width: 250px;">
                        <h4 style="margin:0">{row['ward']}</h4>
                        <p style="margin:5px 0">
                            <b>Risk:</b> {row['risk_score']:.1%} 
                            <span style="color:{row['color']}">● {row['risk_level']}</span>
                        </p>
                        <hr style="margin:5px 0">
                        <p style="margin:5px 0"><b>Current Cases:</b></p>
                        <ul style="margin:2px 0">
                            <li>Cholera: {row['cholera_cases']}</li>
                            <li>Typhoid: {row['typhoid_cases']}</li>
                            <li>Dysentery: {row['dysentery_cases']}</li>
                            <li>Hepatitis: {row['hepatitis_cases']}</li>
                        </ul>
                        <hr style="margin:5px 0">
                        <p style="margin:5px 0">
                            <b>Water Quality:</b><br>
                            Chlorine: {row['chlorine']:.2f} mg/L<br>
                            Turbidity: {row['turbidity']:.1f} NTU
                        </p>
                    </div>
                    """
                    
                    # Scale radius by risk
                    radius = 10 + int(row['risk_score'] * 20)
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=folium.Popup(popup_html, max_width=300),
                        color=row['color'],
                        fill=True,
                        fillColor=row['color'],
                        fillOpacity=0.7,
                        weight=2,
                        tooltip=f"{row['ward']}: {row['risk_level']} Risk"
                    ).add_to(m)
            
            folium_static(m, width=800)
        
        with col2:
            st.subheader("Risk Legend")
            st.markdown("""
            <div style="padding: 10px; background: black; border-radius: 5px;">
                <p><span style="color: green;">●</span> <b>Low Risk</b> (&lt;30%)</p>
                <p><span style="color: orange;">●</span> <b>Medium Risk</b> (30-60%)</p>
                <p><span style="color: red;">●</span> <b>High Risk</b> (>60%)</p>
                <p><span style="color: purple;">●</span> Circle size = risk level</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Total Wards", len(results))
            st.metric("High Risk Wards", len(results[results['risk_level'] == 'High']))
            st.metric("Medium Risk Wards", len(results[results['risk_level'] == 'Medium']))
    
    # ============================================================================
    # TAB 2: DISEASE ANALYSIS
    # ============================================================================
    with tab2:
        st.subheader("Disease-Specific Analysis")
        
        # Disease selector
        selected_disease = st.selectbox("Select Disease:", diseases, format_func=lambda x: disease_names[x])
        disease_col = f"{selected_disease}_cases"
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cases = results[disease_col].sum()
            st.metric(f"Total {disease_names[selected_disease]} Cases", int(total_cases))
        with col2:
            avg_cases = results[disease_col].mean()
            st.metric("Average per Ward", f"{avg_cases:.1f}")
        with col3:
            max_cases = results[disease_col].max()
            st.metric("Max Cases", int(max_cases))
        with col4:
            wards_with_cases = (results[disease_col] > 0).sum()
            st.metric("Wards Affected", wards_with_cases)
        
        # Bar chart of disease distribution
        st.subheader(f"{disease_names[selected_disease]} Distribution by Ward")
        
        # Sort wards by cases for better visualization
        sorted_results = results.sort_values(disease_col, ascending=True)
        
        fig = px.bar(sorted_results, 
                     x='ward', 
                     y=disease_col,
                     title=f"{disease_names[selected_disease]} Cases by Ward",
                     color=disease_col,
                     color_continuous_scale='Reds',
                     labels={disease_col: 'Number of Cases', 'ward': 'Ward'})
        
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Disease comparison chart
        st.subheader("Disease Comparison by Ward")
        
        # Prepare data for comparison (top 10 wards by total cases)
        results['total_cases'] = results[['cholera_cases', 'typhoid_cases', 'dysentery_cases', 'hepatitis_cases']].sum(axis=1)
        top_wards = results.nlargest(10, 'total_cases')['ward'].tolist()
        
        comparison_data = []
        for _, row in results[results['ward'].isin(top_wards)].iterrows():
            for disease in diseases:
                comparison_data.append({
                    'ward': row['ward'],
                    'disease': disease_names[disease],
                    'cases': row[f'{disease}_cases']
                })
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(comp_df, x='ward', y='cases', color='disease',
                     title="Disease Cases by Ward (Top 10 Wards)",
                     barmode='group',
                     color_discrete_map={'Cholera': '#1f77b4', 'Typhoid': '#ff7f0e', 
                                        'Dysentery': '#2ca02c', 'Hepatitis': '#d62728'})
        
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart of disease distribution
        st.subheader("Overall Disease Distribution")
        
        total_by_disease = [
            results['cholera_cases'].sum(),
            results['typhoid_cases'].sum(),
            results['dysentery_cases'].sum(),
            results['hepatitis_cases'].sum()
        ]
        
        fig = px.pie(values=total_by_disease, 
                     names=[disease_names[d] for d in diseases],
                     title="Proportion of Cases by Disease",
                     color_discrete_map={'Cholera': '#1f77b4', 'Typhoid': '#ff7f0e', 
                                        'Dysentery': '#2ca02c', 'Hepatitis': '#d62728'})
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # TAB 3: FORECAST
    # ============================================================================
    with tab3:
        st.subheader("7-Day Risk Forecast")
        
        # Generate simple forecast (extend current trends)
        forecast_dates = [latest_date + timedelta(days=i) for i in range(1, 8)]
        forecast_data = []
        
        for ward in wards:
            ward_current = results[results['ward'] == ward].iloc[0]
            base_risk = ward_current['risk_score']
            
            for i, fdate in enumerate(forecast_dates):
                # Simple trend: risk changes based on rainfall forecast
                month = fdate.month
                rain_factor = 1.2 if 6 <= month <= 9 else 1.0
                trend = 1 + (i * 0.02)  # slight increase over time
                forecast_risk = min(0.95, base_risk * rain_factor * trend)
                
                forecast_data.append({
                    'ward': ward,
                    'date': fdate,
                    'risk_score': forecast_risk,
                    'risk_level': risk_category(forecast_risk)
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Ward selector
        selected_ward_forecast = st.selectbox("Select Ward for Forecast:", wards)
        
        # Filter for selected ward
        ward_forecast = forecast_df[forecast_df['ward'] == selected_ward_forecast].copy()
        ward_forecast = ward_forecast.sort_values('date')
        
        # Create forecast chart
        fig = go.Figure()
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=ward_forecast['date'],
            y=ward_forecast['risk_score'],
            mode='lines+markers',
            name='Forecasted Risk',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        # Add current risk as point
        current_risk = results[results['ward'] == selected_ward_forecast].iloc[0]['risk_score']
        fig.add_trace(go.Scatter(
            x=[latest_date],
            y=[current_risk],
            mode='markers',
            name='Current Risk',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                     annotation_text="High Risk")
        
        fig.update_layout(
            title=f"7-Day Risk Forecast - {selected_ward_forecast}",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            yaxis_range=[0, 1],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast table
        st.subheader("Daily Forecast Details")
        forecast_display = ward_forecast[['date', 'risk_score', 'risk_level']].copy()
        forecast_display['risk_score'] = forecast_display['risk_score'].apply(lambda x: f"{x:.1%}")
        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(forecast_display, use_container_width=True)
        
        # Risk summary for all wards
        st.subheader("7-Day Risk Summary - All Wards")
        
        # Calculate average forecast risk for each ward
        avg_forecast = forecast_df.groupby('ward')['risk_score'].mean().reset_index()
        avg_forecast = avg_forecast.sort_values('risk_score', ascending=False)
        avg_forecast['risk_level'] = avg_forecast['risk_score'].apply(risk_category)
        
        fig = px.bar(avg_forecast.head(15), 
                     x='ward', 
                     y='risk_score',
                     color='risk_level',
                     color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                     title="Top 15 Wards by Average Forecasted Risk (Next 7 Days)",
                     labels={'risk_score': 'Average Risk Score', 'ward': 'Ward'})
        
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # TAB 4: WARD DETAILS
    # ============================================================================
    with tab4:
        st.subheader("Detailed Ward Information")
        
        selected_ward = st.selectbox("Select Ward for Details:", wards, key="ward_details")
        
        # Get ward data
        ward_result = results[results['ward'] == selected_ward].iloc[0]
        ward_pop = ward_population[selected_ward]
        ward_infra = ward_infrastructure[selected_ward]
        
        # Ward header with risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"## {selected_ward}")
            st.markdown(f"**Infrastructure:** {ward_infra.title()}")
            st.markdown(f"**Population:** {ward_pop:,}")
        
        with col2:
            risk_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}[ward_result['risk_level']]
            st.markdown(f"## Risk: <span style='color:{risk_color}'>{ward_result['risk_level']}</span>", 
                       unsafe_allow_html=True)
            st.markdown(f"**Risk Score:** {ward_result['risk_score']:.1%}")
        
        with col3:
            st.markdown("## Current Cases")
            st.markdown(f"**Cholera:** {ward_result['cholera_cases']}")
            st.markdown(f"**Typhoid:** {ward_result['typhoid_cases']}")
            st.markdown(f"**Dysentery:** {ward_result['dysentery_cases']}")
            st.markdown(f"**Hepatitis:** {ward_result['hepatitis_cases']}")
        
        # Environmental factors
        st.subheader("Environmental Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chlorine_status = "✅ Good" if ward_result['chlorine'] > 0.5 else "⚠️ Low" if ward_result['chlorine'] > 0.3 else "❌ Critical"
            st.metric("Chlorine Level", f"{ward_result['chlorine']:.2f} mg/L", chlorine_status)
        
        with col2:
            turbidity_status = "✅ Good" if ward_result['turbidity'] < 3 else "⚠️ High" if ward_result['turbidity'] < 5 else "❌ Very High"
            st.metric("Turbidity", f"{ward_result['turbidity']:.1f} NTU", turbidity_status)
        
        with col3:
            st.metric("Rainfall (7 days)", f"{ward_result['rain_7d']:.1f} mm")
        
        # Historical chart for this ward
        st.subheader("Historical Disease Trends")
        
        ward_history = health[health['ward'] == selected_ward].copy()
        ward_history = ward_history.sort_values('date')
        
        # Resample to monthly for cleaner visualization
        ward_history['year_month'] = ward_history['date'].dt.to_period('M')
        monthly_history = ward_history.groupby('year_month').agg({
            'cholera_cases': 'sum',
            'typhoid_cases': 'sum',
            'dysentery_cases': 'sum',
            'hepatitis_cases': 'sum'
        }).reset_index()
        monthly_history['year_month'] = monthly_history['year_month'].astype(str)
        
        fig = go.Figure()
        
        for disease in diseases:
            fig.add_trace(go.Scatter(
                x=monthly_history['year_month'],
                y=monthly_history[f'{disease}_cases'],
                mode='lines+markers',
                name=disease_names[disease],
                line=dict(color=disease_colors[disease], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"Disease Trends (Monthly) - {selected_ward}",
            xaxis_title="Month",
            yaxis_title="Number of Cases",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent cases table
        st.subheader("Recent Weekly Cases")
        recent_cases = ward_history.tail(8)[['date', 'cholera_cases', 'typhoid_cases', 
                                             'dysentery_cases', 'hepatitis_cases']].copy()
        recent_cases['date'] = recent_cases['date'].dt.strftime('%Y-%m-%d')
        recent_cases.columns = ['Date', 'Cholera', 'Typhoid', 'Dysentery', 'Hepatitis']
        st.dataframe(recent_cases, use_container_width=True)
        
        # Early warning actions for this ward
        if ward_result['risk_level'] in ['High', 'Medium']:
            st.subheader("🚨 Recommended Actions")
            
            if ward_result['risk_level'] == 'High':
                st.error("**CRITICAL - Immediate Actions Required:**")
                actions = [
                    "Deploy rapid response team immediately",
                    "Issue public health advisory",
                    "Conduct emergency water chlorination",
                    "Set up temporary medical camps",
                    "Distribute oral rehydration salts",
                    "Activate community health workers"
                ]
            else:
                st.warning("**ALERT - Preventive Actions Recommended:**")
                actions = [
                    "Increase water quality monitoring frequency",
                    "Alert local health clinics",
                    "Conduct public awareness campaign",
                    "Stockpile essential medicines",
                    "Review sanitation measures"
                ]
            
            for action in actions:
                st.markdown(f"- {action}")

    # ============================================================================
    # EXPORT AND NOTIFICATION
    # ============================================================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Risk Report"):
            # Create report
            report = results[['ward', 'risk_score', 'risk_level', 'cholera_cases', 
                             'typhoid_cases', 'dysentery_cases', 'hepatitis_cases',
                             'chlorine', 'turbidity', 'rain_7d']].copy()
            report['risk_score'] = report['risk_score'].apply(lambda x: f"{x:.2%}")
            report.to_csv('risk_report.csv', index=False)
            st.success("✅ Report saved as 'risk_report.csv'")

    with col2:
        if st.button("📧 Send Email Alerts"):
            st.info("📧 Email alerts would be sent to: health-officer@coimbatore.gov.in, water-board@coimbatore.gov.in")
            # In production, you'd integrate with SMTP here

    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()

else:
    st.error("Model not loaded. Please run train_model.py first.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>🏥 Coimbatore District Health Department | 🌧️ Indian Meteorological Department | 💧 TWAD Board</p>
    <p>🕐 Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
""", unsafe_allow_html=True)
