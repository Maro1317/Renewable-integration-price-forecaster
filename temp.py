import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# UPDATE THIS PATH IF NEEDED
FILE_PATH = r"C:\Users\ammar\Downloads\Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv"

# Weather Configuration (Calgary Proxy)
LATITUDE = 51.05
LONGITUDE = -114.07

# ==========================================
# STEP 1: LOAD & CLEAN GRID DATA
# ==========================================
def process_aeso_data(file_path):
    print(f"Loading data from: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return None

    # Decoder Ring
    wind_codes = ['AKE1', 'CR1', 'KHW1', 'NEP1', 'BUL1', 'BUL2', 'CRR1', 'CRR2', 'RIV1', 'WHT1', 'WHT2', 'RTL1', 'FMG1', 'GDP1', 'GOC1']
    solar_codes = ['TVS1', 'STR1', 'STR2', 'HYS1', 'JER1', 'BRK1', 'BRK2', 'COL1', 'BUR1', 'BSC1', 'CLR1', 'CLR2', 'HUL1', 'INF1', 'VXH1']

    # Filter for existing assets
    actual_wind = [c for c in wind_codes if c in df.columns]
    actual_solar = [c for c in solar_codes if c in df.columns]

    print(f"Found {len(actual_wind)} Wind Assets and {len(actual_solar)} Solar Assets.")

    # Aggregation
    df['Total_Wind'] = df[actual_wind].sum(axis=1)
    df['Total_Solar'] = df[actual_solar].sum(axis=1)
    df['Net_Load'] = df['ACTUAL_AIL'] - (df['Total_Wind'] + df['Total_Solar'])

    # Select Columns
    usable_columns = [
        'Date_Begin_GMT',
        'Date_Begin_Local',
        'ACTUAL_POOL_PRICE',
        'ACTUAL_AIL',
        'Total_Wind',
        'Total_Solar',
        'Net_Load'
    ]

    clean_df = df[usable_columns].copy()
    
    # Time Conversion
    clean_df['Date_Begin_Local'] = pd.to_datetime(clean_df['Date_Begin_Local'])
    clean_df = clean_df.sort_values('Date_Begin_Local')
    
    print("Grid data processing complete.")
    return clean_df

# ==========================================
# STEP 2: FETCH WEATHER (Original Calgary Only)
# ==========================================
def get_historical_weather(df_clean):
    print("Fetching weather data from Open-Meteo API...")
    
    df_clean['Date_Begin_GMT'] = pd.to_datetime(df_clean['Date_Begin_GMT'])
    start_date = df_clean['Date_Begin_GMT'].min().strftime('%Y-%m-%d')
    end_date = df_clean['Date_Begin_GMT'].max().strftime('%Y-%m-%d')
    
    print(f"Requesting data from {start_date} to {end_date}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_100m", "shortwave_radiation"],
        "timezone": "GMT"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("Error downloading weather data!")
        return None

    data = response.json()
    hourly_data = data['hourly']
    
    weather_df = pd.DataFrame({
        'Date_Begin_GMT': pd.to_datetime(hourly_data['time']),
        'Temp_C': hourly_data['temperature_2m'],
        'Wind_Speed_100m': hourly_data['wind_speed_100m'],
        'Solar_Irradiance': hourly_data['shortwave_radiation']
    })
    
    merged_df = pd.merge(df_clean, weather_df, on='Date_Begin_GMT', how='inner')
    print("Weather data merged successfully!")
    return merged_df

# ==========================================
# STEP 3: TRAIN MODELS (NOW WITH LAGS)
# ==========================================
def train_component_models(final_df):
    print("\n--- TRAINING 3 SPECIALIST MODELS (WITH MEMORY) ---")
    
    # 1. Setup Time Features
    final_df['Hour'] = final_df['Date_Begin_Local'].dt.hour
    final_df['Month'] = final_df['Date_Begin_Local'].dt.month
    final_df['DayOfWeek'] = final_df['Date_Begin_Local'].dt.dayofweek
    final_df['Is_Weekend'] = (final_df['DayOfWeek'] >= 5).astype(int)

    # 2. FEATURE ENGINEERING: LAGS (The Fix)
    # We create a column that shows what happened exactly 24 hours ago
    print("Creating Lag-24 Features (Memory of Yesterday)...")
    final_df['Wind_Lag24'] = final_df['Total_Wind'].shift(24)
    final_df['Solar_Lag24'] = final_df['Total_Solar'].shift(24)
    final_df['Load_Lag24'] = final_df['ACTUAL_AIL'].shift(24)

    # IMPORTANT: Drop the first 24 rows (NaNs) or the model will crash
    final_df = final_df.dropna()

    # Split Data
    split_date = '2024-01-01'
    train = final_df[final_df['Date_Begin_Local'] < split_date].copy()
    test = final_df[final_df['Date_Begin_Local'] >= split_date].copy()

    # ------------------------------------------
    # MODEL 1: WIND PREDICTOR (With Lag)
    # ------------------------------------------
    print("1. Training Wind Model...")
    # New Input List: Wind Speed + Lag24
    features_wind = ['Wind_Speed_100m', 'Wind_Lag24', 'Month', 'Hour'] 
    
    model_wind = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
    model_wind.fit(train[features_wind], train['Total_Wind'])
    
    pred_wind = np.maximum(0, model_wind.predict(test[features_wind]))
    print(f"   > Wind Accuracy (R²): {r2_score(test['Total_Wind'], pred_wind):.2f}")

    # ------------------------------------------
    # MODEL 2: SOLAR PREDICTOR (With Lag)
    # ------------------------------------------
    print("2. Training Solar Model...")
    # New Input List: Solar Irradiance + Lag24
    features_solar = ['Solar_Irradiance', 'Solar_Lag24', 'Hour', 'Month']
    
    model_solar = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
    model_solar.fit(train[features_solar], train['Total_Solar'])
    
    pred_solar = np.maximum(0, model_solar.predict(test[features_solar]))
    print(f"   > Solar Accuracy (R²): {r2_score(test['Total_Solar'], pred_solar):.2f}")

    # ------------------------------------------
    # MODEL 3: DEMAND PREDICTOR (With Lag)
    # ------------------------------------------
    print("3. Training Demand Model...")
    # New Input List: Temp + Lag24
    features_load = ['Temp_C', 'Load_Lag24', 'Hour', 'DayOfWeek', 'Month', 'Is_Weekend']

    model_load = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
    model_load.fit(train[features_load], train['ACTUAL_AIL'])
    
    pred_load = model_load.predict(test[features_load])
    print(f"   > Demand Accuracy (R²): {r2_score(test['ACTUAL_AIL'], pred_load):.2f}")

    # ------------------------------------------
    # NET LOAD & REPORT
    # ------------------------------------------
    pred_net_load = pred_load - (pred_wind + pred_solar)
    actual_net_load = test['Net_Load']
    print(f"\n   > Net Load Accuracy (R²): {r2_score(actual_net_load, pred_net_load):.2f}")

    # JAN 1ST REPORT CARD
    print("\n=========================================================================")
    print("               JANUARY 1st 2024: DETAILED REPORT CARD")
    print("=========================================================================")
    
    jan1 = test.iloc[:24].copy()
    jan1['Pred_Wind'] = pred_wind[:24]
    jan1['Pred_Solar'] = pred_solar[:24]
    jan1['Pred_Load'] = pred_load[:24]
    jan1['Pred_Net_Load'] = pred_net_load[:24]

    # Rename for clarity
    jan1 = jan1.rename(columns={
        'Temp_C': '[IN] Temp',
        'Wind_Speed_100m': '[IN] Wind',
        'Solar_Irradiance': '[IN] Solar'
    })

    cols = [
        'Hour', 
        '[IN] Temp', 'Pred_Load',          
        '[IN] Wind', 'Pred_Wind',        
        '[IN] Solar', 'Pred_Solar',      
        'Net_Load', 'Pred_Net_Load'
    ]
    print(jan1[cols].to_string(index=False, float_format="%.0f"))
    
    # Visualization
    plt.figure(figsize=(15, 10))
    subset = 72
    plt.subplot(2, 1, 1)
    plt.plot(pred_load[:subset], label='Predicted Demand', color='black', linestyle='--')
    plt.plot(pred_wind[:subset], label='Predicted Wind', color='blue', alpha=0.6)
    plt.plot(pred_solar[:subset], label='Predicted Solar', color='orange', alpha=0.6)
    plt.title("Step 1: The Components (With Memory)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(actual_net_load.values[:subset], label='ACTUAL', color='grey', linewidth=3, alpha=0.5)
    plt.plot(pred_net_load[:subset], label='PREDICTED', color='red', linewidth=2)
    plt.title("Step 2: The Duck Curve (With Memory)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    df_grid = process_aeso_data(FILE_PATH)
    if df_grid is not None:
        df_final = get_historical_weather(df_grid)
        if df_final is not None:

            train_component_models(df_final)
