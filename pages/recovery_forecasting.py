import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import StringIO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score

# === Helper Functions ===
def convert_timestamp(ts):
    try:
        minutes, seconds = map(float, ts.split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan

@st.cache_data
def load_and_clean(csv_data):
    df = pd.read_csv(csv_data)
    df['timestamp_sec'] = df['timestamp'].apply(convert_timestamp)
    required = [
        'timestamp_sec', 'recovery_time_sec', 'workload_size_gb',
        'net_tx_mbps', 'net_rx_mbps', 'disk_write_MBps',
        'disk_reads_completed', 'cpu_user_delta', 'cpu_system_delta', 'cpu_iowait_delta'
    ]
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    return df_clean

@st.cache_resource
def train_models(df_clean):
    # Regression
    df_forecast = df_clean[df_clean['recovery_time_sec'] > 0].copy()
    features = [
        'workload_size_gb', 'net_tx_mbps', 'net_rx_mbps',
        'disk_write_MBps', 'disk_reads_completed',
        'cpu_user_delta', 'cpu_system_delta', 'cpu_iowait_delta'
    ]
    for feat in features:
        for lag in range(1, 4):
            df_forecast[f'{feat}_lag{lag}'] = df_forecast[feat].shift(lag)
    df_forecast.dropna(inplace=True)

    X = df_forecast[[c for c in df_forecast.columns if 'lag' in c]]
    y = df_forecast['recovery_time_sec']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Classification
    df_class = df_clean.copy()
    df_class['is_recovering_next'] = (df_class['recovery_time_sec'].shift(-1) > 0).astype(int)
    for feat in features:
        for lag in range(1, 4):
            df_class[f'{feat}_lag{lag}'] = df_class[feat].shift(lag)
    df_class.dropna(inplace=True)

    Xc = df_class[[c for c in df_class.columns if 'lag' in c]]
    yc = df_class['is_recovering_next']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42
    )
    model_cls = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_cls.fit(Xc_train, yc_train)
    y_cls_pred = (model_cls.predict(Xc_test) > 0.5).astype(int)
    cls_report = classification_report(yc_test, y_cls_pred, output_dict=True)

    return {
        'scaler': scaler,
        'model': model,
        'r2': r2,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'cls_report': cls_report,
        'features': features,
        'df_forecast': df_forecast
    }

@st.cache_data
def simulate_forecast(df_forecast, _scaler, _model, features, horizon):
    last = df_forecast.iloc[-1]
    sim = []
    for minute in range(horizon + 3):
        row = {}
        for f in features:
            if 'workload' in f:
                row[f] = last[f] + minute * 2
            elif 'net_tx' in f or 'net_rx' in f:
                row[f] = last[f] + minute * 1.5
            elif 'disk' in f:
                row[f] = last[f] + minute * 0.3
            else:
                row[f] = min(100.0, last[f] + minute * 0.2)
        sim.append(row)
    sim_df = pd.DataFrame(sim)
    inputs = []
    for i in range(horizon):
        r = {}
        for lag in range(1, 4):
            for f in features:
                r[f + f'_lag{lag}'] = sim_df.iloc[i + 3 - lag][f]
        inputs.append(r)
    inp_df = pd.DataFrame(inputs)
    cols = list(_scaler.feature_names_in_)
    inp_df = inp_df.reindex(columns=cols, fill_value=0)
    preds = _model.predict(_scaler.transform(inp_df))
    return pd.DataFrame({'minute': range(1, horizon+1), 'predicted_recovery_time_sec': preds})

# Streamlit Page
st.title('📈 Recovery Time Forecast Dashboard')
st.markdown('Combine forecasting and classification to monitor system recovery.')

# Data Input
uploaded = st.sidebar.file_uploader('Upload ts-analysis.csv', type='csv')
default_file = 'dataset/ts-analysis.csv'
csv_source = uploaded if uploaded is not None else default_file
df_clean = load_and_clean(csv_source)

# Train & Metrics
res = train_models(df_clean)

# Classification Section
st.subheader('▶️ Will Recovery Start Next?')
st.write('Predict if the system begins recovery in the next timeframe.')
cls = res['cls_report']
acc = cls['accuracy']; prec = cls['macro avg']['precision']
rec = cls['macro avg']['recall']; f1 = cls['macro avg']['f1-score']
col1, col2, col3, col4 = st.columns(4)
col1.metric('Accuracy', f"{acc:.4f}")
col2.metric('Precision', f"{prec:.4f}")
col3.metric('Recall', f"{rec:.4f}")
col4.metric('F1 Score', f"{f1:.4f}")

# Regression Section
st.subheader('▶️ Recovery Time Forecast (R²)')
st.metric('R² Score', f"{res['r2']:.4f}")

st.subheader('🔍 Actual vs Predicted')
df_vis = pd.DataFrame({'Actual': res['y_test'].values, 'Predicted': res['y_pred']})
st.line_chart(df_vis)

# Multi-Horizon Forecast Input
st.subheader('⏳ Multi-Horizon Forecast')
horizon = st.number_input('Horizon (minutes)', min_value=1, max_value=30, value=5)
fc = simulate_forecast(res['df_forecast'], res['scaler'], res['model'], res['features'], horizon)
st.line_chart(fc.set_index('minute'))
st.download_button('Download CSV', data=fc.to_csv(index=False), file_name=f'forecast_{horizon}min.csv', mime='text/csv')
