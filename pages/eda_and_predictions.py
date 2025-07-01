import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime
import joblib
import os

st.set_page_config(page_title="Ceph Accuracy Booster", layout="wide")
st.title("📊 Ceph Recovery & Replication Prediction (Final Optimized)")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    required_cols = [
        "timestamp", "workload_size_gb", "recovery_time_sec", "replication_rate_mbps",
        "net_tx_mbps", "net_rx_mbps", "disk_write_MBps", "cpu_iowait_delta", "cpu_system_delta"
    ]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Required columns missing.")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Feature engineering
    def add_engineered_features(df):
        df = df.copy()
        df["cpu_ratio"] = df["cpu_system_delta"] / (df["cpu_iowait_delta"] + 1)
        df["net_io_product"] = df["net_tx_mbps"] * df["net_rx_mbps"]
        df["disk_util_ratio"] = df["disk_write_MBps"] / (df["workload_size_gb"] + 1)
        return df

    df = add_engineered_features(df)
    os.makedirs("models", exist_ok=True)
    raw_df = df.copy()

    # Outlier removal
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

    df_recovery = remove_outliers_iqr(df[df["recovery_time_sec"] > 0], "recovery_time_sec")
    df_replication = remove_outliers_iqr(df[df["replication_rate_mbps"] > 0], "replication_rate_mbps")

    features = [
        "workload_size_gb", "net_tx_mbps", "net_rx_mbps", "disk_write_MBps",
        "cpu_iowait_delta", "cpu_system_delta", "hour", "dayofweek", "day", "month",
        "cpu_ratio", "net_io_product", "disk_util_ratio"
    ]

    Xr, yr = df_recovery[features], np.log1p(df_recovery["recovery_time_sec"])
    Xp, yp = df_replication[features], np.log1p(df_replication["replication_rate_mbps"])

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)

    @st.cache_resource
    def tune_and_train(X_train, y_train, model_name):
        param_grid = {
            'xgb': {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.08, 0.1]
            },
            'lgb': {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.08, 0.1]
            },
            'rf': {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 20, 25]
            }
        }

        models = {
            'xgb': XGBRegressor(objective='reg:squarederror', random_state=42),
            'lgb': LGBMRegressor(random_state=42),
            'rf': RandomForestRegressor(random_state=42)
        }

        tuned_models = {}
        for name in models:
            search = RandomizedSearchCV(
                models[name], param_distributions=param_grid[name],
                n_iter=10, cv=4, scoring='r2', n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            tuned_models[name] = search.best_estimator_
            st.write(f"✅ Best {name.upper()} for {model_name}:", search.best_params_)

        return tuned_models

    def is_model_compatible(model_path, expected_features):
        if not os.path.exists(model_path):
            return False
        try:
            model = joblib.load(model_path)
            base_est = model.estimators_[0][1]
            return hasattr(base_est, "feature_names_in_") and \
                   list(base_est.feature_names_in_) == expected_features
        except Exception:
            return False

    current_features = list(Xr_train.columns)

    if is_model_compatible("models/recovery_model.pkl", current_features) and \
       is_model_compatible("models/replication_model.pkl", current_features):
        
        st.success("✅ Loaded compatible trained models from disk.")
        recovery_model = joblib.load("models/recovery_model.pkl")
        replication_model = joblib.load("models/replication_model.pkl")

    else:
        st.warning("⚠️ Model mismatch or not found. Retraining...")

        tuned_r = tune_and_train(Xr_train, yr_train, "Recovery")
        recovery_model = StackingRegressor(
            estimators=[(k, tuned_r[k]) for k in tuned_r],
            final_estimator=LGBMRegressor()
        )
        recovery_model.fit(Xr_train, yr_train)
        joblib.dump(recovery_model, "models/recovery_model.pkl")

        tuned_p = tune_and_train(Xp_train, yp_train, "Replication")
        replication_model = StackingRegressor(
            estimators=[(k, tuned_p[k]) for k in tuned_p],
            final_estimator=XGBRegressor(objective='reg:squarederror')
        )
        replication_model.fit(Xp_train, yp_train)
        joblib.dump(replication_model, "models/replication_model.pkl")

    # === Evaluation ===
    r2_r = r2_score(np.expm1(yr_test), np.expm1(recovery_model.predict(Xr_test)))
    st.success(f"📈 Recovery Time R² Score: **{r2_r:.3f}**")

    r2_p = r2_score(np.expm1(yp_test), np.expm1(replication_model.predict(Xp_test)))
    st.success(f"📉 Replication Rate R² Score: **{r2_p:.3f}**")

    # === UI for Prediction ===
    mode = st.radio("Choose Mode", ["Analysis", "Predict Recovery Time", "Predict Replication Rate"])

    def get_user_input(ts):
        ts = pd.to_datetime(ts)
        input_data = {
            "workload_size_gb": st.number_input("Workload (GB)", min_value=0.0),
            "net_tx_mbps": st.number_input("Net TX (Mbps)", min_value=0.0),
            "net_rx_mbps": st.number_input("Net RX (Mbps)", min_value=0.0),
            "disk_write_MBps": st.number_input("Disk Write (MBps)", min_value=0.0),
            "cpu_iowait_delta": st.number_input("CPU IOWait", min_value=0.0),
            "cpu_system_delta": st.number_input("CPU System", min_value=0.0),
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "day": ts.day,
            "month": ts.month
        }
        input_data["cpu_ratio"] = input_data["cpu_system_delta"] / (input_data["cpu_iowait_delta"] + 1)
        input_data["net_io_product"] = input_data["net_tx_mbps"] * input_data["net_rx_mbps"]
        input_data["disk_util_ratio"] = input_data["disk_write_MBps"] / (input_data["workload_size_gb"] + 1)
        return pd.DataFrame([input_data])

    ts_default = str(df["timestamp"].iloc[-1])

    if mode == "Analysis":
        st.subheader("📈 Time Series Analysis")
        def plot_series(metric1, metric2, label1, label2):
            plt.figure(figsize=(12, 4))
            plt.plot(raw_df["timestamp"], raw_df[metric1], label=label1)
            plt.plot(raw_df["timestamp"], raw_df[metric2], label=label2)
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)

        st.markdown("#### Recovery Time Trends")
        plot_series("recovery_time_sec", "workload_size_gb", "Recovery Time (s)", "Workload (GB)")
        plot_series("recovery_time_sec", "net_tx_mbps", "Recovery Time (s)", "Net TX (Mbps)")
        plot_series("recovery_time_sec", "net_rx_mbps", "Recovery Time (s)", "Net RX (Mbps)")
        plot_series("recovery_time_sec", "disk_write_MBps", "Recovery Time (s)", "Disk Write (MBps)")
        plot_series("recovery_time_sec", "cpu_system_delta", "Recovery Time (s)", "CPU System Delta")
        plot_series("recovery_time_sec", "cpu_iowait_delta", "Recovery Time (s)", "CPU IOWait Delta")

        st.markdown("#### Replication Rate Trends")
        plot_series("replication_rate_mbps", "workload_size_gb", "Replication Rate (Mbps)", "Workload (GB)")
        plot_series("replication_rate_mbps", "net_tx_mbps", "Replication Rate (Mbps)", "Net TX (Mbps)")
        plot_series("replication_rate_mbps", "net_rx_mbps", "Replication Rate (Mbps)", "Net RX (Mbps)")
        plot_series("replication_rate_mbps", "disk_write_MBps", "Replication Rate (Mbps)", "Disk Write (MBps)")
        plot_series("replication_rate_mbps", "cpu_system_delta", "Replication Rate (Mbps)", "CPU System Delta")
        plot_series("replication_rate_mbps", "cpu_iowait_delta", "Replication Rate (Mbps)", "CPU IOWait Delta")

    elif mode == "Predict Recovery Time":
        st.subheader("🔍 Predict Recovery Time")

        workload_r = st.number_input("Workload Size (GB)", min_value=0.0, key="w_r")
        net_tx_r = st.number_input("Network TX (Mbps)", min_value=0.0, key="tx_r")
        net_rx_r = st.number_input("Network RX (Mbps)", min_value=0.0, key="rx_r")
        disk_write_r = st.number_input("Disk Write (MBps)", min_value=0.0, key="dw_r")
        cpu_iowait_r = st.number_input("CPU IOWait Delta", min_value=0.0, key="iowait_r")
        cpu_system_r = st.number_input("CPU System Delta", min_value=0.0, key="sys_r")
        ts_input_r = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=str(df["timestamp"].iloc[-1]), key="ts_r")

        if st.button("🔍 Predict Recovery"):
            try:
                ts = pd.to_datetime(ts_input_r)
                input_r = {
                    "workload_size_gb": workload_r,
                    "net_tx_mbps": net_tx_r,
                    "net_rx_mbps": net_rx_r,
                    "disk_write_MBps": disk_write_r,
                    "cpu_iowait_delta": cpu_iowait_r,
                    "cpu_system_delta": cpu_system_r,
                    "hour": ts.hour,
                    "dayofweek": ts.dayofweek,
                    "day": ts.day,
                    "month": ts.month,
                    "cpu_ratio": cpu_system_r / (cpu_iowait_r + 1),
                    "net_io_product": net_tx_r * net_rx_r,
                    "disk_util_ratio": disk_write_r / (workload_r + 1)
                    }
                df_r = pd.DataFrame([input_r])
                pred_r = np.expm1(recovery_model.predict(df_r)[0])
                st.success(f"📈 Predicted Recovery Time: **{pred_r:.2f} seconds**")
            except Exception as e:
                st.error(f"❌ Recovery Prediction Error: {e}")


    elif mode == "Predict Replication Rate":
        st.subheader("🔁 Predict Replication Rate")
        workload_p = st.number_input("Workload Size (GB)", min_value=0.0, key="w_p")
        net_tx_p = st.number_input("Network TX (Mbps)", min_value=0.0, key="tx_p")
        net_rx_p = st.number_input("Network RX (Mbps)", min_value=0.0, key="rx_p")
        disk_write_p = st.number_input("Disk Write (MBps)", min_value=0.0, key="dw_p")
        cpu_iowait_p = st.number_input("CPU IOWait Delta", min_value=0.0, key="iowait_p")
        cpu_system_p = st.number_input("CPU System Delta", min_value=0.0, key="sys_p")
        ts_input_p = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=str(df["timestamp"].iloc[-1]), key="ts_p")
        if st.button("🔁 Predict Replication"):
            try:
                ts = pd.to_datetime(ts_input_p)
                input_p = {
                    "workload_size_gb": workload_p,
                    "net_tx_mbps": net_tx_p,
                    "net_rx_mbps": net_rx_p,
                    "disk_write_MBps": disk_write_p,
                    "cpu_iowait_delta": cpu_iowait_p,
                    "cpu_system_delta": cpu_system_p,
                    "hour": ts.hour,
                    "dayofweek": ts.dayofweek,
                    "day": ts.day,
                    "month": ts.month,
                    "cpu_ratio": cpu_system_p / (cpu_iowait_p + 1),
                    "net_io_product": net_tx_p * net_rx_p,
                    "disk_util_ratio": disk_write_p / (workload_p + 1)
                    }
                df_p = pd.DataFrame([input_p])
                pred_p = np.expm1(replication_model.predict(df_p)[0])
                st.success(f"📡 Predicted Replication Rate: **{pred_p:.2f} Mbps**")
            except Exception as e:
                st.error(f"❌ Replication Prediction Error: {e}")
