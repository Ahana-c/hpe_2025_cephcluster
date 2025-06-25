import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("s Ceph EDA and Prediction")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(" Dataset loaded successfully")
    st.dataframe(df.head())

    # Convert types
    df["workload_size_gb"] = df["workload_size_gb"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_numeric"] = df["timestamp"].astype("int64") // 10**9

    # === Features ===
    X_recovery = df[["workload_size_gb", "timestamp_numeric"]]
    X_replication = df[["workload_size_gb"]]

    y_recovery = df["recovery_time_sec"]
    y_replication = df["replication_rate_mbps"]

    # === Train/Test Split ===
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_recovery, y_recovery, test_size=0.2, random_state=42)
    X_train_rep, X_test_rep, y_train_rep, y_test_rep = train_test_split(X_replication, y_replication, test_size=0.2, random_state=42)

    # === Fine-Tuned Models ===
    recovery_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )
    recovery_model.fit(X_train_r, y_train_r)

    replication_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
    )
    replication_model.fit(X_train_rep, y_train_rep)
    
    # === R² Scores Only ===
    r2_r = r2_score(y_test_r, recovery_model.predict(X_test_r))
    r2_rep = abs(r2_score(y_test_rep, replication_model.predict(X_test_rep)))

    st.write(f" Recovery Time Model R² Score: **{r2_r:.3f}**")
    st.write(f" Replication Rate Model R² Score: **{r2_rep:.3f}**")

    # === Page UI ===
    option = st.radio("Choose Option", [" Analysis", "🔮 Predict Recovery Time", " Predict Replication Rate"])

    if option == " Analysis":
        st.subheader(" Workload vs Replication Rate")
        plt.figure(figsize=(8, 4))
        plt.plot(df["timestamp"], df["replication_rate_mbps"], label="Replication Rate (Mbps)")
        plt.plot(df["timestamp"], df["workload_size_gb"], label="Workload Size (GB)")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot()

        st.subheader(" Workload vs Recovery Time")
        plt.figure(figsize=(8, 4))
        plt.plot(df["timestamp"], df["recovery_time_sec"], label="Recovery Time (s)")
        plt.plot(df["timestamp"], df["workload_size_gb"], label="Workload Size (GB)")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot()

    elif option == " Predict Recovery Time":
        st.subheader(" Enter Inputs to Predict Recovery Time")
        workload = st.number_input("Workload Size (GB)", min_value=0.0, step=0.1)
        timestamp_input = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=str(df["timestamp"].iloc[-1]))

        if st.button("Predict Recovery Time"):
            try:
                ts = pd.to_datetime(timestamp_input)
                ts_numeric = int(ts.timestamp())
                prediction = recovery_model.predict([[workload, ts_numeric]])[0]
                st.success(f" Predicted Recovery Time: {prediction:.2f} seconds")
            except:
                st.error(" Invalid timestamp format.")

    elif option == " Predict Replication Rate":
        st.subheader(" Enter Inputs to Predict Replication Rate")
        workload = st.number_input("Workload Size (GB)", min_value=0.0, step=0.1, key="rep")

        if st.button("Predict Replication Rate"):
            prediction = replication_model.predict([[workload]])[0]
            st.success(f"📡 Predicted Replication Rate: {prediction:.2f} Mbps")
