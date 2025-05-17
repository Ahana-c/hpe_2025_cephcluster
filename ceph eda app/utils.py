import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# Directory for saving models and plots
MODEL_DIR = 'models'
PLOT_DIR = 'uploads/static/plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def preprocess(df):
    df = df.copy()
    le_data = LabelEncoder()
    le_recovery = LabelEncoder()

    # Encode categorical columns
    df['Data Type'] = le_data.fit_transform(df['Data Type'])
    df['Recovery Config'] = le_recovery.fit_transform(df['Recovery Config'])

    # Convert Workload Size from string like '10Gb' to float 10.0
    df['Workload Size'] = df['Workload Size'].str.replace('Gb', '', regex=False).astype(float)

    # Convert Timestamp to UNIX timestamp in seconds (float)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 1e9

    return df


def train_models(df):
    df = preprocess(df)

    # Train recovery time model
    X_rec = df[['Workload Size', 'Data Type', 'Active I/O Ops',
                'Recovery Config', 'Number of Objects', 'Timestamp']]
    y_rec = df['Recovery Time (sec)']
    model_rec = RandomForestRegressor(random_state=42)
    model_rec.fit(X_rec, y_rec)
    joblib.dump(model_rec, os.path.join(MODEL_DIR, 'recovery_model.pkl'))

    # Train replication rate model
    X_rep = df[['Workload Size', 'Data Type', 'Replication Factor', 'Timestamp']]
    y_rep = df['Replication Rate (MB/s)']
    model_rep = RandomForestRegressor(random_state=42)
    model_rep.fit(X_rep, y_rep)
    joblib.dump(model_rep, os.path.join(MODEL_DIR, 'replication_model.pkl'))


def predict_recovery_time(data):
    model = joblib.load(os.path.join(MODEL_DIR, 'recovery_model.pkl'))
    input_df = pd.DataFrame([data])

    # Clean and convert inputs to correct types
    input_df['Workload Size'] = input_df['Workload Size'].str.replace('Gb', '', regex=False).astype(float)
    input_df['Data Type'] = input_df['Data Type'].map({'block data': 0, 'RGW': 1, 'file data': 2, 'object data': 3})
    input_df['Recovery Config'] = input_df['Recovery Config'].map({'aggressive': 0, 'default': 1, 'throttled': 2})
    input_df['Active I/O Ops'] = input_df['Active I/O Ops'].astype(int)
    input_df['Number of Objects'] = input_df['Number of Objects'].astype(int)
    input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp']).astype(int) / 1e9

    pred = model.predict(input_df)[0]
    acc = model.score(input_df, [pred])
    return round(pred, 2), round(acc * 100, 2)


def predict_replication_rate(data):
    model = joblib.load(os.path.join(MODEL_DIR, 'replication_model.pkl'))
    input_df = pd.DataFrame([data])

    # Clean and convert inputs to correct types
    input_df['Workload Size'] = input_df['Workload Size'].str.replace('Gb', '', regex=False).astype(float)
    input_df['Data Type'] = input_df['Data Type'].map({'block data': 0, 'RGW': 1, 'file data': 2, 'object data': 3})
    input_df['Replication Factor'] = input_df['Replication Factor'].astype(int)
    input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp']).astype(int) / 1e9

    pred = model.predict(input_df)[0]
    acc = model.score(input_df, [pred])
    return round(pred, 2), round(acc * 100, 2)


def generate_analysis_plots(df):
    plots = []
    df = df.copy()

    # Convert columns as needed for plotting
    df['Workload Size'] = df['Workload Size'].str.replace('Gb', '', regex=False).astype(float)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Plot 1: Workload Size vs Recovery Time over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Timestamp', y='Recovery Time (sec)', hue='Workload Size', data=df)
    plt.title('Workload Size vs Recovery Time over Time')
    plt.xticks(rotation=45)
    file1 = os.path.join(PLOT_DIR, 'workload_recovery.png')
    plt.tight_layout()
    plt.savefig(file1)
    plots.append(file1)
    plt.close()

    # Plot 2: Workload Size vs Replication Rate over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Timestamp', y='Replication Rate (MB/s)', hue='Workload Size', data=df)
    plt.title('Workload Size vs Replication Rate over Time')
    plt.xticks(rotation=45)
    file2 = os.path.join(PLOT_DIR, 'workload_replication.png')
    plt.tight_layout()
    plt.savefig(file2)
    plots.append(file2)
    plt.close()

    # Plot 3: Data Type vs Recovery Time (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Data Type', y='Recovery Time (sec)', data=df)
    plt.title('Data Type vs Recovery Time')
    file3 = os.path.join(PLOT_DIR, 'datatype_recovery.png')
    plt.tight_layout()
    plt.savefig(file3)
    plots.append(file3)
    plt.close()

    # Plot 4: Data Type vs Replication Rate (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Data Type', y='Replication Rate (MB/s)', data=df)
    plt.title('Data Type vs Replication Rate')
    file4 = os.path.join(PLOT_DIR, 'datatype_replication.png')
    plt.tight_layout()
    plt.savefig(file4)
    plots.append(file4)
    plt.close()

    # Plot 5: Active I/O Ops vs Recovery Time (Scatter plot over time)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Timestamp', y='Recovery Time (sec)', hue='Active I/O Ops', data=df)
    plt.title('Active I/O Operations vs Recovery Time')
    file5 = os.path.join(PLOT_DIR, 'ioops_recovery.png')
    plt.tight_layout()
    plt.savefig(file5)
    plots.append(file5)
    plt.close()

    # Plot 6: Recovery Config vs Recovery Time (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Recovery Config', y='Recovery Time (sec)', data=df)
    plt.title('Recovery Config vs Recovery Time')
    file6 = os.path.join(PLOT_DIR, 'recoveryconfig_recovery.png')
    plt.tight_layout()
    plt.savefig(file6)
    plots.append(file6)
    plt.close()

    # Plot 7: Number of Objects vs Recovery Time (Scatter plot over time)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Timestamp', y='Recovery Time (sec)', hue='Number of Objects', data=df)
    plt.title('Number of Objects vs Recovery Time')
    file7 = os.path.join(PLOT_DIR, 'objects_recovery.png')
    plt.tight_layout()
    plt.savefig(file7)
    plots.append(file7)
    plt.close()

    # Plot 8: Replication Factor vs Replication Rate (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Replication Factor', y='Replication Rate (MB/s)', data=df)
    plt.title('Replication Factor vs Replication Rate')
    file8 = os.path.join(PLOT_DIR, 'repfactor_replication.png')
    plt.tight_layout()
    plt.savefig(file8)
    plots.append(file8)
    plt.close()

    # Return relative paths for HTML usage (assuming 'static/' is root for static files)
    return [f'plots/{os.path.basename(plot)}' for plot in plots]

