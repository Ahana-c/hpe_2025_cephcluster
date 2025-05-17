# Ceph Cluster Time Series Analysis and Prediction

This project focuses on performing **Time Series Analysis** on a Ceph cluster to understand and predict **Recovery Time** and **Replication Rate**. The work spans data analysis using public datasets, evaluation of a custom Ceph cluster, and deployment of an interactive web application for Exploratory Data Analysis (EDA) and prediction.

---

## 📌 Project Structure

- **📁 [`Time_Series_Analysis_of_Ceph_Cluster_(Recovery).ipynb`](https://github.com/Ahana-c/hpe_2025_cephcluster/blob/main/Time_Series_Analysis_of_Ceph_Cluster_(Recovery).ipynb)**  
  Time series analysis and prediction of metrics affecting **Recovery Time** using Prophet model.

- **📁 [`Time_Series_Analysis_of_Ceph_Cluster_(Replication).ipynb`](https://github.com/Ahana-c/hpe_2025_cephcluster/blob/main/Time_Series_Analysis_of_Ceph_Cluster_(Replication).ipynb)**  
  Time series analysis and prediction of metrics affecting **Replication Rate** using Prophet model.

- **📁 [`Evaluation_of_Cluster_Metrics_and_Prediction_Uisng_Linear_Regression.ipynb`](https://github.com/Ahana-c/hpe_2025_cephcluster/blob/main/Evaluation_of_Cluster_Metrics_and_Prediction_Uisng_Linear_Regression.ipynb)**  
  Linear regression-based prediction on metrics collected from a self-hosted **Ceph Pacific** cluster.

- **📁 [`ceph eda app`](https://github.com/Ahana-c/hpe_2025_cephcluster/tree/main/ceph%20eda%20app)**  
  Web application for interactive EDA and prediction based on Ceph cluster metric data.


---

## 🔍 Metrics Considered

### For **Recovery Time**:
- `OSD Recovery Time (s)` – Time taken for Object Storage Daemons to recover.
- `PG Recovery Time (s)` – Time for Placement Groups to recover.
- `Backfill Rate (MB/s)` – Data rate at which backfilling occurs.
- `Recovery Throughput (MB/s)` – Total throughput during recovery.
- `Degraded PGs Count` – Number of degraded Placement Groups.

### For **Replication Rate**:
- `ceph_osd_objects_unfound` – Count of objects that couldn’t be located.
- `ceph_osd_scrub_count` – Number of regular data scrubbing operations.
- `ceph_osd_deep_scrub_count` – Number of deep scrubbing operations.
- `ceph_osd_replication_latency` – Latency experienced during replication.
- `ceph_osd_recovery_bw` – Bandwidth used for recovery.
- `ceph_osd_primary` – Number of primary OSDs.
- `ceph_osd_peering_state` – Status of OSDs in the peering process.

---

## 🔬 Models Used

- **Prophet** for Time Series Forecasting
- **Linear Regression** for cluster metric prediction in real-time scenarios

---

## 🌐 Web Application Features

**Folder:** [`ceph eda app`](https://github.com/Ahana-c/hpe_2025_cephcluster/tree/main/ceph%20eda%20app)  
**Setup Instructions:** See `technical setup.txt`

### 📂 Inputs Required (CSV/Excel):
| Column | Description |
|--------|-------------|
| `timestamp` | Time of metric capture |
| `workload size` | Size of workload (e.g., 10GB, 5.5GB) |
| `data type` | Type of data: block, RGW, file, object |
| `active i/o operations` | Number of concurrent input operations |
| `recovery config` | Config mode: aggressive, default, throttled |
| `number of objects` | Count of objects in the cluster |
| `replication factor` | Number of replicas (e.g., 2, 3) |
| `recovery time` | Actual measured recovery time |
| `replication rate` | Actual measured replication rate |

---

## ⚙️ App Functionality

### 🔍 EDA (Analysis)
- Plot workload vs. recovery/replication time
- Plot data type vs. recovery/replication time
- Analyze active I/O operations, config, and object count impact on recovery
- Plot replication factor impact on replication rate

### 📈 Recovery Time Prediction
- Inputs: workload, data type, I/O operations, config, object count, timestamp
- Output: Predicted recovery time with model accuracy

### 📈 Replication Rate Prediction
- Inputs: workload, data type, replication factor, timestamp
- Output: Predicted replication rate with model accuracy

---

## 🧠 Key Learnings

- Ceph metrics vary significantly across versions; Ceph Pacific's metrics were carefully selected for visual and statistical correlation.
- Recovery time and replication rate are influenced by configuration, workload, data type, and object size.
- Real-time analysis and prediction is feasible using lightweight regression models backed by historical metric behavior.

---

## 📌 Technologies Used

- Python, Jupyter Notebooks
- Facebook Prophet, scikit-learn
- Flask, HTML/CSS, JavaScript
- Pandas, NumPy, Matplotlib, Seaborn

---

## 🏁 How to Run the App

1. Install required Python packages (`requirements.txt`)
2. Navigate to the app folder:
   ```bash
   cd "ceph eda app"
   python app.py
