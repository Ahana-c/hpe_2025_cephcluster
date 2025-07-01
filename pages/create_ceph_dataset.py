import streamlit as st
import pandas as pd
import json
import tempfile
from datetime import datetime
import io

def json_to_dataframe(json_file_path: str) -> pd.DataFrame:
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

    samples_data = data.get('samples', [])
    if not isinstance(samples_data, list):
        return pd.DataFrame()

    processed_rows = []
    for sample in samples_data:
        if 'timestamp' not in sample or 'metrics' not in sample:
            continue
        row = {'timestamp': sample['timestamp']}
        for k, v in sample['metrics'].items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    row[f"{k}_{sub_k}"] = sub_v
            else:
                row[k] = v
        processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    return df.reset_index()

def convert_to_samples_format(uploaded_file) -> str:
    content = uploaded_file.read().decode("utf-8")
    try:
        raw_json = json.loads(content)
        if 'samples' in raw_json:
            structured = raw_json
        else:
            structured = {'samples': raw_json}
    except:
        segments = content.split('][')
        fixed_entries = []
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if i == 0:
                seg += "]"
            elif i == len(segments) - 1:
                seg = "[" + seg
            else:
                seg = "[" + seg + "]"
            try:
                entries = json.loads(seg)
                if isinstance(entries, list):
                    for entry in entries:
                        if 'timestamp' in entry and 'metrics' in entry:
                            fixed_entries.append(entry)
            except:
                continue
        structured = {'samples': fixed_entries}

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmpfile:
        json.dump(structured, tmpfile)
        return tmpfile.name

def calculate_custom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(by='timestamp')
    df['workload_size_gb'] = df['ceph_cluster_total_used_bytes'] / (1024 ** 3)
    df['recovery_time_sec'] = 0.0
    df['replication_rate_mbps'] = 0.0

    # Network Bandwidth
    df['net_tx_mbps'] = df['node_network_transmit_bytes_total'].diff() * 8 / (df['timestamp'].diff().dt.total_seconds() * 1e6)
    df['net_rx_mbps'] = df['node_network_receive_bytes_total'].diff() * 8 / (df['timestamp'].diff().dt.total_seconds() * 1e6)

    # Disk I/O
    df['disk_write_MBps'] = df['node_disk_written_bytes_total'].diff() / (1024 ** 2) / df['timestamp'].diff().dt.total_seconds()
    df['disk_reads_completed'] = df['node_disk_reads_completed_total'].diff() / df['timestamp'].diff().dt.total_seconds()

    # CPU Load
    df['cpu_user_delta'] = df['node_cpu_seconds_total{mode="user"}'].diff() / df['timestamp'].diff().dt.total_seconds()
    df['cpu_system_delta'] = df['node_cpu_seconds_total{mode="system"}'].diff() / df['timestamp'].diff().dt.total_seconds()
    df['cpu_iowait_delta'] = df['node_cpu_seconds_total{mode="iowait"}'].diff() / df['timestamp'].diff().dt.total_seconds()

    # Recovery Time Calculation
    recovery_start_idx = df[(df['ceph_pg_degraded'] > 0) | (df['ceph_num_objects_degraded'] > 0)].index.min()
    if pd.notna(recovery_start_idx):
        t_start = df.loc[recovery_start_idx, 'timestamp']
        df.loc[recovery_start_idx:, 'recovery_time_sec'] = (df.loc[recovery_start_idx:, 'timestamp'] - t_start).dt.total_seconds()

        for i in range(recovery_start_idx + 1, len(df)):
            delta_bytes = df.loc[i, 'ceph_osd_recovery_bytes'] - df.loc[i - 1, 'ceph_osd_recovery_bytes']
            delta_time = (df.loc[i, 'timestamp'] - df.loc[i - 1, 'timestamp']).total_seconds()
            if delta_time > 0 and delta_bytes > 0:
                df.loc[i, 'replication_rate_mbps'] = (delta_bytes * 8) / (delta_time * 1e6)

    return df

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        'timestamp',
        'workload_size_gb',
        'ceph_osd_recovery_bytes',
        'ceph_osd_recovery_ops',
        'ceph_pg_backfilling',
        'ceph_pg_recovering',
        'ceph_pg_forced_recovery',
        'ceph_osd_flag_nobackfill',
        'ceph_pg_recovery_wait',
        'ceph_pg_recovery_unfound',
        'ceph_mds_cache_recovery_started',
        'ceph_mds_cache_recovery_completed',
        'ceph_num_objects_degraded',
        'ceph_pg_degraded',
        'recovery_time_sec',
        'replication_rate_mbps',
        'net_tx_mbps',
        'net_rx_mbps',
        'disk_write_MBps',
        'disk_reads_completed',
        'cpu_user_delta',
        'cpu_system_delta',
        'cpu_iowait_delta'
    ]
    return df[[col for col in required if col in df.columns]]

# === Streamlit Interface ===
st.title("Ceph Metrics Analyzer (Multi-File Support)")
st.write("Upload one or more Ceph JSON files. The app will compute and combine metrics across all files.")

uploaded_files = st.file_uploader("Upload Ceph JSON file(s)", type="json", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("⏳ Processing files..."):
        combined_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            fixed_path = convert_to_samples_format(uploaded_file)
            df = json_to_dataframe(fixed_path)
            if not df.empty:
                df_metrics = calculate_custom_metrics(df)
                combined_df = pd.concat([combined_df, df_metrics], ignore_index=True)

        if not combined_df.empty:
            combined_df = combined_df.sort_values(by='timestamp')
            df_final = filter_columns(combined_df)

            st.success("✅ Files processed successfully!")
            st.dataframe(df_final.head(50))

            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Combined CSV", data=csv, file_name="ceph_combined_metrics.csv", mime="text/csv")
        else:
            st.error("❌ No data could be extracted from the uploaded file(s).")
