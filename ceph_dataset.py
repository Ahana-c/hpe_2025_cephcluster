import os
import json
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Ceph Dataset Creator", layout="wide")
st.title("Ceph JSON to Dataset Converter")

uploaded_files = st.file_uploader(
    "Upload one or more Ceph JSON files:",
    type="json",
    accept_multiple_files=True
)

if st.button("Create Dataset"):
    all_rows = []

    for uploaded_file in uploaded_files:
        raw = json.load(uploaded_file)

        use_case = raw.get("use_case", {})
        workload_size = f"{use_case.get('workload_size', 0)} GB"
        data_type = use_case.get("data_type", "Unknown")
        active_io_ops = use_case.get("active_io_ops", 0)
        recovery_config = use_case.get("recovery_config", "default")
        num_objects = use_case.get("num_objects", 0)
        replication_factor = use_case.get("replication_factor", 1)

        metrics = raw.get("metrics", [])

        last_recovery_ops = {}
        last_recovery_bytes = {}
        last_timestamp = None

        for m in metrics:
            timestamp = m.get("timestamp")
            data = m.get("metrics", {})

            # Compute recovery time (proxy): total recovering PGs
            pg_recovering = data.get("ceph_pg_recovering", [])
            total_recovering = 0
            if isinstance(pg_recovering, list):
                for pg in pg_recovering:
                    try:
                        if isinstance(pg, dict) and "value" in pg and isinstance(pg["value"], list):
                            total_recovering += int(float(pg["value"][1]))
                    except (ValueError, TypeError, IndexError):
                        continue
            recovery_time = total_recovering  # Simplified proxy in seconds

            # Compute replication rate (delta bytes / delta time)
            recovery_bytes = data.get("ceph_osd_recovery_bytes", [])
            curr_bytes = {}
            if isinstance(recovery_bytes, list):
                for b in recovery_bytes:
                    try:
                        if isinstance(b, dict) and "metric" in b and "value" in b:
                            daemon = b["metric"].get("ceph_daemon", "unknown") if isinstance(b["metric"], dict) else "unknown"
                            value = float(b["value"][1]) if isinstance(b["value"], list) and len(b["value"]) > 1 else 0
                            curr_bytes[daemon] = int(value)
                    except (ValueError, TypeError, IndexError):
                        continue


            replication_rate = 0.0
            if last_recovery_bytes and last_timestamp:
                delta_t = pd.to_datetime(timestamp) - pd.to_datetime(last_timestamp)
                seconds = delta_t.total_seconds()
                if seconds > 0:
                    delta_bytes = sum(
                        curr_bytes.get(k, 0) - last_recovery_bytes.get(k, 0)
                        for k in curr_bytes.keys()
                    )
                    replication_rate = round(delta_bytes / (1024 * 1024) / seconds, 2)  # MB/s

            last_recovery_bytes = curr_bytes
            last_timestamp = timestamp

            all_rows.append({
                "Timestamp": timestamp,
                "Workload Size": workload_size,
                "Data Type": data_type,
                "Active I/O Ops": active_io_ops,
                "Recovery Config": recovery_config,
                "Number of Objects": num_objects,
                "Replication Factor": replication_factor,
                "Recovery Time (sec)": recovery_time,
                "Replication Rate (MB/s)": replication_rate
            })

    df = pd.DataFrame(all_rows)
    st.dataframe(df.head(20))

    towrite = BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Ceph Dataset")
    towrite.seek(0)

    st.download_button(
        label="Download Excel File",
        data=towrite,
        file_name="ceph_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
