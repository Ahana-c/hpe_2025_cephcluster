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
        try:
            raw = json.load(uploaded_file)
        except Exception as e:
            st.warning(f"Failed to read {uploaded_file.name}: {e}")
            continue

        use_case = raw.get("use_case", {})
        workload_size = f"{use_case.get('workload_size', 0)} Gb"
        data_type = use_case.get("data_type", "unknown").lower()
        active_io_ops = use_case.get("active_io_ops", 0)
        recovery_config = use_case.get("recovery_config", "default").lower()
        num_objects = use_case.get("num_objects", 0)
        replication_factor = use_case.get("replication_factor", 1)

        metrics = raw.get("metrics", [])

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
                    except Exception:
                        continue
            recovery_time = total_recovering  # Simplified proxy in seconds

            # Compute replication rate (delta bytes / delta time)
            recovery_bytes = data.get("ceph_osd_recovery_bytes", [])
            curr_bytes = {}
            if isinstance(recovery_bytes, list):
                for b in recovery_bytes:
                    try:
                        if isinstance(b, dict) and "metric" in b and "value" in b:
                            daemon = b["metric"].get("ceph_daemon", "unknown")
                            value = float(b["value"][1]) if isinstance(b["value"], list) and len(b["value"]) > 1 else 0
                            curr_bytes[daemon] = int(value)
                    except Exception:
                        continue

            replication_rate = 0.0
            if last_recovery_bytes and last_timestamp:
                try:
                    delta_t = pd.to_datetime(timestamp) - pd.to_datetime(last_timestamp)
                    seconds = delta_t.total_seconds()
                    if seconds > 0:
                        delta_bytes = sum(
                            curr_bytes.get(k, 0) - last_recovery_bytes.get(k, 0)
                            for k in curr_bytes.keys()
                        )
                        replication_rate = round(delta_bytes / (1024 * 1024) / seconds, 2)  # MB/s
                except Exception:
                    replication_rate = 0.0

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

    if not all_rows:
        st.warning("No valid data extracted from uploaded files.")
    else:
        df = pd.DataFrame(all_rows)

        # Ensure column names are exactly as expected
        df = df[[
            "Timestamp",
            "Workload Size",
            "Data Type",
            "Active I/O Ops",
            "Recovery Config",
            "Number of Objects",
            "Replication Factor",
            "Recovery Time (sec)",
            "Replication Rate (MB/s)"
        ]]

        st.success(f"✅ Dataset created with {len(df)} rows.")
        st.dataframe(df.head(20))

        towrite = BytesIO()
        df.to_csv(towrite, index=False)
        towrite.seek(0)

        st.download_button(
            label="Download CSV File",
            data=towrite,
            file_name="ceph_dataset.csv",
            mime="text/csv"
        )
