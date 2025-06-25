import pandas as pd
from datetime import datetime

def parse_ceph_json(entries):
    timestamps = []
    workloads = []
    recovery_ops = []
    degraded_pg = []
    degraded_objs = []

    for entry in entries:
        try:
            ts = entry["timestamp"]
            metrics = entry["metrics"]
            timestamps.append(datetime.fromisoformat(ts))
            workloads.append(metrics["ceph_cluster_total_used_bytes"] / (1024 ** 3))  # GB
            recovery_ops.append(metrics["ceph_osd_recovery_ops"])
            degraded_pg.append(metrics["ceph_pg_degraded"])
            degraded_objs.append(metrics["ceph_num_objects_degraded"])
        except Exception:
            continue

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": timestamps,
        "workload_size": workloads,
        "recovery_ops": recovery_ops,
        "pg_degraded": degraded_pg,
        "objects_degraded": degraded_objs
    })

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Estimate recovery time from degraded activity
    recovery_df = df[(df["pg_degraded"] > 0) | (df["objects_degraded"] > 0)]
    if not recovery_df.empty and len(recovery_df) > 1:
        t_start = recovery_df["timestamp"].iloc[0]
        t_end = recovery_df["timestamp"].iloc[-1]
        recovery_time = (t_end - t_start).total_seconds()
    else:
        # fallback to entire dataset duration
        t_start = df["timestamp"].iloc[0]
        t_end = df["timestamp"].iloc[-1]
        recovery_time = (t_end - t_start).total_seconds()

    # Calculate replication rate using recovery_ops
    delta_ops = df["recovery_ops"].iloc[-1] - df["recovery_ops"].iloc[0]
    avg_obj_size_bytes = 4 * 1024 * 1024  # 4 MB typical object
    delta_bytes = delta_ops * avg_obj_size_bytes
    delta_time = (t_end - t_start).total_seconds()
    replication_rate = (delta_bytes * 8) / (delta_time * 1_000_000) if delta_time > 0 else 0

    return pd.DataFrame([{
        "timestamp": df["timestamp"].iloc[-1],
        "workload_size": df["workload_size"].mean(),
        "recovery_time": recovery_time,
        "replication_rate": replication_rate
    }])
