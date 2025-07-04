#!/usr/bin/env python3
import json
import time
import requests
import os
import math
from datetime import datetime

# ===== Configuration =====
CEPH_EXPORTER_URL = "http://192.168.93.52:9283/metrics"
NODE_EXPORTER_URL = "http://192.168.93.52:9100/metrics"
SAMPLE_INTERVAL = 5
TOTAL_DURATION = 30
OUTPUT_DIR = "results"

# ===== COMPLETE METRIC LIST =====
METRICS = [
    # Ceph Metrics
    'ceph_mds_cache_recovery_completed', 'ceph_mds_cache_recovery_started', 'ceph_num_objects_degraded',
    'ceph_osd_flag_nobackfill', 'ceph_osd_recovery_bytes', 'ceph_osd_recovery_ops',
    'ceph_pg_backfilling', 'ceph_pg_degraded', 'ceph_pg_forced_recovery', 'ceph_pg_recovering',
    'ceph_pg_recovery_unfound', 'ceph_pg_recovery_wait', 'ceph_cluster_total_bytes',
    'ceph_cluster_total_used_bytes', 'ceph_osd_up', 'ceph_osd_in', 'ceph_health_status',
    'ceph_mon_quorum_status',

    # Node Exporter Metrics
    'node_cpu_seconds_total{mode="iowait"}', 'node_cpu_seconds_total{mode="system"}',
    'node_cpu_seconds_total{mode="user"}', 'node_memory_Cached_bytes', 'node_memory_Buffers_bytes',
    'node_memory_MemAvailable_bytes', 'node_disk_written_bytes_total',
    'node_disk_reads_completed_total', 'node_filesystem_avail_bytes',
    'node_network_transmit_bytes_total', 'node_network_receive_bytes_total',
    'node_network_receive_drop_total', 'node_network_transmit_packets_total', 'node_network_receive_packets_total'
]

# ===== AGGREGATION METHODS =====
AGGREGATION_METHODS = {
    # Base name mapping to method
    'ceph_osd_up': 'status_count', 'ceph_osd_in': 'status_count',
    'ceph_health_status': 'value', 'ceph_mon_quorum_status': 'quorum_status',
    'node_memory_Cached_bytes': 'value', 'node_memory_Buffers_bytes': 'value',
    'node_memory_MemAvailable_bytes': 'value',
    'node_filesystem_avail_bytes': 'per_mountpoint',
    # Default for all other base names will be 'sum'
}

def _parse_line(line):
    """Parses a single Prometheus metric line into (base_name, labels_dict, value)."""
    try:
        if line.startswith("#") or not line.strip():
            return None, None, None
        parts = line.split()
        full_name = parts[0]
        value = float(parts[-1])
        base_name = full_name
        labels = {}
        if "{" in full_name:
            base_name, label_part = full_name.split("{", 1)
            label_part = label_part.rstrip("}")
            if label_part:
                for item in label_part.split(","):
                    if "=" in item:
                        key, val = item.split("=", 1)
                        labels[key.strip()] = val.strip().strip('"')
        return base_name, labels, value
    except Exception:
        return None, None, None

def aggregate_metric(metric_key, lines):
    """Finds all matching lines for a metric key and aggregates their values."""
    target_base_name, target_labels, _ = _parse_line(metric_key + " 0")
    if not target_base_name:
        return "invalid_key"

    method = AGGREGATION_METHODS.get(target_base_name, 'sum')

    collected_data = []
    for line in lines:
        line_base_name, line_labels, value = _parse_line(line)
        if not line_base_name or line_base_name != target_base_name:
            continue
        
        # Check if all required labels from the key exist in the line's labels
        is_match = all(line_labels.get(key) == val for key, val in target_labels.items())
        
        if is_match:
            collected_data.append({'value': value, 'labels': line_labels})

    if not collected_data:
        return None

    # Apply Aggregation Logic
    if method == 'sum':
        return sum(item['value'] for item in collected_data)
    if method == 'value':
        return collected_data[0]['value']
    if method == 'status_count':
        normal_count = sum(1 for item in collected_data if math.isclose(item['value'], 1.0))
        return {'normal': normal_count, 'abnormal': len(collected_data) - normal_count, 'total': len(collected_data)}
    if method == 'quorum_status':
        return sum(1 for item in collected_data if math.isclose(item['value'], 1.0))
    if method == 'per_mountpoint':
        return {item['labels'].get('mountpoint', 'unknown'): item['value'] for item in collected_data}
    return "unknown_method"

def collect_metrics():
    """Collects metrics from both Ceph and Node exporters."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()
    filename = os.path.join(OUTPUT_DIR, f"ceph_node_recovery_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    
    samples_needed = TOTAL_DURATION // SAMPLE_INTERVAL
    result = {
        "config": {"sample_interval": SAMPLE_INTERVAL, "total_duration": TOTAL_DURATION},
        "start_time": start_time.isoformat(), "samples": []
    }
    
    print(f"Starting data collection: {samples_needed} samples over {TOTAL_DURATION}s.")
    print("Press Ctrl+C to stop early and save collected data.")
    
    try:
        for i in range(samples_needed):
            sample_time = datetime.now()
            sample_metrics = {}
            
            ceph_lines, node_lines = [], []
            try:
                response_ceph = requests.get(CEPH_EXPORTER_URL, timeout=10)
                response_ceph.raise_for_status()
                ceph_lines = response_ceph.text.splitlines()
            except requests.RequestException as e:
                print(f"ERROR fetching Ceph metrics: {e}")

            try:
                response_node = requests.get(NODE_EXPORTER_URL, timeout=10)
                response_node.raise_for_status()
                node_lines = response_node.text.splitlines()
            except requests.RequestException as e:
                print(f"ERROR fetching Node Exporter metrics: {e}")

            for metric_key in METRICS:
                source_lines = ceph_lines if metric_key.startswith('ceph_') else node_lines
                value = aggregate_metric(metric_key, source_lines)
                sample_metrics[metric_key] = value if value is not None else "not_found"

            if i > 0 and result["samples"]:
                prev_metrics = result["samples"][-1].get("metrics", {})
                time_delta = (sample_time - datetime.fromisoformat(result["samples"][-1]["timestamp"])).total_seconds()
                if time_delta > 0:
                    current_bytes = sample_metrics.get('ceph_osd_recovery_bytes')
                    prev_bytes = prev_metrics.get('ceph_osd_recovery_bytes')
                    if isinstance(current_bytes, (int, float)) and isinstance(prev_bytes, (int, float)):
                        byte_diff = current_bytes - prev_bytes
                        if byte_diff >= 0:
                            sample_metrics['recovery_rate_mbps'] = (byte_diff / time_delta) / 1e6

            result["samples"].append({"timestamp": sample_time.isoformat(), "metrics": sample_metrics})
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            degraded_pgs = sample_metrics.get('ceph_pg_degraded', 'N/A')
            recovery_rate = sample_metrics.get('recovery_rate_mbps', 0)
            
            print(f"[{i+1}/{samples_needed}] {sample_time.strftime('%H:%M:%S')} - "
                  f"Degraded PGs: {degraded_pgs} | "
                  f"Recovery: {recovery_rate:.2f} MB/s")
            
            if i < samples_needed - 1:
                time.sleep(SAMPLE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nCollection stopped by user.")
    
    finally:
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Final results saved to {filename}")

if __name__ == "__main__":
    collect_metrics()
