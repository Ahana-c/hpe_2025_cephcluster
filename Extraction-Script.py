#!/usr/bin/env python3
import json
import time
import requests
import os
import math
from datetime import datetime

EXPORTER_URL = "http://localhost:9283/metrics"
SAMPLE_INTERVAL = 15  # seconds
TOTAL_DURATION = 7200  # seconds
OUTPUT_DIR = "results"

METRICS = [
    'ceph_mds_cache_recovery_completed',
    'ceph_mds_cache_recovery_started',
    'ceph_num_objects_degraded',
    'ceph_osd_flag_nobackfill',
    'ceph_osd_recovery_bytes',
    'ceph_osd_recovery_ops',
    'ceph_pg_backfilling',
    'ceph_pg_degraded',
    'ceph_pg_forced_recovery',
    'ceph_pg_recovering',
    'ceph_pg_recovery_unfound',
    'ceph_pg_recovery_wait',
    "ceph_cluster_total_bytes",
    "ceph_cluster_total_used_bytes",
    "ceph_osd_up",
    "ceph_osd_in",
    "ceph_health_status",
    "ceph_mon_quorum_status"
]

# Define aggregation methods for each metric
AGGREGATION_METHODS = {
    # OSD status metrics
    'ceph_osd_up': {
        'method': 'status_count',
        'status_key': 'up',
        'normal_value': 1.0
    },
    'ceph_osd_in': {
        'method': 'status_count',
        'status_key': 'in',
        'normal_value': 1.0
    },
    
    # Recovery progress metrics
    'ceph_osd_recovery_bytes': 'sum',
    'ceph_osd_recovery_ops': 'sum',
    
    # PG state metrics
    'ceph_pg_backfilling': 'sum',
    'ceph_pg_degraded': 'sum',
    'ceph_pg_forced_recovery': 'sum',
    'ceph_pg_recovering': 'sum',
    'ceph_pg_recovery_unfound': 'sum',
    'ceph_pg_recovery_wait': 'sum',
    
    # Metadata server metrics
    'ceph_mds_cache_recovery_started': 'max_active',  # Focus on active MDS
    'ceph_mds_cache_recovery_completed': 'max_active',
    
    # Cluster health metrics
    'ceph_num_objects_degraded': 'value',
    'ceph_osd_flag_nobackfill': 'value',
    'ceph_cluster_total_bytes': 'value',
    'ceph_cluster_total_used_bytes': 'value',
    'ceph_health_status': 'value',
    'ceph_mon_quorum_status': 'quorum_status'  # Special handling
}

def aggregate_metric(metric, lines):
    """Apply appropriate aggregation method based on metric type"""
    if metric not in AGGREGATION_METHODS:
        return "metric not configured"
    
    method = AGGREGATION_METHODS[metric]
    values = []
    labeled_values = {}
    
    # Collect all values for this metric
    for line in lines:
        if line.startswith(metric) and not line.startswith("#"):
            try:
                parts = line.split()
                value = float(parts[-1])
                
                # Extract labels if present
                labels = {}
                if "{" in line:
                    label_part = line.split("{")[1].split("}")[0]
                    for item in label_part.split(","):
                        if "=" in item:
                            key, val = item.split("=")
                            labels[key.strip()] = val.strip().strip('"')
                
                values.append(value)
                
                # Store labeled values for status metrics
                if isinstance(method, dict) and method['method'] == 'status_count':
                    daemon = labels.get('ceph_daemon', 'unknown')
                    labeled_values[daemon] = value
                    
            except Exception as e:
                print(f"Error parsing {line}: {str(e)}")
                continue
    
    if not values:
        return None
    
    # Apply aggregation method
    if isinstance(method, dict):
        if method['method'] == 'status_count':
            normal_count = sum(1 for v in labeled_values.values() 
                              if math.isclose(v, method['normal_value'], rel_tol=1e-5))
            abnormal_count = len(labeled_values) - normal_count
            return {
                'normal': normal_count,
                'abnormal': abnormal_count,
                'total': len(labeled_values)
            }
    
    elif method == 'sum':
        return sum(values)
    
    elif method == 'value':
        return values[0]  # For single-value cluster metrics
    
    elif method == 'max_active':
        # For MDS, we care about the active instance (non-zero values)
        active_values = [v for v in values if v > 0]
        return max(active_values) if active_values else 0
    
    elif method == 'quorum_status':
        # Return count of monitors in quorum (status=1)
        return sum(1 for v in values if math.isclose(v, 1.0, rel_tol=1e-5))
    
    return None

def collect_metrics():
    """Collect metrics from exporter and save to file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()
    filename = os.path.join(OUTPUT_DIR, f"ceph_recovery_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    
    # Calculate number of samples needed
    samples_needed = TOTAL_DURATION // SAMPLE_INTERVAL
    collected_samples = []
    
    # Create result structure
    result = {
        "config": {
            "sample_interval": SAMPLE_INTERVAL,
            "total_duration": TOTAL_DURATION,
            "metrics_config": AGGREGATION_METHODS
        },
        "start_time": start_time.isoformat(),
        "samples": collected_samples
    }
    
    print(f"Starting data collection for {samples_needed} samples...")
    print("Press Ctrl+C to stop early and save collected data")
    
    try:
        for i in range(samples_needed):
            sample_time = datetime.now()
            timestamp = sample_time.isoformat()
            sample_metrics = {}
            
            try:
                response = requests.get(EXPORTER_URL, timeout=10)
                response.raise_for_status()
                lines = response.text.splitlines()
                
                for metric in METRICS:
                    value = aggregate_metric(metric, lines)
                    if value is not None:
                        sample_metrics[metric] = value
                    else:
                        sample_metrics[metric] = "not_found"
                
                # Add recovery progress estimation
                if i > 0:
                    prev = collected_samples[-1]["metrics"]
                    time_delta = (sample_time - datetime.fromisoformat(
                        collected_samples[-1]["timestamp"])).total_seconds()
                    
                    # Calculate recovery rate in bytes/sec
                    if ('ceph_osd_recovery_bytes' in sample_metrics and 
                        'ceph_osd_recovery_bytes' in prev):
                        bytes_diff = sample_metrics['ceph_osd_recovery_bytes'] - prev['ceph_osd_recovery_bytes']
                        sample_metrics['recovery_rate_bps'] = bytes_diff / time_delta
                    
                    # Calculate operations rate
                    if ('ceph_osd_recovery_ops' in sample_metrics and 
                        'ceph_osd_recovery_ops' in prev):
                        ops_diff = sample_metrics['ceph_osd_recovery_ops'] - prev['ceph_osd_recovery_ops']
                        sample_metrics['recovery_rate_ops'] = ops_diff / time_delta
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {str(e)}")
                for metric in METRICS:
                    sample_metrics[metric] = f"error: {str(e)}"
            
            # Add sample to collection
            collected_samples.append({
                "timestamp": timestamp,
                "metrics": sample_metrics
            })
            
            # Update JSON file
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Progress output
            degraded = sample_metrics.get('ceph_pg_degraded', 'N/A')
            recovery_rate = sample_metrics.get('recovery_rate_bps', 0)
            print(f"[{i+1}/{samples_needed}] {timestamp} - "
                  f"Degraded: {degraded} | "
                  f"Recovery: {recovery_rate/1e6:.2f} MB/s")
            
            # Sleep until next sample (except last iteration)
            if i < samples_needed - 1:
                time.sleep(SAMPLE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nCollection stopped by user. Saving collected data...")
    
    print(f"Collection complete. Results saved to {filename}")
    return result

if name == "main":
    collect_metrics()