import json
import pandas as pd

def json_to_dataframe(json_file_path: str) -> pd.DataFrame:
    """
    Loads Ceph metrics from a JSON file, processes the 'samples' array,
    and converts it into a Pandas DataFrame.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: A DataFrame with timestamps as index and flattened metrics
                      as columns. Returns an empty DataFrame if 'samples' is
                      not found or is empty, or if the file is not found/invalid.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return pd.DataFrame()

    #ignoring config and start_time
    samples_data = data.get('samples')

    if not samples_data or not isinstance(samples_data, list):
        print("Warning: 'samples' key not found or is not a list. Returning empty DataFrame.")
        return pd.DataFrame()

    processed_rows = []
    for sample in samples_data:
        row_dict = {}
        timestamp = sample.get('timestamp')
        if timestamp is None:
            print("Warning: Sample missing timestamp. Skipping this sample.")
            continue
        row_dict['timestamp'] = timestamp

        metrics = sample.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                #flatten here.
                for sub_key, sub_value in metric_value.items():
                    row_dict[f"{metric_name}_{sub_key}"] = sub_value
            else:
                row_dict[metric_name] = metric_value
        processed_rows.append(row_dict)

    if not processed_rows:
        print("Warning: No valid samples found to process. Returning empty DataFrame.")
        return pd.DataFrame()

    
    df = pd.DataFrame(processed_rows)

    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index() 
    else:
        print("Warning: 'timestamp' column was not created. Cannot set index.")

    return df

if __name__ == '__main__':
    file_path = 'ceph_recovery_20250624_120913.json'


    print(f"Attempting to load data from: {file_path}")
    df_ceph_metrics = json_to_dataframe(file_path)

    if not df_ceph_metrics.empty:
        print("\nDataFrame Head:")
        print(df_ceph_metrics.head())
        print("\nDataFrame Info:")
        df_ceph_metrics.info()
        print("\nDataFrame Description (summary statistics for numerical columns):")
        print(df_ceph_metrics.describe(include='all')) 

        
        if 'ceph_osd_up_normal' in df_ceph_metrics.columns:
            print("\n'ceph_osd_up_normal' column sample:")
            print(df_ceph_metrics['ceph_osd_up_normal'].head())
    else:
        print("DataFrame is empty. Please check the file path and JSON content.")

csv_file_name = 'ceph_metrics_output.csv'
try:
    df_ceph_metrics.to_csv(csv_file_name, index=True)
    print(f"\nSuccessfully saved DataFrame to '{csv_file_name}'")
    print(f"You can now open '{csv_file_name}' with spreadsheet software.")
except IOError as e:
    print(f"\nError: Could not save DataFrame to CSV file '{csv_file_name}': {e}")