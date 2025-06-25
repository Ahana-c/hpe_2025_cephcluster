import streamlit as st
import pandas as pd
import json
from utils.processing import parse_ceph_json

st.title("📂 Ceph Dataset Creator")

def load_json_stream(file):
    """Handle multiple JSON arrays separated by ][ and flatten into one list."""
    raw = file.read().decode("utf-8").strip()

    # Fix invalid format by joining multiple arrays
    fixed_raw = raw.replace("][", "],[")
    try:
        parsed = json.loads(f"[{fixed_raw}]")  # Wrap in [] to make one big list
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding failed: {e}")
        return []

    # Flatten if it's a list of lists
    if isinstance(parsed, list) and all(isinstance(p, list) for p in parsed):
        flat = []
        for sublist in parsed:
            flat.extend(sublist)
        return flat
    return parsed

uploaded_files = st.file_uploader(
    "Upload Ceph JSON Files", type="json", accept_multiple_files=True
)

if uploaded_files:
    if st.button("Create Dataset"):
        final_df = pd.DataFrame()
        for file in uploaded_files:
            entries = load_json_stream(file)
            if entries:
                parsed_df = parse_ceph_json(entries)
                final_df = pd.concat([final_df, parsed_df], ignore_index=True)
            else:
                st.warning(f"No valid entries found in {file.name}")

        if not final_df.empty:
            st.success("✅ Dataset Created!")
            st.dataframe(final_df.head())

            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "ceph_dataset.csv", "text/csv")
        else:
            st.warning("No valid data extracted from any uploaded files.")
