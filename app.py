import streamlit as st

st.set_page_config(page_title="Ceph Analyzer", layout="wide")
st.title("Ceph EDA & Prediction Web App")

st.markdown("""
Welcome to the Ceph Analysis Platform.

- Go to **Create Ceph Dataset** to upload your JSON files and generate the dataset.
- Then go to **EDA and Predictions** to explore data and predict recovery/replication metrics.
""")

