import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

st.title("🔍 Energy Anomaly Detector")
# df = pd.read_csv('../data/processed_data.csv')
df = pd.read_csv('../data/processed_data.csv')
st.line_chart(df['Appliances'])
anomalies = df[df['anomaly'] == -1]
st.scatter_chart(anomalies[['Appliances']])

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
#
# st.title("🔍 Energy Anomaly Detector")
#
# # Fix the file path to use relative path instead of absolute
# try:
#     df = pd.read_csv('../data/processed_data.csv')
# except FileNotFoundError:
#     # Fallback paths in case the relative path doesn't work
#     st.error("Could not find the data file at '../data/processed_data.csv'")
#     st.info("Attempting alternative paths...")
#
#     try_paths = ['./data/processed_data.csv', 'data/processed_data.csv', '/data/processed_data.csv']
#
#     for path in try_paths:
#         try:
#             df = pd.read_csv(path)
#             st.success(f"Successfully loaded data from {path}")
#             break
#         except FileNotFoundError:
#             continue
#     else:
#         st.error("Could not find the data file in any of the expected locations.")
#         st.stop()
#
# # Display the energy consumption data
# st.subheader("Energy Consumption Over Time")
# st.line_chart(df['Appliances'])
#
# # Check if 'anomaly' column exists, if not, create it
# if 'anomaly' not in df.columns:
#     st.info("Detecting anomalies using Isolation Forest...")
#
#     # Apply Isolation Forest algorithm to detect anomalies
#     model = IsolationForest(contamination=0.05, random_state=42)
#     df['anomaly'] = model.fit_predict(df[['Appliances']])
#
# # Display anomalies
# anomalies = df[df['anomaly'] == -1]
# if not anomalies.empty:
#     st.subheader("Detected Anomalies")
#     st.scatter_chart(anomalies[['Appliances']])
#     st.info(f"Found {len(anomalies)} anomalies out of {len(df)} data points ({len(anomalies) / len(df) * 100:.2f}%)")
# else:
#     st.info("No anomalies detected.")
