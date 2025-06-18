import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("🔍 Energy Anomaly Detector")

try:
    # Load the raw data
    df = pd.read_csv('../data/processed_data.csv', parse_dates=['date'])

    # Data cleaning and validation
    if 'Appliances' not in df.columns:
        st.error("The 'Appliances' column is not found in the dataset.")
        st.stop()

    # Clean the data: handle missing values and ensure proper data types
    df['Appliances'] = pd.to_numeric(df['Appliances'], errors='coerce')
    df_clean = df.dropna(subset=['Appliances'])

    if df_clean.empty:
        st.error("No valid data found in the 'Appliances' column after cleaning.")
        st.stop()

    st.info(f"Loaded {len(df_clean)} valid data points")

    # Anomaly detection method selection
    method = st.selectbox("Choose anomaly detection method:",
                          ["Z-Score", "Isolation Forest", "Rolling Z-Score"])

    if method == "Z-Score":
        # Simple Z-score based anomaly detection
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        mean = df_clean['Appliances'].mean()
        std = df_clean['Appliances'].std()
        df_clean['z_score'] = (df_clean['Appliances'] - mean) / std
        df_clean['anomaly'] = np.where(df_clean['z_score'].abs() > threshold, -1, 1)

    elif method == "Isolation Forest":
        # Isolation Forest anomaly detection
        contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01)
        features = ['Appliances']

        # Add more features if available
        available_features = ['T1', 'RH_1', 'T_out', 'lights']
        for feature in available_features:
            if feature in df_clean.columns:
                features.append(feature)

        # Prepare data for Isolation Forest
        feature_data = df_clean[features].fillna(df_clean[features].mean())
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df_clean['anomaly'] = iso_forest.fit_predict(scaled_data)

    else:  # Rolling Z-Score
        # Dynamic/Rolling Z-score
        window = st.slider("Rolling Window Size", 10, 200, 72, 10)
        threshold = st.slider("Dynamic Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

        df_clean['rolling_mean'] = df_clean['Appliances'].rolling(window).mean()
        df_clean['rolling_std'] = df_clean['Appliances'].rolling(window).std()
        df_clean['dynamic_z'] = (df_clean['Appliances'] - df_clean['rolling_mean']) / df_clean['rolling_std']
        df_clean['anomaly'] = np.where(df_clean['dynamic_z'].abs() > threshold, -1, 1)

    # Display results
    st.subheader("Energy Consumption Over Time")
    st.line_chart(df_clean.set_index('date')['Appliances'])

    # Show anomalies
    anomalies = df_clean[df_clean['anomaly'] == -1]
    if not anomalies.empty:
        st.subheader("Detected Anomalies")
        st.write(f"Found {len(anomalies)} anomalies ({len(anomalies) / len(df_clean) * 100:.1f}% of data)")

        # Anomaly visualization
        fig_data = df_clean.set_index('date')
        normal_data = fig_data[fig_data['anomaly'] == 1][['Appliances']]
        anomaly_data = fig_data[fig_data['anomaly'] == -1][['Appliances']]

        # Plot normal data as line chart
        st.line_chart(normal_data)

        # Overlay anomalies as scatter plot
        if not anomaly_data.empty:
            st.scatter_chart(anomaly_data)

        # Show anomaly statistics
        st.subheader("Anomaly Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        with col2:
            st.metric("Anomaly Rate", f"{len(anomalies) / len(df_clean) * 100:.1f}%")
        with col3:
            st.metric("Max Anomaly Value", f"{anomalies['Appliances'].max():.1f}")

        # Show recent anomalies
        st.subheader("Recent Anomalies")
        display_cols = ['date', 'Appliances']
        if 'z_score' in anomalies.columns:
            display_cols.append('z_score')
        if 'dynamic_z' in anomalies.columns:
            display_cols.append('dynamic_z')

        st.dataframe(anomalies[display_cols].tail(10))
    else:
        st.info("No anomalies detected with current settings.")

except FileNotFoundError:
    st.error("Could not find the data file at '../data/processed_data.csv'")
    st.info("Please ensure the file exists at the specified path.")

except Exception as e:
    st.error(f"An error occurred while processing the data: {str(e)}")
