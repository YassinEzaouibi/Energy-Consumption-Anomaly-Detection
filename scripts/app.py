import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

st.set_page_config(page_title="Energy Anomaly Detector", page_icon="", layout="wide")

st.title(" Energy Anomaly Detector")
st.markdown("### Intelligent detection of unusual energy consumption patterns")

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

    # Add temporal features for better analysis
    df_clean['hour'] = df_clean['date'].dt.hour
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    df_clean['month'] = df_clean['date'].dt.month

    # Sidebar controls
    st.sidebar.header("️ Detection Settings")

    # Method selection
    method = st.sidebar.selectbox("Choose anomaly detection method:",
                                  ["Z-Score", "Isolation Forest", "Rolling Z-Score"])

    # Date range selector
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=[df_clean['date'].min().date(), df_clean['date'].max().date()],
        min_value=df_clean['date'].min().date(),
        max_value=df_clean['date'].max().date()
    )

    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_clean[(df_clean['date'].dt.date >= start_date) &
                               (df_clean['date'].dt.date <= end_date)]
    else:
        df_filtered = df_clean

    st.info(
        f" Analyzing {len(df_filtered)} data points from {df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}")

    # Method-specific parameters and detection
    if method == "Z-Score":
        threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        mean = df_filtered['Appliances'].mean()
        std = df_filtered['Appliances'].std()
        df_filtered['z_score'] = (df_filtered['Appliances'] - mean) / std
        df_filtered['anomaly'] = np.where(df_filtered['z_score'].abs() > threshold, -1, 1)

    elif method == "Isolation Forest":
        contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01)
        features = ['Appliances']

        # Add more features if available
        available_features = ['T1', 'RH_1', 'T_out', 'lights']
        selected_features = st.sidebar.multiselect(
            "Select additional features:",
            [f for f in available_features if f in df_filtered.columns],
            default=[f for f in available_features if f in df_filtered.columns][:2]
        )
        features.extend(selected_features)

        # Prepare data for Isolation Forest
        feature_data = df_filtered[features].fillna(df_filtered[features].mean())
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df_filtered['anomaly'] = iso_forest.fit_predict(scaled_data)

    else:  # Rolling Z-Score
        window = st.sidebar.slider("Rolling Window Size", 10, 200, 72, 10)
        threshold = st.sidebar.slider("Dynamic Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

        df_filtered['rolling_mean'] = df_filtered['Appliances'].rolling(window).mean()
        df_filtered['rolling_std'] = df_filtered['Appliances'].rolling(window).std()
        df_filtered['dynamic_z'] = (df_filtered['Appliances'] - df_filtered['rolling_mean']) / df_filtered[
            'rolling_std']
        df_filtered['anomaly'] = np.where(df_filtered['dynamic_z'].abs() > threshold, -1, 1)

    # Main dashboard with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(" Energy Consumption Timeline")

        # Interactive Plotly chart
        fig = go.Figure()

        # Normal data
        normal_data = df_filtered[df_filtered['anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['Appliances'],
            mode='lines',
            name='Normal Consumption',
            line=dict(color='#1f77b4', width=1),
            opacity=0.7
        ))

        # Anomalies
        anomaly_data = df_filtered[df_filtered['anomaly'] == -1]
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['date'],
                y=anomaly_data['Appliances'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='diamond'),
                text=anomaly_data['date'].dt.strftime('%Y-%m-%d %H:%M'),
                hovertemplate='<b>Anomaly Detected</b><br>Date: %{text}<br>Energy: %{y:.1f} Wh<extra></extra>'
            ))

        # Add rolling mean if using Rolling Z-Score
        if method == "Rolling Z-Score" and 'rolling_mean' in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['rolling_mean'],
                mode='lines',
                name='Rolling Mean',
                line=dict(color='orange', dash='dash'),
                opacity=0.5
            ))

        fig.update_layout(
            title=f"Energy Consumption Analysis ({method})",
            xaxis_title="Date",
            yaxis_title="Energy (Wh)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(" Detection Summary")

        anomalies = df_filtered[df_filtered['anomaly'] == -1]
        total_points = len(df_filtered)
        anomaly_count = len(anomalies)
        anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0

        # Metrics
        st.metric("Total Data Points", f"{total_points:,}")
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

        if not anomalies.empty:
            st.metric("Peak Anomaly", f"{anomalies['Appliances'].max():.1f} Wh")
            st.metric("Avg Anomaly", f"{anomalies['Appliances'].mean():.1f} Wh")

    # Additional visualizations if anomalies found
    if not anomalies.empty:
        st.subheader(" Anomaly Analysis")

        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(
            [" Temporal Patterns", " Distribution", "️ Environmental Correlation", " Data Table"])

        with tab1:
            # Hourly and daily patterns
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Anomalies by Hour of Day**")
                hourly_anomalies = anomalies['hour'].value_counts().sort_index()

                # Create hourly distribution chart
                fig_hour = go.Figure()
                fig_hour.add_trace(go.Bar(
                    x=list(range(24)),
                    y=[hourly_anomalies.get(i, 0) for i in range(24)],
                    marker_color='lightcoral',
                    name='Anomalies'
                ))
                fig_hour.update_layout(
                    title="Anomaly Distribution by Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Anomalies",
                    showlegend=False
                )
                st.plotly_chart(fig_hour, use_container_width=True)

            with col2:
                st.write("**Anomalies by Day of Week**")
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                daily_anomalies = anomalies['day_of_week'].value_counts().sort_index()

                fig_day = go.Figure()
                fig_day.add_trace(go.Bar(
                    x=[day_names[i] for i in range(7)],
                    y=[daily_anomalies.get(i, 0) for i in range(7)],
                    marker_color='lightblue',
                    name='Anomalies'
                ))
                fig_day.update_layout(
                    title="Anomaly Distribution by Day",
                    xaxis_title="Day of Week",
                    yaxis_title="Number of Anomalies",
                    showlegend=False
                )
                st.plotly_chart(fig_day, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Energy Consumption Distribution**")

                # Create histogram comparison
                normal_data_hist = df_filtered[df_filtered['anomaly'] == 1]['Appliances']
                anomaly_data_hist = anomalies['Appliances']

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=normal_data_hist,
                    name='Normal',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='lightblue'
                ))
                fig_hist.add_trace(go.Histogram(
                    x=anomaly_data_hist,
                    name='Anomalies',
                    opacity=0.7,
                    nbinsx=15,
                    marker_color='red'
                ))
                fig_hist.update_layout(
                    title="Distribution Comparison",
                    xaxis_title="Energy (Wh)",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                st.write("**Statistical Summary**")

                # Create summary statistics
                normal_stats = df_filtered[df_filtered['anomaly'] == 1]['Appliances'].describe()
                anomaly_stats = anomalies['Appliances'].describe()

                stats_df = pd.DataFrame({
                    'Normal': normal_stats,
                    'Anomalies': anomaly_stats
                })
                st.dataframe(stats_df.round(2))

                # Box plot comparison
                box_data = []
                box_data.extend([{'Energy': val, 'Type': 'Normal'} for val in normal_data_hist])
                box_data.extend([{'Energy': val, 'Type': 'Anomaly'} for val in anomaly_data_hist])
                box_df = pd.DataFrame(box_data)

                fig_box = px.box(box_df, x='Type', y='Energy', title="Energy Distribution Comparison")
                st.plotly_chart(fig_box, use_container_width=True)

        with tab3:
            if 'T_out' in df_filtered.columns and 'lights' in df_filtered.columns:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Temperature vs Energy**")

                    # Create scatter plot for temperature correlation
                    fig_temp = go.Figure()

                    # Normal points
                    normal_temp = df_filtered[df_filtered['anomaly'] == 1]
                    fig_temp.add_trace(go.Scatter(
                        x=normal_temp['T_out'],
                        y=normal_temp['Appliances'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', opacity=0.5, size=4)
                    ))

                    # Anomaly points
                    fig_temp.add_trace(go.Scatter(
                        x=anomalies['T_out'],
                        y=anomalies['Appliances'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', opacity=0.8, size=6)
                    ))

                    fig_temp.update_layout(
                        title="Temperature vs Energy Consumption",
                        xaxis_title="Outside Temperature",
                        yaxis_title="Energy (Wh)"
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)

                with col2:
                    st.write("**Lights vs Energy**")

                    # Create scatter plot for lights correlation
                    fig_lights = go.Figure()

                    # Normal points
                    fig_lights.add_trace(go.Scatter(
                        x=normal_temp['lights'],
                        y=normal_temp['Appliances'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', opacity=0.5, size=4)
                    ))

                    # Anomaly points
                    fig_lights.add_trace(go.Scatter(
                        x=anomalies['lights'],
                        y=anomalies['Appliances'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', opacity=0.8, size=6)
                    ))

                    fig_lights.update_layout(
                        title="Lights vs Energy Consumption",
                        xaxis_title="Lights Energy",
                        yaxis_title="Appliances Energy (Wh)"
                    )
                    st.plotly_chart(fig_lights, use_container_width=True)
            else:
                st.info("Environmental data (temperature, lights) not available for correlation analysis.")

        with tab4:
            st.write("**Recent Anomalies**")
            display_cols = ['date', 'Appliances', 'hour', 'day_of_week']

            # Add method-specific columns
            if 'z_score' in anomalies.columns:
                display_cols.append('z_score')
            if 'dynamic_z' in anomalies.columns:
                display_cols.append('dynamic_z')

            # Sort by date and show recent anomalies
            recent_anomalies = anomalies[display_cols].sort_values('date', ascending=False).head(20)
            st.dataframe(recent_anomalies, use_container_width=True)

            # Download option
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label=" Download All Anomalies as CSV",
                data=csv,
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    else:
        st.info(" No anomalies detected with current settings. Try adjusting the threshold parameters.")

        # Show normal data distribution
        st.subheader(" Normal Data Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Energy distribution histogram
            fig_normal = go.Figure()
            fig_normal.add_trace(go.Histogram(
                x=df_filtered['Appliances'],
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7
            ))
            fig_normal.update_layout(
                title="Energy Consumption Distribution",
                xaxis_title="Energy (Wh)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_normal, use_container_width=True)

        with col2:
            # Hourly average pattern
            hourly_avg = df_filtered.groupby('hour')['Appliances'].mean()

            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Scatter(
                x=list(range(24)),
                y=[hourly_avg.get(i, 0) for i in range(24)],
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            fig_hourly.update_layout(
                title="Average Energy by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Average Energy (Wh)"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

    # Footer with model information
    st.markdown("---")
    with st.expander("ℹ️ Model Information"):
        st.write(f"""
        **Detection Method:** {method}

        **Data Summary:**
        - Total Records: {len(df_filtered):,}
        - Date Range: {df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}
        - Sampling Interval: 10 minutes

        **Method Details:**
        """)

        if method == "Z-Score":
            st.write(f"- Threshold: {threshold}")
            st.write(f"- Mean Energy: {df_filtered['Appliances'].mean():.2f} Wh")
            st.write(f"- Standard Deviation: {df_filtered['Appliances'].std():.2f} Wh")
        elif method == "Isolation Forest":
            st.write(f"- Contamination Rate: {contamination}")
            st.write(f"- Features Used: {', '.join(features)}")
        else:
            st.write(f"- Rolling Window: {window} points")
            st.write(f"- Threshold: {threshold}")

except FileNotFoundError:
    st.error("❌ Could not find the data file at '../data/processed_data.csv'")
    st.info("Please ensure the file exists at the specified path.")

except Exception as e:
    st.error(f"❌ An error occurred while processing the data: {str(e)}")
    st.info("Please check your data format and try again.")
