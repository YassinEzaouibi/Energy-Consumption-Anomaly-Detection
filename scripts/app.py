import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Energy Anomaly Detection Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">⚡ Energy Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent detection and comparison of unusual energy consumption patterns")


@st.cache_data
def load_data():
    """Load and preprocess data with caching"""
    try:
        # Try multiple possible paths
        possible_paths = [
            '../data/processed_data.csv',
            './data/processed_data.csv',
            'data/processed_data.csv'
        ]

        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['date'])
                break

        if df is None:
            st.error("Could not find processed_data.csv in any expected location")
            return None

        # Data validation and cleaning
        required_columns = ['date', 'Appliances']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Required columns {required_columns} not found in dataset")
            return None

        # Clean and prepare data
        df['Appliances'] = pd.to_numeric(df['Appliances'], errors='coerce')
        df = df.dropna(subset=['Appliances'])

        # Add temporal features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_name'] = df['date'].dt.day_name()

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def detect_zscore_anomalies(df, threshold=3.0):
    """Z-Score based anomaly detection"""
    mean_val = df['Appliances'].mean()
    std_val = df['Appliances'].std()
    df['z_score'] = (df['Appliances'] - mean_val) / std_val
    df['is_anomaly_zscore'] = np.abs(df['z_score']) > threshold
    return df


def detect_isolation_forest_anomalies(df, contamination=0.05, features=['Appliances']):
    """Isolation Forest based anomaly detection"""
    # Prepare features
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        available_features = ['Appliances']

    feature_data = df[available_features].fillna(df[available_features].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_iso'] = iso_forest.fit_predict(scaled_data)
    df['is_anomaly_iso'] = df['anomaly_iso'] == -1

    return df


def create_sequences(data, window_size=24):
    """Create sequences for LSTM"""
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:(i + window_size)])
    return np.array(X)


# def detect_lstm_anomalies(df, window_size=24, threshold_percentile=95):
#     """LSTM Autoencoder based anomaly detection"""
#     try:
#         # Normalize data
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(df[['Appliances']])
#
#         # Create sequences
#         X = create_sequences(scaled_data.flatten(), window_size)
#
#         if len(X) == 0:
#             st.warning("Not enough data for LSTM analysis")
#             df['lstm_error'] = 0
#             df['is_anomaly_lstm'] = False
#             return df
#
#         # Check if pre-trained model exists
#         model_path = 'lstm_autoencoder_model.h5'
#         if os.path.exists(model_path):
#             try:
#                 model = load_model(model_path)
#                 st.info("✅ Loaded pre-trained LSTM model")
#             except:
#                 model = build_simple_lstm_autoencoder(window_size)
#                 st.info("🔄 Training new LSTM model (this may take a moment)...")
#                 model = train_lstm_model(model, X)
#         else:
#             model = build_simple_lstm_autoencoder(window_size)
#             st.info("🔄 Training new LSTM model (this may take a moment)...")
#             model = train_lstm_model(model, X)
#
#         # Get predictions and calculate reconstruction error
#         X_pred = model.predict(X, verbose=0)
#         reconstruction_error = np.mean(np.square(X_pred - X), axis=(1, 2))
#
#         # Set threshold
#         threshold = np.percentile(reconstruction_error, threshold_percentile)
#
#         # Initialize columns
#         df['lstm_error'] = 0.0
#         df['is_anomaly_lstm'] = False
#
#         # Map errors back to dataframe
#         start_idx = window_size - 1
#         end_idx = start_idx + len(reconstruction_error)
#         df.iloc[start_idx:end_idx, df.columns.get_loc('lstm_error')] = reconstruction_error
#         df.iloc[start_idx:end_idx, df.columns.get_loc('is_anomaly_lstm')] = reconstruction_error > threshold
#
#         return df
#
#     except Exception as e:
#         st.error(f"Error in LSTM anomaly detection: {str(e)}")
#         df['lstm_error'] = 0
#         df['is_anomaly_lstm'] = False
#         return df

def detect_lstm_anomalies(df, window_size=24, threshold_percentile=95):
    """LSTM Autoencoder based anomaly detection"""
    try:
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Appliances']])

        # Create sequences - fix the shape issue here
        X = create_sequences(scaled_data.flatten(), window_size)

        # Reshape X to match expected LSTM input shape (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Add this line

        if len(X) == 0:
            st.warning("Not enough data for LSTM analysis")
            df['lstm_error'] = 0
            df['is_anomaly_lstm'] = False
            return df

        # Check if pre-trained model exists
        model_path = 'lstm_autoencoder_model.h5'
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                st.info("✅ Loaded pre-trained LSTM model")
            except:
                model = build_simple_lstm_autoencoder(window_size)
                st.info("🔄 Training new LSTM model (this may take a moment)...")
                model = train_lstm_model(model, X)
        else:
            model = build_simple_lstm_autoencoder(window_size)
            st.info("🔄 Training new LSTM model (this may take a moment)...")
            model = train_lstm_model(model, X)

        # Get predictions and calculate reconstruction error
        X_pred = model.predict(X, verbose=0)

        # Fix the shape mismatch by ensuring both arrays have the same shape
        if X_pred.shape != X.shape:
            # If X_pred has shape (n, timesteps, 1) and X has shape (n, timesteps)
            if len(X_pred.shape) == 3 and len(X.shape) == 3:
                reconstruction_error = np.mean(np.square(X_pred.squeeze() - X.squeeze()), axis=1)
            else:
                reconstruction_error = np.mean(np.square(X_pred - X), axis=(1, 2))
        else:
            reconstruction_error = np.mean(np.square(X_pred - X), axis=(1, 2))

        # Set threshold
        threshold = np.percentile(reconstruction_error, threshold_percentile)

        # Initialize columns
        df['lstm_error'] = 0.0
        df['is_anomaly_lstm'] = False

        # Map errors back to dataframe
        start_idx = window_size - 1
        end_idx = start_idx + len(reconstruction_error)
        df.iloc[start_idx:end_idx, df.columns.get_loc('lstm_error')] = reconstruction_error
        df.iloc[start_idx:end_idx, df.columns.get_loc('is_anomaly_lstm')] = reconstruction_error > threshold

        return df

    except Exception as e:
        st.error(f"Error in LSTM anomaly detection: {str(e)}")
        df['lstm_error'] = 0
        df['is_anomaly_lstm'] = False
        return df


# def build_simple_lstm_autoencoder(window_size, n_features=1):
#     """Build a simple LSTM autoencoder"""
#     model = Sequential([
#         Input(shape=(window_size, n_features)),
#         LSTM(32, return_sequences=False),
#         RepeatVector(window_size),
#         LSTM(32, return_sequences=True),
#         TimeDistributed(Dense(n_features))
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model


def build_simple_lstm_autoencoder(window_size):
    """Build a simple LSTM autoencoder"""
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=False),
        RepeatVector(window_size),
        LSTM(50, return_sequences=True),
        TimeDistributed(Dense(1, activation='linear'))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(model, X, epochs=20, validation_split=0.2):
    """Train LSTM model with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Training progress: {epoch + 1}/{epochs} epochs - Loss: {logs.get("loss", 0):.6f}')

    model.fit(X, X, epochs=epochs, batch_size=32, validation_split=validation_split,
              verbose=0, callbacks=[ProgressCallback()])

    progress_bar.empty()
    status_text.empty()

    # Save model
    try:
        model.save('lstm_autoencoder_model.h5')
        st.success("✅ Model trained and saved successfully")
    except:
        st.warning("⚠️ Model trained but could not be saved")

    return model


def create_comparison_plots(df, methods_used):
    """Create comprehensive comparison plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series Comparison', 'Method Detection Rates',
                        'Hourly Anomaly Patterns', 'Anomaly Overlap Analysis'),
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # Plot 1: Time Series with all anomalies
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Appliances'], mode='lines',
                   name='Energy Consumption', line=dict(color='blue', width=1)),
        row=1, col=1
    )

    colors = ['red', 'orange', 'purple']
    method_names = ['Z-Score', 'Isolation Forest', 'LSTM Autoencoder']
    anomaly_cols = ['is_anomaly_zscore', 'is_anomaly_iso', 'is_anomaly_lstm']

    for i, (method, col, color) in enumerate(zip(method_names, anomaly_cols, colors)):
        if col in df.columns and method in methods_used:
            anomalies = df[df[col]]
            if len(anomalies) > 0:
                fig.add_trace(
                    go.Scatter(x=anomalies['date'], y=anomalies['Appliances'],
                               mode='markers', name=f'{method} Anomalies',
                               marker=dict(color=color, size=4)),
                    row=1, col=1
                )

    # Plot 2: Detection rates comparison
    detection_rates = []
    method_labels = []
    for method, col in zip(method_names, anomaly_cols):
        if col in df.columns and method in methods_used:
            rate = (df[col].sum() / len(df)) * 100
            detection_rates.append(rate)
            method_labels.append(method)

    fig.add_trace(
        go.Bar(x=method_labels, y=detection_rates, name='Detection Rate (%)',
               marker_color=['red', 'orange', 'purple'][:len(method_labels)]),
        row=1, col=2
    )

    # Plot 3: Hourly patterns
    hourly_data = {}
    for method, col in zip(method_names, anomaly_cols):
        if col in df.columns and method in methods_used:
            hourly_anomalies = df[df[col]].groupby('hour').size()
            hourly_data[method] = [hourly_anomalies.get(h, 0) for h in range(24)]

    for method, data in hourly_data.items():
        fig.add_trace(
            go.Bar(x=list(range(24)), y=data, name=f'{method} Hourly',
                   opacity=0.7),
            row=2, col=1
        )

    # Plot 4: Anomaly overlap (if multiple methods)
    if len([col for col in anomaly_cols if col in df.columns]) > 1:
        # Create overlap analysis
        overlap_data = []
        for method1, col1 in zip(method_names, anomaly_cols):
            if col1 in df.columns:
                for method2, col2 in zip(method_names, anomaly_cols):
                    if col2 in df.columns and method1 != method2:
                        overlap = df[df[col1] & df[col2]].shape[0]
                        total1 = df[col1].sum()
                        if total1 > 0:
                            overlap_pct = (overlap / total1) * 100
                            overlap_data.append({
                                'Method1': method1, 'Method2': method2,
                                'Overlap': overlap_pct
                            })

        if overlap_data:
            overlap_df = pd.DataFrame(overlap_data)
            for _, row in overlap_df.iterrows():
                fig.add_trace(
                    go.Scatter(x=[row['Method1']], y=[row['Overlap']],
                               mode='markers', name=f"Overlap with {row['Method2']}",
                               marker=dict(size=10)),
                    row=2, col=2
                )

    fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Anomaly Detection Analysis")
    return fig


# Load data
df = load_data()

if df is not None:
    # Sidebar configuration
    st.sidebar.header("🔧 Detection Configuration")

    # Method selection
    st.sidebar.subheader("📊 Analysis Methods")
    use_zscore = st.sidebar.checkbox("Z-Score Method", value=True)
    use_isolation_forest = st.sidebar.checkbox("Isolation Forest", value=True)
    use_lstm = st.sidebar.checkbox("LSTM Autoencoder", value=False)

    if not any([use_zscore, use_isolation_forest, use_lstm]):
        st.warning("Please select at least one detection method")
        st.stop()

    # Date range selector
    st.sidebar.subheader("📅 Data Range")
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=[df['date'].min().date(), df['date'].max().date()],
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )

    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'].dt.date >= start_date) &
                         (df['date'].dt.date <= end_date)].copy()
    else:
        df_filtered = df.copy()

    # Method-specific parameters
    methods_used = []

    if use_zscore:
        st.sidebar.subheader("📈 Z-Score Parameters")
        z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        methods_used.append("Z-Score")

    if use_isolation_forest:
        st.sidebar.subheader("🌲 Isolation Forest Parameters")
        contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01)
        available_features = ['Appliances', 'T_out', 'lights', 'RH_1']
        selected_features = st.sidebar.multiselect(
            "Select features:",
            [f for f in available_features if f in df_filtered.columns],
            default=['Appliances']
        )
        methods_used.append("Isolation Forest")

    if use_lstm:
        st.sidebar.subheader("🧠 LSTM Parameters")
        window_size = st.sidebar.slider("Window Size (hours)", 12, 72, 24, 6)
        threshold_percentile = st.sidebar.slider("Threshold Percentile", 90, 99, 95, 1)
        methods_used.append("LSTM Autoencoder")

    # Data info
    st.sidebar.info(f"""
    **Dataset Information:**
    - Total Records: {len(df_filtered):,}
    - Date Range: {df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}
    - Methods Selected: {len(methods_used)}
    """)

    # Run detection
    with st.spinner("🔍 Running anomaly detection..."):

        # Apply selected methods
        if use_zscore:
            df_filtered = detect_zscore_anomalies(df_filtered, z_threshold)

        if use_isolation_forest:
            df_filtered = detect_isolation_forest_anomalies(
                df_filtered, contamination, selected_features
            )

        if use_lstm:
            df_filtered = detect_lstm_anomalies(
                df_filtered, window_size, threshold_percentile
            )

    # Main dashboard
    st.header("📊 Detection Results Overview")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data Points", f"{len(df_filtered):,}")

    with col2:
        if use_zscore and 'is_anomaly_zscore' in df_filtered.columns:
            zscore_count = df_filtered['is_anomaly_zscore'].sum()
            zscore_pct = (zscore_count / len(df_filtered)) * 100
            st.metric("Z-Score Anomalies", f"{zscore_count:,} ({zscore_pct:.1f}%)")

    with col3:
        if use_isolation_forest and 'is_anomaly_iso' in df_filtered.columns:
            iso_count = df_filtered['is_anomaly_iso'].sum()
            iso_pct = (iso_count / len(df_filtered)) * 100
            st.metric("Isolation Forest Anomalies", f"{iso_count:,} ({iso_pct:.1f}%)")

    with col4:
        if use_lstm and 'is_anomaly_lstm' in df_filtered.columns:
            lstm_count = df_filtered['is_anomaly_lstm'].sum()
            lstm_pct = (lstm_count / len(df_filtered)) * 100
            st.metric("LSTM Anomalies", f"{lstm_count:,} ({lstm_pct:.1f}%)")

    # Comprehensive visualization
    st.header("📈 Comprehensive Analysis")

    if len(methods_used) > 0:
        fig = create_comparison_plots(df_filtered, methods_used)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed analysis tabs
    st.header("🔍 Detailed Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["🕒 Temporal Patterns", "📊 Statistical Analysis", "📋 Data Export"])

    with tab1:
        st.subheader("Temporal Anomaly Patterns")

        col1, col2 = st.columns(2)

        with col1:
            # Hourly patterns for each method
            st.write("**Hourly Anomaly Distribution**")
            hourly_fig = go.Figure()

            colors = ['red', 'orange', 'purple']
            method_names = ['Z-Score', 'Isolation Forest', 'LSTM Autoencoder']
            anomaly_cols = ['is_anomaly_zscore', 'is_anomaly_iso', 'is_anomaly_lstm']

            for method, col, color in zip(method_names, anomaly_cols, colors):
                if col in df_filtered.columns and method in methods_used:
                    hourly_anomalies = df_filtered[df_filtered[col]].groupby('hour').size()
                    hourly_data = [hourly_anomalies.get(h, 0) for h in range(24)]
                    hourly_fig.add_trace(
                        go.Bar(x=list(range(24)), y=hourly_data, name=method,
                               marker_color=color, opacity=0.7)
                    )

            hourly_fig.update_layout(
                title="Anomalies by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Number of Anomalies",
                barmode='group'
            )
            st.plotly_chart(hourly_fig, use_container_width=True)

        with col2:
            # Daily patterns
            st.write("**Daily Anomaly Distribution**")
            daily_fig = go.Figure()

            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            for method, col, color in zip(method_names, anomaly_cols, colors):
                if col in df_filtered.columns and method in methods_used:
                    daily_anomalies = df_filtered[df_filtered[col]].groupby('day_of_week').size()
                    daily_data = [daily_anomalies.get(d, 0) for d in range(7)]
                    daily_fig.add_trace(
                        go.Bar(x=day_names, y=daily_data, name=method,
                               marker_color=color, opacity=0.7)
                    )

            daily_fig.update_layout(
                title="Anomalies by Day of Week",
                xaxis_title="Day",
                yaxis_title="Number of Anomalies",
                barmode='group'
            )
            st.plotly_chart(daily_fig, use_container_width=True)

    with tab2:
        st.subheader("Statistical Analysis")

        # Create statistical comparison
        stats_data = []
        for method, col in zip(method_names, anomaly_cols):
            if col in df_filtered.columns and method in methods_used:
                anomalies = df_filtered[df_filtered[col]]
                normal = df_filtered[~df_filtered[col]]

                stats_data.append({
                    'Method': method,
                    'Total Anomalies': len(anomalies),
                    'Anomaly Rate (%)': (len(anomalies) / len(df_filtered)) * 100,
                    'Avg Energy (Normal)': normal['Appliances'].mean(),
                    'Avg Energy (Anomaly)': anomalies['Appliances'].mean() if len(anomalies) > 0 else 0,
                    'Max Energy (Anomaly)': anomalies['Appliances'].max() if len(anomalies) > 0 else 0
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.round(2), use_container_width=True)

        # Distribution comparison
        if len(methods_used) > 0:
            st.write("**Energy Distribution: Normal vs Anomalous**")
            dist_fig = go.Figure()

            # Normal data
            normal_data = df_filtered['Appliances']
            dist_fig.add_trace(
                go.Histogram(x=normal_data, name='All Data', opacity=0.7,
                             nbinsx=50, marker_color='blue')
            )

            # Anomalous data for each method
            for method, col, color in zip(method_names, anomaly_cols, colors):
                if col in df_filtered.columns and method in methods_used:
                    anomaly_data = df_filtered[df_filtered[col]]['Appliances']
                    if len(anomaly_data) > 0:
                        dist_fig.add_trace(
                            go.Histogram(x=anomaly_data, name=f'{method} Anomalies',
                                         opacity=0.6, nbinsx=30, marker_color=color)
                        )

            dist_fig.update_layout(
                title="Energy Consumption Distribution Comparison",
                xaxis_title="Energy (Wh)",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(dist_fig, use_container_width=True)

    with tab3:
        st.subheader("Method Comparison")

        if len(methods_used) > 1:
            # Calculate overlaps
            st.write("**Anomaly Detection Overlap Analysis**")

            overlap_matrix = pd.DataFrame(index=methods_used, columns=methods_used)

            method_col_map = {
                'Z-Score': 'is_anomaly_zscore',
                'Isolation Forest': 'is_anomaly_iso',
                'LSTM Autoencoder': 'is_anomaly_lstm'
            }

            for method1 in methods_used:
                for method2 in methods_used:
                    if method1 == method2:
                        overlap_matrix.loc[method1, method2] = 100.0
                    else:
                        col1 = method_col_map[method1]
                        col2 = method_col_map[method2]

                        if col1 in df_filtered.columns and col2 in df_filtered.columns:
                            overlap = df_filtered[df_filtered[col1] & df_filtered[col2]].shape[0]
                            total1 = df_filtered[col1].sum()
                            overlap_pct = (overlap / total1 * 100) if total1 > 0 else 0
                            overlap_matrix.loc[method1, method2] = overlap_pct
                        else:
                            overlap_matrix.loc[method1, method2] = 0

            overlap_matrix = overlap_matrix.astype(float)

            # Create heatmap
            import plotly.figure_factory as ff

            fig_heatmap = ff.create_annotated_heatmap(
                z=overlap_matrix.values,
                x=list(overlap_matrix.columns),
                y=list(overlap_matrix.index),
                annotation_text=overlap_matrix.round(1).astype(str),
                colorscale='Viridis'
            )
            fig_heatmap.update_layout(
                title="Method Overlap Matrix (% of Method 1 anomalies also detected by Method 2)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Consensus analysis
            st.write("**Consensus Anomalies (Detected by Multiple Methods)**")
            consensus_anomalies = df_filtered.copy()
            consensus_anomalies['consensus_count'] = 0

            for method, col in zip(method_names, anomaly_cols):
                if col in df_filtered.columns and method in methods_used:
                    consensus_anomalies['consensus_count'] += consensus_anomalies[col].astype(int)

            consensus_summary = consensus_anomalies['consensus_count'].value_counts().sort_index()

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Consensus Distribution:**")
                for count, freq in consensus_summary.items():
                    if count > 0:
                        pct = (freq / len(df_filtered)) * 100
                        st.write(f"- Detected by {count} method(s): {freq:,} points ({pct:.2f}%)")

            with col2:
                # High consensus anomalies
                high_consensus = consensus_anomalies[consensus_anomalies['consensus_count'] >= 2]
                if len(high_consensus) > 0:
                    st.write("**High Confidence Anomalies (Multiple Methods):**")
                    st.write(f"Total: {len(high_consensus):,} anomalies")
                    st.write(f"Peak Energy: {high_consensus['Appliances'].max():.1f} Wh")
                    st.write(f"Average Energy: {high_consensus['Appliances'].mean():.1f} Wh")
        else:
            st.info("Select multiple methods to see comparison analysis")

    with tab4:
        st.subheader("Data Export")

        # Prepare export data
        export_columns = ['date', 'Appliances', 'hour', 'day_of_week', 'day_name']

        if use_zscore and 'is_anomaly_zscore' in df_filtered.columns:
            export_columns.extend(['z_score', 'is_anomaly_zscore'])

        if use_isolation_forest and 'is_anomaly_iso' in df_filtered.columns:
            export_columns.extend(['is_anomaly_iso'])

        if use_lstm and 'is_anomaly_lstm' in df_filtered.columns:
            export_columns.extend(['lstm_error', 'is_anomaly_lstm'])

        export_df = df_filtered[export_columns].copy()

        # Summary of export data
        st.write("**Export Summary:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(export_df))

        with col2:
            total_anomalies = 0
            if 'is_anomaly_zscore' in export_df.columns:
                total_anomalies += export_df['is_anomaly_zscore'].sum()
            if 'is_anomaly_iso' in export_df.columns:
                total_anomalies += export_df['is_anomaly_iso'].sum()
            if 'is_anomaly_lstm' in export_df.columns:
                total_anomalies += export_df['is_anomaly_lstm'].sum()
            st.metric("Total Anomaly Detections", total_anomalies)

        with col3:
            st.metric("Columns Included", len(export_columns))

        # Preview
        st.write("**Data Preview:**")
        st.dataframe(export_df.head(10), use_container_width=True)

        # Download buttons
        col1, col2 = st.columns(2)

        with col1:
            # Full dataset download
            csv_full = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results",
                data=csv_full,
                file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Anomalies only download
            anomaly_mask = False
            for col in ['is_anomaly_zscore', 'is_anomaly_iso', 'is_anomaly_lstm']:
                if col in export_df.columns:
                    anomaly_mask = anomaly_mask | export_df[col]

            if anomaly_mask is not False and anomaly_mask.any():
                anomalies_only = export_df[anomaly_mask]
                csv_anomalies = anomalies_only.to_csv(index=False)
                st.download_button(
                    label="📥 Download Anomalies Only",
                    data=csv_anomalies,
                    file_name=f"detected_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

else:
    st.error("Unable to load data. Please check that the processed_data.csv file exists in the correct location.")
    st.info("""
    Expected file locations:
    - `../data/processed_data.csv`
    - `./data/processed_data.csv` 
    - `data/processed_data.csv`
    """)

# Footer
st.markdown("---")
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    ### Energy Anomaly Detection Dashboard

    This interactive dashboard implements three different anomaly detection approaches:

    **🔢 Z-Score Method**: Statistical approach using standard deviation thresholds
    - Fast and interpretable
    - Good for detecting extreme outliers
    - Configurable threshold (1-5 standard deviations)

    **🌲 Isolation Forest**: Machine learning ensemble method
    - Handles multivariate data
    - Robust to outliers
    - Configurable contamination rate and features

    **🧠 LSTM Autoencoder**: Deep learning approach for temporal patterns
    - Captures sequential dependencies
    - Learns complex patterns
    - Configurable window size and threshold

    **Features:**
    - Real-time parameter adjustment
    - Method comparison and overlap analysis
    - Comprehensive temporal analysis
    - Data export functionality
    - Pre-trained model loading (LSTM)

    **Data Requirements:**
    - CSV file with 'date' and 'Appliances' columns
    - Optional environmental features (T_out, lights, RH_1)
    - 10-minute interval time series data
    """)
