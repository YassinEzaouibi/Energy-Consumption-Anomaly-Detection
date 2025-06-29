
# Energy Consumption Anomaly Detection

#### This project developed by `Yassin Ezaouibi & Hamza El Moukadam & Mohamed Essalhi`

## Overview

This project focuses on detecting unusual energy consumption patterns using a household energy dataset. By leveraging
multiple machine learning and deep learning techniques, we aim to identify anomalies that could indicate equipment
malfunctions, inefficient usage, or other significant events through comprehensive comparative analysis.

## 🎯 Goal

The primary objective is to build and compare multiple robust anomaly detection models capable of identifying abnormal
energy consumption patterns in time-series data. This enables proactive insights into potential issues and aids in
energy efficiency analysis by providing different perspectives on anomaly detection.

## 📊 Dataset

### Source

This project utilizes
the [UCI Appliances Energy Dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), which
provides detailed measurements of appliance energy consumption alongside environmental conditions.

### Data Description

- **Time Period**: Data collected in 10-minute intervals.
- **Format**: YYYY-MM-DD HH:MM:SS

### Features

1. **Energy Consumption**:
   * `Appliances`: Energy consumption of major appliances in Wh.
   * `lights`: Energy consumption of lights in Wh.

2. **Temperature Readings**:
   * `T1` to `T9`: Temperature measurements from various indoor sensors.
   * `T_out`: Outdoor temperature.

3. **Humidity Measurements**:
   * `RH_1` to `RH_9`: Relative humidity from various indoor sensors.
   * `RH_out`: Outdoor relative humidity.

4. **Weather Conditions**:
   * `Press_mm_hg`: Pressure in mm Hg.
   * `Windspeed`: Wind speed.
   * `Visibility`: Visibility.
   * `Tdewpoint`: Dew point temperature.

5. **Additional Variables**:
   * `rv1`, `rv2`: Random variables (used for testing purposes).

### File Format

- CSV (Comma-Separated Values).
- Each row represents one time interval.
- Missing values are handled during preprocessing.

### Data Location

The dataset is expected to be located in the `data/processed_data.csv` file within this project structure.

## 🔬 Methodology

This project implements **three distinct anomaly detection approaches**, each with its own dedicated notebook for
focused analysis and comparison:

### 1. **Statistical Approach: Z-Score Method**

**Notebook**: `notebooks/appliances_anomaly_detection_zscore.ipynb`

- **Technique**: Statistical outlier detection using Z-score analysis
- **Threshold**: Configurable standard deviation multiplier (default: 3σ)
- **Advantages**: Simple, interpretable, computationally efficient
- **Best for**: Detecting extreme outliers in energy consumption patterns

### 2. **Machine Learning Approach: Isolation Forest**

**Notebook**: `notebooks/appliances_anomaly_detection_isolationforest.ipynb`

- **Technique**: Ensemble-based anomaly detection using random forest isolation
- **Features**: Multivariate analysis using `['Appliances', 'T_out', 'lights', 'RH_1']`
- **Contamination Rate**: Configurable (default: 5%)
- **Advantages**: Handles multivariate patterns, robust to outliers
- **Best for**: Identifying complex anomalous patterns in feature space

### 3. **Deep Learning Approach: LSTM Autoencoder**

**Notebook**: `notebooks/appliances_anomaly_detection_tensorflow.ipynb`

- **Technique**: Deep learning with LSTM autoencoder for sequence reconstruction
- **Architecture**:
   - Encoder: Multi-layer LSTM with dropout
   - Decoder: Symmetric reconstruction layers
   - Window size: 24 hours (configurable)
- **Threshold**: Multiple options (95th/99th percentile, statistical)
- **Advantages**: Captures temporal dependencies and seasonal patterns
- **Best for**: Time-series pattern anomalies and sequence-based detection

## 🔄 Preprocessing Pipeline

All approaches share a common preprocessing pipeline:

1. **Data Loading**: Parse time-series data with proper datetime indexing
2. **Missing Value Handling**: Remove incomplete records using `dropna()`
3. **Feature Engineering**: Extract temporal features (hour, day_of_week, month)
4. **Normalization**: Apply `MinMaxScaler` for neural networks and distance-based models
5. **Temporal Features**: Generate hour, day-of-week, and monthly patterns

## 📈 Analysis Components

Each notebook includes comprehensive analysis:

### **Exploratory Data Analysis**

- Time-series visualization of energy consumption patterns
- Distribution analysis and statistical summaries
- Correlation matrix of environmental factors
- Hourly and daily usage pattern identification

### **Anomaly Detection & Evaluation**

- Model-specific anomaly identification
- Multiple threshold comparison (where applicable)
- Performance metrics and detection statistics
- Temporal pattern analysis of detected anomalies

### **In-depth Anomaly Characterization**

- **Temporal Analysis**: Hourly, daily, and seasonal anomaly patterns
- **Feature Distribution Comparison**: Normal vs. anomalous data characteristics
- **K-Means Clustering**: Grouping anomalies into distinct types
- **Environmental Correlation**: Relationship with weather and usage patterns

## 📊 Results & Visualizations

The project generates comprehensive visualizations for each approach:

### Statistical (Z-Score)

- Time-series plots with Z-score thresholds
- Distribution analysis of Z-scores
- Temporal anomaly patterns

### Machine Learning (Isolation Forest)

- Time-series anomaly highlighting
- Feature space scatter plots
- Hourly and daily anomaly distributions
- Environmental factor correlations

### Deep Learning (LSTM Autoencoder)

- Reconstruction error analysis
- Training history visualization
- Multi-threshold comparison
- Temporal pattern analysis

Example outputs:

![Isolation Forest Anomalies](images/anomalies_detected_by_isolation_forest.png)

![LSTM Autoencoder Analysis](images/time_series_with_anomalies_&_reconstruction_error.png)

## 📁 Project Structure

```
Energy-Consumption-Anomaly-Detection/
├── data/
│   ├── processed_data.csv              # Main dataset
│   └── anomaly_details.csv             # Detected anomaly details
├── notebooks/
│   ├── appliances_anomaly_detection_zscore.ipynb         # Statistical approach
│   ├── appliances_anomaly_detection_isolationforest.ipynb # ML approach
│   └── appliances_anomaly_detection_tensorflow.ipynb      # Deep learning approach
├── images/
│   ├── anomalies_detected_by_isolation_forest.png
│   ├── time_series_with_anomalies_&_reconstruction_error.png
│   └── *.png                           # Generated visualizations
├── scripts/
│   └── app.py                          # Streamlit dashboard
├── requirements.txt                     # Project dependencies
├── .gitignore                          # Git ignore file
└── README.md                           # Project documentation
```

## 🚀 How to Run

### Prerequisites
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YassinEzaouibi/Energy-Consumption-Anomaly-Detection
   cd Energy-Consumption-Anomaly-Detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Individual Approaches

Choose the approach you want to explore:

#### 1. Statistical Z-Score Method

```bash
jupyter notebook notebooks/appliances_anomaly_detection_zscore.ipynb
```

- **Quick setup**: Minimal computational requirements
- **Runtime**: ~2-3 minutes
- **Best for**: Initial exploration and baseline comparison

#### 2. Isolation Forest (Machine Learning)

```bash
jupyter notebook notebooks/appliances_anomaly_detection_isolationforest.ipynb
```

- **Moderate setup**: Standard machine learning requirements
- **Runtime**: ~5-10 minutes
- **Best for**: Multivariate anomaly analysis

#### 3. LSTM Autoencoder (Deep Learning)

```bash
jupyter notebook notebooks/appliances_anomaly_detection_tensorflow.ipynb
```

- **Advanced setup**: TensorFlow/GPU recommended
- **Runtime**: ~15-30 minutes (depending on hardware)
- **Best for**: Temporal pattern analysis and advanced modeling

### Optional: Interactive Dashboard

```bash
streamlit run scripts/app.py
```

## 🔧 Requirements

The project dependencies are optimized for all three approaches:

```txt
pandas~=2.2.2          # Data manipulation
matplotlib~=3.9.2      # Basic plotting
numpy~=2.1.1           # Numerical computing
scikit-learn~=1.5.1    # Machine learning algorithms
seaborn~=3.0.2         # Statistical visualizations
tensorflow~=2.19.0     # Deep learning framework
plotly                 # Interactive visualizations
streamlit~=1.45.1      # Dashboard framework
```

## 🎯 Model Comparison

| Approach             | Complexity | Interpretability | Temporal Awareness | Multivariate | Computational Cost |
|----------------------|------------|------------------|--------------------|--------------|--------------------|
| **Z-Score**          | Low        | High             | No                 | No           | Very Low           |
| **Isolation Forest** | Medium     | Medium           | No                 | Yes          | Low                |
| **LSTM Autoencoder** | High       | Low              | Yes                | Configurable | High               |

## 📋 Output Files

Each approach generates specific output files:

### Z-Score Method

- Anomaly statistics and temporal analysis
- Z-score distribution plots

### Isolation Forest

- `../data/anomaly_details.csv`: Detailed anomaly information
- Feature distribution comparisons
- Cluster analysis results

### LSTM Autoencoder

- `lstm_autoencoder_model.h5`: Trained model
- `lstm_anomaly_results.csv`: Complete results
- `detailed_lstm_anomalies.csv`: Anomaly-specific details
- `lstm_model_config.json`: Model configuration and metrics

## 🔮 Future Work

### Short-term Enhancements

- **Ensemble Methods**: Combine predictions from all three approaches
- **Hyperparameter Optimization**: Automated tuning for each model
- **Real-time Processing**: Streaming anomaly detection capabilities

### Advanced Features

- **Explainable AI (XAI)**: SHAP/LIME integration for model interpretability
- **Multivariate LSTM**: Incorporate all sensor features simultaneously
- **Online Learning**: Adaptive models for evolving patterns
- **Seasonal Decomposition**: Enhanced temporal pattern recognition

### Deployment & Integration

- **Docker Containerization**: Easy deployment and scaling
- **REST API**: Service-oriented architecture
- **Real-time Dashboard**: Live monitoring capabilities
- **Alert System**: Automated anomaly notifications

## 👥 Contributors

- **Mouncef Filali Bouami** - Supervisor
- **Yassin Ezaouibi** - Project Lead & Deep Learning Implementation
- **Hamza El Moukadam** - Machine Learning & Statistical Analysis
- **Mohamed Essalhi** - Data Analysis & Visualization
