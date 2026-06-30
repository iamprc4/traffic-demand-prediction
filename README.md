# 🚦 Traffic Demand Prediction | Flipkart Grid 6.0 - GridLock Hackathon

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Overview

This repository contains our solution for **Flipkart Grid 6.0 - GridLock Hackathon**, where the objective was to accurately predict traffic demand using historical traffic patterns, temporal information, road infrastructure, weather conditions, and geographical data.

The solution leverages advanced feature engineering techniques combined with ensemble machine learning models (LightGBM and CatBoost) to learn complex traffic demand patterns and generate highly accurate predictions with a score of **91.18**.

---

## 🎯 Problem Statement

Urban traffic demand changes dynamically based on multiple factors:

- **Temporal patterns**: Time of day, day of week, hourly variations
- **Road infrastructure**: Road type, number of lanes, large vehicle permissions
- **Weather conditions**: Temperature, weather patterns (Sunny, Rainy, etc.)
- **Geographic location**: Geohash-encoded location data
- **Point of Interest**: Presence of nearby landmarks

The goal was to build a robust regression model capable of accurately forecasting future traffic demand using the provided multivariate time-series dataset.

---

## 📊 Dataset

### Train Dataset (77,299 records)
- **Index**: Unique identifier
- **geohash**: Encoded geographic location
- **day**: Day number
- **timestamp**: Hour:Minute format
- **demand**: Target variable (traffic demand)
- **RoadType**: Type of road (Residential, Highway, etc.)
- **NumberofLanes**: Number of lanes on the road
- **LargeVehicles**: Whether large vehicles are allowed
- **Landmarks**: Presence of nearby landmarks
- **Temperature**: Temperature in Celsius
- **Weather**: Weather condition (Sunny, Rainy, etc.)

### Test Dataset (41,778 records)
Same features as training data, excluding the target variable `demand`.

---

## 🔄 Project Workflow

```
Data Loading
    ↓
Exploratory Data Analysis
    ↓
Data Cleaning & Preprocessing
    ↓
Advanced Feature Engineering
    ↓
Feature Encoding (Label, Target, Frequency)
    ↓
Model Training (LightGBM + CatBoost)
    ↓
K-Fold Cross Validation
    ↓
Hyperparameter Tuning
    ↓
Ensemble Prediction
    ↓
Submission Generation
```

---

## 🛠️ Feature Engineering

Our solution implements extensive feature engineering to capture traffic patterns:

### 1️⃣ Temporal Features

Extracted from `timestamp` and `day`:

- **hour**: Hour of the day (0-23)
- **minute**: Minute of the hour
- **time_slot**: Categorical time periods (Morning, Afternoon, Evening, Night)
- **hour_sin/hour_cos**: Cyclical encoding of hours (captures circular nature of time)
- **is_peak_hour**: Binary feature for rush hours (7-9 AM, 5-7 PM)
- **is_weekend**: Derived from day patterns

### 2️⃣ Geospatial Features

Decoded from `geohash`:

- **latitude**: Geographic latitude
- **longitude**: Geographic longitude
- **geo_prefix_4**: First 4 characters of geohash (regional clustering)
- **geo_prefix_5**: First 5 characters of geohash (sub-regional clustering)

These hierarchical features help capture regional and local traffic patterns.

### 3️⃣ Statistical Aggregation Features

Location-based statistical features:

- **mean_demand_by_geo**: Average demand per geohash location
- **median_demand_by_geo**: Median demand per geohash
- **std_demand_by_geo**: Demand variability per location
- **mean_demand_by_hour**: Average demand per hour across all locations
- **mean_demand_by_road**: Average demand by road type

### 4️⃣ Frequency Encoding

Count-based features to capture location and infrastructure popularity:

- **geohash_count**: Frequency of each geohash in training data
- **roadtype_count**: Frequency of each road type
- **landmark_count**: Frequency of landmark presence

### 5️⃣ Interaction Features

Combined features to capture complex relationships:

- **geo_hour_interaction**: Geohash × Hour interaction
- **road_lane_interaction**: Road Type × Number of Lanes
- **weather_temp_interaction**: Weather × Temperature bands

### 6️⃣ Target Encoding (Out-of-Fold)

Applied on high-cardinality categorical variables:

- **geohash_target_encoded**
- **geo_prefix_4_target_encoded**

Uses K-Fold out-of-fold encoding to prevent target leakage while preserving predictive information.

---

## 🧹 Data Preprocessing

### Missing Value Handling
- **Numerical features**: Median imputation
- **Categorical features**: Mode imputation or 'Unknown' category
- **RoadType**: Filled with most frequent value or 'Other'
- **Temperature/Weather**: Strategic imputation based on time and location patterns

### Encoding Strategies
- **Label Encoding**: For ordinal and categorical features
- **Target Encoding**: For high-cardinality features
- **Frequency Encoding**: For rare category handling

### Target Transformation
Applied log transformation to handle skewness:

```python
y_train = np.log1p(demand)  # Log transformation
predictions = np.expm1(y_pred)  # Inverse transformation
```

---

## 🤖 Machine Learning Models

### 1️⃣ LightGBM (Light Gradient Boosting Machine)

**Why LightGBM?**
- Extremely fast training on large datasets
- Excellent performance on tabular data
- Efficient handling of categorical features
- Built-in support for missing values
- Leaf-wise tree growth strategy

**Hyperparameters:**
```python
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
```

### 2️⃣ CatBoost (Categorical Boosting)

**Why CatBoost?**
- Superior handling of categorical features
- Robust to overfitting
- Minimal hyperparameter tuning required
- Built-in ordered target encoding
- Excellent generalization

**Hyperparameters:**
```python
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'verbose': False
}
```

### 3️⃣ Ensemble Learning

Final predictions generated using weighted ensemble:

```python
final_prediction = 0.5 * lgb_predictions + 0.5 * catboost_predictions
```

**Benefits:**
- Reduces model variance
- Improves generalization
- Captures different aspects of data patterns
- More robust to outliers

---

## ✅ Validation Strategy

### K-Fold Cross Validation (5 Folds)
- Ensures model stability across different data splits
- Prevents overfitting
- Provides reliable performance estimates

### Out-of-Fold Predictions
- Used for target encoding
- Prevents data leakage
- Improves feature quality

### Early Stopping
- Monitors validation performance
- Stops training when no improvement
- Prevents overfitting

### Evaluation Metrics
- **Primary**: R² Score (Coefficient of Determination)
- **Secondary**: RMSE (Root Mean Squared Error)
- **Monitoring**: MAE (Mean Absolute Error)

---

## 🏆 Results

### Model Performance
- **Final Score**: **91.18** on leaderboard
- **Cross-Validation R² Score**: 0.89+
- **RMSE**: Low error rate on validation set

### Key Success Factors
1. **Comprehensive Feature Engineering**: Created 30+ engineered features
2. **Effective Handling of Missing Data**: Strategic imputation strategies
3. **Geospatial Intelligence**: Decoded and utilized location hierarchy
4. **Temporal Pattern Recognition**: Cyclical encoding and time-based features
5. **Ensemble Approach**: Combined strengths of LightGBM and CatBoost
6. **Proper Validation**: K-Fold CV with out-of-fold target encoding

---

## 💻 Tech Stack

### Core Libraries
- **Python 3.11+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-Learn**: Machine learning utilities

### Machine Learning
- **LightGBM**: Gradient boosting framework
- **CatBoost**: Gradient boosting on decision trees

### Geospatial
- **PyGeoHash**: Geohash encoding/decoding

### Development Environment
- **Google Colab**: Cloud-based Jupyter environment
- **Jupyter Notebook**: Interactive development

---

## 📁 Repository Structure

```
traffic-demand-prediction/
│
├── data/
│   ├── train.csv              # Training dataset (77,299 records)
│   └── test.csv               # Test dataset (41,778 records)
│
├── notebooks/
│   ├── Traffic_Demand_Final_NoErrorsnewer.ipynb  # Main solution notebook
│   └── Traffic Demand Prediction - Flipkart Grid 2.0 (1).docx
│
├── outputs/
│   └── submission.csv         # Final predictions
│
├── README.md                  # Project documentation
│
└── requirements.txt           # Python dependencies
```

---

## 🚀 Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/iamprc4/traffic-demand-prediction.git
cd traffic-demand-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
catboost>=1.2.0
pygeohash>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Run the Notebook

**Option 1: Jupyter Notebook**
```bash
jupyter notebook Traffic_Demand_Final_NoErrorsnewer.ipynb
```

**Option 2: Google Colab**
- Upload the notebook to Google Colab
- Upload `train.csv` and `test.csv` when prompted
- Run all cells sequentially

### Generate Predictions

The notebook will automatically:
1. Load and preprocess data
2. Engineer features
3. Train models
4. Generate predictions
5. Create `submission.csv` file

---

## 📈 Model Training Pipeline

### Step 1: Data Loading
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

### Step 2: Feature Engineering
```python
# Temporal features
train['hour'] = train['timestamp'].str.split(':').str[0].astype(int)
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)

# Geospatial features
train['latitude'], train['longitude'] = zip(*train['geohash'].apply(decode_geohash))

# Statistical features
train['mean_demand_by_geo'] = train.groupby('geohash')['demand'].transform('mean')
```

### Step 3: Model Training
```python
# LightGBM
lgb_model = lgb.train(lgb_params, train_data, valid_sets=[valid_data])

# CatBoost
cat_model = CatBoostRegressor(**catboost_params)
cat_model.fit(X_train, y_train)
```

### Step 4: Ensemble Prediction
```python
final_pred = 0.5 * lgb_pred + 0.5 * cat_pred
```

---

## 🔍 Key Insights

### Traffic Patterns Discovered
1. **Peak Hours**: Strong demand during 7-9 AM and 5-7 PM
2. **Weekend Effect**: Reduced demand on weekends
3. **Weather Impact**: Rainy weather reduces traffic demand
4. **Location Clusters**: Certain geohashes consistently show high demand
5. **Infrastructure Influence**: Roads with more lanes handle higher demand

### Feature Importance Analysis
Top 10 most important features:
1. hour (temporal)
2. geohash_target_encoded
3. mean_demand_by_geo
4. latitude/longitude
5. NumberofLanes
6. hour_sin/hour_cos
7. RoadType_encoded
8. Temperature
9. is_peak_hour
10. geo_hour_interaction

---

## 🔮 Future Improvements

### Advanced Modeling
- **Deep Learning**: LSTM/GRU for sequential patterns
- **Graph Neural Networks**: For spatial relationships
- **Transformer Models**: Attention-based spatio-temporal prediction
- **XGBoost Integration**: Additional ensemble member

### Feature Engineering
- **Traffic Flow Networks**: Graph-based features
- **External Data**: Holiday calendars, event data, POI data
- **Lag Features**: Previous day/week demand
- **Rolling Statistics**: Moving averages and trends

### Optimization
- **Bayesian Hyperparameter Optimization**: Using Optuna or Hyperopt
- **AutoML**: Automated feature selection and model tuning
- **Stacking**: Multi-level ensemble models
- **Calibration**: Probability calibration for better predictions

### Deployment
- **Real-time Prediction API**: FastAPI/Flask deployment
- **Model Monitoring**: Track prediction drift
- **A/B Testing**: Compare model versions
- **Containerization**: Docker deployment

---

## 👥 Team

**Developed for Flipkart Grid 6.0 - GridLock Hackathon**

### Contributors
- Data Analysis & Feature Engineering
- Model Development & Optimization
- Validation & Testing

---

## 📝 License

This project is developed for **Flipkart Grid 6.0 Hackathon** and is intended for educational and competition purposes.

---

## 🙏 Acknowledgments

- **Flipkart**: For organizing the GridLock Hackathon
- **Competition Organizers**: For providing the dataset and platform
- **Open Source Community**: For the amazing ML libraries

---

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or repository discussions.

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover this work.

---

**Built with ❤️ for Flipkart Grid 6.0 GridLock Hackathon**
