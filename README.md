# LoomIQ — Production Intelligence System

> An ML-powered production monitoring and intelligence dashboard for analyzing machine-level manufacturing performance, predicting production output, and identifying unusual operating behavior.

**Live Demo:** https://loomiq-appuction-intelligence-system-xetzupcccpcsjaavn6b4vt.streamlit.app/  
**GitHub:** https://github.com/Pratyush0110/LoomIQ-Production-Intelligence-System

---

## Overview

**LoomIQ — Production Intelligence System** is a Streamlit-based machine-learning application designed for production monitoring and operational analysis.

Users can upload compatible CSV or Excel production datasets and use the dashboard to:

- monitor machine-level production performance
- compare actual production with an expected baseline when delivery-speed data is available
- detect unusual operating patterns using Isolation Forest
- inspect individual machine behavior and historical records
- generate production estimates using a trained Random Forest model
- explore production and anomaly results through interactive Plotly visualizations

The application is designed as a **decision-support and monitoring tool**, not as a replacement for engineering or operational judgment.

---

## Key Features

### 📊 Production Analytics
- Production and runtime KPIs
- Machine-level production monitoring
- Actual vs. expected production analysis
- Production deviation and deviation percentage analysis
- Interactive production tables and charts

### 🤖 Machine Learning
- **Random Forest** model for production prediction
- **Isolation Forest** for unsupervised anomaly detection
- Dynamic anomaly-model training on the uploaded dataset
- Automatic selection of suitable numerical operating features
- Standardized feature processing before anomaly detection

### 🏭 Machine Monitoring
- Machine selector for detailed inspection
- Historical production analysis
- Runtime and operating-parameter analysis
- Recent machine records
- Machine-level anomaly summaries

### 📁 Flexible Data Input
- CSV upload
- XLSX upload
- XLS upload
- Automatic column cleaning and numerical conversion
- Duplicate-record removal
- Removal of report summary/total rows
- Missing-value handling for supported analysis

---

## How It Works

### 1. Upload Production Data

The user uploads a compatible CSV or Excel production dataset through the Streamlit dashboard.

### 2. Data Processing

The application:

1. cleans column names
2. removes duplicate records
3. removes report summary rows such as `Total`
4. converts dates to datetime
5. converts supported numerical fields to numeric values
6. removes records missing the core fields required for analysis

### 3. Expected Production Baseline

When a genuine `Del Spd` (delivery speed) column is available, the application calculates:

**Expected Production = Delivery Speed × Runtime**

It then derives production-ratio and deviation metrics, including:

- Expected Production
- Deviation (m)
- Deviation (%)
- Absolute Deviation (%)

If `Del Spd` is not available, the application does **not** create an artificial production baseline. The operating-behavior analysis can still be performed.

### 4. Anomaly Detection

The application dynamically identifies suitable numerical operating features from the uploaded dataset.

The selected features are:

- cleaned and converted to numeric values
- imputed where necessary
- filtered to remove constant features
- standardized using `StandardScaler`

An **Isolation Forest** model is then trained on the uploaded operating records.

The model labels records as:

- `Normal`
- `Anomaly`

An anomaly indicates that the operating pattern is unusual relative to the data provided. It does not automatically indicate equipment failure.

### 5. Production Prediction

The application also loads a pre-trained **Random Forest production model** from:

```text
models/production_model.pkl
```

The prediction interface allows the user to enter operating conditions and obtain an estimated production value.

The result is compared with the selected machine's historical average and median production when historical data is available.

---

## Dataset Requirements

### Required Columns

| Column | Description |
|---|---|
| `Mcno` | Machine identifier |
| `Date` | Production date |
| `Runtime(Min)` | Machine runtime in minutes |
| `Prod Mtrs` | Production output in meters |

### Optional Production-Baseline Column

| Column | Description |
|---|---|
| `Del Spd` | Delivery speed |

`Del Spd` is required only for the expected-production and production-deviation analysis.

### Supported Operating Variables

The application can use available numerical operating variables such as:

- `Shift`
- `Count`
- `Stoptime(Min)`
- `Tar.Eff%`
- `A%`
- `Del Spd`
- `Tar RPM`
- `RPM`
- `TPM`
- `Total Dofftime`
- `kW/Hr`
- `UKG`
- `Total Units`

The exact feature set used by the anomaly detector depends on which valid numerical variables are present in the uploaded dataset.

---

## Column-Name Handling

The application intentionally distinguishes between different field names.

For example:

```text
Del Spd
delivery
delivery speed
```

are not blindly treated as the same business field.

The application uses explicit column normalization rules so that a similarly named field is not incorrectly interpreted as delivery speed and used to calculate an expected-production baseline.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and ML development |
| Streamlit | Interactive web dashboard |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model serialization/loading |
| Plotly | Interactive visualizations |
| OpenPyXL | XLSX file handling |
| xlrd | XLS file handling |

---

## Project Structure

```text
LoomIQ-Production-Intelligence-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── production_model.pkl
│
└── assets/
    ├── textile_machine.jpg
    └── textile_industry.jpg
```

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pratyush0110/LoomIQ-Production-Intelligence-System.git
cd LoomIQ-Production-Intelligence-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For compatibility with the serialized production model, the project pins the required scikit-learn version in `requirements.txt`.

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

https://loomiq-appuction-intelligence-system-xetzupcccpcsjaavn6b4vt.streamlit.app/

---

## Data Privacy

This repository is intended for application code and deployment assets, not confidential production records.

**Do not upload:**

- confidential company production datasets
- proprietary business data
- credentials or API keys
- internal reports
- sensitive operational information

For public demonstrations, use a **synthetic or sanitized dataset** and ensure you have permission to use any model or data derived from organizational information.

---

## Limitations

- Anomaly detection is relative to the uploaded dataset and its operating patterns.
- An anomaly does not automatically mean a machine has failed.
- Expected-production analysis requires a valid `Del Spd` field.
- Prediction quality depends on how representative the model's training data is of the production conditions being evaluated.
- Uploaded datasets must contain the required core columns for production analysis.

---

## Disclaimer

This system is a **production monitoring and decision-support tool**.

Its predictions and anomaly flags should be interpreted together with machine history, operating conditions, maintenance information, and engineering judgment. The application should not be treated as an autonomous fault-diagnosis or safety system.

---

## Author

**Pratyush Singh Yadav**

Machine Learning / Data Science

[GitHub](https://github.com/Pratyush0110)
