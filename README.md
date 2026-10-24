# Production Intelligence System

A machine-learning-based production monitoring and anomaly detection dashboard built with Python, Streamlit, Pandas, Scikit-learn, and Plotly.

The system allows users to upload compatible production datasets and automatically analyzes machine-level operating behavior, production performance, and unusual operating patterns.

## Features

- Production performance analysis
- Machine-learning-based anomaly detection using Isolation Forest
- Machine-level monitoring
- Actual vs expected production analysis when delivery-speed data is available
- Detection of unusual operating behavior
- Interactive production and anomaly tables
- CSV and Excel dataset upload
- Interactive Streamlit dashboard
- Dynamic model training from the uploaded dataset

## How It Works

### 1. Dataset Upload

The user uploads a CSV or Excel production dataset through the dashboard.

### 2. Data Processing

The application automatically detects the dataset header, cleans column names, converts dates and numerical fields, removes duplicate records, removes total/summary rows, and handles missing values required for analysis.

### 3. Production Baseline

If the dataset contains a valid delivery-speed column (`Del Spd`), the application calculates an expected production baseline using:

**Expected Production = Delivery Speed × Runtime**

Production deviation can then be analyzed against the historical operating range.

If delivery-speed information is not available, production-deviation analysis is disabled while machine-learning-based operating analysis remains available.

### 4. Machine Learning

The application dynamically selects suitable numerical operating features from the uploaded dataset and trains an Isolation Forest model.

The model identifies records whose overall operating behavior differs from the learned historical pattern.

### 5. Machine Monitoring

Users can select individual machines and inspect production performance, runtime, operating characteristics, historical feature ranges, and recent machine records.

## Dataset Requirements

### Required Columns

| Column | Description |
|---|---|
| `Mcno` | Machine identifier |
| `Date` | Production date |
| `Runtime(Min)` | Machine runtime |
| `Prod Mtrs` | Production in meters |

### Optional Production Baseline Column

| Column | Description |
|---|---|
| `Del Spd` | Delivery speed |

If `Del Spd` is available, the dashboard can perform expected-production and production-deviation analysis.

### Other Numerical Features

The application can also use other numerical operating variables available in the dataset, such as:

- `Shift`
- `Count`
- `Stoptime(Min)`
- `RPM`
- `TPM`
- `Total Units`
- `kW/Hr`
- `Tar RPM`
- `Tar.Eff%`
- `A%`
- `UKG`

The exact features used by the anomaly model depend on the uploaded dataset.

## Important Note About Column Names

`delivery` and `Del Spd` are treated as different fields.

The application does not assume that a generic `delivery` column represents delivery speed. This prevents incorrect production-baseline calculations when datasets use different meanings for similarly named columns.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd production_intelligence_app
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
production_intelligence_app/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL

## Privacy

Do not publish confidential company production data, proprietary datasets, credentials, or internal business information to a public GitHub repository.

For demonstrations, use a synthetic or sanitized dataset.

## Disclaimer

The anomaly detection output identifies records with unusual operating patterns based on the uploaded dataset. An anomaly does not automatically mean that a machine has failed or that a production defect has occurred.

The system should be treated as a decision-support and monitoring tool rather than a replacement for engineering inspection or operational judgment.

## Author

Pratyush Singh Yadav
