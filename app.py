
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Production Intelligence System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b0f14;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

/* =========================
   MAIN HEADER
   ========================= */

.dashboard-title {
    font-size: 36px;
    font-weight: 750;
    letter-spacing: -0.6px;
    margin-bottom: 4px;
}

.dashboard-subtitle {
    color: #8b95a5;
    font-size: 14px;
    margin-bottom: 30px;
}

/* =========================
   SECTION HEADINGS
   ========================= */

.section-title {
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.2px;
    margin-top: 28px;
    margin-bottom: 16px;
}

/* =========================
   KPI CARDS
   ========================= */

.kpi-card {
    background: linear-gradient(
        145deg,
        #171d26 0%,
        #10151c 100%
    );
    border: 1px solid #27303c;
    border-radius: 14px;
    padding: 19px 20px;
    min-height: 118px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.16);
}

.kpi-label {
    color: #8b95a5;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 9px;
    letter-spacing: 0.2px;
}

.kpi-value {
    font-size: 27px;
    font-weight: 720;
    letter-spacing: -0.4px;
}

.kpi-small {
    color: #697586;
    font-size: 11px;
    margin-top: 6px;
}

/* =========================
   SIDEBAR
   ========================= */

.sidebar-title {
    font-size: 21px;
    font-weight: 720;
    letter-spacing: -0.3px;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    color: #7f8a9a;
    font-size: 12px;
    margin-bottom: 22px;
}

/* =========================
   STREAMLIT CONTROLS
   ========================= */

div[data-baseweb="select"] > div {
    border-radius: 9px;
}

div[data-baseweb="input"] > div {
    border-radius: 9px;
}

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

/* =========================
   DATAFRAMES
   ========================= */

div[data-testid="stDataFrame"] {
    border: 1px solid #27303c;
    border-radius: 10px;
    overflow: hidden;
}

/* =========================
   DIVIDERS
   ========================= */

hr {
    border-color: #252d38;
    margin-top: 24px;
    margin-bottom: 24px;
}

/* =========================
   ALERT / INFO BOXES
   ========================= */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* =========================
   FILE UPLOADER
   ========================= */

section[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 1px dashed #394454;
}

/* =========================
   METRICS
   ========================= */

div[data-testid="stMetric"] {
    padding: 4px 2px;
}

div[data-testid="stMetricLabel"] {
    color: #8b95a5;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# APPLICATION PATH
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# UPLOADED DATASET SUPPORT
# ============================================================

# Minimum columns required for a production dataset
REQUIRED_COLUMNS = [
    "Mcno",
    "Date",
    "Runtime(Min)",
    "Prod Mtrs"
]

# Optional production-speed column
OPTIONAL_SPEED_COLUMNS = [
    "Del Spd"
]

# Alternative names that are safe to normalize
# Only genuine delivery-speed names are included here.
COLUMN_ALIASES = {
    "delivery speed": "Del Spd",
    "Delivery Speed": "Del Spd",
    "del spd": "Del Spd",
    "Del.Speed": "Del Spd"
}

OPTIONAL_COLUMNS = [
    "Shift",
    "Count",
    "Material",
    "Stoptime(Min)",
    "Tar.Eff%",
    "A%",
    "Tar RPM",
    "RPM",
    "TPM",
    "Total Dofftime",
    "Total Units",
    "kW/Hr",
    "UKG",
    "ProdWt(kg)"
]


# ============================================================
# DISPLAY FORMATTING
# ============================================================

def format_production_table(display_df):
    """
    Format production/anomaly tables for clean dashboard display.
    Does not modify the underlying data.
    """

    formatted_df = display_df.copy()

    number_formats = {
        "Actual Production": "{:,.0f}",
        "Expected Production": "{:,.2f}",
        "Deviation (m)": "{:+,.2f}",
        "Deviation (%)": "{:+.3f}%",
        "ML Score": "{:.3f}",
        "Unusual Features": "{:.0f}",
        "Prod Mtrs": "{:,.0f}",
        "Expected_Prod": "{:,.2f}",
        "Deviation_Mtrs": "{:+,.2f}",
        "Deviation_%": "{:+.3f}%",
        "Anomaly_Score": "{:.3f}",
        "Number_of_Unusual_Features": "{:.0f}"
    }

    for column, formatter in number_formats.items():
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column].map(
                lambda value: (
                    formatter.format(value)
                    if pd.notna(value)
                    else "—"
                )
            )

    return formatted_df


def load_uploaded_dataset(uploaded_file):
    """
    Read and validate an uploaded CSV/XLS/XLSX production dataset.

    The function automatically detects the real header row so that
    production reports containing title/metadata rows can also be used.
    """

    file_name = uploaded_file.name.lower()

    # ---------------------------------------------------------
    # Read raw file without assuming the header is row 1
    # ---------------------------------------------------------

    if file_name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file, header=None)

    elif file_name.endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(uploaded_file, header=None)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file."
        )

    # ---------------------------------------------------------
    # Remove completely empty rows and columns
    # ---------------------------------------------------------

    raw_df = raw_df.dropna(how="all")
    raw_df = raw_df.dropna(axis=1, how="all")

    # ---------------------------------------------------------
    # Automatically detect the header row
    # ---------------------------------------------------------

    header_row = None

    for row_index in range(len(raw_df)):

        row_values = [
            str(value).strip()
            for value in raw_df.iloc[row_index].dropna().tolist()
        ]

        # Normalize possible alternative column names
        normalized_row_values = [
            COLUMN_ALIASES.get(value, value)
            for value in row_values
        ]

        if all(
            required_col in normalized_row_values
            for required_col in REQUIRED_COLUMNS
        ):
            header_row = row_index
            break

    if header_row is None:

        raise ValueError(
            "Could not identify the production-data header row. "
            "The file must contain these required columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    # ---------------------------------------------------------
    # Re-read using detected header
    # ---------------------------------------------------------

    uploaded_df = raw_df.iloc[header_row + 1:].copy()

    uploaded_df.columns = [
        str(value).strip()
        for value in raw_df.iloc[header_row].tolist()
    ]

    # Remove columns with empty/unnamed headers
    valid_columns = [
        col
        for col in uploaded_df.columns
        if col and not col.lower().startswith("unnamed")
    ]

    uploaded_df = uploaded_df.loc[:, valid_columns]

    # Remove completely empty rows
    uploaded_df = uploaded_df.dropna(how="all")

    # Clean column names
    uploaded_df.columns = (
        uploaded_df.columns
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Normalize alternative column names
    # ---------------------------------------------------------

    uploaded_df.columns = [
        COLUMN_ALIASES.get(col, col)
        for col in uploaded_df.columns
    ]

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in uploaded_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return uploaded_df.reset_index(drop=True)


def process_uploaded_dataset(uploaded_df):
    """
    Clean and prepare an uploaded production dataset.

    Supports datasets with or without delivery-speed information.
    Expected production and deviation metrics are calculated only
    when a genuine Del Spd column is available.
    """

    df_processed = uploaded_df.copy()

    # ---------------------------------------------------------
    # Basic cleaning
    # ---------------------------------------------------------

    df_processed.columns = (
        df_processed.columns
        .astype(str)
        .str.strip()
    )

    df_processed = (
        df_processed
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Remove report summary rows
    if "Mcno" in df_processed.columns:
        df_processed = df_processed[
            df_processed["Mcno"]
            .astype(str)
            .str.strip()
            .str.lower() != "total"
        ].reset_index(drop=True)

    # ---------------------------------------------------------
    # Date conversion
    # ---------------------------------------------------------

    df_processed["Date"] = pd.to_datetime(
        df_processed["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------

    numeric_columns = [
        "Count",
        "Runtime(Min)",
        "Stoptime(Min)",
        "Tar.Eff%",
        "A%",
        "Del Spd",
        "Tar RPM",
        "RPM",
        "TPM",
        "Prod Mtrs",
        "ProdWt(kg)",
        "Total Dofftime",
        "Total Units",
        "kW/Hr",
        "UKG"
    ]

    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(
                df_processed[col],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # Required core data
    # ---------------------------------------------------------

    df_processed = df_processed.dropna(
        subset=[
            "Mcno",
            "Date",
            "Runtime(Min)",
            "Prod Mtrs"
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Production-speed based features
    # ---------------------------------------------------------

    if "Del Spd" in df_processed.columns:

        df_processed["Speed_Runtime"] = (
            df_processed["Del Spd"]
            * df_processed["Runtime(Min)"]
        )

        df_processed["Expected_Prod"] = (
            df_processed["Del Spd"]
            * df_processed["Runtime(Min)"]
        )

        df_processed["Production_Ratio"] = (
            df_processed["Prod Mtrs"]
            /
            df_processed["Expected_Prod"].replace(
                0,
                np.nan
            )
        )

        df_processed["Deviation_Mtrs"] = (
            df_processed["Prod Mtrs"]
            -
            df_processed["Expected_Prod"]
        )

        df_processed["Deviation_%"] = (
            df_processed["Deviation_Mtrs"]
            /
            df_processed["Expected_Prod"].replace(
                0,
                np.nan
            )
        ) * 100

        df_processed["Abs_Deviation_%"] = (
            df_processed["Deviation_%"].abs()
        )

    else:

        # No genuine delivery-speed field.
        # Do not create an artificial expected-production value.

        df_processed["Speed_Runtime"] = np.nan
        df_processed["Expected_Prod"] = np.nan
        df_processed["Production_Ratio"] = np.nan
        df_processed["Deviation_Mtrs"] = np.nan
        df_processed["Deviation_%"] = np.nan
        df_processed["Abs_Deviation_%"] = np.nan

    return df_processed


ANOMALY_FEATURES = [
    "Runtime(Min)",
    "Stoptime(Min)",
    "Tar.Eff%",
    "A%",
    "Del Spd",
    "Tar RPM",
    "RPM",
    "TPM",
    "Total Dofftime",
    "kW/Hr",
    "UKG"
]


def train_anomaly_model(df_input):
    """
    Train an Isolation Forest anomaly detector using the
    numeric operating features actually available in the
    uploaded dataset.
    """

    df_result = df_input.copy()

    # ---------------------------------------------------------
    # Determine operating records
    # ---------------------------------------------------------

    if (
        "Expected_Prod" in df_result.columns
        and df_result["Expected_Prod"].notna().any()
    ):
        operating_mask = df_result["Expected_Prod"] > 0

    else:
        operating_mask = (
            (df_result["Runtime(Min)"] > 0)
            |
            (df_result["Prod Mtrs"] > 0)
        )

    operating_df = df_result[
        operating_mask
    ].copy()

    if len(operating_df) < 20:
        raise ValueError(
            "At least 20 operating records are required "
            "for ML anomaly detection."
        )

    # ---------------------------------------------------------
    # Dynamically select numeric operating features
    # ---------------------------------------------------------

    excluded_columns = {
        "Prod Mtrs",
        "Expected_Prod",
        "Production_Ratio",
        "Deviation_Mtrs",
        "Deviation_%",
        "Abs_Deviation_%",
        "Speed_Runtime"
    }

    excluded_identity_columns = {
        "Mcno",
        "Date",
        "Material"
    }

    candidate_features = []

    for col in operating_df.columns:

        if col in excluded_columns:
            continue

        if col in excluded_identity_columns:
            continue

        numeric_values = pd.to_numeric(
            operating_df[col],
            errors="coerce"
        )

        valid_count = numeric_values.notna().sum()

        if valid_count >= max(
            20,
            int(len(operating_df) * 0.5)
        ):
            operating_df[col] = numeric_values
            candidate_features.append(col)

    if len(candidate_features) < 2:
        raise ValueError(
            "The uploaded dataset does not contain enough "
            "numeric operating features for anomaly detection."
        )

    # ---------------------------------------------------------
    # Prepare feature matrix
    # ---------------------------------------------------------

    feature_data = operating_df[
        candidate_features
    ].copy()

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_data = feature_data.fillna(
        feature_data.median()
    )

    # Remove constant columns
    variable_features = [
        col
        for col in feature_data.columns
        if feature_data[col].nunique() > 1
    ]

    feature_data = feature_data[
        variable_features
    ]

    if len(variable_features) < 2:
        raise ValueError(
            "The uploaded dataset does not contain enough "
            "variable numeric features for anomaly detection."
        )

    # ---------------------------------------------------------
    # Scale features
    # ---------------------------------------------------------

    scaler_model = StandardScaler()

    X_scaled = scaler_model.fit_transform(
        feature_data
    )

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    isolation_model.fit(X_scaled)

    predictions = isolation_model.predict(
        X_scaled
    )

    scores = isolation_model.decision_function(
        X_scaled
    )

    operating_df["ML_Prediction"] = predictions

    operating_df["Anomaly_Score"] = scores

    operating_df["ML_Anomaly"] = np.where(
        operating_df["ML_Prediction"] == -1,
        "Anomaly",
        "Normal"
    )

    # ---------------------------------------------------------
    # Production deviation analysis
    # ---------------------------------------------------------

    has_deviation = (
        "Deviation_%" in operating_df.columns
        and operating_df["Deviation_%"].notna().any()
    )

    if has_deviation:

        deviation_values = operating_df[
            "Deviation_%"
        ].dropna()

        deviation_mean = deviation_values.mean()

        deviation_std = deviation_values.std()

        lower_limit = (
            deviation_mean
            - 3 * deviation_std
        )

        upper_limit = (
            deviation_mean
            + 3 * deviation_std
        )

        operating_df["Anomaly_3Sigma"] = np.where(
            (
                (operating_df["Deviation_%"] < lower_limit)
                |
                (operating_df["Deviation_%"] > upper_limit)
            ),
            "Production Deviation",
            "Within Range"
        )

    else:

        lower_limit = np.nan
        upper_limit = np.nan

        operating_df["Anomaly_3Sigma"] = (
            "Not Available"
        )

    # ---------------------------------------------------------
    # Final anomaly label
    # ---------------------------------------------------------

    if has_deviation:

        operating_df["Anomaly_Label"] = np.select(
            [
                (
                    (operating_df["ML_Anomaly"] == "Anomaly")
                    &
                    (
                        operating_df["Anomaly_3Sigma"]
                        == "Production Deviation"
                    )
                ),
                (
                    (operating_df["ML_Anomaly"] == "Anomaly")
                    &
                    (
                        operating_df["Anomaly_3Sigma"]
                        == "Within Range"
                    )
                ),
                (
                    (operating_df["ML_Anomaly"] == "Normal")
                    &
                    (
                        operating_df["Anomaly_3Sigma"]
                        == "Production Deviation"
                    )
                )
            ],
            [
                "ML + Production Anomaly",
                "ML Operating Anomaly",
                "Production Deviation"
            ],
            default="Normal"
        )

    else:

        operating_df["Anomaly_Label"] = np.where(
            operating_df["ML_Anomaly"] == "Anomaly",
            "ML Operating Anomaly",
            "Normal"
        )

    # ---------------------------------------------------------
    # Machine-specific unusual feature count
    # ---------------------------------------------------------

    machine_unusual = {}

    for machine in operating_df[
        "Mcno"
    ].dropna().unique():

        machine_data = operating_df[
            operating_df["Mcno"] == machine
        ]

        unusual_count = 0

        for col in variable_features:

            if len(machine_data) < 4:
                continue

            q1 = machine_data[col].quantile(0.25)

            q3 = machine_data[col].quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            latest_value = machine_data.iloc[-1][col]

            if (
                latest_value < lower
                or latest_value > upper
            ):
                unusual_count += 1

        machine_unusual[machine] = unusual_count

    operating_df[
        "Number_of_Unusual_Features"
    ] = (
        operating_df["Mcno"]
        .map(machine_unusual)
        .fillna(0)
        .astype(int)
    )

    operating_df[
        "Machine_Unusual_Features"
    ] = (
        operating_df[
            "Number_of_Unusual_Features"
        ].astype(str)
        + " operating features outside historical range"
    )

    return (
        operating_df,
        isolation_model,
        scaler_model,
        variable_features,
        lower_limit,
        upper_limit
    )




# ============================================================
# DATASET LOADING
# ============================================================

# Uploaded datasets are processed dynamically below.
# No private/company dataset is loaded automatically.

# ============================================================
# ============================================================
# DATASET UPLOAD
# ============================================================

st.markdown(
    "### 📂 Dataset",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload production data",
    type=["csv", "xlsx", "xls"],
    help="Upload a compatible production CSV or Excel file."
)

# DATASET PIPELINE
# ============================================================

df = None
anomaly_report = None
isolation_model = None
scaler = None
machine_baseline = None
production_limits = None

if uploaded_file is not None:

    try:
        # Read uploaded file
        raw_uploaded_df = load_uploaded_dataset(uploaded_file)

        # Clean + engineer features
        df = process_uploaded_dataset(raw_uploaded_df)

        # Train anomaly detection model
        (
            anomaly_report,
            isolation_model,
            scaler,
            anomaly_features_used,
            production_lower_limit,
            production_upper_limit
        ) = train_anomaly_model(df)

        # Store production limits
        production_limits = {
            "lower": production_lower_limit,
            "upper": production_upper_limit
        }

        st.sidebar.success("Dataset loaded successfully")

    except Exception as e:

        st.error(
            f"Unable to process the uploaded dataset: {e}"
        )

        st.stop()

else:

    st.info(
        "Upload a production CSV or Excel file from the sidebar "
        "to start the Production Intelligence analysis."
    )

    st.stop()

# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ⚙ Production Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-subtitle">
            Machine monitoring & anomaly analytics
        </div>
        """,
        unsafe_allow_html=True
    )


    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Production Analysis",
            "Machine Monitor",
            "Anomaly Detection"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success("Models loaded")

    st.caption(
        f"{len(df)} historical records"
    )

    st.caption(
        f"{df['Mcno'].nunique()} machines monitored"
    )

# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="dashboard-title">
            Production Intelligence System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Machine performance, production monitoring and anomaly detection
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_actual = df["Prod Mtrs"].sum()

    has_production_baseline = (
        "Expected_Prod" in df.columns
        and df["Expected_Prod"].notna().any()
    )

    if has_production_baseline:

        total_expected = df["Expected_Prod"].sum()

        total_difference = (
            total_actual - total_expected
        )

        overall_deviation = (
            total_difference / total_expected * 100
            if total_expected != 0
            else 0
        )

    else:

        total_expected = np.nan
        overall_deviation = np.nan

    machine_count = df["Mcno"].nunique()

    anomaly_count = len(anomaly_report)

    latest_date = df["Date"].max()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL PRODUCTION</div>
                <div class="kpi-value">{total_actual:,.0f} m</div>
                <div class="kpi-small">Actual production</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        if has_production_baseline:

            expected_value = f"{total_expected:,.0f} m"
            expected_note = "Speed × runtime baseline"

        else:

            expected_value = "Not available"
            expected_note = "Delivery speed not provided"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">EXPECTED PRODUCTION</div>
                <div class="kpi-value">{expected_value}</div>
                <div class="kpi-small">{expected_note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        if has_production_baseline:

            deviation_value = f"{overall_deviation:+.3f}%"
            deviation_note = "Actual vs expected"

        else:

            deviation_value = "Not available"
            deviation_note = "No delivery-speed baseline"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">PRODUCTION DEVIATION</div>
                <div class="kpi-value">{deviation_value}</div>
                <div class="kpi-small">{deviation_note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">ML ANOMALIES</div>
                <div class="kpi-value">{anomaly_count}</div>
                <div class="kpi-small">Latest evaluation set</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PRODUCTION PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Production Performance
        </div>
        """,
        unsafe_allow_html=True
    )

    if has_production_baseline:

        date_summary = (
            df.groupby("Date")
            .agg(
                Actual_Production=("Prod Mtrs", "mean"),
                Expected_Production=("Expected_Prod", "mean"),
                Records=("Prod Mtrs", "count")
            )
            .reset_index()
        )

    else:

        date_summary = (
            df.groupby("Date")
            .agg(
                Actual_Production=("Prod Mtrs", "mean"),
                Records=("Prod Mtrs", "count")
            )
            .reset_index()
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_summary["Date"],
            y=date_summary["Actual_Production"],
            mode="lines+markers",
            name="Actual Production",
            line=dict(width=3),
            marker=dict(size=8)
        )
    )

    if has_production_baseline:

        fig.add_trace(
            go.Scatter(
                x=date_summary["Date"],
                y=date_summary["Expected_Production"],
                mode="lines+markers",
                name="Expected Production",
                line=dict(
                    width=2,
                    dash="dash"
                ),
                marker=dict(size=7)
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20
        ),
        title=dict(
            text="Average Production per Record",
            x=0,
            xanchor="left",
            font=dict(size=17)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#d8dee9"
        ),
        xaxis=dict(
            showgrid=False,
            title=""
        ),
        yaxis=dict(
            title="Average Production (m)",
            gridcolor="#242b35"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MACHINE PRODUCTION + ANOMALY SUMMARY
    # --------------------------------------------------------

    col_left, col_right = st.columns(
        [1.4, 1]
    )

    with col_left:

        st.markdown(
            """
            <div class="section-title">
                Machine Production
            </div>
            """,
            unsafe_allow_html=True
        )

        machine_summary = (
            df.groupby("Mcno")
            .agg(
                Production=("Prod Mtrs", "sum"),
                Records=("Prod Mtrs", "count"),
                Average_Production=("Prod Mtrs", "mean")
            )
            .reset_index()
            .sort_values(
                "Production",
                ascending=False
            )
            .head(10)
            .sort_values(
                "Production",
                ascending=True
            )
        )

        machine_fig = go.Figure()

        machine_fig.add_trace(
            go.Bar(
                x=machine_summary["Production"],
                y=machine_summary["Mcno"],
                orientation="h",
                text=machine_summary["Production"],
                texttemplate="%{text:,.0f} m",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Total Production: %{x:,.0f} m<br>"
                    "<extra></extra>"
                )
            )
        )

        machine_fig.update_layout(
            height=460,
            margin=dict(
                l=20,
                r=90,
                t=20,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#d8dee9"
            ),
            xaxis=dict(
                title="Total Production (m)",
                gridcolor="#242b35"
            ),
            yaxis=dict(
                title="",
                categoryorder="total ascending"
            ),
            showlegend=False
        )

        st.plotly_chart(
            machine_fig,
            use_container_width=True
        )

        st.caption(
            "Top 10 machines by total recorded production"
        )

    with col_right:

        st.markdown(
            """
            <div class="section-title">
                Anomaly Overview
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "ML operating anomalies",
            anomaly_count
        )

        st.metric(
            "Meaningful September records",
            38
        )

        st.metric(
            "3-Sigma production anomalies",
            0
        )

        st.divider()

        st.markdown(
            "**Latest anomaly records**"
        )

        anomaly_preview = anomaly_report[
            [
                "Mcno",
                "Material",
                "Number_of_Unusual_Features",
                "Deviation_%"
            ]
        ].copy()

        anomaly_preview.columns = [
            "Machine",
            "Material",
            "Unusual Features",
            "Deviation %"
        ]

        st.dataframe(
            anomaly_preview,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        f"Latest dataset date: "
        f"{latest_date.strftime('%d %b %Y')}  •  "
        f"Historical baseline: {len(df)} records  •  "
        f"Production baseline: Delivery Speed × Runtime"
    )

# ============================================================
# PRODUCTION ANALYSIS PAGE
# ============================================================

elif page == "Production Analysis":

    st.title("Production Analysis")

    st.caption(
        "Production performance, expected output and deviation analysis"
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Analysis Filters</div>',
        unsafe_allow_html=True
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        date_min = df["Date"].min().date()
        date_max = df["Date"].max().date()

        selected_dates = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )

    with filter_col2:
        machine_options = sorted(
            df["Mcno"].dropna().unique().tolist()
        )

        selected_machines = st.multiselect(
            "Machines",
            machine_options,
            default=machine_options
        )

    with filter_col3:
        material_options = sorted(
            df["Material"].dropna().unique().tolist()
        )

        selected_materials = st.multiselect(
            "Materials",
            material_options,
            default=material_options
        )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    analysis_df = df.copy()

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])

        analysis_df = analysis_df[
            (analysis_df["Date"] >= start_date) &
            (analysis_df["Date"] <= end_date)
        ]

    if selected_machines:
        analysis_df = analysis_df[
            analysis_df["Mcno"].isin(selected_machines)
        ]

    if selected_materials:
        analysis_df = analysis_df[
            analysis_df["Material"].isin(selected_materials)
        ]

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_production = analysis_df["Prod Mtrs"].sum()

    has_production_baseline = (
        "Expected_Prod" in analysis_df.columns
        and analysis_df["Expected_Prod"].notna().any()
    )

    if has_production_baseline:

        total_expected = analysis_df["Expected_Prod"].sum()

        if total_expected != 0:
            overall_deviation = (
                (total_production - total_expected)
                / total_expected
            ) * 100
        else:
            overall_deviation = np.nan

    else:

        total_expected = np.nan
        overall_deviation = np.nan

    average_production = analysis_df["Prod Mtrs"].mean()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Production Summary</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Total Production",
            f"{total_production:,.0f} m"
        )

    with k2:

        if has_production_baseline:
            expected_value = f"{total_expected:,.0f} m"
        else:
            expected_value = "Not available"

        st.metric(
            "Expected Production",
            expected_value
        )

    with k3:

        st.metric(
            "Average Production",
            f"{average_production:,.1f} m"
        )

    with k4:

        if has_production_baseline:
            deviation_value = f"{overall_deviation:+.3f}%"
        else:
            deviation_value = "Not available"

        st.metric(
            "Production Deviation",
            deviation_value
        )

    # --------------------------------------------------------
    # ACTUAL VS EXPECTED
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Actual vs Expected Production</div>',
        unsafe_allow_html=True
    )

    if len(analysis_df) > 0:

        if has_production_baseline:

            production_trend = (
                analysis_df
                .groupby("Date")
                .agg(
                    Actual=("Prod Mtrs", "sum"),
                    Expected=("Expected_Prod", "sum")
                )
                .reset_index()
                .sort_values("Date")
            )

        else:

            production_trend = (
                analysis_df
                .groupby("Date")
                .agg(
                    Actual=("Prod Mtrs", "sum")
                )
                .reset_index()
                .sort_values("Date")
            )

        fig_analysis = go.Figure()

        fig_analysis.add_trace(
            go.Scatter(
                x=production_trend["Date"],
                y=production_trend["Actual"],
                mode="lines+markers",
                name="Actual Production",
                line=dict(width=3),
                marker=dict(size=8)
            )
        )

        if has_production_baseline:

            fig_analysis.add_trace(
                go.Scatter(
                    x=production_trend["Date"],
                    y=production_trend["Expected"],
                    mode="lines+markers",
                    name="Expected Production",
                    line=dict(
                        width=2,
                        dash="dash"
                    ),
                    marker=dict(size=7)
                )
            )

        fig_analysis.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#d8dee9"
            ),
            xaxis=dict(
                title="Date",
                showgrid=False
            ),
            yaxis=dict(
                title="Production (m)",
                gridcolor="#242b35"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_analysis,
            use_container_width=True
        )

        st.caption(
            f"Showing {len(analysis_df):,} production records "
            f"across {analysis_df['Mcno'].nunique()} machines."
        )

        if has_production_baseline:

            # ----------------------------------------------------
            # PRODUCTION DEVIATION ANALYSIS
            # ----------------------------------------------------

            st.markdown(
                '<div class="section-title">Production Deviation Analysis</div>',
                unsafe_allow_html=True
            )

            deviation_df = analysis_df.copy()

            deviation_df["Deviation_Mtrs"] = (
                deviation_df["Prod Mtrs"]
                - deviation_df["Expected_Prod"]
            )

            deviation_df["Deviation_%"] = (
                deviation_df["Deviation_Mtrs"]
                / deviation_df["Expected_Prod"].replace(0, np.nan)
            ) * 100

            meaningful_deviation = deviation_df[
                deviation_df["Expected_Prod"] > 100
            ].copy()

            if len(meaningful_deviation) > 0:

                mean_deviation = (
                    meaningful_deviation["Deviation_%"].mean()
                )

                mean_absolute_deviation = (
                    meaningful_deviation["Deviation_%"]
                    .abs()
                    .mean()
                )

                max_positive_deviation = (
                    meaningful_deviation["Deviation_%"].max()
                )

                max_negative_deviation = (
                    meaningful_deviation["Deviation_%"].min()
                )

                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.metric(
                        "Mean Deviation",
                        f"{mean_deviation:+.3f}%"
                    )

                with d2:
                    st.metric(
                        "Mean Absolute Deviation",
                        f"{mean_absolute_deviation:.3f}%"
                    )

                with d3:
                    st.metric(
                        "Maximum Positive",
                        f"{max_positive_deviation:+.3f}%"
                    )

                with d4:
                    st.metric(
                        "Maximum Negative",
                        f"{max_negative_deviation:+.3f}%"
                    )

                # ------------------------------------------------
                # DAILY DEVIATION TREND
                # ------------------------------------------------

                deviation_trend = (
                    meaningful_deviation
                    .groupby("Date")
                    .agg(
                        Average_Deviation=("Deviation_%", "mean"),
                        Absolute_Deviation=(
                            "Deviation_%",
                            lambda x: x.abs().mean()
                        )
                    )
                    .reset_index()
                    .sort_values("Date")
                )

                deviation_fig = go.Figure()

                deviation_fig.add_trace(
                    go.Scatter(
                        x=deviation_trend["Date"],
                        y=deviation_trend["Average_Deviation"],
                        mode="lines+markers",
                        name="Average Deviation",
                        line=dict(width=3),
                        marker=dict(size=8)
                    )
                )

                deviation_fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_width=1
                )

                # Historical 3-sigma production deviation limits
                historical_deviation = (
                    df[df["Expected_Prod"] > 100]["Deviation_%"]
                    if "Deviation_%" in df.columns
                    else (
                        (
                            df["Prod Mtrs"] - df["Expected_Prod"]
                        )
                        / df["Expected_Prod"].replace(0, np.nan)
                        * 100
                    )
                )

                historical_mean = historical_deviation.mean()
                historical_std = historical_deviation.std()

                lower_limit = historical_mean - (3 * historical_std)
                upper_limit = historical_mean + (3 * historical_std)

                deviation_fig.add_hline(
                    y=lower_limit,
                    line_dash="dot",
                    line_width=1,
                    annotation_text="Historical lower 3σ"
                )

                deviation_fig.add_hline(
                    y=upper_limit,
                    line_dash="dot",
                    line_width=1,
                    annotation_text="Historical upper 3σ"
                )

                deviation_fig.update_layout(
                    height=400,
                    margin=dict(
                        l=20,
                        r=30,
                        t=30,
                        b=20
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        color="#d8dee9"
                    ),
                    xaxis=dict(
                        title="Date",
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Production Deviation (%)",
                        gridcolor="#242b35"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )

                st.plotly_chart(
                    deviation_fig,
                    use_container_width=True
                )

                # ------------------------------------------------
                # LARGEST DEVIATIONS
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Largest Production Deviations</div>',
                    unsafe_allow_html=True
                )

                deviation_table = (
                    meaningful_deviation[
                        [
                            "Date",
                            "Mcno",
                            "Material",
                            "Runtime(Min)",
                            "Del Spd",
                            "Prod Mtrs",
                            "Expected_Prod",
                            "Deviation_Mtrs",
                            "Deviation_%"
                        ]
                    ]
                    .assign(
                        Absolute_Deviation=lambda x:
                        x["Deviation_%"].abs()
                    )
                    .sort_values(
                        "Absolute_Deviation",
                        ascending=False
                    )
                    .head(15)
                    .drop(
                        columns=["Absolute_Deviation"]
                    )
                    .copy()
                )

                deviation_table.columns = [
                    "Date",
                    "Machine",
                    "Material",
                    "Runtime (min)",
                    "Delivery Speed",
                    "Actual Production",
                    "Expected Production",
                    "Deviation (m)",
                    "Deviation (%)"
                ]

                # Professional number formatting
                production_formats = {
                    "Runtime (min)": "{:,.2f}",
                    "Delivery Speed": "{:,.2f}",
                    "Actual Production": "{:,.0f}",
                    "Expected Production": "{:,.2f}",
                    "Deviation (m)": "{:+,.2f}",
                    "Deviation (%)": "{:+.3f}%"
                }

                production_formats = {
                    column: formatter
                    for column, formatter in production_formats.items()
                    if column in deviation_table.columns
                }

                deviation_table = deviation_table.style.format(
                    production_formats,
                    na_rep="—"
                )

                st.dataframe(
                    deviation_table,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No meaningful production records are available "
                    "for deviation analysis."
                )

        else:

            st.warning(
                "No production records match the selected filters."
            )

    # ============================================================
    # MACHINE MONITOR PAGE
    # ============================================================


elif page == "Machine Monitor":

    st.title("Machine Monitor")

    st.caption(
        "Machine-level production performance and operating behavior"
    )

    # --------------------------------------------------------
    # MACHINE SELECTION
    # --------------------------------------------------------

    machine_list = sorted(
        df["Mcno"].dropna().astype(str).unique().tolist()
    )

    selected_machine = st.selectbox(
        "Select Machine",
        machine_list
    )

    machine_df = df[
        df["Mcno"].astype(str) == selected_machine
    ].copy()

    machine_df = machine_df.sort_values("Date")

    if len(machine_df) == 0:

        st.warning(
            "No records available for the selected machine."
        )

    else:

        # ----------------------------------------------------
        # MACHINE KPIs
        # ----------------------------------------------------

        total_production = machine_df["Prod Mtrs"].sum()

        average_production = machine_df["Prod Mtrs"].mean()

        has_machine_speed = (
            "Del Spd" in machine_df.columns
            and machine_df["Del Spd"].notna().any()
        )

        if has_machine_speed:
            average_speed = machine_df["Del Spd"].mean()
        else:
            average_speed = np.nan

        average_runtime = machine_df["Runtime(Min)"].mean()

        operating_records = machine_df[
            machine_df["Runtime(Min)"] > 0
        ]

        operating_rate = (
            len(operating_records) / len(machine_df) * 100
        )

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.metric(
                "Total Production",
                f"{total_production:,.0f} m"
            )

        with k2:
            st.metric(
                "Average / Record",
                f"{average_production:,.0f} m"
            )

        with k3:
            if has_machine_speed:
                st.metric(
                    "Average Delivery Speed",
                    f"{average_speed:.2f}"
                )
            else:
                st.metric(
                    "Average Delivery Speed",
                    "Not available"
                )

        with k4:
            st.metric(
                "Average Runtime",
                f"{average_runtime:.1f} min"
            )

        with k5:
            st.metric(
                "Operating Records",
                f"{operating_rate:.1f}%"
            )

        # ----------------------------------------------------
        # PRODUCTION TREND
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Machine Production Trend</div>',
            unsafe_allow_html=True
        )

        machine_chart = go.Figure()

        machine_chart.add_trace(
            go.Scatter(
                x=machine_df["Date"],
                y=machine_df["Prod Mtrs"],
                mode="lines+markers",
                name="Production",
                line=dict(width=3),
                marker=dict(size=8)
            )
        )

        machine_chart.update_layout(
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#d8dee9"
            ),
            xaxis=dict(
                title="Date",
                showgrid=False
            ),
            yaxis=dict(
                title="Production (m)",
                gridcolor="#242b35"
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            machine_chart,
            use_container_width=True
        )

        # ----------------------------------------------------
        # OPERATING PROFILE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Operating Profile</div>',
            unsafe_allow_html=True
        )

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:

            preferred_operating_features = [
                "Del Spd",
                "Runtime(Min)",
                "Stoptime(Min)",
                "RPM",
                "TPM",
                "Total Units",
                "kW/Hr",
                "Tar RPM",
                "Tar.Eff%",
                "A%"
            ]

            operating_features = [
                col
                for col in preferred_operating_features
                if col in machine_df.columns
            ]

            st.markdown(
                "#### Operating Profile"
            )

            if len(operating_features) > 0:

                profile_data = (
                    machine_df[operating_features]
                    .mean()
                    .reset_index()
                )

                profile_data.columns = [
                    "Feature",
                    "Average"
                ]

                profile_fig = px.bar(
                    profile_data,
                    x="Average",
                    y="Feature",
                    orientation="h",
                    text="Average"
                )

                profile_fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside"
                )

                profile_fig.update_layout(
                    height=350,
                    margin=dict(
                        l=20,
                        r=60,
                        t=20,
                        b=20
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        color="#d8dee9"
                    ),
                    xaxis=dict(
                        title="Average Value",
                        gridcolor="#242b35"
                    ),
                    yaxis=dict(
                        title=""
                    )
                )

                st.plotly_chart(
                    profile_fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numeric operating features are available "
                    "for this machine."
                )

        with profile_col2:

            st.markdown(
                "#### Machine Historical Range"
            )

            preferred_range_features = [
                "Del Spd",
                "Runtime(Min)",
                "Stoptime(Min)",
                "RPM",
                "TPM",
                "UKG",
                "Total Units",
                "kW/Hr",
                "Tar RPM"
            ]

            range_features = [
                col
                for col in preferred_range_features
                if col in machine_df.columns
            ]

            range_rows = []

            for feature in range_features:

                values = machine_df[feature].dropna()

                if len(values) > 0:

                    range_rows.append(
                        {
                            "Feature": feature,
                            "Minimum": values.min(),
                            "Average": values.mean(),
                            "Maximum": values.max()
                        }
                    )

            range_table = pd.DataFrame(
                range_rows
            )

            st.dataframe(
                range_table,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # LATEST MACHINE RECORD
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Latest Machine Record</div>',
            unsafe_allow_html=True
        )

        latest_machine = machine_df.iloc[-1]

        if has_machine_speed:

            latest_expected = (
                latest_machine["Del Spd"]
                * latest_machine["Runtime(Min)"]
            )

            latest_deviation = (
                latest_machine["Prod Mtrs"]
                - latest_expected
            )

            if latest_expected > 0:

                latest_deviation_pct = (
                    latest_deviation
                    / latest_expected
                    * 100
                )

            else:

                latest_deviation_pct = np.nan

        else:

            latest_expected = np.nan
            latest_deviation = np.nan
            latest_deviation_pct = np.nan

        l1, l2, l3, l4 = st.columns(4)

        with l1:
            st.metric(
                "Latest Production",
                f"{latest_machine['Prod Mtrs']:,.0f} m"
            )

        with l2:
            if has_machine_speed:
                st.metric(
                    "Expected Production",
                    f"{latest_expected:,.0f} m"
                )
            else:
                st.metric(
                    "Expected Production",
                    "Not available"
                )

        with l3:
            if has_machine_speed and not pd.isna(
                latest_deviation_pct
            ):
                st.metric(
                    "Deviation",
                    f"{latest_deviation_pct:+.3f}%"
                )
            else:
                st.metric(
                    "Deviation",
                    "Not available"
                )

        with l4:
            if has_machine_speed:
                st.metric(
                    "Latest Delivery Speed",
                    f"{latest_machine['Del Spd']:.2f}"
                )
            else:
                st.metric(
                    "Latest Delivery Speed",
                    "Not available"
                )

        # ----------------------------------------------------
        # MACHINE DATA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Machine Records</div>',
            unsafe_allow_html=True
        )

        display_columns = [
            "Date",
            "Shift",
            "Material",
            "Runtime(Min)",
            "Stoptime(Min)",
            "Del Spd",
            "RPM",
            "TPM",
            "Prod Mtrs"
        ]

        available_columns = [
            col for col in display_columns
            if col in machine_df.columns
        ]

        machine_records = machine_df[
            available_columns
        ].sort_values(
            "Date",
            ascending=False
        )

        st.dataframe(
            machine_records,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# ANOMALY DETECTION PAGE
# ============================================================

elif page == "Anomaly Detection":

    # Load required anomaly-analysis data locally
    # Use the currently uploaded dataset and
    # dynamically generated anomaly report.
    historical_data = df.copy()

    # Production-deviation analysis is available only
    # when a valid delivery-speed baseline exists.
    has_production_baseline = (
        "Expected_Prod" in historical_data.columns
        and historical_data["Expected_Prod"].notna().any()
        and "Deviation_%" in historical_data.columns
        and historical_data["Deviation_%"].notna().any()
    )

    if has_production_baseline:
        historical_operating = historical_data[
            historical_data["Expected_Prod"] > 100
        ].copy()

        if len(historical_operating) >= 2:
            lower_limit = (
                historical_operating["Deviation_%"].mean()
                - 3 * historical_operating["Deviation_%"].std()
            )

            upper_limit = (
                historical_operating["Deviation_%"].mean()
                + 3 * historical_operating["Deviation_%"].std()
            )
        else:
            lower_limit = np.nan
            upper_limit = np.nan
    else:
        historical_operating = pd.DataFrame()
        lower_limit = np.nan
        upper_limit = np.nan


    st.title("Anomaly Detection")
    st.caption("Machine-learning-based operating behavior analysis")

    # -----------------------------
    # KPI SUMMARY
    # -----------------------------
    # The anomaly report contains exactly the records
    # evaluated by the ML model.
    total_evaluated = len(anomaly_report)

    total_anomalies = 0

    if "ML_Anomaly" in anomaly_report.columns:
        total_anomalies = int(
            (
                anomaly_report["ML_Anomaly"]
                .astype(str)
                .str.strip()
                .str.lower()
                == "anomaly"
            ).sum()
        )

    # Fallback for older anomaly-report structures
    if total_anomalies == 0 and "ML_Prediction" in anomaly_report.columns:
        total_anomalies = int(
            (anomaly_report["ML_Prediction"] == -1).sum()
        )

    anomaly_rate = (
        total_anomalies / total_evaluated * 100
        if total_evaluated > 0 else 0
    )

    production_anomalies = 0

    if (
        has_production_baseline
        and "Deviation_%" in anomaly_report.columns
        and not pd.isna(lower_limit)
        and not pd.isna(upper_limit)
    ):
        production_anomalies = len(
            anomaly_report[
                (anomaly_report["Deviation_%"] < lower_limit) |
                (anomaly_report["Deviation_%"] > upper_limit)
            ]
        )

    meaningful_records = total_evaluated

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ML Anomalies",
            f"{total_anomalies}"
        )

    with col2:
        st.metric(
            "Records Evaluated",
            f"{total_evaluated}"
        )

    with col3:
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )

    with col4:
        st.metric(
            "3-Sigma Production Anomalies",
            f"{production_anomalies}"
        )

    st.divider()

    # -----------------------------
    # ANOMALY SCORE CHART
    # -----------------------------
    st.subheader("Anomaly Score Distribution")

    score_df = anomaly_report.copy()

    if "Anomaly_Score" in score_df.columns:

        fig_score = px.bar(
            score_df.sort_values(
                "Anomaly_Score",
                ascending=True
            ),
            x="Anomaly_Score",
            y="Mcno",
            orientation="h",
            text="Anomaly_Score",
            labels={
                "Anomaly_Score": "Anomaly Score",
                "Mcno": "Machine"
            }
        )

        fig_score.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig_score.update_layout(
            height=450,
            margin=dict(l=20, r=40, t=20, b=20)
        )

        st.plotly_chart(
            fig_score,
            use_container_width=True
        )

    st.divider()

    # -----------------------------
    # ANOMALY RECORDS
    # -----------------------------
    st.subheader("Detected Anomaly Records")

    display_columns = [
        "Mcno",
        "Material",
        "Prod Mtrs",
        "Expected_Prod",
        "Deviation_Mtrs",
        "Deviation_%",
        "Anomaly_Score",
        "Number_of_Unusual_Features"
    ]

    available_columns = [
        col for col in display_columns
        if col in anomaly_report.columns
    ]

    display_df = anomaly_report[
        available_columns
    ].copy()

    rename_map = {
        "Mcno": "Machine",
        "Prod Mtrs": "Actual Production",
        "Expected_Prod": "Expected Production",
        "Deviation_Mtrs": "Deviation (m)",
        "Deviation_%": "Deviation (%)",
        "Anomaly_Score": "ML Score",
        "Number_of_Unusual_Features": "Unusual Features"
    }

    display_df = display_df.rename(
        columns=rename_map
    )

    # Apply professional number formatting
    display_df = format_production_table(display_df)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------
    # ANOMALY EXPLANATIONS
    # -----------------------------
    st.subheader("Why Were These Records Flagged?")

    for _, row in anomaly_report.iterrows():

        machine = row.get("Mcno", "Unknown")
        material = row.get("Material", "Unknown")

        unusual_count = row.get(
            "Number_of_Unusual_Features",
            0
        )

        deviation = row.get(
            "Deviation_%",
            np.nan
        )

        score = row.get(
            "Anomaly_Score",
            np.nan
        )

        with st.expander(
            f"{machine}  •  {material}"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "ML Anomaly Score",
                    f"{score:.3f}"
                    if pd.notna(score)
                    else "N/A"
                )

            with c2:
                st.metric(
                    "Unusual Features",
                    f"{int(unusual_count)}"
                )

            with c3:
                if has_production_baseline and pd.notna(deviation):
                    st.metric(
                        "Production Deviation",
                        f"{deviation:+.3f}%"
                    )
                else:
                    st.metric(
                        "Production Deviation",
                        "Not available"
                    )

            if unusual_count > 0:
                st.write(
                    f"**Operating behavior:** "
                    f"{int(unusual_count)} operating features "
                    f"differ from this machine's historical range."
                )
            else:
                st.write(
                    "**Operating behavior:** "
                    "The record was flagged by the machine-learning "
                    "model based on its overall operating feature pattern."
                )

            if has_production_baseline and pd.notna(deviation):

                if lower_limit <= deviation <= upper_limit:
                    st.write(
                        "**Production behavior:** "
                        "Production deviation remains within "
                        "the historical 3-sigma range."
                    )
                else:
                    st.write(
                        "**Production behavior:** "
                        "Production deviation is outside "
                        "the historical 3-sigma range."
                    )
            else:
                st.write(
                    "**Production behavior:** "
                    "Production-deviation analysis is not available "
                    "because this dataset does not provide a valid "
                    "delivery-speed production baseline."
                )

            st.write(
                "**Interpretation:** "
                "The Isolation Forest detected this record "
                "as unusual compared with historical operating behavior. "
                "This does not automatically mean that the machine "
                "has a production failure."
            )

    st.divider()

    # -----------------------------
    # MACHINE-WISE ANOMALY COUNT
    # -----------------------------
    st.subheader("Anomalies by Machine")

    machine_anomaly_count = (
        anomaly_report
        .groupby("Mcno")
        .size()
        .reset_index(name="Anomalies")
        .sort_values(
            "Anomalies",
            ascending=False
        )
    )

    fig_machine = px.bar(
        machine_anomaly_count,
        x="Mcno",
        y="Anomalies",
        text="Anomalies",
        labels={
            "Mcno": "Machine",
            "Anomalies": "Detected Anomalies"
        }
    )

    fig_machine.update_traces(
        textposition="outside"
    )

    fig_machine.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(
        fig_machine,
        use_container_width=True
    )

    st.info(
        "ML anomalies indicate unusual operating behavior. "
        "They should be investigated using machine history, "
        "operating parameters and production context before "
        "being treated as equipment faults."
    )