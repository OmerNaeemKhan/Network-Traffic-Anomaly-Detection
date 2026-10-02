import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import html

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Network Security Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "Outputs"

PREDICTION_RESULTS_PATH = OUTPUTS_DIR / "prediction_results.csv"
ANOMALY_RESULTS_PATH = OUTPUTS_DIR / "anomaly_detection_results.csv"

# ============================================================
# THEME
# ============================================================

THEMES = {
    "Dark": {
        "bg": "#070d18",
        "surface": "#0f1a2b",
        "surface_2": "#152338",
        "sidebar": "#0a1322",
        "border": "#22374f",
        "text": "#eaf1fa",
        "muted": "#8fa3ba",
        "accent": "#22d3ee",
        "accent_2": "#5b8def",
        "success": "#34d399",
        "danger": "#f87171",
        "warning": "#fbbf24",
        "grid": "#1c2f47",
        "shadow": "0 4px 24px rgba(0, 0, 0, 0.45)",
        "shadow_hover": "0 8px 32px rgba(34, 211, 238, 0.16)",
        "card_gradient": "linear-gradient(160deg, #0f1a2b 0%, #131f33 100%)",
        "hero_glow": "radial-gradient(1100px 380px at 8% -40%, rgba(34,211,238,0.13), transparent 62%)",
    },
    "Light": {
        "bg": "#f2f6fb",
        "surface": "#ffffff",
        "surface_2": "#f7fafd",
        "sidebar": "#e9f1f9",
        "border": "#d5e0ec",
        "text": "#111a2b",
        "muted": "#5b6b81",
        "accent": "#0e9bb8",
        "accent_2": "#2563eb",
        "success": "#15a34a",
        "danger": "#dc2626",
        "warning": "#d97706",
        "grid": "#dde6f0",
        "shadow": "0 2px 14px rgba(19, 42, 74, 0.07)",
        "shadow_hover": "0 8px 26px rgba(14, 155, 184, 0.16)",
        "card_gradient": "linear-gradient(160deg, #ffffff 0%, #f7fafd 100%)",
        "hero_glow": "radial-gradient(1100px 380px at 8% -40%, rgba(14,155,184,0.10), transparent 62%)",
    },
}

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"

# ------------------------------------------------------------
# PLOTLY VERSION COMPATIBILITY
# Rounded bars need plotly >= 5.19 and font weight needs >= 5.23.
# These are purely cosmetic, so they degrade silently on older versions.
# ------------------------------------------------------------

def _plotly_supports(builder):
    try:
        builder()
        return True
    except Exception:
        return False


_ROUNDED_BARS = _plotly_supports(
    lambda: go.Bar(x=["a"], y=[1], marker=dict(cornerradius=8))
)
_FONT_WEIGHT = _plotly_supports(
    lambda: go.Figure().update_layout(title=dict(font=dict(weight=700)))
)

BAR_MARKER = (
    dict(line=dict(width=0), cornerradius=8)
    if _ROUNDED_BARS
    else dict(line=dict(width=0))
)
HIST_MARKER = (
    dict(line=dict(width=0), cornerradius=3, opacity=0.9)
    if _ROUNDED_BARS
    else dict(line=dict(width=0), opacity=0.9)
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(file_path: str):
    path = Path(file_path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        return None


prediction_df = load_csv(str(PREDICTION_RESULTS_PATH))
anomaly_df = load_csv(str(ANOMALY_RESULTS_PATH))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_anomaly_column(df):
    if df is None or df.empty:
        return None

    possible_columns = [
        "anomaly",
        "is_anomaly",
        "anomaly_label",
        "anomaly_prediction",
        "prediction",
        "predictions",
        "label",
        "if_anomaly",
        "classification",
        "class",
        "result",
        "detected_anomaly",
    ]

    lower_columns = {str(col).lower().strip(): col for col in df.columns}

    for possible in possible_columns:
        if possible in lower_columns:
            return lower_columns[possible]

    # Secondary matching for columns with additional words.
    for lower_name, original_name in lower_columns.items():
        if "anomal" in lower_name:
            return original_name

    return None


def build_anomaly_mask(series):
    """
    Detect anomalies from common result formats.

    Handles:
    - Isolation Forest: -1 = anomaly, 1 = normal
    - Binary labels: 1 = anomaly, 0 = normal
    - Text labels: anomaly, attack, malicious, etc.
    """
    values = series.copy()
    normalized = values.astype(str).str.lower().str.strip()

    unique_values = set(normalized.dropna().unique())

    # Isolation Forest convention.
    if unique_values and unique_values.issubset({"-1", "1"}):
        return normalized.eq("-1")

    # Common binary classification convention.
    if unique_values and unique_values.issubset({"0", "1"}):
        return normalized.eq("1")

    anomaly_terms = {
        "-1",
        "anomaly",
        "anomalous",
        "true",
        "yes",
        "attack",
        "malicious",
        "intrusion",
        "abnormal",
        "suspicious",
    }

    return normalized.isin(anomaly_terms)


def calculate_statistics(df):
    if df is None or df.empty:
        return {
            "total_records": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_rate": 0.0,
            "anomaly_column": None,
            "anomaly_mask": pd.Series(dtype=bool),
        }

    total_records = len(df)
    anomaly_column = find_anomaly_column(df)

    if anomaly_column is None:
        anomaly_count = 0
        normal_count = total_records
        anomaly_mask = pd.Series(False, index=df.index)
    else:
        anomaly_mask = build_anomaly_mask(df[anomaly_column])
        anomaly_count = int(anomaly_mask.sum())
        normal_count = total_records - anomaly_count

    anomaly_rate = (
        anomaly_count / total_records * 100
        if total_records > 0
        else 0.0
    )

    return {
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_rate": anomaly_rate,
        "anomaly_column": anomaly_column,
        "anomaly_mask": anomaly_mask,
    }


def get_primary_data():
    if prediction_df is not None and not prediction_df.empty:
        return prediction_df, "prediction_results.csv"

    if anomaly_df is not None and not anomaly_df.empty:
        return anomaly_df, "anomaly_detection_results.csv"

    return None, "No data file found"


def safe_numeric_columns(df):
    if df is None:
        return []

    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def escape(value):
    return html.escape(str(value))


main_df, data_source = get_primary_data()
stats = calculate_statistics(main_df)

total_records = stats["total_records"]
anomaly_count = stats["anomaly_count"]
normal_count = stats["normal_count"]
anomaly_rate = stats["anomaly_rate"]
anomaly_column = stats["anomaly_column"]
anomaly_mask = stats["anomaly_mask"]

feature_count = 43


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">CYBERSECURITY AI</div>
                <div class="brand-subtitle">Network Traffic Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>🎨 DASHBOARD THEME</div>", unsafe_allow_html=True)

    theme_choice = st.selectbox(
        "Theme",
        options=["Dark", "Light"],
        index=0 if st.session_state.theme_choice == "Dark" else 1,
        label_visibility="collapsed",
    )
    st.session_state.theme_choice = theme_choice

    st.markdown("<div class='sidebar-section-title'>NAVIGATION</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Anomaly Analysis",
            "Prediction Results",
            "Model Performance",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)

    status_text = "SYSTEM ONLINE" if main_df is not None else "DATA FILE NOT FOUND"
    status_class = "status-online" if main_df is not None else "status-offline"

    st.markdown(
        f"""
        <div class="system-card">
            <div class="{status_class}"><span class="pulse-dot"></span>{status_text}</div>
            <div class="system-detail">
                {total_records:,} records available
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-card-title">PROJECT INFORMATION</div>
            <div class="project-row"><span>Project</span><b>Network Traffic Anomaly Detection</b></div>
            <div class="project-row"><span>Algorithm</span><b>Isolation Forest</b></div>
            <div class="project-row"><span>Dataset</span><b>UNSW-NB15</b></div>
            <div class="project-row"><span>Features Used</span><b>{feature_count}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLY THEME CSS
# ============================================================

T = THEMES[st.session_state.theme_choice]

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        .stApp {{
            background: {T["bg"]};
            color: {T["text"]};
        }}

        [data-testid="stAppViewContainer"] {{
            background:
                {T["hero_glow"]},
                {T["bg"]};
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: {T["sidebar"]};
            border-right: 1px solid {T["border"]};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: {T["sidebar"]};
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.8rem;
        }}

        .block-container {{
            max-width: 1560px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }}

        /* ---------- scrollbar ---------- */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: {T["border"]};
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: {T["muted"]}; }}

        /* ---------- sidebar brand ---------- */
        .brand-block {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.35rem 0.1rem 1.35rem 0.1rem;
        }}

        .brand-icon {{
            font-size: 1.85rem;
            line-height: 1;
            filter: drop-shadow(0 0 10px {T["accent"]}55);
        }}

        .brand-name {{
            font-size: 1.06rem;
            font-weight: 900;
            letter-spacing: 0.055em;
            color: {T["text"]};
            line-height: 1.15;
        }}

        .brand-subtitle {{
            color: {T["muted"]};
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 0.2rem;
            letter-spacing: 0.01em;
        }}

        .sidebar-rule {{
            height: 1px;
            background: linear-gradient(
                90deg, {T["border"]} 0%, {T["border"]}00 100%
            );
            margin: 1rem 0 1.3rem 0;
        }}

        .sidebar-label,
        .sidebar-section-title {{
            color: {T["muted"]};
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }}

        .sidebar-section-title {{
            margin-top: 1.4rem;
        }}

        .system-card,
        .project-card {{
            background: {T["card_gradient"]};
            border: 1px solid {T["border"]};
            border-radius: 14px;
            padding: 1rem 1.05rem;
            color: {T["muted"]};
            font-size: 0.87rem;
            box-shadow: {T["shadow"]};
        }}

        .project-card-title {{
            color: {T["text"]};
            font-weight: 800;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            margin-bottom: 0.9rem;
            text-transform: uppercase;
        }}

        .project-row {{
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            padding: 0.5rem 0;
            border-top: 1px solid {T["border"]}66;
        }}

        .project-row:first-of-type {{ border-top: none; padding-top: 0; }}

        .project-row span {{
            color: {T["muted"]};
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .project-row b {{
            color: {T["text"]};
            font-size: 0.88rem;
            font-weight: 600;
            line-height: 1.35;
        }}

        .status-online,
        .status-offline {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 800;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
        }}

        .status-online {{ color: {T["success"]}; }}
        .status-offline {{ color: {T["danger"]}; }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 0 0 currentColor;
            animation: pulse 2.2s infinite;
            flex-shrink: 0;
        }}

        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }}
            70%  {{ box-shadow: 0 0 0 9px rgba(52, 211, 153, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
        }}

        .system-detail {{
            color: {T["muted"]};
            margin-top: 0.55rem;
            font-size: 0.82rem;
        }}

        /* ---------- hero: forced to a single line ---------- */
        .hero-title {{
            color: {T["text"]};
            font-size: clamp(1.85rem, 3.15vw, 3.05rem);
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin: 0.15rem 0 0.7rem 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .hero-icon {{
            font-size: 0.92em;
            line-height: 1;
            flex-shrink: 0;
            filter: drop-shadow(0 0 14px {T["accent"]}4d);
        }}

        .hero-text {{
            overflow: hidden;
            text-overflow: ellipsis;
            background: linear-gradient(
                92deg, {T["text"]} 30%, {T["accent"]} 130%
            );
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .hero-subtitle {{
            color: {T["muted"]};
            font-size: 1.02rem;
            font-weight: 400;
            line-height: 1.65;
            max-width: 900px;
            margin-bottom: 1.4rem;
        }}

        .section-title {{
            color: {T["text"]};
            font-size: 1.32rem;
            font-weight: 800;
            margin: 2rem 0 1rem 0;
            letter-spacing: -0.015em;
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }}

        .section-title::before {{
            content: "";
            width: 3px;
            height: 1.05em;
            border-radius: 3px;
            background: linear-gradient(
                180deg, {T["accent"]}, {T["accent_2"]}
            );
            flex-shrink: 0;
        }}

        .status-banner {{
            background: {T["card_gradient"]};
            border: 1px solid {T["border"]};
            border-left: 3px solid {T["accent"]};
            border-radius: 14px;
            padding: 1.05rem 1.3rem;
            margin: 1.1rem 0 1.6rem 0;
            box-shadow: {T["shadow"]};
        }}

        .status-title {{
            color: {T["text"]};
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-description {{
            color: {T["muted"]};
            font-size: 0.94rem;
            margin-top: 0.4rem;
        }}

        .status-description b {{ color: {T["text"]}; font-weight: 600; }}

        /* ---------- metric cards ---------- */
        .metric-card {{
            position: relative;
            background: {T["card_gradient"]};
            border: 1px solid {T["border"]};
            border-radius: 16px;
            padding: 1.15rem 1.3rem;
            min-height: 128px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 0.8rem;
            overflow: hidden;
            box-shadow: {T["shadow"]};
            transition: transform 0.18s ease, box-shadow 0.18s ease,
                        border-color 0.18s ease;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(
                90deg, {T["accent"]}, {T["accent_2"]}
            );
            opacity: 0.85;
        }}

        .metric-card:hover {{
            transform: translateY(-3px);
            border-color: {T["accent"]}66;
            box-shadow: {T["shadow_hover"]};
        }}

        .metric-label {{
            color: {T["muted"]};
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }}

        .metric-value {{
            color: {T["text"]};
            font-size: clamp(1.6rem, 2.3vw, 2.3rem);
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.035em;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .panel-card,
        .info-card {{
            background: {T["card_gradient"]};
            border: 1px solid {T["border"]};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            height: 100%;
            box-shadow: {T["shadow"]};
            transition: transform 0.18s ease, border-color 0.18s ease;
        }}

        .info-card {{
            min-height: 150px;
            margin-bottom: 1rem;
        }}

        .info-card:hover {{
            transform: translateY(-2px);
            border-color: {T["accent"]}55;
        }}

        .info-title {{
            color: {T["text"]};
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            letter-spacing: -0.01em;
        }}

        .info-text {{
            color: {T["muted"]};
            font-size: 0.9rem;
            line-height: 1.65;
        }}

        .info-text b {{ color: {T["text"]}; font-weight: 600; }}

        .kpi-row {{
            margin-top: 0.25rem;
        }}

        .footer {{
            color: {T["muted"]};
            text-align: center;
            padding: 2rem 0 0.5rem 0;
            font-size: 0.82rem;
            border-top: 1px solid {T["border"]};
            margin-top: 2.5rem;
        }}

        .stSelectbox label,
        .stRadio label,
        .stMarkdown,
        p,
        li {{
            color: {T["text"]};
        }}

        /* ---------- native widgets ---------- */
        div[data-testid="stMetric"] {{
            background: {T["card_gradient"]};
            border: 1px solid {T["border"]};
            border-radius: 16px;
            padding: 1rem;
            box-shadow: {T["shadow"]};
        }}

        div[data-testid="stMetricLabel"] {{
            color: {T["muted"]};
        }}

        div[data-testid="stMetricValue"] {{
            color: {T["text"]};
        }}

        .stDataFrame {{
            border: 1px solid {T["border"]};
            border-radius: 12px;
            overflow: hidden;
            box-shadow: {T["shadow"]};
        }}

        .stDownloadButton > button,
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid {T["border"]};
            background: {T["surface_2"]};
            color: {T["text"]};
            font-weight: 600;
            padding: 0.5rem 1.15rem;
            transition: all 0.18s ease;
        }}

        .stDownloadButton > button:hover,
        .stButton > button:hover {{
            border-color: {T["accent"]};
            color: {T["accent"]};
            box-shadow: {T["shadow_hover"]};
            transform: translateY(-1px);
        }}

        [data-testid="stSidebar"] .stRadio > label {{
            padding: 0.18rem 0;
        }}

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
            border-radius: 8px;
            padding: 0.34rem 0.5rem;
            margin-bottom: 0.1rem;
            transition: background 0.15s ease;
        }}

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {{
            background: {T["surface_2"]};
        }}

        div[data-baseweb="select"] > div {{
            background: {T["surface"]};
            border-color: {T["border"]};
            border-radius: 10px;
        }}

        .stTextInput input {{
            background: {T["surface"]};
            border: 1px solid {T["border"]};
            border-radius: 10px;
            color: {T["text"]};
        }}

        .stTextInput input:focus {{
            border-color: {T["accent"]};
            box-shadow: 0 0 0 2px {T["accent"]}25;
        }}

        .stSlider [data-baseweb="slider"] {{ padding-top: 0.4rem; }}

        .js-plotly-plot {{
            border: 1px solid {T["border"]};
            border-radius: 16px;
            overflow: hidden;
            box-shadow: {T["shadow"]};
        }}

        /* ---------- responsive ---------- */
        @media (max-width: 1150px) {{
            .hero-title {{
                white-space: normal;
                font-size: clamp(1.6rem, 4.4vw, 2.3rem);
            }}
        }}

        @media (max-width: 900px) {{
            .block-container {{
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .hero-title {{
                font-size: 1.85rem;
            }}

            .metric-card {{ min-height: 108px; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def hero(title, subtitle=None, icon="🛡️"):
    st.markdown(
        f'<div class="hero-title">'
        f'<span class="hero-icon">{icon}</span>'
        f'<span class="hero-text">{escape(title)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="hero-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def section_title(title):
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def no_data_message():
    st.error(
        "No results file was found. Expected one of these files:\n\n"
        f"- `{PREDICTION_RESULTS_PATH}`\n"
        f"- `{ANOMALY_RESULTS_PATH}`"
    )


def apply_plotly_theme(fig, height=430):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=T["text"],
            family='Inter, "Segoe UI", Arial, sans-serif',
            size=12,
        ),
        title=dict(
            font=(
                dict(size=15, color=T["text"], weight=700)
                if _FONT_WEIGHT
                else dict(size=15, color=T["text"])
            ),
            x=0.02,
            xanchor="left",
            y=0.95,
        ),
        margin=dict(l=30, r=30, t=68, b=40),
        legend=dict(
            font=dict(color=T["muted"], size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(
            bgcolor=T["surface_2"],
            bordercolor=T["border"],
            font=dict(color=T["text"], size=12),
        ),
        xaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["grid"],
            color=T["muted"],
            linecolor=T["border"],
            tickfont=dict(size=11),
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["grid"],
            color=T["muted"],
            linecolor=T["border"],
            tickfont=dict(size=11),
            gridwidth=1,
        ),
    )
    return fig


def classification_dataframe():
    return pd.DataFrame(
        {
            "Traffic Type": ["Normal Traffic", "Anomalous Traffic"],
            "Records": [normal_count, anomaly_count],
        }
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":
    hero(
        "Network Security Command Center",
        "AI-powered anomaly detection and network traffic intelligence using the "
        "Isolation Forest machine learning algorithm.",
        "🛡️",
    )

    if main_df is None:
        no_data_message()

    else:
        st.markdown(
            f"""
            <div class="status-banner">
                <div class="status-title"><span class="pulse-dot"></span>SYSTEM ONLINE</div>
                <div class="status-description">
                    Monitoring {total_records:,} network traffic records from
                    <b>{escape(data_source)}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Records", f"{total_records:,}")
        with c2:
            metric_card("Features Analyzed", f"{feature_count}")
        with c3:
            metric_card("Anomalies Detected", f"{anomaly_count:,}")
        with c4:
            metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%")

        section_title("Traffic Intelligence Overview")

        left_col, right_col = st.columns(2)

        classification_df = classification_dataframe()

        with left_col:
            fig = px.bar(
                classification_df,
                x="Traffic Type",
                y="Records",
                text="Records",
                color="Traffic Type",
                color_discrete_map={
                    "Normal Traffic": T["success"],
                    "Anomalous Traffic": T["danger"],
                },
            )
            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(size=12, color=T["text"]),
                cliponaxis=False,
                marker=BAR_MARKER,
                width=0.5,
                hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
            )
            fig.update_layout(
                showlegend=False,
                title="Network Traffic Classification",
                xaxis_title=None,
                yaxis_title="Records",
                bargap=0.45,
            )
            apply_plotly_theme(fig, height=430)
            st.plotly_chart(fig, width="stretch")

        with right_col:
            fig = px.pie(
                classification_df,
                names="Traffic Type",
                values="Records",
                hole=0.68,
                color="Traffic Type",
                color_discrete_map={
                    "Normal Traffic": T["success"],
                    "Anomalous Traffic": T["danger"],
                },
            )
            fig.update_traces(
                textinfo="percent",
                textfont=dict(size=13, color="#ffffff"),
                marker=dict(line=dict(color=T["bg"], width=3)),
                hovertemplate="%{label}<br>%{value:,} records<br>%{percent}<extra></extra>",
            )
            fig.update_layout(title="Security Risk Summary")
            fig.add_annotation(
                text=f"<b>{anomaly_rate:.1f}%</b><br><span style='font-size:11px'>anomalous</span>",
                x=0.5, y=0.5,
                font=dict(size=24, color=T["text"]),
                showarrow=False,
            )
            apply_plotly_theme(fig, height=430)
            st.plotly_chart(fig, width="stretch")

        section_title("Detection Summary")

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-title">Data Source</div>
                    <div class="info-text">
                        The dashboard is currently analyzing
                        <b>{total_records:,}</b> records from
                        <b>{escape(data_source)}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            column_message = (
                f"The anomaly classification column is <b>{escape(anomaly_column)}</b>."
                if anomaly_column is not None
                else "No anomaly classification column was automatically identified."
            )
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-title">Detection Logic</div>
                    <div class="info-text">
                        {column_message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s3:
            risk_level = (
                "High" if anomaly_rate >= 40
                else "Elevated" if anomaly_rate >= 20
                else "Moderate" if anomaly_rate >= 5
                else "Low"
            )
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-title">Current Risk Level</div>
                    <div class="info-text">
                        Based on the detected anomaly rate of
                        <b>{anomaly_rate:.2f}%</b>, the current traffic risk level
                        is classified as <b>{risk_level}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ANOMALY ANALYSIS PAGE
# ============================================================

elif page == "Anomaly Analysis":
    hero(
        "Anomaly Analysis",
        "Explore detected anomalies, compare normal and suspicious traffic, and inspect the records identified by the model.",
        "🔍",
    )

    if main_df is None:
        no_data_message()

    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Records", f"{total_records:,}")
        with c2:
            metric_card("Normal Traffic", f"{normal_count:,}")
        with c3:
            metric_card("Anomalies", f"{anomaly_count:,}")
        with c4:
            metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%")

        section_title("Detected Network Traffic")

        classification_df = classification_dataframe()

        fig = px.bar(
            classification_df,
            x="Traffic Type",
            y="Records",
            text="Records",
            color="Traffic Type",
            color_discrete_map={
                "Normal Traffic": T["success"],
                "Anomalous Traffic": T["danger"],
            },
        )
        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(size=13, color=T["text"]),
            cliponaxis=False,
            marker=BAR_MARKER,
            width=0.42,
            hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
        )
        fig.update_layout(
            title="Normal vs Anomalous Network Traffic",
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Number of Records",
            bargap=0.5,
        )
        apply_plotly_theme(fig, height=470)
        st.plotly_chart(fig, width="stretch")

        section_title("Anomaly Detection Details")

        if anomaly_column is None:
            st.warning(
                "An anomaly classification column could not be automatically identified."
            )
            st.info("Available columns in the selected results file:")
            st.dataframe(
                pd.DataFrame({"Column Name": list(main_df.columns)}),
                width="stretch",
                hide_index=True,
            )
        else:
            detected_anomalies = main_df[anomaly_mask].copy()

            st.markdown(
                f"""
                <div class="status-banner">
                    <div class="status-title"><span class="pulse-dot"></span>{anomaly_count:,} ANOMALIES DETECTED</div>
                    <div class="status-description">
                        Classification column: <b>{escape(anomaly_column)}</b>.
                        Showing the most relevant records below.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if detected_anomalies.empty:
                st.info(
                    "No anomaly rows were identified using the detected classification values."
                )
            else:
                max_rows = min(1000, len(detected_anomalies))
                rows_to_show = st.slider(
                    "Number of anomaly records to display",
                    min_value=min(10, max_rows),
                    max_value=max_rows,
                    value=min(100, max_rows),
                    step=min(10, max_rows) if max_rows >= 10 else 1,
                )

                st.dataframe(
                    detected_anomalies.head(rows_to_show),
                    width="stretch",
                    height=520,
                )

                csv_data = detected_anomalies.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Detected Anomalies CSV",
                    data=csv_data,
                    file_name="detected_anomalies.csv",
                    mime="text/csv",
                )

                numeric_cols = safe_numeric_columns(detected_anomalies)

                if numeric_cols:
                    section_title("Anomaly Feature Exploration")

                    selected_feature = st.selectbox(
                        "Select a numeric feature",
                        numeric_cols,
                        key="anomaly_feature",
                    )

                    feature_df = detected_anomalies[[selected_feature]].dropna()

                    if not feature_df.empty:
                        fig = px.histogram(
                            feature_df,
                            x=selected_feature,
                            nbins=40,
                            title=f"Distribution of {selected_feature} for Detected Anomalies",
                            color_discrete_sequence=[T["accent"]],
                        )
                        fig.update_traces(
                            marker=HIST_MARKER,
                            hovertemplate="%{x}<br>%{y:,} records<extra></extra>",
                        )
                        fig.update_layout(
                            showlegend=False,
                            yaxis_title="Count",
                            bargap=0.06,
                        )
                        apply_plotly_theme(fig, height=430)
                        st.plotly_chart(fig, width="stretch")


# ============================================================
# PREDICTION RESULTS PAGE
# ============================================================

elif page == "Prediction Results":
    hero(
        "Network Traffic Prediction Results",
        "Browse the generated prediction dataset and inspect the data used by the dashboard.",
        "📊",
    )

    if prediction_df is None or prediction_df.empty:
        st.warning(
            "The prediction results file was not found or could not be loaded."
        )
        st.code(str(PREDICTION_RESULTS_PATH))
    else:
        p_total = len(prediction_df)
        p_columns = len(prediction_df.columns)
        p_anomaly_column = find_anomaly_column(prediction_df)

        p1, p2, p3 = st.columns(3)
        with p1:
            metric_card("Total Predictions", f"{p_total:,}")
        with p2:
            metric_card("Available Columns", f"{p_columns}")
        with p3:
            metric_card(
                "Classification Column",
                str(p_anomaly_column) if p_anomaly_column is not None else "Not Found",
            )

        section_title("Prediction Dataset")

        search_term = st.text_input(
            "Search within prediction data",
            placeholder="Type text to filter rows...",
        )

        display_df = prediction_df

        if search_term:
            search_mask = prediction_df.astype(str).apply(
                lambda col: col.str.contains(search_term, case=False, na=False)
            )
            display_df = prediction_df[search_mask.any(axis=1)]

        st.caption(
            f"Displaying {len(display_df):,} of {len(prediction_df):,} records."
        )

        st.dataframe(
            display_df,
            width="stretch",
            height=560,
        )

        csv_data = prediction_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Prediction Results",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv",
        )

        section_title("Dataset Columns")

        columns_df = pd.DataFrame(
            {
                "Column Name": prediction_df.columns,
                "Data Type": [str(dtype) for dtype in prediction_df.dtypes],
                "Missing Values": prediction_df.isna().sum().values,
                "Unique Values": prediction_df.nunique(dropna=True).values,
            }
        )

        st.dataframe(
            columns_df,
            width="stretch",
            hide_index=True,
            height=500,
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":
    hero(
        "Model Performance",
        "Performance overview of the Isolation Forest anomaly detection model and the generated classification results.",
        "📈",
    )

    if main_df is None:
        no_data_message()

    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Records Processed", f"{total_records:,}")
        with c2:
            metric_card("Features", f"{feature_count}")
        with c3:
            metric_card("Detected Anomalies", f"{anomaly_count:,}")
        with c4:
            metric_card("Detection Rate", f"{anomaly_rate:.2f}%")

        section_title("Model Detection Distribution")

        performance_data = pd.DataFrame(
            {
                "Classification": ["Normal", "Anomaly"],
                "Records": [normal_count, anomaly_count],
            }
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=performance_data["Classification"],
                    y=performance_data["Records"],
                    text=performance_data["Records"],
                    texttemplate="%{text:,}",
                    textposition="outside",
                    textfont=dict(size=13, color=T["text"]),
                    marker_color=[T["success"], T["danger"]],
                    marker=BAR_MARKER,
                    width=0.42,
                    hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="Isolation Forest Detection Results",
            xaxis_title=None,
            yaxis_title="Number of Records",
            showlegend=False,
            bargap=0.5,
        )
        apply_plotly_theme(fig, height=500)
        st.plotly_chart(fig, width="stretch")

        section_title("Performance Summary")

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Algorithm",
                    "Records Processed",
                    "Features Used",
                    "Normal Traffic",
                    "Detected Anomalies",
                    "Detection Rate",
                    "Data Source",
                ],
                "Value": [
                    "Isolation Forest",
                    f"{total_records:,}",
                    feature_count,
                    f"{normal_count:,}",
                    f"{anomaly_count:,}",
                    f"{anomaly_rate:.2f}%",
                    data_source,
                ],
            }
        )

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "This page summarizes the generated detection results. Traditional "
            "supervised-model metrics such as accuracy, precision, recall, and "
            "F1 score are not calculated here because the current dashboard "
            "code only has the prediction result files available."
        )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":
    hero(
        "About This Project",
        "A machine learning and cybersecurity analytics project focused on identifying unusual network behavior.",
        "ℹ️",
    )

    section_title("Network Traffic Anomaly Detection")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Project Overview</div>
            <div class="info-text">
                This project is an AI-powered cybersecurity solution designed to
                identify unusual and potentially malicious activity within network
                traffic. The dashboard transforms generated model results into
                clear operational insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Project Objective")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Identify Unusual Network Behavior</div>
            <div class="info-text">
                The main objective is to analyze network traffic records and use
                machine learning to identify patterns that differ significantly
                from expected or normal behavior.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Machine Learning Algorithm")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Isolation Forest</div>
            <div class="info-text">
                Isolation Forest is used for anomaly detection. It works by
                isolating observations through randomly selected features and
                decision boundaries. Records that are easier to isolate can be
                treated as more unusual and are potential anomalies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Dataset")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">UNSW-NB15 Network Intrusion Dataset</div>
            <div class="info-text">
                The project uses network traffic features from the UNSW-NB15
                dataset to support anomaly detection and cybersecurity analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Project Workflow")

    w1, w2, w3, w4 = st.columns(4)

    workflow_items = [
        (
            "1. Data Inspection",
            "Load, inspect, and understand the network traffic records.",
        ),
        (
            "2. Data Processing",
            "Clean and prepare relevant features for machine learning.",
        ),
        (
            "3. AI Detection",
            "Use Isolation Forest to identify unusual network traffic patterns.",
        ),
        (
            "4. Security Insights",
            "Present detection results through an interactive dashboard.",
        ),
    ]

    for col, (title, text) in zip([w1, w2, w3, w4], workflow_items):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-title">{escape(title)}</div>
                    <div class="info-text">{escape(text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Technologies Used")

    tech_df = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "Scikit-learn",
                "Isolation Forest",
                "Streamlit",
                "Plotly",
            ],
            "Purpose": [
                "Core application and data processing",
                "Dataset manipulation and analysis",
                "Machine learning utilities",
                "Anomaly detection algorithm",
                "Interactive web dashboard",
                "Interactive data visualization",
            ],
        }
    )

    st.dataframe(
        tech_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ Network Traffic Anomaly Detection &nbsp;|&nbsp;
        AI-Powered Cybersecurity Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)