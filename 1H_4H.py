# app.py
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Page & Sidebar Settings
# =========================
st.set_page_config(
    page_title="1H → Session-aware 4H OHLC Converter",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Settings")

keep_partial = st.sidebar.selectbox(
    "Include partial 4H bars (last bucket in a session with 1–3 hours)?",
    options=["Yes (recommended)", "No (drop partial)"],
    index=0
)
tolerance_min = st.sidebar.number_input(
    "Gap tolerance (minutes, 0 = strict 60-min steps)",
    min_value=0, max_value=30, value=0, step=1,
    help="Allow small timestamp jitter if your feed occasionally drifts."
)
show_chart = st.sidebar.checkbox("Show 4H candlestick chart", value=True)

st.title("1H → Session-aware 4H OHLC Converter")
st.caption("Build 4H candles within sessions only. Any non-1H gap ends the session; the final 4H bar per session can be partial.")

# =========================
# Helpers (no placeholders)
# =========================
REQ_OHLC = ["Open", "High", "Low", "Close"]
ORDERED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "Open Int"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a tz-naive 'timestamp' from either (Date+Time) or a single datetime column.
    Keep all original columns intact. No timezone conversions are performed.

    If parsing fails for any row, raise a ValueError that includes example rows (index, raw Date, raw Time)
    so the UI can show where the CSV has bad values.
    """
    df = df.copy()
    cols = set(df.columns)
    # Prefer explicit Date + Time from your sample
    if {"Date", "Time"}.issubset(cols):
        # Keep original raw values for diagnostics
        date_raw = df["Date"].astype(object)
        time_raw = df["Time"].astype(object)

        # Trim strings (safe even if values are not strings)
        try:
            date_str = date_raw.astype(str).str.strip()
            time_str = time_raw.astype(str).str.strip()
        except Exception:
            date_str = date_raw.apply(lambda x: "" if pd.isna(x) else str(x)).str.strip()
            time_str = time_raw.apply(lambda x: "" if pd.isna(x) else str(x)).str.strip()

        combo = date_str + " " + time_str
        ts = pd.to_datetime(combo, errors="coerce", infer_datetime_format=True)

        if ts.isna().any():
            failed_idx = ts[ts.isna()].index
            total_failed = len(failed_idx)
            examples = []
            for i in failed_idx[:50]:
                examples.append(f"{i}: Date={repr(df.at[i,'Date'])}, Time={repr(df.at[i,'Time'])}")
            example_text = "\n".join(examples) if examples else "(no examples)"
            msg = (
                "Failed to parse Date+Time into timestamps for some rows.\n"
                f"Total failed rows: {total_failed}.\n\n"
                "Example problematic rows (index: Date, Time):\n"
                f"{example_text}\n\n"
                "Common causes: empty/NULL Date or Time cells, mixed date formats (e.g. DD/MM vs MM/DD),\n"
                "unexpected separators, non-printable characters, or stray header/footer rows in the CSV.\n\n"
                "Fix the rows shown above (or re-export CSV with consistent Date+Time format) and re-upload."
            )
            raise ValueError(msg)

        df["timestamp"] = ts

    else:
        # Try single-column candidates
        candidates = [c for c in ["timestamp", "Timestamp", "Datetime", "DateTime", "date_time", "datetime", "Date"]
                      if c in cols]
        if not candidates:
            raise ValueError("Provide 'Date' + 'Time' OR a single datetime column (e.g., 'timestamp').")
        col = candidates[0]
        ts = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        if ts.isna().any():
            failed_idx = ts[ts.isna()].index
            total_failed = len(failed_idx)
            examples = []
            for i in failed_idx[:50]:
                examples.append(f"{i}: {col}={repr(df.at[i,col])}")
            example_text = "\n".join(examples) if examples else "(no examples)"
            msg = (
                f"Failed to parse datetimes from column '{col}'.\n"
                f"Total failed rows: {total_failed}.\n\n"
                "Example problematic rows (index: value):\n"
                f"{example_text}\n\n"
                "Common causes: empty values, mixed formats, or stray header/footer rows in the CSV.\n"
                "Please fix the examples above and re-upload."
            )
            raise ValueError(msg)
        df["timestamp"] = ts

    # Strict sort by timestamp; drop exact duplicate timestamps, keep first to avoid misalignment
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df

def require_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ_OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Expected: {REQ_OHLC}")
    df = df.copy()
    for c in REQ_OHLC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=REQ_OHLC)
    if len(df) == 0:
        raise ValueError("All rows dropped due to invalid OHLC values.")
    if len(df) < before:
        st.warning(f"Dropped {before - len(df)} rows with invalid OHLC.")
    return df

def assign_sessions(df: pd.DataFrame, tolerance_minutes: int = 0) -> pd.DataFrame:
    """
    A session is a run of strictly consecutive 1H bars.
    Break the session whenever delta != (3600 ± tolerance).
    """
    df = df.copy()
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
    deltas = df["timestamp"].diff().dt.total_seconds()
    tol = tolerance_minutes * 60
    session_break = deltas.isna() | (deltas < (3600 - tol)) | (deltas > (3600 + tol))
    df["session_id"] = session_break.cumsum().astype(int)
    df["row_in_session"] = df.groupby("session_id").cumcount()
    return df

def prepare_inc_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-bar incremental volume series 'inc_vol'.
    Prefer 'Inc Vol'. Otherwise derive from session-wise diffs of 'Volume'.
    If neither exists, set NaN.
    """
    df = df.copy()
    cols = set(df.columns)
    if "Inc Vol" in cols:
        inc = pd.to_numeric(df["Inc Vol"], errors="coerce")
        if not inc.isna().all():
            df["inc_vol"] = inc.clip(lower=0)
            return df
        st.warning("'Inc Vol' exists but is non-numeric; attempting to derive from 'Volume'.")
    if "Volume" in cols:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        if df["Volume"].isna().all():
            df["inc_vol"] = np.nan
            return df

        def _diff_session(g: pd.DataFrame) -> pd.Series:
            d = g["Volume"].diff()
            d.iloc[0] = g["Volume"].iloc[0]
            return d.clip(lower=0)

        df["inc_vol"] = df.groupby("session_id", group_keys=False).apply(_diff_session)
        return df

    df["inc_vol"] = np.nan
    return df

def detect_open_interest_column(df: pd.DataFrame):
    for cand in ["Open Int", "OpenInt", "OpenInterest", "Open_Interest", "Open interest"]:
        if cand in df.columns:
            return cand
    return None

def aggregate_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate within each session into 4H buckets (row_in_session // 4).
    Produce Date (start), OHLC, Volume (sum of inc_vol), and Open Int (last in bucket if present).
    """
    df = df.copy()
    df["h4_bucket"] = (df["row_in_session"] // 4).astype(int)
    oi_col = detect_open_interest_column(df)

    # Named aggregations to get clear column names
    agg_map = {
        "timestamp_first": ("timestamp", "first"),
        "timestamp_last": ("timestamp", "last"),
        "Open": ("Open", "first"),
        "High": ("High", "max"),
        "Low": ("Low", "min"),
        "Close": ("Close", "last"),
        "Volume_sum": ("inc_vol", "sum"),
        "bars_in_4h": ("timestamp", "size"),
    }
    if oi_col:
        agg_map["OpenInt_last"] = (oi_col, "last")

    grouped = df.groupby(["session_id", "h4_bucket"], as_index=False).agg(**agg_map)

    out = pd.DataFrame({
        "Date": grouped["timestamp_first"],
        "Open": grouped["Open"],
        "High": grouped["High"],
        "Low": grouped["Low"],
        "Close": grouped["Close"],
        "Volume": grouped["Volume_sum"],
        "bars_in_4h": grouped["bars_in_4h"],
    })

    out["is_complete"] = out["bars_in_4h"] == 4

    if oi_col:
        out["Open Int"] = grouped["OpenInt_last"]
    else:
        out["Open Int"] = np.nan

    # Keep only requested final columns in exact order requested by user
    out = out[["Date", "Open", "High", "Low", "Close", "Volume", "Open Int", "bars_in_4h", "is_complete"]]
    out = out.sort_values("Date").reset_index(drop=True)
    return out

def reorder_and_sort_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a preview of the raw 1H file strictly ordered by Date and with columns
    in the requested order. We never alter row values beyond sorting.
    """
    df = df.copy()
    # Ensure we have a Date column matching timestamps exactly
    df["Date"] = df["timestamp"]
    # Pick a Volume for preview: prefer 'Volume', else 'Inc Vol'
    if "Volume" in df.columns:
        vol = df["Volume"]
    elif "Inc Vol" in df.columns:
        vol = df["Inc Vol"]
    else:
        vol = pd.Series([np.nan] * len(df), index=df.index, name="Volume")

    # Detect open interest and create a consistent "Open Int" column (may be NaN)
    oi_col = detect_open_interest_column(df)
    if oi_col:
        oi_series = df[oi_col]
    else:
        oi_series = pd.Series([np.nan] * len(df), index=df.index, name="Open Int")

    preview = pd.DataFrame({
        "Date": df["Date"],
        "Open": df["Open"],
        "High": df["High"],
        "Low": df["Low"],
        "Close": df["Close"],
        "Volume": vol,
        "Open Int": oi_series
    })
    preview = preview.sort_values("Date").reset_index(drop=True)
    return preview

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# =========================
# File upload
# =========================
uploaded = st.file_uploader(
    "Upload 1H OHLC CSV (e.g., Date, Time, Open, High, Low, Close, Volume, Inc Vol, Open Int)",
    type=["csv"],
    accept_multiple_files=False
)

@st.cache_data(show_spinner=True, ttl=3600, max_entries=4)
def load_and_process(file_bytes: bytes, tolerance_minutes: int):
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    df = normalize_columns(df)
    df = parse_timestamp(df)
    df = require_ohlc(df)
    df = assign_sessions(df, tolerance_minutes=tolerance_minutes)
    df = prepare_inc_volume(df)

    # Raw preview (1H), sorted and re-ordered (no value swaps)
    raw_preview = reorder_and_sort_for_preview(df)

    # 4H aggregation within sessions
    h4 = aggregate_4h(df)

    return raw_preview, h4

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

file_bytes = uploaded.read()

try:
    with st.spinner("Parsing and aggregating..."):
        raw_preview, df_4h_full = load_and_process(file_bytes, tolerance_minutes=int(tolerance_min))
except Exception as e:
    st.error(f"Error while processing file: {e}")
    st.stop()

# Optionally drop partial 4H bars
if keep_partial.startswith("No"):
    df_4h = df_4h_full[df_4h_full["is_complete"]].reset_index(drop=True)
else:
    df_4h = df_4h_full.copy()

# =========================
# Raw & Processed Previews
# =========================
st.subheader("Raw File Preview (All Rows, Sorted & Re-Ordered)")
# Show raw preview with exactly requested column order
st.dataframe(raw_preview[ORDERED_COLS].copy(), use_container_width=True, height=420)

st.subheader("Processed File Preview — Session-aware 4H (All Rows)")
# Ensure final processed view has exactly the requested final columns and order
processed_view = df_4h[["Date", "Open", "High", "Low", "Close", "Volume", "Open Int", "bars_in_4h", "is_complete"]].copy()
st.dataframe(processed_view, use_container_width=True, height=420)

# =========================
# Date Range Filter (Processed)
# =========================
st.markdown("### Date Range Filter")
if len(df_4h) == 0:
    st.warning("No 4H rows to filter.")
    filtered_df = df_4h.copy()
else:
    min_d, max_d = pd.to_datetime(df_4h["Date"].min()).date(), pd.to_datetime(df_4h["Date"].max()).date()
    start_date, end_date = st.date_input(
        "Select date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )
    # Inclusive filter (include day-end)
    mask = (df_4h["Date"] >= pd.Timestamp(start_date)) & (df_4h["Date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    filtered_df = df_4h.loc[mask].reset_index(drop=True)

st.markdown("### Filtered Data Preview")
filtered_view = filtered_df[["Date", "Volume", "Open", "High", "Low", "Close", "Open Int"]].copy()
filtered_view = filtered_view[["Date", "Volume", "Open", "High", "Low", "Close", "Open Int"]].sort_values("Date").reset_index(drop=True)
st.dataframe(filtered_view, use_container_width=True, height=360)

# =========================
# Chart (optional)
# =========================
if show_chart and len(df_4h) > 0:
    st.subheader("4H Candlestick Chart (Preview)")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_4h["Date"],
                open=df_4h["Open"],
                high=df_4h["High"],
                low=df_4h["Low"],
                close=df_4h["Close"],
                name="4H"
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Time (4H start)",
        yaxis_title="Price",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Save Options (Download + Write to Folder)
# =========================
st.markdown("## Save Options")

# Defaults: use uploaded filename (exact) as requested
uploaded_name = uploaded.name if hasattr(uploaded, "name") else "processed.csv"
default_folder = "C:/Users/yash.patel/Python Projects/Algo_Builder/data/Product Data/New/"
default_full_name = uploaded_name
default_filtered_name = uploaded_name

folder_path = st.text_input("Folder Path", value=default_folder)
full_processed_file = st.text_input("Full Processed File Name", value=default_full_name)
filtered_file = st.text_input("Filtered File Name", value=default_filtered_name)

c_a, c_b = st.columns(2)
with c_a:
    st.download_button(
        "Download Complete Processed CSV",
        data=to_csv_bytes(processed_view[["Date", "Open", "High", "Low", "Close", "Volume", "Open Int", "bars_in_4h", "is_complete"]]),
        file_name=full_processed_file,
        mime="text/csv"
    )
with c_b:
    st.download_button(
        "Download Filtered CSV",
        data=to_csv_bytes(filtered_view),
        file_name=filtered_file,
        mime="text/csv"
    )

c1, c2 = st.columns(2)
with c1:
    if st.button("Save Complete Processed CSV to Folder"):
        try:
            target_dir = Path(folder_path.strip())
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / full_processed_file.strip()
            # Save processed with requested final columns order
            processed_view[["Date", "Open", "High", "Low", "Close", "Volume", "Open Int", "bars_in_4h", "is_complete"]].to_csv(target_path, index=False)
            st.success(f"File saved to {target_path}")
        except Exception as e:
            st.error(f"Failed to save processed CSV: {e}")

with c2:
    if st.button("Save Filtered CSV to Folder"):
        try:
            target_dir = Path(folder_path.strip())
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filtered_file.strip()
            filtered_view.to_csv(target_path, index=False)
            st.success(f"File saved to {target_path}")
        except Exception as e:
            st.error(f"Failed to save filtered CSV: {e}")

# Footnote
with st.expander("Notes", expanded=False):
    st.markdown(
        "- Sorting is strictly by **Date**; OHLC/Volume/Open Int values remain aligned per timestamp.\n"
        "- 4H aggregation is *session-aware* (no cross-gap merging). Final 4H in a session may be partial.\n"
        "- `Volume` in 4H is the **sum of per-hour incremental volume**; `Open Int` is the **last** value in the 4H bucket if provided."
    )
