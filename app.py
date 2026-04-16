import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# -----------------------------------------------------------------------------
# Page & white-label styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="A/B Testing — Decision Console",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_TRUST_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    .block-container { padding-top: 1rem; max-width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; color: #253858; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
    h1 { color: #0052CC !important; font-weight: 700 !important; }
    h2, h3 { color: #253858 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
"""
st.markdown(_TRUST_CSS, unsafe_allow_html=True)

st.title("A/B Testing — Decision Console")
st.caption(
    "Two-proportion Z-test on conversion rates. Upload a CSV or simulate data; "
    "all controls live in the sidebar."
)


def _wald_ci(success: int, n: int, z: float) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = success / n
    se = np.sqrt(max(p * (1 - p) / n, 0.0))
    return p, max(0.0, p - z * se), min(1.0, p + z * se)


def _normal_pdf_grid(mean: float, se: float, label: str) -> pd.DataFrame:
    """Build a smooth normal approximation curve for uncertainty visualization."""
    if se <= 0:
        x = np.array([mean])
        y = np.array([1.0])
    else:
        lo = max(0.0, mean - 4 * se)
        hi = min(1.0, mean + 4 * se)
        x = np.linspace(lo, hi, 240)
        y = stats.norm.pdf(x, loc=mean, scale=se)
    return pd.DataFrame({"rate": x, "density": y, "group": label})


@st.cache_data(show_spinner=False)
def cached_read_upload(upload_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(upload_bytes))


@st.cache_data(show_spinner=False)
def run_ab_analysis(df: pd.DataFrame, confidence_level: float) -> dict:
    """Core Z-test and conversion summary (hashable inputs for cache)."""
    z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    conversion_table = df.groupby("group")["converted"].agg(["sum", "count"])
    conversion_table["conversion_rate"] = conversion_table["sum"] / conversion_table["count"]

    conv_a = float(conversion_table.loc["A", "sum"])
    n_a = int(conversion_table.loc["A", "count"])
    conv_b = float(conversion_table.loc["B", "sum"])
    n_b = int(conversion_table.loc["B", "count"])

    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    rate_a = conversion_table.loc["A", "conversion_rate"]
    rate_b = conversion_table.loc["B", "conversion_rate"]
    z_score = float((rate_b - rate_a) / se) if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    pa, lo_a, hi_a = _wald_ci(int(conv_a), n_a, z_crit)
    pb, lo_b, hi_b = _wald_ci(int(conv_b), n_b, z_crit)
    lift_pp = float((rate_b - rate_a) * 100)
    rel_lift = float((rate_b / rate_a - 1) * 100) if rate_a > 0 else float("nan")

    return {
        "conversion_table": conversion_table,
        "z_score": z_score,
        "p_value": p_value,
        "rate_a": float(rate_a),
        "rate_b": float(rate_b),
        "n_a": n_a,
        "n_b": n_b,
        "lo_a": lo_a,
        "hi_a": hi_a,
        "lo_b": lo_b,
        "hi_b": hi_b,
        "lift_pp": lift_pp,
        "rel_lift": rel_lift,
    }


@st.cache_data(show_spinner=False)
def run_simulation_batch(
    p_a: float,
    p_b: float,
    n_users_a: int,
    n_users_b: int,
    n_simulations: int,
    confidence_level: float,
    rng_seed: int,
) -> pd.DataFrame:
    alpha = 1 - confidence_level
    rng = np.random.default_rng(rng_seed)
    z_scores = []
    p_values = []
    for _ in range(n_simulations):
        sim_a = rng.binomial(1, p_a, n_users_a)
        sim_b = rng.binomial(1, p_b, n_users_b)
        conv_a_sim = float(sim_a.sum())
        conv_b_sim = float(sim_b.sum())
        p_pool_sim = (conv_a_sim + conv_b_sim) / (n_users_a + n_users_b)
        se_sim = np.sqrt(p_pool_sim * (1 - p_pool_sim) * (1 / n_users_a + 1 / n_users_b))
        if se_sim <= 0:
            continue
        z = (sim_b.mean() - sim_a.mean()) / se_sim
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        z_scores.append(z)
        p_values.append(p)
    return pd.DataFrame({"z_score": z_scores, "p_value": p_values})


# --- Sidebar: all interactive controls ---
st.sidebar.header("Data source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Columns: user_id, group (A/B), converted (0/1)")

st.sidebar.header("Or: simulate")
simulate = st.sidebar.checkbox("Simulate sample dataset", value=False)
n_users_a = n_users_b = 250
p_a = 0.2
p_b = 0.25
n_simulations = 100
rng_seed = 42
if simulate:
    n_users_a = st.sidebar.number_input("Users in Group A", min_value=10, value=250, step=10)
    n_users_b = st.sidebar.number_input("Users in Group B", min_value=10, value=250, step=10)
    p_a = st.sidebar.slider("Conversion rate — Group A", 0.0, 1.0, 0.2)
    p_b = st.sidebar.slider("Conversion rate — Group B", 0.0, 1.0, 0.25)
    n_simulations = st.sidebar.number_input("Monte Carlo runs", min_value=1, value=100, step=10)
    rng_seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.header("Statistics")
confidence_level = st.sidebar.selectbox(
    "Confidence level",
    options=[0.90, 0.95, 0.99],
    format_func=lambda x: f"{int(x * 100)}%",
)
alpha = 1 - confidence_level

df = None
if uploaded_file is not None:
    try:
        df = cached_read_upload(uploaded_file.getvalue())
        st.sidebar.success("CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

if simulate:
    rng = np.random.default_rng(rng_seed)
    df_a = pd.DataFrame(
        {
            "user_id": range(1, n_users_a + 1),
            "group": "A",
            "converted": rng.binomial(1, p_a, n_users_a),
        }
    )
    df_b = pd.DataFrame(
        {
            "user_id": range(n_users_a + 1, n_users_a + n_users_b + 1),
            "group": "B",
            "converted": rng.binomial(1, p_b, n_users_b),
        }
    )
    df = pd.concat([df_a, df_b], ignore_index=True)
    st.sidebar.success("Simulation ready — scroll main panel for results.")

if df is None:
    st.info("Upload a CSV or enable **Simulate sample dataset** in the sidebar to begin.")
    st.stop()

required_cols = {"user_id", "group", "converted"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: `{required_cols}`. Found: `{set(df.columns)}`")
    st.stop()

out = run_ab_analysis(df, confidence_level)
alpha_level = alpha
sig = out["p_value"] < alpha_level

# --- Top KPI row (Staff-level: deltas + sample size) ---
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Conv. rate — A",
        f"{out['rate_a']:.2%}",
        delta=f"n = {out['n_a']:,}",
        delta_color="off",
        help=f"95% Wald-style interval (display): [{out['lo_a']:.1%}, {out['hi_a']:.1%}]",
    )
with k2:
    st.metric(
        "Conv. rate — B",
        f"{out['rate_b']:.2%}",
        delta=f"{out['lift_pp']:+.2f} pp vs A",
        delta_color="normal",
        help=f"95% Wald-style interval (display): [{out['lo_b']:.1%}, {out['hi_b']:.1%}]",
    )
with k3:
    rel = out["rel_lift"]
    delta_txt = "Lift" if out["lift_pp"] >= 0 else "Drop"
    st.metric(
        "Lift / drop (B vs A)",
        f"{rel:+.2f}%" if rel == rel else "—",
        delta=delta_txt,
        delta_color="normal",
        help="Percentage points (pp) and relative % change vs group A.",
    )
with k4:
    st.metric(
        "Sample size",
        f"{out['n_a'] + out['n_b']:,}",
        delta=f"α = {alpha_level:.3f}",
        delta_color="off",
        help="Total observations used in this test.",
    )

if sig and out["rate_b"] > out["rate_a"]:
    st.success(
        "**STATISTICALLY SIGNIFICANT: Variant B is the winner.**\n\n"
        "Observed lift is significant at the selected confidence level. Validate guardrails "
        "(SRM, novelty effects, seasonality) before full rollout."
    )
elif sig and out["rate_b"] < out["rate_a"]:
    st.warning(
        "**STATISTICALLY SIGNIFICANT: Variant A is the winner.**\n\n"
        "Variant B underperforms at the selected confidence level. Investigate UX and traffic "
        "quality before another treatment launch."
    )
else:
    st.info(
        "**INCONCLUSIVE: More data required.**\n\n"
        "Current evidence does not separate variants at the selected alpha. Extend runtime, "
        "increase sample size, or segment by key cohorts."
    )

st.divider()

# --- Input parameters (read-only summary; widgets in sidebar) ---
st.subheader("Input parameters")
st.caption("All tunables are in the **sidebar** — this section is an audit trail of the current run.")
ip1, ip2, ip3 = st.columns(3)
with ip1:
    st.write(f"**Confidence:** {int(confidence_level * 100)}% (α = {alpha_level:.3f})")
with ip2:
    st.write(f"**Simulated:** {'Yes' if simulate else 'No'}")
with ip3:
    st.write(f"**Total users in analysis:** {len(df):,}")

st.divider()

# --- Visual analysis ---
st.subheader("Visual analysis")

ci_chart_df = pd.DataFrame(
    {
        "group": ["A", "B"],
        "conversion_rate": [out["rate_a"], out["rate_b"]],
        "ci_low": [out["lo_a"], out["lo_b"]],
        "ci_high": [out["hi_a"], out["hi_b"]],
    }
)

points = (
    alt.Chart(ci_chart_df)
    .mark_point(filled=True, size=120, color="#0052CC")
    .encode(
        x=alt.X("group:N", title="Variant"),
        y=alt.Y("conversion_rate:Q", title="Conversion rate", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("group:N", title="Group"),
            alt.Tooltip("conversion_rate:Q", title="Rate", format=".2%"),
            alt.Tooltip("ci_low:Q", title="CI low", format=".2%"),
            alt.Tooltip("ci_high:Q", title="CI high", format=".2%"),
        ],
    )
)
error_bars = (
    alt.Chart(ci_chart_df)
    .mark_errorbar(color="#253858")
    .encode(
        x="group:N",
        y=alt.Y("ci_low:Q", title="Conversion rate"),
        y2="ci_high:Q",
    )
)
chart_conv = (error_bars + points).properties(
    title="Conversion rate with approximate 95% intervals (Wald)",
    height=400,
    width=500,
)
st.altair_chart(chart_conv, use_container_width=True)

st.markdown("#### Confidence interval overlap (uncertainty distribution)")
se_a = np.sqrt(max(out["rate_a"] * (1 - out["rate_a"]) / out["n_a"], 0.0))
se_b = np.sqrt(max(out["rate_b"] * (1 - out["rate_b"]) / out["n_b"], 0.0))
dist_df = pd.concat(
    [
        _normal_pdf_grid(out["rate_a"], se_a, "Group A"),
        _normal_pdf_grid(out["rate_b"], se_b, "Group B"),
    ],
    ignore_index=True,
)
dist_chart = (
    alt.Chart(dist_df)
    .mark_area(opacity=0.35)
    .encode(
        x=alt.X("rate:Q", title="Conversion rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("density:Q", title="Relative likelihood"),
        color=alt.Color(
            "group:N",
            title="Group",
            scale=alt.Scale(domain=["Group A", "Group B"], range=["#253858", "#0052CC"]),
        ),
        tooltip=[
            alt.Tooltip("group:N", title="Group"),
            alt.Tooltip("rate:Q", title="Rate", format=".2%"),
            alt.Tooltip("density:Q", title="Density", format=".3f"),
        ],
    )
    .properties(height=320, title="Approximate overlap of conversion-rate uncertainty")
)
st.altair_chart(dist_chart, use_container_width=True)

dl1, dl2 = st.columns([1, 4])
with dl1:
    st.download_button(
        "Download summary CSV",
        data=out["conversion_table"].reset_index().to_csv(index=False).encode("utf-8"),
        file_name="ab_conversion_summary.csv",
        mime="text/csv",
    )

if simulate and n_simulations > 1:
    st.markdown("**Monte Carlo — distribution of Z-scores and p-values** (cached for responsiveness)")
    sim_df = run_simulation_batch(p_a, p_b, n_users_a, n_users_b, n_simulations, confidence_level, rng_seed)
    z_chart = (
        alt.Chart(sim_df)
        .mark_bar(color="#0052CC", opacity=0.85)
        .encode(
            alt.X("z_score:Q", bin=alt.Bin(maxbins=30), title="Z-score"),
            y=alt.Y("count()", title="Count"),
        )
        .properties(height=280)
    )
    p_chart = (
        alt.Chart(sim_df)
        .mark_bar(color="#253858", opacity=0.85)
        .encode(
            alt.X("p_value:Q", bin=alt.Bin(maxbins=30), title="p-value"),
            y="count()",
        )
        .properties(height=280)
    )
    st.altair_chart(z_chart, use_container_width=True)
    st.altair_chart(p_chart, use_container_width=True)
    significant = int((sim_df["p_value"] < alpha_level).sum())
    st.caption(
        f"{significant} of {len(sim_df)} runs significant at {int(confidence_level * 100)}% "
        "(same α as primary test)."
    )
    st.download_button(
        "Download simulation draws CSV",
        data=sim_df.to_csv(index=False).encode("utf-8"),
        file_name="ab_simulation_results.csv",
        mime="text/csv",
    )

st.divider()

# --- Raw data ---
st.subheader("Raw data")
st.dataframe(df.head(500), use_container_width=True)
st.download_button(
    "Download input sample (first 500 rows)",
    data=df.head(500).to_csv(index=False).encode("utf-8"),
    file_name="ab_input_sample.csv",
    mime="text/csv",
)

with st.expander("Technical methodology & assumptions"):
    st.markdown(
        """
- **Test:** Two-proportion pooled Z-test; p-value is two-sided vs equal conversion rates.
- **Intervals on chart:** Normal (Wald) approximation for binomial proportion per group — use Wilson or
  exact (Clopper–Pearson) for regulatory submissions if required.
- **Independence:** Assumes i.i.d. Bernoulli conversions and no interference between units.
- **Multiple testing:** One primary metric shown; adjust for multiple comparisons if you run many tests.
- **Simulations:** Monte Carlo re-draws under sidebar probabilities; seed controls reproducibility.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Sherriff Abdul-Hamid · Portfolio tooling")
