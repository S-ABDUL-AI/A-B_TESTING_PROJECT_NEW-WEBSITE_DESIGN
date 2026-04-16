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
    .ab-kpi-row { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; }
    .ab-kpi-card {
        flex: 1 1 200px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(37,56,88,0.08);
        border: 1px solid #e2e8f0;
        border-left-width: 5px;
        min-height: 118px;
    }
    .ab-kpi-label { color: #64748b; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase; }
    .ab-kpi-value { color: #253858; font-size: 1.55rem; font-weight: 700; line-height: 1.2; margin-top: 6px; }
    .ab-kpi-sub { color: #475569; font-size: 0.88rem; margin-top: 8px; line-height: 1.35; }
    .ab-insight-box {
        border-radius: 12px;
        padding: 20px 22px;
        margin: 16px 0 8px 0;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .ab-insight-kicker { color: #0052CC; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .ab-insight-lead { color: #253858; font-size: 1.25rem; font-weight: 800; line-height: 1.35; margin: 10px 0 12px 0; }
    .ab-insight-body { color: #334155; font-size: 0.98rem; line-height: 1.55; }
    .ab-chart-narrative {
        margin: 14px 0 20px 0;
        padding: 14px 16px 16px 16px;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0052CC;
    }
    .ab-chart-narrative .ab-chart-narrative-title {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .ab-chart-narrative p { margin: 0 0 8px 0; font-size: 0.95rem; line-height: 1.55; color: #334155; }
    .ab-chart-narrative p:last-child { margin-bottom: 0; }
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


def _kpi_card_html(label: str, value: str, sub: str, accent: str) -> str:
    return (
        f'<div class="ab-kpi-card" style="border-left-color:{accent};">'
        f'<div class="ab-kpi-label">{label}</div>'
        f'<div class="ab-kpi-value">{value}</div>'
        f'<div class="ab-kpi-sub">{sub}</div></div>'
    )


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
b_ahead = out["rate_b"] > out["rate_a"]
b_wins = sig and b_ahead
a_wins = sig and not b_ahead and out["rate_b"] < out["rate_a"]
tie_rates = abs(out["rate_b"] - out["rate_a"]) < 1e-12

# Accent palette (high-trust, respectful)
COL_WIN = "#0d9488"
COL_CONTROL = "#64748b"
COL_TREAT = "#0052CC"
COL_LIFT_POS = "#047857"
COL_LIFT_NEG = "#b45309"
COL_NEUTRAL = "#475569"

accent_a = COL_WIN if a_wins else (COL_CONTROL if b_wins else COL_CONTROL)
accent_b = COL_WIN if b_wins else (COL_CONTROL if a_wins else COL_TREAT)
if tie_rates and not sig:
    accent_a = accent_b = COL_CONTROL

lift_pp = out["lift_pp"]
rel = out["rel_lift"]
if sig:
    accent_lift = COL_LIFT_POS if lift_pp > 0 else (COL_LIFT_NEG if lift_pp < 0 else COL_NEUTRAL)
else:
    accent_lift = COL_NEUTRAL
rel_line = f"Relative change vs A: <strong>{rel:+.2f}%</strong>" if rel == rel else "Relative change: —"
lift_sub = (
    f"{rel_line}<br/>{int(confidence_level * 100)}% CI — A: [{out['lo_a']:.1%}, {out['hi_a']:.1%}] · "
    f"B: [{out['lo_b']:.1%}, {out['hi_b']:.1%}]"
)

kpi_html = (
    '<div class="ab-kpi-row">'
    + _kpi_card_html(
        "Group A · Conversion",
        f"{out['rate_a']:.2%}",
        f"n = {out['n_a']:,} · Control / baseline",
        accent_a,
    )
    + _kpi_card_html(
        "Group B · Conversion",
        f"{out['rate_b']:.2%}",
        f"n = {out['n_b']:,} · Treatment",
        accent_b,
    )
    + _kpi_card_html(
        "Lift (B vs A)",
        f"{lift_pp:+.2f} pp",
        lift_sub,
        accent_lift,
    )
    + _kpi_card_html(
        "Evidence strength",
        "Significant" if sig else "Not significant",
        f"p-value = {out['p_value']:.4f} · α = {alpha_level:.3f} · two-sided Z-test",
        COL_WIN if sig else COL_NEUTRAL,
    )
    + "</div>"
)
st.markdown(kpi_html, unsafe_allow_html=True)

# --- Statistical verdict (success = significant; error = inconclusive) ---
if b_wins:
    st.success(
        "**STATISTICALLY SIGNIFICANT** — Variant **B** outperforms A on conversion at your chosen "
        f"confidence level (p = {out['p_value']:.4f} < α = {alpha_level:.3f})."
    )
elif a_wins:
    st.success(
        "**STATISTICALLY SIGNIFICANT** — Variant **A** outperforms B on conversion at your chosen "
        f"confidence level (p = {out['p_value']:.4f} < α = {alpha_level:.3f}). "
        "Do not roll out treatment B as tested."
    )
elif tie_rates and sig:
    st.success(
        "**STATISTICALLY SIGNIFICANT** — Estimated rates are tied; the test detects a negligible "
        f"effect (p = {out['p_value']:.4f}). Prefer business tie-breakers or more precision."
    )
else:
    st.error(
        "**NOT STATISTICALLY SIGNIFICANT** — Results are **inconclusive** at α = "
        f"{alpha_level:.3f} (p = {out['p_value']:.4f}). **Recommendation:** extend the experiment "
        "or increase power before making a launch decision."
    )

# --- Storytelling: bold executive insight ---
if b_wins:
    insight_lead = (
        f"Ship variant **B** toward staged rollout: observed **+{lift_pp:.2f} pp** "
        f"({rel:+.1f}% relative vs A) with statistical backing."
        if rel == rel
        else f"Ship variant **B** toward staged rollout: observed **+{lift_pp:.2f} pp** with statistical backing."
    )
    insight_body = (
        "Next: confirm **sample ratio mismatch (SRM)**, **novelty/holdout**, and **cohort slices** "
        "(device, geography) before promoting to 100%. Document the decision in your experiment log."
    )
elif a_wins:
    insight_lead = (
        f"**Hold** treatment B: it trails control by **{abs(lift_pp):.2f} pp** "
        f"({rel:.1f}% relative vs A) with significance."
        if rel == rel
        else f"**Hold** treatment B: it trails control by **{abs(lift_pp):.2f} pp** with significance."
    )
    insight_body = (
        "Next: qualitative review of B (UX, latency, audience mismatch), then iterate the treatment "
        "or run a follow-up with a clearer hypothesis."
    )
else:
    insight_lead = (
        "**No launch decision yet** — uncertainty intervals overlap enough that the test cannot "
        "separate A and B at this alpha."
    )
    insight_body = (
        "Next: run longer, raise traffic split, or pre-register a **minimum detectable effect (MDE)** "
        "so stakeholders know when to stop."
    )

st.markdown(
    f"""
<div class="ab-insight-box">
  <div class="ab-insight-kicker">Executive insight</div>
  <div class="ab-insight-lead">{insight_lead}</div>
  <div class="ab-insight-body">{insight_body}</div>
</div>
""",
    unsafe_allow_html=True,
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
st.caption(
    "Point estimates with 95% confidence intervals (Wald). Overlap signals remaining uncertainty; "
    "significance still follows the formal two-proportion test."
)

ci_chart_df = pd.DataFrame(
    {
        "group": ["A", "B"],
        "conversion_rate": [out["rate_a"], out["rate_b"]],
        "ci_low": [out["lo_a"], out["lo_b"]],
        "ci_high": [out["hi_a"], out["hi_b"]],
    }
)

# Same hexes as the KPI card left-border accents (straight-through brand consistency)
ci_scale = alt.Scale(domain=["A", "B"], range=[accent_a, accent_b])
# labelOverlap=False + explicit bottom orient prevents Vega-Lite from auto-rotating A/B when the
# chart is narrow (e.g. Streamlit full-width layout).
x_axis_variant = alt.X(
    "group:N",
    title="Variant",
    sort=["A", "B"],
    axis=alt.Axis(
        orient="bottom",
        labelAngle=0,
        labelOverlap=False,
        labelAlign="center",
        labelBaseline="alphabetic",
        labelFontWeight="bold",
        labelFontSize=14,
        titleFontWeight="bold",
        titlePadding=12,
        labelPadding=10,
    ),
)
_rate_scale = alt.Scale(domain=[0, 1])
_rate_axis = alt.Axis(format="%", titleFontWeight="bold")

points = (
    alt.Chart(ci_chart_df)
    .mark_point(filled=True, size=140, stroke="#fff", strokeWidth=2)
    .encode(
        x=x_axis_variant,
        y=alt.Y(
            "conversion_rate:Q",
            title="Conversion rate",
            scale=_rate_scale,
            axis=_rate_axis,
        ),
        color=alt.Color("group:N", scale=ci_scale, legend=alt.Legend(title="Variant")),
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
    .mark_errorbar(thickness=2)
    .encode(
        x=x_axis_variant,
        y=alt.Y("ci_low:Q", scale=_rate_scale, axis=None),
        y2="ci_high:Q",
        color=alt.Color("group:N", scale=ci_scale, legend=None),
    )
)
chart_conv = (
    (error_bars + points)
    .properties(
        title=alt.TitleParams(
            text=f"{int(confidence_level * 100)}% confidence intervals — conversion rate by variant",
            fontSize=16,
            fontWeight="bold",
            color="#253858",
            anchor="start",
        ),
        width=520,
        height=420,
    )
    .configure_axisX(
        labelAngle=0,
        labelOverlap=False,
        labelFontWeight="bold",
        labelFontSize=14,
        titleFontWeight="bold",
    )
    .configure_axisY(titleFontWeight="bold")
    .configure_axisBottom(
        labelAngle=0,
        labelOverlap=False,
        labelFontWeight="bold",
        labelFontSize=14,
    )
    .configure_view(stroke=None)
)
# theme=None avoids Streamlit's Altair theme re-tuning categorical label angles.
st.altair_chart(chart_conv, use_container_width=True, theme=None)

chart_story_a = (
    f'<span style="color:{accent_a};font-weight:800;">Group A</span> (control) is at '
    f"<strong>{out['rate_a']:.2%}</strong> with a {int(confidence_level * 100)}% interval "
    f"<strong>{out['lo_a']:.1%}–{out['hi_a']:.1%}</strong>."
)
chart_story_b = (
    f'<span style="color:{accent_b};font-weight:800;">Group B</span> (treatment) is at '
    f"<strong>{out['rate_b']:.2%}</strong> with a {int(confidence_level * 100)}% interval "
    f"<strong>{out['lo_b']:.1%}–{out['hi_b']:.1%}</strong>."
)
chart_story_lift = (
    f"The point gap is <strong>{lift_pp:+.2f} pp</strong>"
    + (
        f" (<strong>{rel:+.1f}%</strong> relative vs A)."
        if rel == rel
        else "."
    )
    + " Bars use the <strong>same accent colors</strong> as the KPI row above."
)
st.markdown(
    f"""
<div class="ab-chart-narrative" style="border-left-color:{accent_lift};">
  <div class="ab-chart-narrative-title">Insight narrative · matches KPI palette</div>
  <p>{chart_story_a}</p>
  <p>{chart_story_b}</p>
  <p>{chart_story_lift}</p>
</div>
""",
    unsafe_allow_html=True,
)

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
dist_colors = [accent_a, accent_b]

dist_chart = (
    alt.Chart(dist_df)
    .mark_area(opacity=0.38)
    .encode(
        x=alt.X("rate:Q", title="Conversion rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("density:Q", title="Relative likelihood"),
        color=alt.Color(
            "group:N",
            title="Group",
            scale=alt.Scale(domain=["Group A", "Group B"], range=dist_colors),
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
