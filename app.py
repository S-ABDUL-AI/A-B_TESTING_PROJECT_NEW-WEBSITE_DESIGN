import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="A/B testing", page_icon="📊", layout="wide")

st.title("📊 A/B testing — conversion experiment")
st.caption(
    "Two-proportion z-test, lift, and optional Monte Carlo checks. "
    "CSV columns: **user_id**, **group** (A/B), **converted** (0/1)."
)


def normalize_groups(s: pd.Series) -> pd.Series:
    def one(x):
        t = str(x).strip().upper()
        if t in ("A", "CONTROL", "0", "OLD"):
            return "A"
        if t in ("B", "TREATMENT", "1", "NEW", "VARIANT"):
            return "B"
        return t[:1] if t else "A"

    return s.map(one)


def two_proportion_z(conv_a: int, n_a: int, conv_b: int, n_b: int):
    """Pooled two-proportion z-test; returns z, p (two-sided)."""
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def prop_ci(k: int, n: int, conf: float):
    alpha = 1 - conf
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    se = np.sqrt(p * (1 - p) / n) if n else 0
    return p, p - z * se, p + z * se


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
simulate = st.sidebar.checkbox("Use simulated data instead", value=False)

df = None
if simulate:
    np.random.seed(42)
    n_a = st.sidebar.number_input("Users in A", 20, 50_000, 400, 10)
    n_b = st.sidebar.number_input("Users in B", 20, 50_000, 400, 10)
    p_a = st.sidebar.slider("True conversion rate A", 0.01, 0.5, 0.12)
    p_b = st.sidebar.slider("True conversion rate B", 0.01, 0.5, 0.16)
    a = pd.DataFrame({"user_id": np.arange(1, n_a + 1), "group": "A", "converted": np.random.binomial(1, p_a, n_a)})
    b = pd.DataFrame(
        {"user_id": np.arange(n_a + 1, n_a + n_b + 1), "group": "B", "converted": np.random.binomial(1, p_b, n_b)}
    )
    df = pd.concat([a, b], ignore_index=True)
elif uploaded is not None:
    df = pd.read_csv(uploaded)

st.sidebar.header("Test settings")
confidence = st.sidebar.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
alpha = 1 - confidence

if df is None:
    st.info("Upload a CSV or enable **simulated data** in the sidebar.")
    st.stop()

lower = {c.lower().strip(): c for c in df.columns}
for req in ("user_id", "group", "converted"):
    if req not in lower:
        st.error("CSV must include columns: **user_id**, **group**, **converted**.")
        st.stop()
df = df.rename(
    columns={
        lower["user_id"]: "user_id",
        lower["group"]: "group",
        lower["converted"]: "converted",
    }
)

df["group"] = normalize_groups(df["group"])
df["converted"] = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)
df = df[df["group"].isin(["A", "B"])].copy()

if df["group"].nunique() < 2:
    st.error("Need both **A** and **B** rows after cleaning.")
    st.stop()

summary = df.groupby("group")["converted"].agg(["sum", "count"])
summary["rate"] = summary["sum"] / summary["count"]

conv_a = int(summary.loc["A", "sum"])
n_a = int(summary.loc["A", "count"])
conv_b = int(summary.loc["B", "sum"])
n_b = int(summary.loc["B", "count"])

p_a = conv_a / n_a
p_b = conv_b / n_b
rel_lift = (p_b - p_a) / p_a if p_a > 0 else np.nan

z, p_val = two_proportion_z(conv_a, n_a, conv_b, n_b)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Conv. A", f"{p_a:.2%}", f"n={n_a}")
m2.metric("Conv. B", f"{p_b:.2%}", f"n={n_b}")
m3.metric("Absolute lift (B−A)", f"{(p_b - p_a) * 100:.2f} pp")
m4.metric("Relative lift", f"{rel_lift * 100:.1f}%" if np.isfinite(rel_lift) else "—")

st.subheader("Results")
st.dataframe(summary.reset_index(), use_container_width=True, hide_index=True)

st.write(
    f"**Two-sided z-test:** z = {z:.4f}, **p-value** = {p_val:.5f} "
    f"(confidence {confidence:.0%}, α = {alpha:.3f})."
)

if p_val < alpha:
    st.success("Difference is **statistically significant** at the chosen level.")
else:
    st.warning("No significant difference detected at the chosen level (consider power / duration).")

pa_hat, lo_a, hi_a = prop_ci(conv_a, n_a, confidence)
pb_hat, lo_b, hi_b = prop_ci(conv_b, n_b, confidence)
st.caption(
    f"{confidence:.0%} Wilson-style normal approx. CIs: A [{lo_a:.3f}, {hi_a:.3f}], B [{lo_b:.3f}, {hi_b:.3f}]"
)

plot_df = summary.reset_index()[["group", "rate"]]
chart = (
    alt.Chart(plot_df)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("group:N", title="Variant"),
        y=alt.Y("rate:Q", title="Conversion rate", axis=alt.Axis(format="%")),
        color=alt.Color("group:N", legend=None),
        tooltip=["group", alt.Tooltip("rate:Q", format=".2%")],
    )
    .properties(height=320, title="Conversion rate by variant")
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Export")
buf = io.StringIO()
summary.to_csv(buf)
st.download_button("Download summary CSV", buf.getvalue(), "ab_summary.csv", "text/csv")

if st.checkbox("Run simulation of z-scores (optional)", value=False):
    n_sims = st.number_input("Simulations", 100, 5000, 500, 50)
    zs = []
    for _ in range(int(n_sims)):
        sim_a = np.random.binomial(1, p_a, n_a)
        sim_b = np.random.binomial(1, p_b, n_b)
        ca, cb = int(sim_a.sum()), int(sim_b.sum())
        zz, _ = two_proportion_z(ca, n_a, cb, n_b)
        zs.append(zz)
    sim_df = pd.DataFrame({"z": zs})
    hist = (
        alt.Chart(sim_df)
        .mark_bar()
        .encode(alt.X("z:Q", bin=alt.Bin(maxbins=35)), y="count()")
        .properties(height=260, title="Null-style z distribution (uses observed rates as plug-in)")
    )
    st.altair_chart(hist, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sherriff Abdul-Hamid**  \n"
    "[GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)
