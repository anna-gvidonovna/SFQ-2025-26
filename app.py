from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols


st.set_page_config(
    page_title="SFQ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


SCORE_IMP_PAIRS = [
    ("fac_support_score", "fac_support_imp"),
    ("fac_clarity_score", "fac_clarity_imp"),
    ("cur_timely_score", "cur_timely_imp"),
    ("cur_help_score", "cur_help_imp"),
    ("prog_clarity_score", "prog_clarity_imp"),
    ("prog_deadlines_score", "prog_deadlines_imp"),
    ("prog_relevance_score", "prog_relevance_imp"),
    ("prog_workload_score", "prog_workload_imp"),
    ("coord_respect_score", "coord_respect_imp"),
    ("coord_results_score", "coord_results_imp"),
    ("coord_timely_score", "coord_timely_imp"),
    ("coord_help_score", "coord_help_imp"),
]

BLOCKS = {
    "faculty": ["fac_support_score", "fac_clarity_score"],
    "curator": ["cur_timely_score", "cur_help_score"],
    "program": ["prog_clarity_score", "prog_deadlines_score", "prog_relevance_score", "prog_workload_score"],
    "coordinator": ["coord_respect_score", "coord_results_score", "coord_timely_score", "coord_help_score"],
    "assessment": ["assess_criteria_timely", "assess_order_clear", "assess_consistent"],
    "infrastructure": [
        "infra_library",
        "infra_wellbeing",
        "infra_food",
        "infra_software",
        "infra_equipment",
        "infra_classrooms",
        "infra_workshops",
    ],
}

KEY_METRICS = [
    "satisf_overall",
    "satisf_teachers",
    "expect_match",
    "assess_criteria_timely",
    "assess_order_clear",
    "assess_consistent",
    "nps",
]


@dataclass
class FilterState:
    programs: list[str]
    years: list[str]
    devices: list[str]
    min_duration: int
    max_duration: int


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    return df


def num_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def bootstrap_mean_ci(x: pd.Series, n_boot: int = 3000, ci: int = 95, seed: int = 42) -> tuple[float, float, float, int]:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = vals.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    return vals.mean(), np.percentile(draws, alpha), np.percentile(draws, 100 - alpha), n


def cronbach_alpha(df_block: pd.DataFrame) -> float:
    x = df_block.dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 3:
        return np.nan
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_var / total_var))


def build_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Filters")

    programs = sorted(df["program"].dropna().unique().tolist()) if "program" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    devices = sorted(df["device"].dropna().unique().tolist()) if "device" in df.columns else []

    selected_programs = st.sidebar.multiselect("Program", programs, default=programs)
    selected_years = st.sidebar.multiselect("Year", years, default=years)
    selected_devices = st.sidebar.multiselect("Device", devices, default=devices)

    if "duration_sec" in df.columns:
        min_d = int(np.nanmin(df["duration_sec"]))
        max_d = int(np.nanmax(df["duration_sec"]))
        d_range = st.sidebar.slider("Duration sec", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    else:
        d_range = (0, 10_000)

    return FilterState(
        programs=selected_programs,
        years=selected_years,
        devices=selected_devices,
        min_duration=d_range[0],
        max_duration=d_range[1],
    )


def apply_filters(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    out = df.copy()
    if "program" in out.columns and f.programs:
        out = out[out["program"].isin(f.programs)]
    if "year" in out.columns and f.years:
        out = out[out["year"].isin(f.years)]
    if "device" in out.columns and f.devices:
        out = out[out["device"].isin(f.devices)]
    if "duration_sec" in out.columns:
        out = out[(out["duration_sec"] >= f.min_duration) & (out["duration_sec"] <= f.max_duration)]
    return out


def add_block_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for block, cols in BLOCKS.items():
        available = [c for c in cols if c in out.columns]
        if available:
            out[f"{block}_mean"] = out[available].mean(axis=1)
    return out


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", f"{len(df):,}")
    c2.metric("Programs", f"{df['program'].nunique() if 'program' in df.columns else 0}")
    c3.metric("Median NPS", f"{df['nps'].median():.2f}" if "nps" in df.columns else "n/a")
    c4.metric("Mean duration (sec)", f"{df['duration_sec'].mean():.1f}" if "duration_sec" in df.columns else "n/a")

    if "program" in df.columns:
        counts = df["program"].value_counts().sort_values(ascending=False).reset_index()
        counts.columns = ["program", "n"]
        fig = px.bar(counts, x="program", y="n", title="Sample size by program", color="n", color_continuous_scale="Teal")
        fig.update_layout(xaxis_title="", yaxis_title="Responses", xaxis_tickangle=-25, height=460)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "nps" in df.columns:
            fig = px.histogram(df, x="nps", nbins=11, marginal="box", title="NPS distribution", color_discrete_sequence=["#1f77b4"])
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "duration_sec" in df.columns:
            fig = px.histogram(
                df,
                x="duration_sec",
                nbins=30,
                marginal="violin",
                title="Completion duration distribution",
                color_discrete_sequence=["#e15759"],
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)


def render_descriptives(df: pd.DataFrame) -> None:
    st.subheader("Descriptive statistics")
    available_metrics = [c for c in KEY_METRICS if c in df.columns]
    if not available_metrics:
        st.warning("No key metrics found.")
        return

    selected = st.multiselect("Metrics for descriptives", available_metrics, default=available_metrics)
    if not selected:
        st.info("Choose at least one metric.")
        return

    rows = []
    for col in selected:
        mean, lo, hi, n = bootstrap_mean_ci(df[col])
        vals = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "n": n,
                "mean": mean,
                "median": vals.median(),
                "std": vals.std(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    stat_df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    st.dataframe(stat_df, use_container_width=True)

    fig = px.scatter(
        stat_df,
        x="mean",
        y="metric",
        error_x=stat_df["ci_high"] - stat_df["mean"],
        error_x_minus=stat_df["mean"] - stat_df["ci_low"],
        size="n",
        color="std",
        color_continuous_scale="RdYlBu_r",
        title="Means with bootstrap CI",
    )
    fig.update_layout(height=460, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    metric_for_group = st.selectbox("Metric by program", selected, index=0)
    grp_rows = []
    for prog, g in df.groupby("program"):
        mean, lo, hi, n = bootstrap_mean_ci(g[metric_for_group])
        grp_rows.append({"program": prog, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi})
    grp = pd.DataFrame(grp_rows).sort_values("mean")
    fig = px.scatter(
        grp,
        x="mean",
        y="program",
        error_x=grp["ci_high"] - grp["mean"],
        error_x_minus=grp["mean"] - grp["ci_low"],
        size="n",
        color="mean",
        color_continuous_scale="Viridis",
        title=f"{metric_for_group}: mean by program with bootstrap CI",
    )
    fig.update_layout(height=620, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.violin(df, x="program", y=metric_for_group, box=True, points="all", title=f"{metric_for_group} distribution by program")
    fig.update_layout(height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_program_comparison(df: pd.DataFrame) -> None:
    st.subheader("Program comparison")
    if "nps" not in df.columns or "program" not in df.columns:
        st.warning("Required columns are missing.")
        return

    min_n = st.slider("Minimum group size for inferential tests", min_value=3, max_value=20, value=5, step=1)
    vc = df["program"].value_counts()
    valid_programs = vc[vc >= min_n].index
    d = df[df["program"].isin(valid_programs)][["program", "nps"]].dropna()
    st.caption(f"Groups used: {d['program'].nunique()} | observations: {len(d)}")

    if d["program"].nunique() < 2:
        st.info("Not enough groups for tests.")
        return

    group_vals = [g["nps"].values for _, g in d.groupby("program")]
    lev_stat, lev_p = stats.levene(*group_vals, center="median")

    anova_model = ols("nps ~ C(program)", data=d).fit()
    anova_tbl = sm.stats.anova_lm(anova_model, typ=2)
    ss_between = anova_tbl.loc["C(program)", "sum_sq"]
    eta2 = ss_between / anova_tbl["sum_sq"].sum()

    kw_stat, kw_p = stats.kruskal(*group_vals)
    dunn = sp.posthoc_dunn(d, val_col="nps", group_col="program", p_adjust="holm")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ANOVA p", f"{anova_tbl.loc['C(program)', 'PR(>F)']:.4g}")
    k2.metric("Eta^2", f"{eta2:.3f}")
    k3.metric("Kruskal p", f"{kw_p:.4g}")
    k4.metric("Levene p", f"{lev_p:.4g}")

    left, right = st.columns(2)
    with left:
        st.markdown("ANOVA table")
        st.dataframe(anova_tbl, use_container_width=True)
    with right:
        st.markdown("Dunn post-hoc (Holm-adjusted p-values)")
        st.dataframe(dunn, use_container_width=True)

    fig = px.box(d, x="program", y="nps", points="all", color="program", title="NPS by program")
    fig.update_layout(showlegend=False, height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_correlations(df: pd.DataFrame) -> None:
    st.subheader("Correlations")
    numeric = num_cols(df)
    default = [c for c in ["nps", "satisf_overall", "expect_match", "program_mean", "coord_mean", "infra_mean"] if c in numeric]
    selected = st.multiselect("Numeric columns for Spearman matrix", numeric, default=default if default else numeric[:12])
    if len(selected) < 2:
        st.info("Select at least 2 numeric columns.")
        return

    corr = df[selected].corr(method="spearman")
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Spearman correlation matrix",
    )
    fig.update_layout(height=680)
    st.plotly_chart(fig, use_container_width=True)

    target = st.selectbox("Target metric", selected, index=selected.index("nps") if "nps" in selected else 0)
    with_target = corr[target].drop(target).sort_values(ascending=False).rename("rho").to_frame()
    st.dataframe(with_target, use_container_width=True)

    top_pos = with_target.head(5).reset_index().rename(columns={"index": "feature"})
    top_neg = with_target.tail(5).reset_index().rename(columns={"index": "feature"})
    cols = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates(subset="feature")
    fig = px.bar(cols, x="rho", y="feature", orientation="h", color="rho", color_continuous_scale="RdBu")
    fig.update_layout(title=f"Top +/- Spearman with {target}", height=420, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_priority_matrix(df: pd.DataFrame) -> None:
    st.subheader("Priority matrix (importance-performance)")
    rows = []
    for score_col, imp_col in SCORE_IMP_PAIRS:
        if score_col not in df.columns or imp_col not in df.columns:
            continue
        score = pd.to_numeric(df[score_col], errors="coerce")
        imp = pd.to_numeric(df[imp_col], errors="coerce")
        valid = score.notna() & imp.notna()
        if valid.sum() == 0:
            continue
        rows.append(
            {
                "item": score_col.replace("_score", ""),
                "mean_score": score[valid].mean(),
                "mean_importance": imp[valid].mean(),
                "gap_imp_minus_score": imp[valid].mean() - score[valid].mean(),
                "n": int(valid.sum()),
            }
        )

    if not rows:
        st.warning("No score/importance pairs found in current filter.")
        return

    m = pd.DataFrame(rows)
    score_mid = m["mean_score"].mean()
    imp_mid = m["mean_importance"].mean()

    fig = px.scatter(
        m,
        x="mean_score",
        y="mean_importance",
        text="item",
        size="n",
        color="gap_imp_minus_score",
        color_continuous_scale="RdYlGn_r",
        hover_data=["gap_imp_minus_score", "n"],
        title="Importance-performance matrix",
    )
    fig.add_vline(score_mid, line_dash="dash", line_color="gray")
    fig.add_hline(imp_mid, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    m = m.sort_values("gap_imp_minus_score", ascending=False)
    st.markdown("Largest improvement opportunities (high importance, lower score)")
    st.dataframe(m, use_container_width=True)


def render_nps(df: pd.DataFrame) -> None:
    st.subheader("NPS deep dive")
    if "nps" not in df.columns:
        st.warning("Column nps not found.")
        return

    d = df.copy()
    d["segment"] = np.where(d["nps"] >= 9, "Promoter", np.where(d["nps"] >= 7, "Passive", "Detractor"))
    seg = d["segment"].value_counts().rename_axis("segment").reset_index(name="n")
    seg["pct"] = seg["n"] / seg["n"].sum() * 100
    nps_value = float(seg.loc[seg["segment"] == "Promoter", "pct"].sum() - seg.loc[seg["segment"] == "Detractor", "pct"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NPS", f"{nps_value:.1f}")
    c2.metric("Promoters %", f"{seg.loc[seg['segment']=='Promoter', 'pct'].sum():.1f}")
    c3.metric("Passives %", f"{seg.loc[seg['segment']=='Passive', 'pct'].sum():.1f}")
    c4.metric("Detractors %", f"{seg.loc[seg['segment']=='Detractor', 'pct'].sum():.1f}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(seg, names="segment", values="n", hole=0.45, color="segment", color_discrete_map={"Promoter": "#2ca02c", "Passive": "#ffbf00", "Detractor": "#d62728"})
        fig.update_layout(title="NPS segment split", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "program" in d.columns:
            by_prog = d.groupby(["program", "segment"]).size().rename("n").reset_index()
            tot = d.groupby("program").size().rename("total").reset_index()
            by_prog = by_prog.merge(tot, on="program", how="left")
            by_prog["pct"] = by_prog["n"] / by_prog["total"] * 100
            fig = px.bar(by_prog, x="program", y="pct", color="segment", barmode="stack", title="NPS composition by program")
            fig.update_layout(height=430, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    if "program" in d.columns:
        nps_by_program = d.groupby("program").apply(
            lambda x: (x["nps"].ge(9).mean() - x["nps"].le(6).mean()) * 100
        ).rename("nps").reset_index()
        fig = px.bar(nps_by_program.sort_values("nps"), x="nps", y="program", orientation="h", color="nps", color_continuous_scale="RdYlGn")
        fig.update_layout(title="NPS by program", height=620, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def render_modeling(df: pd.DataFrame) -> None:
    st.subheader("Regression and reliability")
    d = add_block_features(df)
    predictors = [
        "satisf_overall",
        "expect_match",
        "faculty_mean",
        "curator_mean",
        "program_mean",
        "coordinator_mean",
        "assessment_mean",
        "infrastructure_mean",
    ]
    predictors = [c for c in predictors if c in d.columns]
    model_df = d[["nps"] + predictors].dropna() if "nps" in d.columns else pd.DataFrame()
    if len(model_df) < 30:
        st.info("Not enough complete rows for robust regression.")
    else:
        X = sm.add_constant(model_df[predictors])
        y = model_df["nps"]
        res = sm.OLS(y, X).fit(cov_type="HC3")
        c1, c2, c3 = st.columns(3)
        c1.metric("Regression n", f"{len(model_df)}")
        c2.metric("R^2", f"{res.rsquared:.3f}")
        c3.metric("Adj. R^2", f"{res.rsquared_adj:.3f}")

        coef = pd.DataFrame(
            {
                "term": res.params.index,
                "coef": res.params.values,
                "p_value": res.pvalues.values,
                "ci_low": res.conf_int()[0].values,
                "ci_high": res.conf_int()[1].values,
            }
        )
        coef = coef[coef["term"] != "const"].sort_values("coef")
        st.dataframe(coef, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=coef["coef"],
                y=coef["term"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=coef["ci_high"] - coef["coef"],
                    arrayminus=coef["coef"] - coef["ci_low"],
                ),
                marker=dict(size=10, color=coef["p_value"], colorscale="Viridis", showscale=True),
            )
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(title="OLS coefficients for NPS (HC3 CI)", height=460, xaxis_title="Coefficient", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    alpha_rows = []
    for block, cols in BLOCKS.items():
        existing = [c for c in cols if c in d.columns]
        if len(existing) >= 2:
            alpha_rows.append({"block": block, "k_items": len(existing), "cronbach_alpha": cronbach_alpha(d[existing])})
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows).sort_values("cronbach_alpha", ascending=False)
        st.markdown("Scale reliability (Cronbach alpha)")
        st.dataframe(alpha_df, use_container_width=True)
        fig = px.bar(alpha_df, x="block", y="cronbach_alpha", color="cronbach_alpha", color_continuous_scale="Tealgrn")
        fig.add_hline(y=0.7, line_dash="dash", line_color="gray")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def render_missingness(df: pd.DataFrame) -> None:
    st.subheader("Missingness profile")
    miss = df.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index()
    miss.columns = ["column", "missing_rate"]
    top_n = st.slider("Columns to show", min_value=10, max_value=min(80, len(miss)), value=min(40, len(miss)))
    show = miss.head(top_n)
    fig = px.bar(show, x="missing_rate", y="column", orientation="h", color="missing_rate", color_continuous_scale="Reds")
    fig.update_layout(height=900, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("SFQ 2025-26 Statistical Dashboard")
    st.caption("Data source: combined_general_agg.csv | Likert + NPS analytics")

    df = load_data("combined_general_agg.csv")
    df = safe_numeric(df, num_cols(df))
    filters = build_filters(df)
    dff = apply_filters(df, filters)

    if dff.empty:
        st.error("No data left after filters.")
        return

    tab_overview, tab_desc, tab_comp, tab_corr, tab_priority, tab_nps, tab_model, tab_missing = st.tabs(
        ["Overview", "Descriptives", "Program Comparison", "Correlations", "Priority Matrix", "NPS", "Modeling", "Missingness"]
    )

    with tab_overview:
        render_overview(dff)
    with tab_desc:
        render_descriptives(dff)
    with tab_comp:
        render_program_comparison(dff)
    with tab_corr:
        render_correlations(add_block_features(dff))
    with tab_priority:
        render_priority_matrix(dff)
    with tab_nps:
        render_nps(dff)
    with tab_model:
        render_modeling(dff)
    with tab_missing:
        render_missingness(dff)


if __name__ == "__main__":
    main()
