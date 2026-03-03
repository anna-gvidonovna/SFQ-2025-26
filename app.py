from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import scikit_posthocs as sp
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.formula.api import ols


st.set_page_config(
    page_title="SFQ Дашборд",
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

COL_LABELS = {
    "program": "Программа",
    "year": "Курс",
    "school": "Школа",
    "nps": "NPS (0-10)",
    "satisf_overall": "Удовлетворенность образовательным процессом",
    "satisf_teachers": "Удовлетворенность преподавателями",
    "expect_match": "Совпадение ожиданий с опытом",
    "assess_criteria_timely": "Критерии оценивания представлены вовремя",
    "assess_order_clear": "Порядок сдачи и оценивания понятен",
    "assess_consistent": "Оценивание соответствует критериям",
    "fac_support_score": "Поддержка преподавательской команды (оценка)",
    "fac_clarity_score": "Понятность объяснений преподавателей (оценка)",
    "cur_timely_score": "Своевременность информации от куратора (оценка)",
    "cur_help_score": "Готовность куратора помочь (оценка)",
    "prog_clarity_score": "Понятность содержания дисциплин (оценка)",
    "prog_deadlines_score": "Реалистичность дедлайнов (оценка)",
    "prog_relevance_score": "Проф. релевантность дисциплин (оценка)",
    "prog_workload_score": "Оптимальность нагрузки (оценка)",
    "coord_respect_score": "Уважительность координатора (оценка)",
    "coord_results_score": "Результативность обращений к координатору (оценка)",
    "coord_timely_score": "Своевременность информации от координатора (оценка)",
    "coord_help_score": "Готовность координатора помочь (оценка)",
    "fac_support_imp": "Поддержка преподавательской команды (важность)",
    "fac_clarity_imp": "Понятность объяснений преподавателей (важность)",
    "cur_timely_imp": "Своевременность информации от куратора (важность)",
    "cur_help_imp": "Готовность куратора помочь (важность)",
    "prog_clarity_imp": "Понятность содержания дисциплин (важность)",
    "prog_deadlines_imp": "Реалистичность дедлайнов (важность)",
    "prog_relevance_imp": "Проф. релевантность дисциплин (важность)",
    "prog_workload_imp": "Оптимальность нагрузки (важность)",
    "coord_respect_imp": "Уважительность координатора (важность)",
    "coord_results_imp": "Результативность обращений к координатору (важность)",
    "coord_timely_imp": "Своевременность информации от координатора (важность)",
    "coord_help_imp": "Готовность координатора помочь (важность)",
    "infra_library": "Библиотека",
    "infra_wellbeing": "Wellbeing",
    "infra_food": "Еда на кампусе",
    "infra_software": "ПО в аудиториях",
    "infra_equipment": "Оборудование в аудиториях",
    "infra_classrooms": "Комфорт аудиторий",
    "infra_workshops": "Комфорт мастерских/ресурсных центров",
    "faculty_mean": "Блок: преподавательская команда (среднее)",
    "curator_mean": "Блок: куратор (среднее)",
    "program_mean": "Блок: программа (среднее)",
    "coordinator_mean": "Блок: координатор (среднее)",
    "assessment_mean": "Блок: оценивание (среднее)",
    "infrastructure_mean": "Блок: инфраструктура (среднее)",
}


@dataclass
class FilterState:
    programs: list[str]
    years: list[str]
    schools: list[str]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_codebook(path: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def label(col: str) -> str:
    return COL_LABELS.get(col, col)


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


def build_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Фильтры")
    st.sidebar.caption("Фильтры применяются ко всем вкладкам дашборда.")

    programs = sorted(df["program"].dropna().unique().tolist()) if "program" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    schools = sorted(df["school"].dropna().unique().tolist()) if "school" in df.columns else []

    selected_programs = st.sidebar.multiselect("Программа", programs, default=programs)
    selected_years = st.sidebar.multiselect("Курс", years, default=years)
    selected_schools = st.sidebar.multiselect("Школа", schools, default=schools)

    return FilterState(programs=selected_programs, years=selected_years, schools=selected_schools)


def apply_filters(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    out = df.copy()
    if "program" in out.columns and f.programs:
        out = out[out["program"].isin(f.programs)]
    if "year" in out.columns and f.years:
        out = out[out["year"].isin(f.years)]
    if "school" in out.columns and f.schools:
        out = out[out["school"].isin(f.schools)]
    return out


def add_block_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for block, cols in BLOCKS.items():
        available = [c for c in cols if c in out.columns]
        if available:
            out[f"{block}_mean"] = out[available].mean(axis=1)
    return out


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Обзор")
    st.info("Что здесь: ключевые показатели по выборке и базовые распределения NPS.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ответов", f"{len(df):,}", help="Количество анкет после применения фильтров.")
    c2.metric("Программ", f"{df['program'].nunique() if 'program' in df.columns else 0}", help="Число уникальных программ в текущем срезе.")
    c3.metric("Школ", f"{df['school'].nunique() if 'school' in df.columns else 0}", help="Число уникальных школ в текущем срезе.")
    c4.metric("Медиана NPS", f"{df['nps'].median():.2f}" if "nps" in df.columns else "н/д", help="Центральное значение готовности рекомендовать школу.")

    if "program" in df.columns:
        st.caption("График: размер выборки по программам.")
        counts = df["program"].value_counts().sort_values(ascending=False).reset_index()
        counts.columns = ["program", "n"]
        fig = px.bar(
            counts,
            x="program",
            y="n",
            title="Размер выборки по программам",
            color="n",
            color_continuous_scale="Teal",
            labels={"program": "Программа", "n": "Количество ответов"},
        )
        fig.update_layout(xaxis_tickangle=-25, height=460)
        st.plotly_chart(fig, use_container_width=True)

    if "nps" in df.columns:
        st.caption("График: распределение NPS (0-10).")
        fig = px.histogram(
            df,
            x="nps",
            nbins=11,
            marginal="box",
            title="Распределение NPS",
            color_discrete_sequence=["#1f77b4"],
            labels={"nps": "NPS"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


def render_descriptives(df: pd.DataFrame) -> None:
    st.subheader("Описательная статистика")
    st.info("Что здесь: средние, медианы, разброс и bootstrap доверительные интервалы для выбранных метрик.")

    available_metrics = [c for c in KEY_METRICS if c in df.columns]
    if not available_metrics:
        st.warning("Ключевые метрики не найдены.")
        return

    selected = st.multiselect(
        "Метрики для анализа",
        available_metrics,
        default=available_metrics,
        format_func=label,
        help="Выберите показатели для расчета описательной статистики.",
    )
    if not selected:
        st.info("Выберите хотя бы одну метрику.")
        return

    rows = []
    for col in selected:
        mean, lo, hi, n = bootstrap_mean_ci(df[col])
        vals = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "Метрика": label(col),
                "n": n,
                "Среднее": mean,
                "Медиана": vals.median(),
                "Std": vals.std(),
                "Q25": vals.quantile(0.25),
                "Q75": vals.quantile(0.75),
                "CI 95% нижняя": lo,
                "CI 95% верхняя": hi,
            }
        )
    stat_df = pd.DataFrame(rows).sort_values("Среднее", ascending=False)
    st.caption("Таблица: описательные статистики по выбранным метрикам.")
    st.dataframe(stat_df, use_container_width=True)

    st.caption("График: средние значения с bootstrap 95% доверительными интервалами.")
    fig = px.scatter(
        stat_df,
        x="Среднее",
        y="Метрика",
        error_x=stat_df["CI 95% верхняя"] - stat_df["Среднее"],
        error_x_minus=stat_df["Среднее"] - stat_df["CI 95% нижняя"],
        size="n",
        color="Std",
        color_continuous_scale="RdYlBu_r",
        title="Средние значения и доверительные интервалы",
    )
    fig.update_layout(height=460, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    metric_for_group = st.selectbox(
        "Метрика по программам",
        selected,
        index=0,
        format_func=label,
        help="Для выбранной метрики показываются средние по программам с доверительными интервалами.",
    )
    grp_rows = []
    for prog, g in df.groupby("program"):
        mean, lo, hi, n = bootstrap_mean_ci(g[metric_for_group])
        grp_rows.append({"program": prog, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi})
    grp = pd.DataFrame(grp_rows).sort_values("mean")

    st.caption("График: средние по программам с bootstrap 95% CI.")
    fig = px.scatter(
        grp,
        x="mean",
        y="program",
        error_x=grp["ci_high"] - grp["mean"],
        error_x_minus=grp["mean"] - grp["ci_low"],
        size="n",
        color="mean",
        color_continuous_scale="Viridis",
        title=f"{label(metric_for_group)}: среднее по программам",
        labels={"program": "Программа", "mean": "Среднее"},
    )
    fig.update_layout(height=620, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("График: распределение значений выбранной метрики по программам.")
    fig = px.violin(
        df,
        x="program",
        y=metric_for_group,
        box=True,
        points="all",
        title=f"{label(metric_for_group)}: распределение по программам",
        labels={"program": "Программа", metric_for_group: label(metric_for_group)},
    )
    fig.update_layout(height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_program_comparison(df: pd.DataFrame) -> None:
    st.subheader("Сравнение программ")
    st.info(
        "Что здесь: статистическое сравнение программ по NPS. "
        "Показаны ANOVA, Kruskal-Wallis, Levene и post-hoc Dunn (Holm-коррекция)."
    )

    if "nps" not in df.columns or "program" not in df.columns:
        st.warning("Для этой вкладки нужны столбцы `program` и `nps`.")
        return

    min_n = st.slider(
        "Минимальный размер группы для тестов",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Программы с размером меньше порога исключаются из инференциальных тестов.",
    )
    vc = df["program"].value_counts()
    valid_programs = vc[vc >= min_n].index
    d = df[df["program"].isin(valid_programs)][["program", "nps"]].dropna()
    st.caption(f"В анализе: программ = {d['program'].nunique()}, наблюдений = {len(d)}.")

    if d["program"].nunique() < 2:
        st.info("Недостаточно программ для проведения тестов.")
        return

    group_vals = [g["nps"].values for _, g in d.groupby("program")]
    _, lev_p = stats.levene(*group_vals, center="median")
    anova_model = ols("nps ~ C(program)", data=d).fit()
    anova_tbl = sm.stats.anova_lm(anova_model, typ=2)
    ss_between = anova_tbl.loc["C(program)", "sum_sq"]
    eta2 = ss_between / anova_tbl["sum_sq"].sum()
    _, kw_p = stats.kruskal(*group_vals)
    dunn = sp.posthoc_dunn(d, val_col="nps", group_col="program", p_adjust="holm")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ANOVA p-value", f"{anova_tbl.loc['C(program)', 'PR(>F)']:.4g}", help="Проверка различий средних NPS между программами.")
    k2.metric("Eta^2", f"{eta2:.3f}", help="Доля дисперсии NPS, объясненная фактором программы.")
    k3.metric("Kruskal p-value", f"{kw_p:.4g}", help="Непараметрический тест различий между программами.")
    k4.metric("Levene p-value", f"{lev_p:.4g}", help="Проверка равенства дисперсий между группами.")

    left, right = st.columns(2)
    with left:
        st.caption("Таблица: результаты ANOVA.")
        st.dataframe(anova_tbl, use_container_width=True)
    with right:
        st.caption("Таблица: Dunn post-hoc, p-value с Holm-коррекцией.")
        st.dataframe(dunn, use_container_width=True)

    st.caption("График: распределение NPS по программам.")
    fig = px.box(
        d,
        x="program",
        y="nps",
        points="all",
        color="program",
        title="NPS по программам",
        labels={"program": "Программа", "nps": "NPS"},
    )
    fig.update_layout(showlegend=False, height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_correlations(df: pd.DataFrame) -> None:
    st.subheader("Корреляции")
    st.info("Что здесь: матрица корреляций Спирмена и наиболее сильные положительные/отрицательные связи с выбранной метрикой.")

    numeric = num_cols(df)
    default = [c for c in ["nps", "satisf_overall", "expect_match", "program_mean", "coordinator_mean", "infrastructure_mean"] if c in numeric]
    selected = st.multiselect(
        "Числовые столбцы для матрицы Спирмена",
        numeric,
        default=default if default else numeric[:12],
        format_func=label,
        help="Выберите 2 и более числовых показателя.",
    )
    if len(selected) < 2:
        st.info("Выберите минимум 2 столбца.")
        return

    corr = df[selected].corr(method="spearman")
    corr_display = corr.copy()
    corr_display.index = [label(i) for i in corr_display.index]
    corr_display.columns = [label(i) for i in corr_display.columns]

    st.caption("График: матрица корреляций Спирмена.")
    fig = px.imshow(
        corr_display,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Корреляционная матрица (Spearman)",
    )
    fig.update_layout(height=680)
    st.plotly_chart(fig, use_container_width=True)

    target = st.selectbox(
        "Целевая метрика",
        selected,
        index=selected.index("nps") if "nps" in selected else 0,
        format_func=label,
        help="Показываются коэффициенты Спирмена остальных признаков с выбранной метрикой.",
    )
    with_target = corr[target].drop(target).sort_values(ascending=False).rename("rho").to_frame()
    with_target.index = [label(i) for i in with_target.index]
    st.caption("Таблица: коэффициенты корреляции с целевой метрикой.")
    st.dataframe(with_target, use_container_width=True)

    top_pos = with_target.head(5).reset_index().rename(columns={"index": "Показатель"})
    top_neg = with_target.tail(5).reset_index().rename(columns={"index": "Показатель"})
    cols = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates(subset="Показатель")
    st.caption("График: топ положительных и отрицательных связей.")
    fig = px.bar(
        cols,
        x="rho",
        y="Показатель",
        orientation="h",
        color="rho",
        color_continuous_scale="RdBu",
        title=f"Топ связей с метрикой: {label(target)}",
        labels={"rho": "Коэффициент Спирмена"},
    )
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_priority_matrix(df: pd.DataFrame) -> None:
    st.subheader("Матрица приоритетов")
    st.info(
        "Что здесь: соотношение важности и оценки по критериям. "
        "Чем выше разрыв (важность - оценка), тем выше приоритет для улучшений."
    )
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
                "Критерий": label(score_col).replace(" (оценка)", ""),
                "Средняя оценка": score[valid].mean(),
                "Средняя важность": imp[valid].mean(),
                "Разрыв (важность - оценка)": imp[valid].mean() - score[valid].mean(),
                "n": int(valid.sum()),
            }
        )

    if not rows:
        st.warning("В текущем срезе нет доступных пар score/importance.")
        return

    m = pd.DataFrame(rows)
    score_mid = m["Средняя оценка"].mean()
    imp_mid = m["Средняя важность"].mean()

    st.caption("График: Importance-Performance matrix с разделением на квадранты по средним значениям.")
    fig = px.scatter(
        m,
        x="Средняя оценка",
        y="Средняя важность",
        text="Критерий",
        size="n",
        color="Разрыв (важность - оценка)",
        color_continuous_scale="RdYlGn_r",
        hover_data=["Разрыв (важность - оценка)", "n"],
        title="Матрица важность-оценка",
    )
    fig.add_vline(score_mid, line_dash="dash", line_color="gray")
    fig.add_hline(imp_mid, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Таблица: приоритеты улучшений (по убыванию разрыва).")
    m = m.sort_values("Разрыв (важность - оценка)", ascending=False)
    st.dataframe(m, use_container_width=True)


def render_nps(df: pd.DataFrame) -> None:
    st.subheader("NPS-аналитика")
    st.info("Что здесь: структура NPS (промоутеры/пассивы/критики), вклад программ и NPS по программам.")
    if "nps" not in df.columns:
        st.warning("Столбец `nps` не найден.")
        return

    d = df.copy()
    d["segment"] = np.where(d["nps"] >= 9, "Промоутер", np.where(d["nps"] >= 7, "Пассив", "Критик"))
    seg = d["segment"].value_counts().rename_axis("segment").reset_index(name="n")
    seg["pct"] = seg["n"] / seg["n"].sum() * 100
    nps_value = float(seg.loc[seg["segment"] == "Промоутер", "pct"].sum() - seg.loc[seg["segment"] == "Критик", "pct"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NPS", f"{nps_value:.1f}", help="NPS = %промоутеров - %критиков.")
    c2.metric("% Промоутеров", f"{seg.loc[seg['segment']=='Промоутер', 'pct'].sum():.1f}")
    c3.metric("% Пассивов", f"{seg.loc[seg['segment']=='Пассив', 'pct'].sum():.1f}")
    c4.metric("% Критиков", f"{seg.loc[seg['segment']=='Критик', 'pct'].sum():.1f}")

    left, right = st.columns(2)
    with left:
        st.caption("График: общий разрез сегментов NPS.")
        fig = px.pie(
            seg,
            names="segment",
            values="n",
            hole=0.45,
            color="segment",
            color_discrete_map={"Промоутер": "#2ca02c", "Пассив": "#ffbf00", "Критик": "#d62728"},
        )
        fig.update_layout(title="Структура NPS по сегментам", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "program" in d.columns:
            st.caption("График: состав NPS-сегментов по программам.")
            by_prog = d.groupby(["program", "segment"]).size().rename("n").reset_index()
            tot = d.groupby("program").size().rename("total").reset_index()
            by_prog = by_prog.merge(tot, on="program", how="left")
            by_prog["pct"] = by_prog["n"] / by_prog["total"] * 100
            fig = px.bar(
                by_prog,
                x="program",
                y="pct",
                color="segment",
                barmode="stack",
                title="Состав NPS по программам",
                labels={"program": "Программа", "pct": "%"},
            )
            fig.update_layout(height=430, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    if "program" in d.columns:
        st.caption("График: NPS по программам.")
        nps_by_program = d.groupby("program").apply(
            lambda x: (x["nps"].ge(9).mean() - x["nps"].le(6).mean()) * 100
        ).rename("nps").reset_index()
        fig = px.bar(
            nps_by_program.sort_values("nps"),
            x="nps",
            y="program",
            orientation="h",
            color="nps",
            color_continuous_scale="RdYlGn",
            title="NPS по программам",
            labels={"program": "Программа", "nps": "NPS"},
        )
        fig.update_layout(height=620, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def render_codebook() -> None:
    st.subheader("Кодбук")
    st.info("Что здесь: описание всех колонок и шкал в данных `combined_general_agg.csv`.")
    codebook = load_codebook("docs/codebook.md")
    if not codebook:
        st.warning("Файл `docs/codebook.md` не найден.")
        return

    lines = codebook.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Введение"
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line.replace("## ", "", 1).strip()
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_title, "\n".join(current_body).strip()))

    for title, body in sections:
        with st.expander(title, expanded=(title == "Введение")):
            if body:
                st.markdown(body)
            else:
                st.caption("Раздел пуст.")


def main() -> None:
    st.title("SFQ 2025-26: Статистический дашборд")
    st.caption("Источник данных: `combined_general_agg.csv`")

    df = load_data("combined_general_agg.csv")
    df = safe_numeric(df, num_cols(df))

    if "school" not in df.columns:
        st.warning("В файле нет столбца `school`. Фильтр по школам будет недоступен.")

    filters = build_filters(df)
    dff = apply_filters(df, filters)

    if dff.empty:
        st.error("После применения фильтров данных не осталось.")
        return

    tabs = st.tabs(
        ["Обзор", "Описательные", "Сравнение программ", "Корреляции", "Матрица приоритетов", "NPS", "Кодбук"]
    )
    with tabs[0]:
        render_overview(dff)
    with tabs[1]:
        render_descriptives(dff)
    with tabs[2]:
        render_program_comparison(dff)
    with tabs[3]:
        render_correlations(add_block_features(dff))
    with tabs[4]:
        render_priority_matrix(dff)
    with tabs[5]:
        render_nps(dff)
    with tabs[6]:
        render_codebook()


if __name__ == "__main__":
    main()
