"""Process semester 2 raw xlsx files into production-ready CSV.

Mirrors the pipeline from notebooks/analysis.ipynb but reads xlsx instead of csv.
Outputs:
  - data/processed/combined_general_sem2.csv   (intermediate, Russian columns)
  - data/processed/combined_teachers_sem2.csv   (intermediate, long-format)
  - data/processed/combined_teachers_agg_sem2.csv (final, Latin columns + school)
  - combined_general_agg_sem2.csv               (final, Latin columns + school)
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Coalescing duplicate (year-branched) survey columns with combine_first emits a
# cosmetic pandas FutureWarning about empty-entry concatenation; output is correct.
warnings.filterwarnings(
    "ignore",
    message="The behavior of array concatenation with empty entries is deprecated",
    category=FutureWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "program_surveys_sem2"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METADATA_COLS = {
    "Номер ответа",
    "Дата ответа",
    "Время выполнения",
    "IP - адрес",
    "Источник распространения",
    "Операционная система",
    "Браузер",
    "Устройство",
}

PROGRAM_NAME_OVERRIDES = {
    "Менеджмент в креативных индустриях (Цифровой маркетинг 2 сем)":
        "Менеджмент в креативных индустриях (сетевая программа)",
}


def strip_qnum(col: str) -> str:
    return re.sub(r"^\d+\.\s*", "", col).strip()


def is_teacher_rating_col(col: str) -> bool:
    s = strip_qnum(col)
    return (
        "оцените работу преподавателей" in s.lower()
        or "оцените работу преподавателей по критериям" in s.lower()
    ) and " - " in s


def is_individual_teacher_col(col: str) -> bool:
    s = unicodedata.normalize("NFKC", strip_qnum(col)).strip()
    low = s.lower()
    if " - " not in s:
        return False
    return (
        low.startswith("оцените преподавателей по дисциплине")
        or low.startswith("оцените преподавателей дисциплины")
        or low.startswith("оцените преподавателей по ")
        or low.startswith("оцените преподавателя по ")
    )


def get_teacher_name(col: str) -> str:
    s = unicodedata.normalize("NFKC", strip_qnum(col))
    return s.rsplit(" - ", 1)[-1].strip()


def is_course_year_col(col: str) -> bool:
    return "выберите курс" in col.lower()


def is_choose_program_col(col: str) -> bool:
    return "выберите программу" in col.lower()


def canonical(col: str) -> str:
    if col in METADATA_COLS:
        return col
    if is_course_year_col(col):
        return "Курс обучения"
    if is_choose_program_col(col):
        return "Программа (из анкеты)"
    if is_teacher_rating_col(col):
        return f"Оценка преподавателя: {get_teacher_name(col)}"
    if is_individual_teacher_col(col):
        return f"Оценка преподавателя: {get_teacher_name(col)}"
    return strip_qnum(col)


def extract_program_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"\s*\(2\s*сем\)", "", name).strip()
    name = re.sub(r"\s*\(2сем\)", "", name).strip()
    return PROGRAM_NAME_OVERRIDES.get(name, name)


def load_program_xlsx(filepath: Path) -> pd.DataFrame:
    program_name = extract_program_name(filepath.name)
    df = pd.read_excel(filepath)

    groups: dict[str, list[str]] = defaultdict(list)
    for col in df.columns:
        groups[canonical(col)].append(col)

    out_series = {}
    for cname, orig_cols in groups.items():
        s = df[orig_cols[0]].copy()
        for c in orig_cols[1:]:
            other = df[c]
            # Skip all-empty duplicates: combine_first on an all-NA Series is a
            # no-op but triggers a pandas FutureWarning about empty concatenation.
            if other.notna().any():
                s = s.combine_first(other)
        out_series[cname] = s

    out = pd.DataFrame(out_series)
    out.insert(0, "Программа", program_name)

    if "Курс обучения" not in out.columns:
        out.insert(1, "Курс обучения", None)
    else:
        out["Курс обучения"] = out["Курс обучения"].astype(object)

    if "Программа (из анкеты)" in out.columns:
        out = out.drop(columns=["Программа (из анкеты)"])

    return out


COL_MAP = {
    "Программа": "program",
    "Курс обучения": "year",
    "Номер ответа": "resp_id",
    "Дата ответа": "date",
    "Время выполнения": "duration_sec",
    "IP - адрес": "ip",
    "Источник распространения": "source",
    "Операционная система": "os",
    "Браузер": "browser",
    "Устройство": "device",
    "Пожалуйста, оцените вашу удовлетворенность образовательным процессом в целом":
        "satisf_overall",
    "Насколько вы в целом удовлетворены работой преподавателей в течение семестра?":
        "satisf_teachers",
    "Насколько ваши ожидания от программы при поступлении совпали с опытом обучения?":
        "expect_match",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - У меня была достаточная поддержка преподавательской команды в профессиональном развитии - Оценка":
        "fac_support_score",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - У меня была достаточная поддержка преподавательской команды в профессиональном развитии - Важность критерия":
        "fac_support_imp",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - Преподавательская команда объясняет сложные вещи понятным языком - Оценка":
        "fac_clarity_score",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - Преподавательская команда объясняет сложные вещи понятным языком - Важность критерия":
        "fac_clarity_imp",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор своевременно сообщает необходимую информацию об образовательном процессе - Оценка":
        "cur_timely_score",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор своевременно сообщает необходимую информацию об образовательном процессе - Важность критерия":
        "cur_timely_imp",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор готов оказать помощь, если у меня появляются вопросы или проблемы - Оценка":
        "cur_help_score",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор готов оказать помощь, если у меня появляются вопросы или проблемы - Важность критерия":
        "cur_help_imp",
    "Здесь вы можете оставить более подробную обратную связь о кураторе":
        "cur_comment",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно содержание дисциплин, которые я изучал(а) в течение семестра, и их взаимосвязь между собой - Оценка":
        "prog_clarity_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно содержание дисциплин, которые я изучал(а) в течение семестра, и их взаимосвязь между собой - Важность критерия":
        "prog_clarity_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Я мог(-ла) выполнить задания преподавателей в нужные сроки в достаточном объеме - Оценка":
        "prog_deadlines_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Я мог(-ла) выполнить задания преподавателей в нужные сроки в достаточном объеме - Важность критерия":
        "prog_deadlines_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно, как изучаемые дисциплины связаны с моей сферой профессиональной деятельности - Оценка":
        "prog_relevance_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно, как изучаемые дисциплины связаны с моей сферой профессиональной деятельности - Важность критерия":
        "prog_relevance_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне кажется, что за прошедший семестр количество занятий в неделю было оптимальным - Оценка":
        "prog_workload_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне кажется, что за прошедший семестр количество занятий в неделю было оптимальным - Важность критерия":
        "prog_workload_imp",
    "Здесь вы можете оставить более подробную обратную связь о программе":
        "prog_comment",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор уважительно взаимодействовал со мной - Оценка":
        "coord_respect_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор уважительно взаимодействовал со мной - Важность критерия":
        "coord_respect_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Мои обращения к координатору приводили к ожидаемому результату - Оценка":
        "coord_results_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Мои обращения к координатору приводили к ожидаемому результату - Важность критерия":
        "coord_results_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор сообщает необходимую информацию об образовательном процессе своевременно - Оценка":
        "coord_timely_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор сообщает необходимую информацию об образовательном процессе своевременно - Важность критерия":
        "coord_timely_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор готов оказать помощь, если возникают вопросы или проблемы - Оценка":
        "coord_help_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор готов оказать помощь, если возникают вопросы или проблемы - Важность критерия":
        "coord_help_imp",
    "Здесь вы можете оставить более подробную обратную связь о взаимодействии с координатором Учебного отдела":
        "coord_comment",
    "Преподаватели своевременно представили критерии оценивания работ":
        "assess_criteria_timely",
    "Преподаватели ясно определили порядок сдачи и оценивания работ":
        "assess_order_clear",
    "Преподаватели проводили оценивание работ в соответствии с заявленными критериями":
        "assess_consistent",
    "Здесь вы можете оставить подробную обратную связь про оценивание на программе":
        "assess_comment",
    "Здесь вы можете оставить более подробную обратную связь о преподавательской команде":
        "fac_comment",
    "Оцените преподавателей по общеобразовательным дисциплинам - Критическое мышление":
        "gen_ed_critical_thinking",
    "Оцените преподавателей по общеобразовательным дисциплинам - История России":
        "gen_ed_history",
    "Оцените преподавателей по общеобразовательным дисциплинам - Иностранный язык":
        "gen_ed_foreign_lang",
    "Оцените преподавателей по общеобразовательным дисциплинам - Безопасность жизнедеятельности":
        "gen_ed_safety",
    "Оцените преподавателей по общеобразовательным дисциплинам - Основы российской государственности":
        "gen_ed_statehood",
    "Оцените преподавателей по гуманитарным дисциплинам: - Критическое мышление":
        "hum_critical_thinking",
    "Оцените преподавателей по гуманитарным дисциплинам: - История России":
        "hum_history",
    "Оцените преподавателей по гуманитарным дисциплинам: - Основы российской государственности":
        "hum_statehood",
    "Оцените преподавателей по гуманитарным дисциплинам: - Иностранный язык":
        "hum_foreign_lang",
    "Оцените преподавателей по гуманитарным дисциплинам: - Философия":
        "hum_philosophy",
    "Оцените преподавателей по гуманитарным дисциплинам: - Теория и практика коммуникации":
        "hum_communication",
    "Оставьте комментарий отдельно о лекционных и семинарских занятиях (укажите преподавателя или семинарский трек). Насколько разнообразными и полезными были семинарские треки:":
        "seminar_comment",
    "Оцените доступность и работу библиотеки": "infra_library",
    "Оцените доступность и качество работы сервиса Wellbeing": "infra_wellbeing",
    "Оцените доступность и качество еды на кампусе": "infra_food",
    "Оцените работоспособность программного обеспечения в аудиториях": "infra_software",
    "Оцените работоспособность оборудования в аудиториях": "infra_equipment",
    "Оцените комфорт пребывания в аудиториях": "infra_classrooms",
    "Оцените комфорт пребывания в мастерских / ресурсных центрах / репетиционных комнатах":
        "infra_workshops",
    "Насколько вероятно, что вы порекомендуете школу друзьям, коллегам или родным, которые планируют обучение":
        "nps",
    "Есть ли у вас дополнительные комментарии или предложения для нас?":
        "comment_final",
    "Насколько дисциплины предыдущего семестра помогли вам в освоении учебного материала текущего семестра?":
        "prev_sem_relevance",
    "Насколько уверенно вы можете применить полученные знания и навыки в реальной профессиональной практике?":
        "skill_confidence",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Продолжение обучения в магистратуре или дополнительном образовании":
        "postgrad_masters",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Работа в выбранной сфере":
        "postgrad_same_field",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Работа в другой сфере":
        "postgrad_other_field",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? -":
        "postgrad_other",
    # New sem2 questions
    "Вы достигли того, ради чего пришли на эту программу?":
        "goal_achieved",
    "Насколько вам было легко ориентироваться в процессе обучения (расписание, задания, дедлайны)?":
        "navigation_ease",
    "Насколько комфортно вы чувствовали себя на занятиях?":
        "class_comfort",
    # NB: "Оставьте свои контакты..." is intentionally NOT mapped — it is PII
    # (student emails / names) and is dropped, matching the semester-1 pipeline.
}


def normalize_text(s):
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKC", str(s)).strip().lower().replace("ё", "е")
    return " ".join(s.split())


def enrich_from_registry(df: pd.DataFrame) -> pd.DataFrame:
    registry_path = REFERENCE_DIR / "Реестр программ.csv"
    if not registry_path.exists():
        print(f"WARNING: Registry not found: {registry_path}")
        df["school"] = np.nan
        return df

    registry = pd.read_csv(registry_path, sep=";")
    if not {"Программа", "Школа", "Курс"}.issubset(registry.columns):
        print("WARNING: Registry missing required columns")
        df["school"] = np.nan
        return df

    registry = registry.copy()
    registry["program_norm"] = registry["Программа"].map(normalize_text)
    registry["Курс"] = pd.to_numeric(registry["Курс"], errors="coerce")

    df["program_norm"] = df["program"].map(normalize_text)

    school_map = (
        registry[["program_norm", "Школа"]]
        .dropna(subset=["program_norm"])
        .drop_duplicates()
        .groupby("program_norm")["Школа"]
        .agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    )
    df["school"] = df["program_norm"].map(school_map)

    course_candidates = (
        registry[["program_norm", "Курс"]]
        .dropna(subset=["program_norm", "Курс"])
        .drop_duplicates()
        .groupby("program_norm")["Курс"]
        .apply(lambda s: sorted(set(int(v) for v in s if pd.notna(v))))
    )
    single_course_map = {k: v[0] for k, v in course_candidates.items() if len(v) == 1}

    missing_year_mask = df["year"].isna() | (df["year"].astype(str).str.strip() == "")
    inferred_year = (
        df.loc[missing_year_mask, "program_norm"]
        .map(single_course_map)
        .map(lambda x: f"{int(x)} курс" if pd.notna(x) else np.nan)
    )
    df.loc[missing_year_mask, "year"] = inferred_year

    print(f"  Missing school after merge: {int(df['school'].isna().sum())}")
    print(f"  Missing year after fill: {int((df['year'].isna() | (df['year'].astype(str).str.strip() == '')).sum())}")

    df = df.drop(columns=["program_norm"])
    return df


def main():
    print("=" * 60)
    print("Processing semester 2 data")
    print("=" * 60)

    # ── 1. Load and combine xlsx files ──────────────────────────────────────
    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        print(f"ERROR: No xlsx files found in {RAW_DIR}")
        return

    dfs = []
    for fp in xlsx_files:
        df = load_program_xlsx(fp)
        print(f"  loaded: {fp.name} → {extract_program_name(fp.name)} ({len(df)} rows, {len(df.columns)} cols)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nCombined: {combined.shape[0]} rows × {combined.shape[1]} columns")

    # ── 2. Reorder columns ──────────────────────────────────────────────────
    meta = ["Программа", "Курс обучения"] + [c for c in METADATA_COLS if c in combined.columns]
    teacher_cols = sorted([c for c in combined.columns if c.startswith("Оценка преподавателя:")])
    other_cols = [c for c in combined.columns if c not in set(meta) and c not in set(teacher_cols)]
    combined = combined[meta + other_cols + teacher_cols]

    # ── 3. Split into general + teachers ────────────────────────────────────
    general_cols = [c for c in combined.columns if c not in teacher_cols]
    df_general = combined[general_cols]
    df_general.to_csv(PROCESSED_DIR / "combined_general_sem2.csv", index=False)
    print(f"\ncombined_general_sem2.csv — {df_general.shape[0]} rows × {df_general.shape[1]} cols")

    id_cols = ["Программа", "Курс обучения", "Номер ответа"]
    id_cols = [c for c in id_cols if c in combined.columns]
    df_teachers = (
        combined[id_cols + teacher_cols]
        .melt(id_vars=id_cols, var_name="Преподаватель", value_name="Оценка")
        .dropna(subset=["Оценка"])
    )
    df_teachers["Преподаватель"] = df_teachers["Преподаватель"].str.removeprefix("Оценка преподавателя: ")
    df_teachers.to_csv(PROCESSED_DIR / "combined_teachers_sem2.csv", index=False)
    print(f"combined_teachers_sem2.csv — {df_teachers.shape[0]} rows × {df_teachers.shape[1]} cols")
    print(f"  Unique teachers: {df_teachers['Преподаватель'].nunique()}")

    # ── 4. Rename to Latin schema ───────────────────────────────────────────
    unmapped = [c for c in df_general.columns if c not in COL_MAP and not c.startswith("Оценка преподавателя:")]
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped columns (will be dropped):")
        for c in unmapped:
            print(f"  {repr(c)}")

    mapped_cols = {c: COL_MAP[c] for c in df_general.columns if c in COL_MAP}
    df_agg = df_general.rename(columns=mapped_cols)
    keep_cols = list(mapped_cols.values())
    extra_cols = [c for c in df_agg.columns if c not in keep_cols]
    if extra_cols:
        df_agg = df_agg.drop(columns=extra_cols)

    print(f"\nRenamed to Latin schema: {df_agg.shape[0]} rows × {df_agg.shape[1]} cols")

    # ── 5. Enrich from registry ─────────────────────────────────────────────
    df_agg = enrich_from_registry(df_agg)

    out_path = PROJECT_ROOT / "combined_general_agg_sem2.csv"
    df_agg.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.name} ({df_agg.shape[0]} rows × {df_agg.shape[1]} cols)")

    # ── 6. Teachers: Latin schema + registry ────────────────────────────────
    teacher_col_map = {
        "Программа": "program",
        "Курс обучения": "year",
        "Номер ответа": "resp_id",
        "Преподаватель": "teacher",
        "Оценка": "rating",
    }
    df_t = df_teachers.rename(columns=teacher_col_map)
    df_t["rating"] = pd.to_numeric(df_t["rating"], errors="coerce")

    df_t = enrich_from_registry(df_t)

    df_t = df_t[["program", "school", "year", "resp_id", "teacher", "rating"]]
    out_t = PROCESSED_DIR / "combined_teachers_agg_sem2.csv"
    df_t.to_csv(out_t, index=False)
    print(f"Saved {out_t.name} ({df_t.shape[0]} rows × {df_t.shape[1]} cols)")

    # ── 7. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Programs: {df_agg['program'].nunique()}")
    print(f"Responses: {len(df_agg)}")
    print(f"Teachers (unique): {df_t['teacher'].nunique()}")
    print(f"Teacher ratings: {len(df_t)}")
    print(f"\nColumns in final general file:")
    for c in df_agg.columns:
        n_valid = df_agg[c].notna().sum()
        print(f"  {c}: {n_valid}/{len(df_agg)} non-null")


if __name__ == "__main__":
    main()
