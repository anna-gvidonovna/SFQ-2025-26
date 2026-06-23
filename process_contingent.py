"""Parse Контингент.xlsx (enrollment) into a clean reference for response-rate analysis.

Output: data/reference/contingent.csv with columns:
    program  — название программы, совпадающее с `program` в данных опроса
    year     — курс (`1 курс`..`5 курс`) или пусто для одно-курсовых/неразбиваемых
    school   — школа (из реестра, по программе)
    contingent — число студентов (численность контингента)

Контингент — снимок численности, применяется к обоим семестрам.
Подготовительные программы (Foundation pre-*, портфолио, интенсивы и т.п.)
в опросе не участвуют и в выходной файл не попадают.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
CONTINGENT_XLSX = REFERENCE_DIR / "Контингент.xlsx"

# Программы опроса, которым в файле контингента соответствует особое название.
ALIAS = {
    "foundation art and design/ дизайн": "Foundation Art and Design",
    "цифровой маркетинг": "Менеджмент в креативных индустриях (сетевая программа)",
}


def normalize_text(s) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower().replace("ё", "е")
    return " ".join(s.split())


def split_year(name: str) -> tuple[str, str | None]:
    """'Кинопроизводство 2 курс' -> ('Кинопроизводство', '2 курс')."""
    m = re.search(r"(\d)\s*курс\s*$", str(name).strip())
    if m:
        prog = re.sub(r"\d\s*курс\s*$", "", str(name)).strip()
        return prog, f"{m.group(1)} курс"
    return str(name).strip(), None


def survey_program_index() -> dict[str, str]:
    """Map normalized program name -> canonical survey program name."""
    progs: set[str] = set()
    for fn in ["combined_general_agg.csv", "combined_general_agg_sem2.csv"]:
        p = PROJECT_ROOT / fn
        if p.exists():
            progs |= set(pd.read_csv(p, usecols=["program"])["program"].dropna())
    return {normalize_text(p): p for p in progs}


def school_map_from_registry() -> dict[str, str]:
    reg_path = REFERENCE_DIR / "Реестр программ.csv"
    if not reg_path.exists():
        return {}
    reg = pd.read_csv(reg_path, sep=";")
    reg["program_norm"] = reg["Программа"].map(normalize_text)
    return (
        reg[["program_norm", "Школа"]]
        .dropna()
        .drop_duplicates()
        .groupby("program_norm")["Школа"]
        .agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
        .to_dict()
    )


def main() -> None:
    cont = pd.read_excel(CONTINGENT_XLSX).dropna(subset=["Контингент"])
    survey_norm = survey_program_index()
    school_map = school_map_from_registry()

    rows = []
    unmatched = []
    for _, r in cont.iterrows():
        prog_raw, year = split_year(r["Программа"])
        pn = normalize_text(prog_raw)
        target = ALIAS.get(pn) or survey_norm.get(pn)
        if target is None:
            unmatched.append((r["Программа"], r["Контингент"]))
            continue
        rows.append(
            {
                "program": target,
                "year": year,
                "school": school_map.get(normalize_text(target), np.nan),
                "contingent": int(r["Контингент"]),
            }
        )

    out = pd.DataFrame(rows)
    # Сворачиваем дубли (program, year) на случай повторов в источнике.
    out = (
        out.groupby(["program", "year"], dropna=False, as_index=False)
        .agg({"school": "first", "contingent": "sum"})
    )

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFERENCE_DIR / "contingent.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path} ({len(out)} rows)")
    print(f"\nКонтингент по программам (всего {int(out['contingent'].sum())} студентов):")
    print(out.groupby("program")["contingent"].sum().to_string())

    survey_progs = set(survey_norm.values())
    matched_progs = set(out["program"])
    missing = survey_progs - matched_progs
    if missing:
        print("\nПрограммы опроса БЕЗ контингента:")
        for p in sorted(missing):
            print(f"  {p}")
    else:
        print("\nВсе программы опроса сопоставлены с контингентом.")

    print(f"\nНесопоставленные строки контингента (подготовительные/прочие): {len(unmatched)}")
    for name, n in unmatched:
        print(f"  {int(n):>4}  {name}")


if __name__ == "__main__":
    main()
