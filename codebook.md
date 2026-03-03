# Codebook — `combined_general_agg.csv`

Student Experience Questionnaire (SFQ), AY 2025–26 semester 2.
Source: Testograf exports, 18 programmes, 236 respondents.

**Scale conventions**
- `_score` columns: satisfaction rating, typically **1–5** (1 = very dissatisfied, 5 = very satisfied).
- `_imp` columns: importance rating for the same criterion, typically **1–5** (1 = not important, 5 = very important).
- Infrastructure & assessment items: **1–5** unless noted.
- `nps`: likelihood to recommend, **0–10** (Net Promoter Score scale).
- `postgrad_*`: binary checkbox (**1** = selected, empty = not selected).
- `_comment` columns: free-text (open-ended), may contain newlines.

---

## Identifiers & metadata

| Column | Type | Description |
|--------|------|-------------|
| `program` | string | Programme name (from source filename, Russian) |
| `year` | string | Year of study selected by respondent (e.g. `1 курс`, `3 курс`). `NA` for single-cohort programmes that did not include this question |
| `resp_id` | int | Response number within the source file (not globally unique) |
| `date` | datetime | Submission timestamp (`DD.MM.YYYY HH:MM:SS`) |
| `duration_sec` | int | Time spent completing the survey, in seconds |
| `ip` | string | Respondent IP address |
| `source` | string | Distribution channel (e.g. `Прямая ссылка`) |
| `os` | string | Operating system detected |
| `browser` | string | Browser detected |
| `device` | string | Device type (`Компьютер`, `Телефон`, `Планшет`) |

---

## Q1–3 — Global satisfaction

| Column | Scale | Question (RU) |
|--------|-------|---------------|
| `satisf_overall` | 1–5 | Пожалуйста, оцените вашу удовлетворенность образовательным процессом в целом |
| `satisf_teachers` | 1–5 | Насколько вы в целом удовлетворены работой преподавателей в течение семестра? |
| `expect_match` | 1–5 | Насколько ваши ожидания от программы при поступлении совпали с опытом обучения? |

> `satisf_teachers` is absent in some single-cohort programmes (Foundation Art and Design, theatre, management).

---

## Q4 — Faculty team (matrix, score + importance)

| Column | Scale | Criterion |
|--------|-------|-----------|
| `fac_support_score` | 1–5 | У меня была достаточная поддержка преподавательской команды в профессиональном развитии — **satisfaction** |
| `fac_support_imp` | 1–5 | То же — **importance** |
| `fac_clarity_score` | 1–5 | Преподавательская команда объясняет сложные вещи понятным языком — **satisfaction** |
| `fac_clarity_imp` | 1–5 | То же — **importance** |
| `fac_comment` | text | Open feedback about the teaching team |

---

## Q5 — Curator (matrix, score + importance)

| Column | Scale | Criterion |
|--------|-------|-----------|
| `cur_timely_score` | 1–5 | Куратор своевременно сообщает необходимую информацию — **satisfaction** |
| `cur_timely_imp` | 1–5 | То же — **importance** |
| `cur_help_score` | 1–5 | Куратор готов оказать помощь, если у меня появляются вопросы или проблемы — **satisfaction** |
| `cur_help_imp` | 1–5 | То же — **importance** |
| `cur_comment` | text | Open feedback about the curator |

---

## Q7 — Programme content (matrix, score + importance)

| Column | Scale | Criterion |
|--------|-------|-----------|
| `prog_clarity_score` | 1–5 | Мне было понятно содержание дисциплин и их взаимосвязь — **satisfaction** |
| `prog_clarity_imp` | 1–5 | То же — **importance** |
| `prog_deadlines_score` | 1–5 | Я мог(-ла) выполнить задания в нужные сроки и в достаточном объёме — **satisfaction** |
| `prog_deadlines_imp` | 1–5 | То же — **importance** |
| `prog_relevance_score` | 1–5 | Мне было понятно, как дисциплины связаны с профессиональной деятельностью — **satisfaction** |
| `prog_relevance_imp` | 1–5 | То же — **importance** |
| `prog_workload_score` | 1–5 | Количество занятий в неделю было оптимальным — **satisfaction** |
| `prog_workload_imp` | 1–5 | То же — **importance** |
| `prog_comment` | text | Open feedback about the programme |

---

## Q9 — Study office coordinator (matrix, score + importance)

| Column | Scale | Criterion |
|--------|-------|-----------|
| `coord_respect_score` | 1–5 | Координатор уважительно взаимодействовал со мной — **satisfaction** |
| `coord_respect_imp` | 1–5 | То же — **importance** |
| `coord_results_score` | 1–5 | Мои обращения к координатору приводили к ожидаемому результату — **satisfaction** |
| `coord_results_imp` | 1–5 | То же — **importance** |
| `coord_timely_score` | 1–5 | Координатор сообщает информацию об образовательном процессе своевременно — **satisfaction** |
| `coord_timely_imp` | 1–5 | То же — **importance** |
| `coord_help_score` | 1–5 | Координатор готов оказать помощь, если возникают вопросы или проблемы — **satisfaction** |
| `coord_help_imp` | 1–5 | То же — **importance** |
| `coord_comment` | text | Open feedback about the study office coordinator |

---

## Q11–13 — Assessment quality

| Column | Scale | Question (RU) |
|--------|-------|---------------|
| `assess_criteria_timely` | 1–5 | Преподаватели своевременно представили критерии оценивания работ |
| `assess_order_clear` | 1–5 | Преподаватели ясно определили порядок сдачи и оценивания работ |
| `assess_consistent` | 1–5 | Преподаватели проводили оценивание в соответствии с заявленными критериями |
| `assess_comment` | text | Open feedback about assessment |

---

## Q17/18 — Humanities & general education teachers

Two sets of columns exist because the question was named differently in different programme surveys. They measure the same construct (rating of teachers of shared/general disciplines) but the subject lists differ slightly.

### "Общеобразовательные дисциплины" — Foundation Art and Design only

| Column | Scale | Subject |
|--------|-------|---------|
| `gen_ed_critical_thinking` | 1–5 | Критическое мышление |
| `gen_ed_history` | 1–5 | История России |
| `gen_ed_foreign_lang` | 1–5 | Иностранный язык |
| `gen_ed_safety` | 1–5 | Безопасность жизнедеятельности |
| `gen_ed_statehood` | 1–5 | Основы российской государственности |

### "Гуманитарные дисциплины" — all other programmes

| Column | Scale | Subject |
|--------|-------|---------|
| `hum_critical_thinking` | 1–5 | Критическое мышление |
| `hum_history` | 1–5 | История России |
| `hum_statehood` | 1–5 | Основы российской государственности |
| `hum_foreign_lang` | 1–5 | Иностранный язык |
| `hum_philosophy` | 1–5 | Философия *(2nd–4th year cohorts only)* |
| `hum_communication` | 1–5 | Теория и практика коммуникации *(2nd–4th year cohorts only)* |

---

## Seminar / lecture comment

| Column | Type | Description |
|--------|------|-------------|
| `seminar_comment` | text | Free-text feedback on lectures and seminar tracks |

---

## Q19–25 — Campus infrastructure

| Column | Scale | Item |
|--------|-------|------|
| `infra_library` | 1–5 | Доступность и работа библиотеки |
| `infra_wellbeing` | 1–5 | Доступность и качество сервиса Wellbeing |
| `infra_food` | 1–5 | Доступность и качество еды на кампусе |
| `infra_software` | 1–5 | Работоспособность программного обеспечения в аудиториях |
| `infra_equipment` | 1–5 | Работоспособность оборудования в аудиториях |
| `infra_classrooms` | 1–5 | Комфорт пребывания в аудиториях |
| `infra_workshops` | 1–5 | Комфорт пребывания в мастерских / ресурсных центрах / репетиционных комнатах |

---

## Q26 — NPS & final comment

| Column | Scale | Description |
|--------|-------|-------------|
| `nps` | 0–10 | Насколько вероятно, что вы порекомендуете школу (Net Promoter Score scale) |
| `comment_final` | text | Additional comments and suggestions |

---

## Additional questions (present in a subset of programmes)

These items appear only in surveys for multi-year programmes or graduating cohorts.

| Column | Scale | Programmes | Question (RU) |
|--------|-------|------------|---------------|
| `prev_sem_relevance` | 1–5 | Most multi-year programmes | Насколько дисциплины предыдущего семестра помогли вам в освоении учебного материала текущего семестра? |
| `skill_confidence` | 1–5 | Multi-year programmes, years 3+ | Насколько уверенно вы можете применить полученные знания и навыки в реальной профессиональной практике? |
| `postgrad_masters` | 0/1 | Graduating cohorts | Планируете продолжение обучения в магистратуре или дополнительном образовании |
| `postgrad_same_field` | 0/1 | Graduating cohorts | Планируете работу в выбранной сфере |
| `postgrad_other_field` | 0/1 | Graduating cohorts | Планируете работу в другой сфере |
| `postgrad_other` | 0/1 | Graduating cohorts | Другой вариант (не уточнён в анкете) |

---

## Related files

| File | Description |
|------|-------------|
| `combined.csv` | All 310 columns including individual teacher ratings (wide format) |
| `combined_general.csv` | 72 columns, original Russian column names |
| `combined_general_agg.csv` | **This file** — 72 columns, short Latin names |
| `combined_teachers.csv` | Long-format teacher ratings: `program`, `year`, `resp_id`, `Преподаватель`, `Оценка` |
