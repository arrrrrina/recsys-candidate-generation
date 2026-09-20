# Candidate Generation for Service Search

Решение задачи кандидатогенерации для поиска услуг. Для каждого поискового
запроса система выбирает до 50 объявлений из корпуса; основная метрика —
`Recall@50`.

## Подход

Финальный пул объединяет четыре retrieval-канала:

- **Local E5** — семантический поиск в той же категории и локации;
- **Local TF-IDF** — точные лексические совпадения в той же категории и локации;
- **Nearby E5** — семантический поиск в соседних локациях радиусом 100 км;
- **Global E5** — глобальный fallback по всему корпусу.

Кандидаты агрегируются Logistic Regression по рангам, similarity-score,
присутствию в каналах и совпадению категории и локации. Локальный
`Recall@50` агрегатора — `0.8096`, результат лучшей benchmark-отправки —
`0.722516`.

## Структура

- `notebooks/01_dataset_inspection.ipynb` — знакомство с данными;
- `notebooks/02_pipeline_development.ipynb` — локальная валидация, retrieval-каналы и обучение агрегатора;
- `notebooks/03_benchmark_submission.ipynb` — полный inference и создание `answer.csv`;
- `notebooks/04_behavioral_retrieval.ipynb` — дополнительные эксперименты с историей и микрокатегориями;
- `notebooks/05_finetuning_e5.ipynb` — эксперимент с дообучением E5;
- `src/` — переиспользуемые функции подготовки данных, retrieval и агрегации;
- `tests/` — небольшие тесты метрик и формата ответа.

## Запуск

Создать окружение и установить зависимости:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Файлы `train.parquet`, `benchmark_queries.parquet` и
`benchmark_items.parquet` нужно положить в `dataset/`.

Затем выполнить блокноты в таком порядке:

1. `02_pipeline_development.ipynb` — создаёт локальные артефакты и обучает агрегатор.
2. `03_benchmark_submission.ipynb` — создаёт и проверяет `answer.csv`.

Тяжёлые результаты сохраняются в `artifacts/`. При повторном запуске готовые
TF-IDF-матрицы, E5-эмбеддинги и retrieval-выдачи загружаются с диска.
