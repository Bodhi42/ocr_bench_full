# OCR Detection Benchmark

Бенчмарк детекции текста на датасете из 113 изображений документов с 4178 размеченными строками (CVAT XML).

## Результаты

![Summary Table](dashboard/00_summary_table.png)

![F1 by Model](dashboard/01_f1_by_model.png)

![Precision by Model](dashboard/02_precision_by_model.png)

![Recall by Model](dashboard/03_recall_by_model.png)

## Модели

| Модель | Детектор | Гранулярность |
|--------|----------|---------------|
| Surya 0.13.1 | Segformer | Строки |
| PaddleOCR PP-OCRv5 server | DB | Строки |
| PaddleOCR PP-OCRv5 mobile | DB | Строки |
| Tesseract 5 | LSTM + merging | Строки (мердж слов) |
| EasyOCR | CRAFT | Фрагменты строк |
| doctr (5 архитектур) | DB / LinkNet | Слова |
| MMOCR DBNet | DBNet | Слова |

## Метрики

- **Precision** — доля предсказанных боксов, совпавших с GT (IoU >= 0.5)
- **Recall** — доля GT боксов, найденных моделью (IoU >= 0.5)
- **F1** — гармоническое среднее precision и recall
- **Area Coverage** — средний максимальный IoU для каждого GT бокса

Matching: венгерский алгоритм (оптимальное назначение).

## Структура

```
src/detectors/       # Обёртки для каждой модели
src/metrics.py       # IoU, matching, precision/recall/F1
src/dashboard.py     # Генерация графиков
scripts/             # Запуск детекторов, метрик, дашборда
predictions/         # Сохранённые боксы моделей (JSON)
results/             # Вычисленные метрики
dashboard/           # Графики и примеры
```
