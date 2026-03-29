# OCR Benchmark

Бенчмарк детекции и распознавания текста на датасете из 113 изображений документов с 4587 размеченными строками.

**Ground Truth**: Yandex Vision OCR (CVAT XML формат).

> Предыдущая разметка была сделана полуавтоматически на основе Surya, что давало неадекватно завышенные результаты для Surya. Текущий GT сгенерирован Yandex Vision OCR для более объективного сравнения.

## Detection

![Summary Table](dashboard/00_summary_table.png)

![F1 by Model](dashboard/01_f1_by_model.png)

![Precision by Model](dashboard/02_precision_by_model.png)

![Recall by Model](dashboard/03_recall_by_model.png)

> **Примечание**: Yandex Vision показывает идеальные метрики (F1=1.000), т.к. является источником GT. Результаты detection зависят от стиля bounding box (tight vs padded), что влияет на IoU — см. раздел "Особенности" ниже.

## Recognition

![Recognition Summary](dashboard/recognition/00_summary_table.png)

![WER by Model](dashboard/recognition/01_wer_by_model.png)

## Топ-5 Recognition (WER micro, ниже = лучше)

| # | Модель | WER micro | WER macro | Тип |
|---|--------|-----------|-----------|-----|
| 1 | Qwen 2.5 VL 72B | 0.164 | 0.264 | VLM (API) |
| 2 | Qwen3 VL 32B | 0.176 | 0.240 | VLM (API) |
| 3 | Qwen 2.5 VL 32B | 0.193 | 0.277 | VLM (API) |
| 4 | Surya 0.13.1 | 0.200 | 0.340 | OCR |
| 5 | PaddleOCR cyrillic | 0.211 | 0.305 | OCR |

## Модели

### Detection

| Модель | Детектор | Гранулярность |
|--------|----------|---------------|
| Yandex Vision | Cloud API | Строки |
| Surya 0.13.1 | Segformer | Строки |
| PaddleOCR PP-OCRv5 server | DB | Строки |
| PaddleOCR PP-OCRv5 mobile | DB | Строки |
| Tesseract 5 | LSTM + merging | Строки (мердж слов) |
| EasyOCR | CRAFT | Фрагменты строк |
| doctr (5 архитектур) | DB / LinkNet | Слова |

### Recognition — классические OCR

| Модель | Движок | Примечание |
|--------|--------|------------|
| Surya 0.13.1 | Собственный | GPU |
| PaddleOCR cyrillic | PP-OCRv4 (rs_cyrillic) | CPU |
| PaddleOCR eslav | PP-OCRv4 (ru) | CPU |
| Tesseract 5 | LSTM | CPU |
| EasyOCR | CRNN | GPU |

### Recognition — VLM (через OpenRouter API)

| Модель | Провайдер | WER micro |
|--------|-----------|-----------|
| Qwen 2.5 VL 72B | Parasail | 0.164 |
| Qwen3 VL 32B | Parasail | 0.176 |
| Qwen 2.5 VL 32B | Parasail | 0.193 |

### Recognition — Docling (OCR × Layout)

24 комбинации: 4 OCR-движка × 6 layout-моделей. Лучшие:

| Комбинация | WER micro |
|-----------|-----------|
| tesseract + egret_large | 0.426 |
| tesseract_cli + heron | 0.426 |
| tesseract + heron | 0.431 |

Layout-модель практически не влияет на качество распознавания текста (разница в пределах 2%).

### Recognition — Docling VLM

| Модель | Размер | WER micro |
|--------|--------|-----------|
| DeepSeek-OCR | 3B | 0.600 |
| Granite-Docling | 258M | 0.903 |
| SmolDocling, Dolphin, GOT-OCR, Granite-Vision, Qwen 3B | <3B | 1.000 |

Маленькие VLM-модели не справляются с русскоязычными документами.

### Recognition — другие

| Модель | Тип | WER micro |
|--------|-----|-----------|
| Yandex Vision | Cloud API | 0.000 (GT) |
| Docling (tesseract) | Pipeline | 0.431 |
| dots.ocr 1.7B | VLM (vLLM) | 0.503 |

## Метрики

### Detection
- **Precision** — доля предсказанных боксов, совпавших с GT (IoU >= 0.5)
- **Recall** — доля GT боксов, найденных моделью (IoU >= 0.5)
- **F1** — гармоническое среднее precision и recall
- **Area Coverage** — средний максимальный IoU для каждого GT бокса

Matching: венгерский алгоритм (оптимальное назначение).

### Recognition
- **WER (micro)** — Word Error Rate на объединённом тексте всех изображений
- **WER (macro)** — среднее WER по изображениям

## Особенности

### Влияние GT на метрики
Yandex Vision создаёт **tight bounding boxes** (средняя высота ~16px), в то время как PaddleOCR — padded (средняя высота ~28px). Это приводит к заниженным IoU для PaddleOCR при detection, хотя текст находится корректно. Для честного сравнения detection рекомендуется использовать порог IoU=0.3 или ориентироваться на Area Coverage.

### VLM vs классический OCR
Большие VLM модели (Qwen 2.5 VL 72B, Qwen3 VL 32B) показывают лучшие результаты recognition, но требуют API-доступа и значительно медленнее. Маленькие VLM (<5B) через Docling не справляются с кириллицей.

## Структура

```
src/detectors/       # Обёртки детекторов
src/recognizers/     # Обёртки распознавателей
src/metrics.py       # IoU, matching, precision/recall/F1
src/recognition_metrics.py  # WER
src/yandex_vision.py # Клиент Yandex Vision API
src/qwen_vl.py       # Клиент OpenRouter для Qwen VL
scripts/             # Запуск моделей, метрик, дашбордов
predictions/         # Сохранённые результаты моделей (JSON)
results/             # Вычисленные метрики
dashboard/           # Графики detection + recognition
```
