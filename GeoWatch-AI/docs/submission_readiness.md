# Submission readiness — 8 сентября 2026

## Реализовано и проверено локально

- Streamlit workspace с dashboard, анализом, temporal comparison, review queue, cases, history и quality pages.
- Проверка качества изображений и совместимости temporal-пары.
- GeoTIFF alignment pipeline с общим CRS/grid/NoData и отчётом качества; JPG/PNG translation-only registration.
- Постоянные IDs comparisons/events, review history, связи с кейсами и SQLite persistence.
- Идемпотентные alerts, persistent jobs с retry/cancel/dedupe и API endpoints.
- ZIP-экспорт кейса с JSON-manifest и доступными evidence-файлами.
- Dataset preparation для SpaceNet 7 с AOI-separated split и provenance manifest.
- Аудит checkpoint/dataset evidence и явный статус `candidate_unverified`.
- 78 автоматических тестов проходят; Python-модули компилируются.

## Экспериментально или частично

- Temporal demo использует синтетические изображения и проверяет продуктовый flow, а не качество change-detection модели.
- Очередь заданий сохраняется и управляется через API, но Earth Engine worker/scheduler ещё не исполняет задания автоматически.
- GeoTIFF код покрыт unit-тестами и безопасной деградацией; runtime без `rasterio` не выполняет реальную репроекцию.
- Qwen является необязательным аналитическим слоем и по умолчанию выключен.

## Внешние блокеры до подтверждённой конкурсной оценки

1. Получить официальный DOTA4 dataset и построить scene-separated train/val/test с четырьмя классами.
2. Установить GPU-окружение с `torch` и `ultralytics` либо завершить Colab run.
3. Обучить/выбрать checkpoint по validation и один раз оценить на независимом test split.
4. Скачать реальные SpaceNet 7 AOI, подготовить пары и обучить building-change baseline.
5. Настроить Google Cloud Project и Earth Engine authentication для реального получения сцен.
6. Установить `rasterio` для GeoTIFF reprojection в целевом runtime.

## Нельзя заявлять до фактического подтверждения

- финальную точность, mAP/F1/IoU или скорость на целевом компьютере;
- реальное автоматическое обнаружение строительства;
- работающий cloud-monitoring без запущенного worker;
- публичный deployment, публикацию GitHub или сдачу решения.

## Acceptance для финальной сдачи

- два последовательных live demo без ручного изменения файлов;
- независимые DOTA4 test metrics с manifest, seed и checkpoint hash;
- минимум три реальных проверенных кейса, включая negative/сложный пример;
- проверенный ZIP проекта без секретов и больших исходных датасетов;
- видео и скриншоты, снятые после фиксации финального интерфейса.

