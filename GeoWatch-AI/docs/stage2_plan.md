# Stage 2 — конкурсная ML-версия

Цель: заменить DOTA8 smoke-модель на воспроизводимый baseline по четырём нейтральным классам: `aircraft`, `ship`, `small vehicle`, `large vehicle`.

## Порядок

1. Получить DOTA только из официального источника и зафиксировать версию/условия использования.
2. Конвертировать исходные oriented annotations в YOLO OBB и remap классов в 0–3 через `scripts/prepare_dota4.py`.
3. Разделять по исходным сценам, а не по тайлам: train/val/test не должны содержать фрагменты одного исходного снимка.
4. Проверить структуру и утечки:

   `python scripts/verify_dataset.py --data data/dota4/dota4.yaml`

5. Обучить baseline минимум с тремя seed (`42`, `43`, `44`) и сохранить параметры каждого запуска.
6. Выбрать модель только по validation; test использовать один раз для финального отчёта.
7. Зафиксировать per-class AP, общие Precision/Recall/F1/mAP50/mAP50-95, false positives на изображение и latency CPU/GPU.
8. Провести ручной error analysis минимум по 50 FP/FN и добавить сложные случаи в data card.

## Рекомендуемый baseline

Для Google Colab используется готовый [ноутбук](../notebooks/GeoWatch_DOTA4_Colab.ipynb): артефакты сохраняются во временной папке `/content/geowatch_runs`, а финальные `best.pt` и `metrics_test.json` скачиваются в последней ячейке. Это не требует предоставлять ноутбуку доступ к Google Drive.

```powershell
python scripts/prepare_dota4.py --source D:\datasets\dota_yolo_obb --output data\dota4 --test-fraction 0.20 --seed 42
python scripts/verify_dataset.py --data data\dota4\dota4.yaml
python scripts/train.py --data data\dota4\dota4.yaml --base yolov8n-obb.pt --epochs 80 --imgsz 1024 --batch 4 --device 0 --seed 42 --name dota4_seed42
python scripts/evaluate.py --data data\dota4\dota4.yaml --model models/dota4_seed42/weights/best.pt --threshold 0.25 --split test
```

## Acceptance gate

- train/val/test существуют, непустые и без пересечений source-scene IDs;
- в test представлены все четыре класса (DOTA8 smoke subset намеренно этот gate не проходит);
- веса, dataset YAML, seed, commit и threshold указаны в отчёте;
- метрики получены скриптом, а не внесены вручную;
- минимум один end-to-end тест: upload → detection → review → JSON/CSV/PDF;
- DOTA8-цифры подписаны только как smoke test;
- демо воспроизводится с чистого окружения по README.

## Необходимые конкурсные материалы

- 3–5 проверенных снимков для live demo;
- один снимок с успешными detections и один сложный/негативный пример;
- скриншоты интерфейса и Swagger;
- 90-секундный сценарий demo-видео;
- таблица метрик и confusion matrix;
- краткая архитектурная схема и честные ограничения.
