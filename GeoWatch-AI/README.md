# GeoWatch AI

Работоспособный конкурсный MVP для первичного анализа открытых космических снимков и формирования проверяемой очереди наблюдений для аналитика.

> GeoWatch AI — MVP для первичного анализа снимков и формирования проверяемой очереди наблюдений для аналитика. Результаты AI не являются окончательным выводом и требуют подтверждения экспертом.

## Соответствие ТЗ

Реализованы загрузка и preprocessing JPG/PNG/GeoTIFF, detection/classification/bounding boxes/confidence, тайлинг малых объектов, class-aware NMS, визуализация, экспертная проверка, накопление feedback, отчётность, evaluation pipeline и интеграционный FastAPI. Официальное ТЗ не задаёт конкретный стек и численные пороги; YOLO, DOTA, Qwen, форматы и endpoints — инженерная реализация MVP.

## Архитектура и pipeline

`Source/AOI → before + after → validation → tiles 1024/20% → YOLO → global coordinates → class-aware NMS → temporal matching → review → case → export`.

Streamlit отвечает за UI, FastAPI — за интеграцию, а `src/` содержит общие сервисы. Анализы, temporal comparisons, события, review history, jobs и связи с кейсами сохраняются в SQLite; экспортируемые артефакты остаются в `data/runs/`. Qwen изолирован и по умолчанию выключен.

## Быстрый запуск

```powershell
cd GeoWatch-AI
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/prepare_data.py
streamlit run app.py
```

API в другом терминале:

```powershell
python scripts/run_api_local.py
```

Swagger: `http://127.0.0.1:8000/docs`; health: `GET /api/v1/health`.

## Сценарий live demo

1. Откройте Streamlit и загрузите подготовленный JPG/PNG/GeoTIFF со спутниковым снимком.
2. Установите confidence threshold и нажмите **«Запустить анализ»**.
3. Покажите слой с bounding boxes, таблицу обнаружений и evidence card выбранного объекта.
4. Для одного объекта сохраните экспертное решение `confirmed`, `rejected` или `needs_review`.
5. Скачайте JSON/CSV/HTML/PDF-отчёт; он содержит модель, threshold, время обработки, результат review и ограничения.

На странице **«Изменения»** можно загрузить два геометрически совмещённых снимка одной территории. Система отдельно анализирует каждый период и формирует события `появилось`, `исчезло` и `сохранилось`. Google Earth Engine предусмотрен как опциональный источник Sentinel-2; подробная настройка описана в `docs/earth_engine_integration.md`.

Для честной демонстрации используйте также один сложный или негативный снимок: отсутствие detections не доказывает отсутствия объектов и должно остаться в статусе экспертной проверки.

## Модель

Обучите DOTA-конвертацию на классах `aircraft`, `ship`, `small vehicle`, `large vehicle` и поместите веса в `models/best.pt`, либо задайте `GEOWATCH_MODEL_PATH`. Не используйте стандартные COCO-веса как доказательство качества на спутниковых снимках. Без файла весов приложение запускается, но честно блокирует inference и не создаёт фиктивные detections.

```powershell
python scripts/train.py --data data/dota4/dota4.yaml --base yolov8n-obb.pt --epochs 80 --imgsz 1024 --batch 4 --device 0 --seed 42 --name dota4_seed42
python scripts/evaluate.py --data data/dota4/dota4.yaml --model models/dota4_seed42/weights/best.pt --threshold 0.25 --split test
pytest -q
```

`prepare_data.py` создаёт только синтетические изображения для smoke-теста plumbing; это не датасет качества. Реальные train/val/test должны быть непересекающимися по исходным изображениям/геосценам.

## Воспроизводимое обучение DOTA4

1. Получите DOTA из официального источника и соблюдайте его условия использования.
2. Подготовьте исходный YOLO-OBB DOTA и создайте внутренний scene-separated split только для `aircraft`, `ship`, `small vehicle`, `large vehicle`:

```powershell
python scripts/prepare_dota4.py --source D:\datasets\dota_yolo_obb --output data\dota4 --test-fraction 0.20 --seed 42
python scripts/verify_dataset.py --data data\dota4\dota4.yaml
```

3. Запустите обучение для seed `42`, `43` и `44`; модель и threshold выбирайте только по validation.
4. Запустите `scripts/evaluate.py --split test` один раз для выбранного checkpoint и сохраните полученный JSON вместе с seed, параметрами запуска, версией кода и manifest датасета.

Для GPU Colab используйте [GeoWatch_DOTA4_Colab.ipynb](notebooks/GeoWatch_DOTA4_Colab.ipynb). Ноутбук автоматически скачивает DOTAv1, строит DOTA4 split, проверяет gate и выгружает `best.pt` и metrics после завершения запуска. Артефакты Colab временно хранятся в `/content/geowatch_runs`, поэтому скачайте итоговые файлы до завершения runtime.

## Метрики: smoke и финальная оценка

В репозитории могут присутствовать DOTA8 weights и метрики как **smoke test pipeline**: этот минимальный набор подтверждает только работоспособность цепочки обучения и inference. Он не является доказательством качества модели и не должен выдаваться за конкурсный результат.

Финальными считаются только результаты на изолированном DOTA4 `test` split, прошедшем `verify_dataset.py`, с представленностью всех четырёх классов. В финальный отчёт включите Precision, Recall, F1, mAP50, mAP50-95, per-class AP, latency, threshold, seed, manifest и краткий error analysis.

## Qwen

Основной pipeline не зависит от Qwen. Для локального запуска установите совместимые `torch`, `transformers`, `accelerate`, обеспечьте ресурсы GPU/CPU, затем:

```env
QWEN_ENABLED=true
QWEN_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
QWEN_DEVICE=auto
QWEN_MAX_NEW_TOKENS=450
```

Qwen получает только allowlisted metadata, detections и expert feedback, не добавляет объекты/назначения/события. При ошибке возвращается безопасный non-critical fallback.

## API examples

```bash
curl -F "file=@image.png" "http://127.0.0.1:8000/api/v1/inference?confidence=0.25"
curl http://127.0.0.1:8000/api/v1/results/IMAGE_ID
curl -H "Content-Type: application/json" -d '{"analysis_id":"...","detection_id":"det_000001","status":"confirmed","comment":"checked"}' http://127.0.0.1:8000/api/v1/review
```

## Данные и лицензии

- DOTA: используйте официальный набор только в рамках его research/usage terms; исходные изображения, аннотации и производные dataset splits не включаются в публичный репозиторий.
- xView: запасной источник; полный набор автоматически не скачивается.
- pipeline samples, созданные `prepare_data.py`: генерируются локально, не содержат внешних данных и не являются accuracy evidence.
- Веса, checkpoints и результаты обучения — отдельные артефакты. Публикуйте их только вместе с provenance, лицензией источника, параметрами запуска и честной маркировкой уровня валидации.

Подробности: `docs/data_card.md`, `docs/limitations.md`, `docs/api_contract.md`.

Конкурсные материалы: `docs/demo_script.md`, `docs/submission_readiness.md` и `submission/GeoWatch_AI_Competition_Deck.pptx`.

## Следующий этап

Подготовлен [план Stage 2](docs/stage2_plan.md), пример `configs/dota_target.example.yaml` и автоматическая проверка train/val/test на пересечения через `scripts/verify_dataset.py`.

Для подготовленного YOLO-OBB DOTA используйте `scripts/prepare_dota4.py`: он оставляет только четыре нейтральных класса, формирует scene-separated внутренний test split и создаёт `manifest.json` с provenance и SHA-256. Исходный DOTA не скачивается скриптом и не должен коммититься в репозиторий.

## Known issues и roadmap

- Включены smoke-веса DOTA8 и отдельно маркированные smoke-метрики; полноценные метрики требуют scene-separated DOTA test split.
- PDF MVP содержит текстовую таблицу; HTML — более полный конкурсный отчёт.
- GeoTIFF alignment реализован с общим CRS/grid, NoData и отчётом качества, но требует установленный `rasterio` в целевом runtime.
- Qwen3-VL тяжёлый и требует отдельной настройки.
- Режим «До/После» реализован для загружаемых georegistered пар; автоматическое получение и совмещение сцен Earth Engine требует Cloud Project и авторизации.
- Persistent job queue поддерживает retry/cancel/dedupe, но автоматический Earth Engine worker и системный планировщик ещё не подключены.

Пилот: согласовать таксономию и режим доступа → собрать/разметить данные → baseline и сравнительные испытания → security review → интеграционный sandbox → опытная версия.

## Дисклеймер

Результаты являются информационно-аналитическими, требуют экспертной проверки и не предназначены для автономного принятия решений.
