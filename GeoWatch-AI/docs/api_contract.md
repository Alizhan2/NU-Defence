# API contract

- `GET /api/v1/health` — состояние API, detector и Qwen.
- `POST /api/v1/quality` — multipart `file`; возвращает метаданные и детерминированные показатели яркости, контраста, резкости и клиппинга. Это не детектор облаков.
- `POST /api/v1/inference?confidence=0.25` — multipart `file`; `200 AnalysisResult`, `400` invalid image, `422` quality gate rejected, `503` model unavailable.
- `POST /api/v1/changes/compare?confidence=0.25&match_iou=0.25` — multipart `before_file` + `after_file`; возвращает два `analysis_id`, quality/pair validation, summary и события `appeared/disappeared/stable`. Несовместимая пара возвращает `422`.
- `GET /api/v1/sources/earth-engine` — безопасный status без автоматической авторизации.
- `GET|POST /api/v1/watch-zones` — список и создание постоянных WGS-84 зон наблюдения.
- `PATCH /api/v1/watch-zones/{zone_id}` — приостановить или возобновить зону.
- `POST /api/v1/watch-zones/{zone_id}/schedule?slot=...` — поставить идемпотентное задание зоны в persistent queue.
- `GET /api/v1/jobs` — список заданий со статусами queued/running/completed/failed/cancelled.
- `POST /api/v1/jobs/{job_id}/cancel` — отменить ожидающее или выполняющееся задание.
- `GET /api/v1/alerts?review_status=needs_review` — детерминированная очередь temporal alerts.
- `GET /api/v1/temporal-comparisons` — сохранённые сравнения с постоянными IDs.
- `GET /api/v1/temporal-comparisons/{comparison_id}` — сравнение, alignment report, события и review history.
- `POST /api/v1/temporal-comparisons/{comparison_id}/events/{event_id}/review` — решение по самому изменению.
- `POST /api/v1/cases/{case_id}/temporal-comparisons/{comparison_id}` — прикрепить comparison к кейсу.
- `GET /api/v1/cases/{case_id}/bundle.zip` — ZIP кейса с manifest, анализами, событиями, решениями и доступными evidence.
- `GET /api/v1/results/{image_id}` — сохранённый результат; `404` if absent.
- `POST /api/v1/review` — `{analysis_id,detection_id,status,comment}`; allowed statuses: confirmed/rejected/needs_review.
- `POST /api/v1/analyst-summary` — `{image_id}`; Qwen failure returns `available:false`, core remains healthy.

OpenAPI доступен в `/docs`. Bbox uses pixel coordinates `[x1,y1,x2,y2]` in original image space. Timestamps are UTC ISO-8601.

Перед temporal comparison GeoTIFF приводятся к общему CRS/grid/extent с учётом NoData, а JPG/PNG проходят translation-only registration с quality gate. Небезопасная пара блокируется. Все `appeared` и `disappeared` остаются в статусе human review.
