# Models

Поместите обученные DOTA-веса в `best.pt` или задайте `GEOWATCH_MODEL_PATH`. Веса не включены, чтобы не выдавать необученную/несогласованную модель за доказательство качества.

После завершения Colab сначала проверьте и затем примените артефакты:

```powershell
python scripts/promote_model.py --weights C:\path\best.pt --metrics C:\path\metrics_test.json --version yolov8n-obb-dota4-seed42-v1
python scripts/promote_model.py --weights C:\path\best.pt --metrics C:\path\metrics_test.json --version yolov8n-obb-dota4-seed42-v1 --apply
```

Первая команда ничего не изменяет. Вторая создаёт backup текущего `best.pt`, атомарно заменяет веса, записывает `model_card.json` и переносит проверенные метрики в UI.

Перед promotion выполните независимый аудит:

```powershell
python scripts/audit_model.py
```

На 8 сентября 2026 локальный `models/best.pt` совпадает с DOTA8 smoke checkpoint.
Наличие файла не означает, что он загрузился или прошёл inference. Эти состояния
раздельно записываются в `data/runs/model_audit.json`. Не применяйте promotion,
пока отчёт имеет `candidate_unverified`.
