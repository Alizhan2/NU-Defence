# GeoWatch AI — конкурсный комплект

Состав:

- `GeoWatch_AI_Competition_Deck_v2.pptx` — актуальная проверенная презентация на 7 слайдов с финальными DOTA4-метриками;
- `screenshots/dashboard.png` — реальный главный экран;
- `screenshots/temporal_workspace.png` — реальный temporal demo workspace;
- `GeoWatch_AI_Source_2026-09-08.zip` — исходники, тесты, документация и локальные demo-артефакты без виртуальных окружений и исходных датасетов.

Запуск интерфейса и API описан в корневом `README.md`. Полный сценарий показа находится в `docs/demo_script.md`, а честные ограничения и внешние блокеры — в `docs/submission_readiness.md`.

Важно: включённый DOTA4 checkpoint имеет статус `verified`; hash и независимые test-метрики находятся в `models/model_card.json` и `data/runs/metrics.json`. Building-change metrics не заявляются до реального SpaceNet 7 test run.
