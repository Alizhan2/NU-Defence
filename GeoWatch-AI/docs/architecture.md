# Архитектура

UI (`app.py`) и API (`src/api.py`) используют единый доменный pipeline. `preprocessing` проверяет содержимое и метаданные; `tiling` гарантирует покрытие; `inference` вызывает локальный YOLO; `postprocessing` переводит/ограничивает bbox и удаляет дубли внутри класса; `review_service` атомарно хранит JSON; `reporting` сериализует результат; `qwen_analyst` — необязательный, fail-safe слой. Модель/данные локальны, секреты во frontend не передаются.

Для production нужны аутентификация/роли, audit log, объектное хранилище, очередь задач, malware scanning, TLS, шифрование, retention policy и согласованный контур ИБ.
