from __future__ import annotations

import io
import json
import os
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.alerting import AlertRule, AlertStore, evaluate_alert_rules
from src.config import ROOT, model_evidence, settings
from src.change_detection import compare_results
from src.demo_case import load_demo_temporal_case
from src.demo_cases import load_demo_case_catalog, load_spacenet_baseline_evidence
from src.earth_engine import EarthEngineAdapter, EarthEngineError, GeoArea, ScenePairRequest
from src.case_service import case_store
from src.image_store import image_store
from src.image_quality import assess_image_quality
from src.geospatial_alignment import AlignmentError, align_geotiff_pair, normalize_aligned_pair, persist_alignment_report, register_translation_pair
from src.inference import ModelUnavailable, YoloDetector, run_inference
from src.models import AnalysisResult, CaseCreateRequest, ReviewRecord, ReviewRequest
from src.preprocessing import ImageValidationError, validate_and_prepare
from src.pair_validation import validate_image_pair
from src.qwen_analyst import QwenAnalyst
from src.reporting import export_batch_zip, export_csv, export_feedback_csv, export_feedback_json, export_html, export_json, export_operational_brief, export_pdf, feedback_records
from src.review_service import store
from src.temporal_store import temporal_store
from src.temporal_ui import REVIEW_STATUS_LABELS, review_decision_hint, review_decision_ready
from src.case_export import export_case_bundle
from src.ui import GLOBAL_CSS, evidence_card, kpi_grid, layer_key, map_empty, readiness, temporal_workbench, timeline, topbar, utc_short
from src.visualization import annotate
from src.watch_zones import WatchZoneStore


st.set_page_config(page_title="GeoWatch AI", page_icon="🛰️", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def review_counts(result: AnalysisResult) -> Counter:
    return Counter(d.review.status if d.review else "needs_review" for d in result.detections)


def result_row(result: AnalysisResult) -> dict:
    reviews = review_counts(result)
    return {
        "Анализ": result.analysis_id[:8],
        "Файл": result.metadata.filename,
        "Время": result.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "Обнаружения": len(result.detections),
        "Подтверждено": reviews["confirmed"],
        "Отклонено": reviews["rejected"],
        "На проверке": reviews["needs_review"],
        "Время, мс": round(result.inference_ms, 1),
        "Модель": result.model_version,
    }


def saved_results() -> list[AnalysisResult]:
    """A damaged historical record must not break a demonstration."""
    results: list[AnalysisResult] = []
    for path in settings.runs_dir.glob("analysis_*.json"):
        try:
            results.append(AnalysisResult.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(results, key=lambda item: item.created_at, reverse=True)


def pending_rows(results: list[AnalysisResult]) -> list[dict]:
    rows = []
    for result in results:
        for detection in result.detections:
            if not detection.review or detection.review.status == "needs_review":
                rows.append({
                    "analysis_id": result.analysis_id,
                    "Файл": result.metadata.filename,
                    "ID": detection.id,
                    "Класс": detection.class_name,
                    "Confidence": detection.confidence,
                    "Порог": result.confidence_threshold,
                    "Время": result.created_at.strftime("%d.%m %H:%M UTC"),
                })
    return sorted(rows, key=lambda item: item["Confidence"], reverse=True)


def detection_rows(result: AnalysisResult, threshold: float, classes: list[str]) -> list[dict]:
    return [
        {
            "ID": d.id,
            "Класс": d.class_name,
            "Confidence": f"{d.confidence:.1%}",
            "BBox": f"{d.bbox.x1:.0f}, {d.bbox.y1:.0f}, {d.bbox.x2:.0f}, {d.bbox.y2:.0f}",
            "Площадь px²": round(d.bbox.area),
            "Статус": d.review.status if d.review else "needs_review",
        }
        for d in result.detections
        if d.class_name in classes and d.confidence >= threshold
    ]


def readiness_stage(title: str, detail: str, ready: bool) -> None:
    marker = "✓" if ready else "→"
    state = "stage-ready" if ready else "stage-wait"
    st.markdown(f'<div class="stage {state}"><b>{marker} {title}</b><br><span>{detail}</span></div>', unsafe_allow_html=True)


def georeferenced_scenes(results: list[AnalysisResult]) -> pd.DataFrame:
    """Return honest map points only when stored coordinates are WGS 84."""
    points = []
    for result in results:
        bounds = result.metadata.bounds
        crs = (result.metadata.crs or "").upper()
        if not bounds or len(bounds) != 4 or "4326" not in crs:
            continue
        left, bottom, right, top = bounds
        lon = (left + right) / 2
        lat = (bottom + top) / 2
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            points.append({"lat": lat, "lon": lon})
    return pd.DataFrame(points)


def navigate(section: str) -> None:
    st.session_state.nav_page = section


def open_demo() -> None:
    st.session_state.nav_page = "Изменения"
    st.session_state.scene_source = "Демо-сценарий"


CHANGE_LABELS = {"appeared": "Появилось", "disappeared": "Исчезло", "stable": "Сохранилось"}
CHANGE_COLORS = {"appeared": "#45e6cf", "disappeared": "#f6bd60", "stable": "#69a9ff"}


def _event_overlay(image, events, boxes: dict[str, tuple[float, float, float, float]], period: str):
    output = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(output)
    for event in events:
        detection_id = event.before_id if period == "before" else event.after_id
        if not detection_id or detection_id not in boxes:
            continue
        box = boxes[detection_id]
        color = CHANGE_COLORS.get(event.status, "#45e6cf")
        draw.rectangle(box, outline=color, width=max(3, output.width // 420))
        label = f"{CHANGE_LABELS.get(event.status, event.status)} · {event.confidence:.0%}"
        x1, y1, _, _ = box
        draw.rectangle((x1, max(0, y1 - 26), x1 + 185, y1), fill="#07151a")
        draw.text((x1 + 6, max(2, y1 - 21)), label, fill=color)
    return output


def render_temporal_workspace(*, before_image, after_image, events, before_boxes, after_boxes, before_label: str, after_label: str, key_prefix: str, review_available: bool = True) -> None:
    classes = sorted({event.class_name for event in events})
    st.markdown(layer_key(), unsafe_allow_html=True)
    control_col, scene_col, evidence_col = st.columns([.78, 2.15, 1.02], gap="medium")
    with control_col:
        st.markdown("#### Слои и фильтры")
        view = st.radio("Режим сцены", ["Разделённый", "До", "После", "Наложение"], key=f"{key_prefix}_view")
        show_overlay = st.toggle("Контуры кандидатов", value=True, key=f"{key_prefix}_overlay")
        status_filter = st.multiselect(
            "Тип изменения", list(CHANGE_LABELS), default=list(CHANGE_LABELS),
            format_func=lambda value: CHANGE_LABELS[value], key=f"{key_prefix}_statuses",
        )
        class_filter = st.multiselect("Классы", classes, default=classes, key=f"{key_prefix}_classes")
        min_confidence = st.slider("Confidence от", 0.0, 1.0, 0.0, .05, key=f"{key_prefix}_confidence")

    visible = [
        event for event in events
        if event.status in status_filter and event.class_name in class_filter and event.confidence >= min_confidence
    ]
    before_view = _event_overlay(before_image, visible, before_boxes, "before") if show_overlay else Image.fromarray(before_image)
    after_view = _event_overlay(after_image, visible, after_boxes, "after") if show_overlay else Image.fromarray(after_image)

    with scene_col:
        st.markdown("#### Временная сцена")
        if view == "Разделённый":
            left, right = st.columns(2, gap="small")
            left.image(before_view, caption=before_label, use_container_width=True)
            right.image(after_view, caption=after_label, use_container_width=True)
        elif view == "До":
            st.image(before_view, caption=before_label, use_container_width=True)
        elif view == "После":
            st.image(after_view, caption=after_label, use_container_width=True)
        else:
            opacity = st.slider("Доля слоя «после»", 0, 100, 50, 5, key=f"{key_prefix}_opacity")
            target = before_view.size
            blended = Image.blend(before_view.convert("RGB"), after_view.convert("RGB").resize(target), opacity / 100)
            st.image(blended, caption=f"Наложение · после {opacity}%", use_container_width=True)
        st.caption(f"Показано кандидатов: {len(visible)} из {len(events)}")

    with evidence_col:
        st.markdown("#### Evidence panel")
        if not visible:
            st.info("Нет кандидатов для выбранных фильтров.")
        else:
            selected_index = st.selectbox(
                "Кандидат", range(len(visible)),
                format_func=lambda index: f"{CHANGE_LABELS.get(visible[index].status, visible[index].status)} · {visible[index].class_name} · {visible[index].confidence:.0%}",
                key=f"{key_prefix}_event",
            )
            selected = visible[selected_index]
            st.markdown(evidence_card(
                event_label=CHANGE_LABELS.get(selected.status, selected.status),
                class_name=selected.class_name,
                confidence=f"{selected.confidence:.1%}",
                overlap=f"{selected.overlap:.2f}",
                before_id=selected.before_id or "—",
                after_id=selected.after_id or "—",
            ), unsafe_allow_html=True)
            if review_available:
                if st.button("Открыть очередь проверки", use_container_width=True, key=f"{key_prefix}_review"):
                    navigate("Очередь проверки")
                    st.rerun()
            else:
                st.caption("Reference/demo-событие не добавлено в рабочую очередь. Для решения аналитика выполните анализ загруженной пары.")

    st.markdown("#### Хронология доказательств")
    timeline_rows = [
        (
            f"{CHANGE_LABELS.get(event.status, event.status)} · {event.class_name}",
            f"Confidence {event.confidence:.1%} · IoU {event.overlap:.2f} · требуется решение аналитика",
            after_label if event.status == "appeared" else before_label,
        )
        for event in visible
    ]
    st.markdown(timeline(timeline_rows, "Измените фильтры или выполните новое сравнение."), unsafe_allow_html=True)

detector = YoloDetector()
model_status = detector.status()
evidence = model_evidence()
alert_store = AlertStore(settings.runs_dir)
zone_store = WatchZoneStore(settings.runs_dir)
alert_rules = [
    AlertRule("temporal-change", "Изменение объекта", frozenset({"appeared", "disappeared"}), min_confidence=0.35),
    AlertRule("high-confidence", "Высокая уверенность", frozenset({"appeared", "disappeared"}), min_confidence=0.70, priority="high"),
]
with st.sidebar:
    st.markdown(
        '<div class="gw-side-brand"><span class="gw-side-brand__mark"></span><div>'
        '<strong>GeoWatch</strong><small>Рабочее пространство</small></div></div>',
        unsafe_allow_html=True,
    )
    groups = {
        "Рабочее пространство": ["Рабочий стол", "Анализ", "Изменения", "Снимки"],
        "Проверка и результаты": ["Очередь проверки", "Кейсы", "Оповещения"],
        "Инструменты": ["Зоны наблюдения", "Пакетный анализ", "История запусков"],
        "Качество и справка": ["Оценка качества", "Feedback", "Ограничения"],
    }
    page = st.session_state.get("nav_page", "Рабочий стол")
    if page not in {item for items in groups.values() for item in items}:
        page = "Рабочий стол"
        st.session_state.nav_page = page
    for group_name, items in groups.items():
        st.markdown(f"## {group_name}")
        for item in items:
            st.button(
                "Экспертная разметка" if item == "Feedback" else item,
                key=f"navigation_{item}",
                type="primary" if page == item else "tertiary",
                use_container_width=True,
                on_click=navigate,
                args=(item,),
            )
    st.markdown("## Статус модели")
    if model_status.state == "ready":
        if evidence["state"] == "promoted":
            system_detail = "DOTA4 · конкурсная модель"
        else:
            system_detail = evidence["label_ru"]
        st.markdown(
            f'<div class="gw-system"><i class="gw-system__dot"></i><div><strong>Контур активен</strong>'
            f'<span>{system_detail}<br>{model_status.model_version}</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(model_status.message_ru)
    st.caption("Все выводы подтверждает аналитик")
    with st.expander("Настройки анализа"):
        threshold = st.slider("Порог уверенности", 0.05, 0.95, float(settings.confidence), 0.05)
        st.caption(f"Тайлы {settings.tile_size}×{settings.tile_size}, overlap {settings.overlap:.0%}")
        if st.button("Сбросить текущий анализ", use_container_width=True):
            for key in ("result", "image", "qwen", "uploaded_image_id"):
                st.session_state.pop(key, None)
            st.rerun()

page_titles = {
    "Рабочий стол": ("Оперативный контур", "Обзор обстановки"),
    "Зоны наблюдения": ("Watch areas", "Зоны наблюдения"),
    "Изменения": ("Temporal intelligence", "Мониторинг изменений"),
    "Оповещения": ("Triage", "Оповещения об изменениях"),
    "Анализ": ("Computer vision", "Анализ снимка"),
    "Пакетный анализ": ("Processing", "Пакетная обработка"),
    "Снимки": ("Imagery", "Каталог снимков"),
    "Очередь проверки": ("Human review", "Очередь решений"),
    "Кейсы": ("Investigations", "Кейсы аналитика"),
    "Feedback": ("Model improvement", "Экспертная разметка"),
    "История запусков": ("Audit trail", "История запусков"),
    "Оценка качества": ("Model evidence", "Оценка качества"),
    "Ограничения": ("Governance", "Ограничения системы"),
}
section_name, page_title = page_titles[page]
st.markdown(topbar(section_name, page_title, model_status.model_version), unsafe_allow_html=True)

if page == "Рабочий стол":
    history = saved_results()
    if not history:
        demo = load_demo_temporal_case()
        st.caption("ИНТЕРАКТИВНОЕ ЗНАКОМСТВО · СИНТЕТИЧЕСКИЕ ДАННЫЕ")
        st.title("Одна территория. Два момента времени.")
        st.write("Посмотрите, как меняется территория: сравните снимки до и после, откройте изменение и изучите его карточку.")
        start, upload = st.columns(2)
        start.button("Открыть интерактивное демо", type="primary", use_container_width=True, on_click=open_demo)
        upload.button("Загрузить свой снимок", use_container_width=True, on_click=navigate, args=("Анализ",))
        before_col, after_col = st.columns(2)
        before_col.image(demo.before, caption=f"ДО · {demo.before_date:%d.%m.%Y}", use_container_width=True)
        after_col.image(demo.after, caption=f"ПОСЛЕ · {demo.after_date:%d.%m.%Y}", use_container_width=True)
        st.markdown(kpi_grid([
            ("Учебный пример", "2 снимка", "одна территория", "neutral"),
            ("Изменение", "1 объект", "добавлен вручную в пример", "neutral"),
            ("Сравнение", "4 режима", "до / после / рядом / наложение", "neutral"),
            ("Проверка", "Карточка", "контекст и контур изменения", "neutral"),
        ]), unsafe_allow_html=True)
        st.info(demo.disclaimer)
        st.subheader("Что попробовать")
        st.write("1. Откройте демо и переключите режим сцены.\n\n2. Включите и выключите контур нового объекта.\n\n3. Выберите изменение, чтобы увидеть его карточку.")
        st.caption("История реальных анализов появится здесь после загрузки ваших снимков.")
        st.stop()
    queue = pending_rows(history)
    cases = case_store.list()
    confirmed_count = sum(review_counts(item)["confirmed"] for item in history)
    open_cases = sum(item.status != "closed" for item in cases)
    st.markdown(
        kpi_grid([
            ("Обработано сцен", str(len(history)), "за всё время", "neutral"),
            ("Требуют решения", str(len(queue)), "кандидаты модели", "warn" if queue else "neutral"),
            ("Активные кейсы", str(open_cases), "аналитические расследования", "signal" if open_cases else "neutral"),
            ("Подтверждено", str(confirmed_count), "решения экспертов", "signal" if confirmed_count else "neutral"),
        ]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gw-section-label">Оперативная обстановка</div>', unsafe_allow_html=True)
    map_column, activity_column = st.columns([1.72, 1], gap="medium")
    map_points = georeferenced_scenes(history)
    with map_column:
        st.markdown(
            '<div class="gw-panel__head"><span class="gw-panel__title">Карта наблюдений</span>'
            f'<span class="gw-panel__meta">{len(map_points)} геопривязанных сцен</span></div>',
            unsafe_allow_html=True,
        )
        if map_points.empty:
            st.markdown(map_empty(), unsafe_allow_html=True)
        else:
            st.map(map_points, latitude="lat", longitude="lon", size=18, use_container_width=True)
    with activity_column:
        audit_events = store.events(limit=6)
        timeline_rows = [
            (
                f"Решение: {item.status}",
                f"Объект {item.detection_id[:10]}" + (f" · {item.comment}" if item.comment else ""),
                utc_short(item.timestamp),
            )
            for item in audit_events
        ]
        st.markdown(
            '<div class="gw-panel"><div class="gw-panel__head"><span class="gw-panel__title">Лента решений</span>'
            '<span class="gw-panel__meta">AUDIT LOG</span></div>'
            + timeline(timeline_rows, "Решения экспертов появятся после проверки кандидатов.")
            + '</div>',
            unsafe_allow_html=True,
        )

    action_a, action_b, action_c, action_d = st.columns(4)
    action_a.button("Новый анализ", type="primary", use_container_width=True, on_click=navigate, args=("Анализ",))
    action_b.button("Сравнить периоды", use_container_width=True, on_click=navigate, args=("Изменения",))
    action_c.button("Открыть очередь", use_container_width=True, on_click=navigate, args=("Очередь проверки",))
    action_d.download_button(
        "Operational brief",
        export_operational_brief(history, cases, store.events(limit=100), evidence),
        "geowatch_operational_brief.json",
        "application/json",
        use_container_width=True,
    )

    st.markdown('<div class="gw-section-label">Готовность контура</div>', unsafe_allow_html=True)
    model_ready = evidence["state"] == "promoted"
    has_review = any(review_counts(item)["confirmed"] + review_counts(item)["rejected"] for item in history)
    lower_left, lower_right = st.columns([1, 1.72], gap="medium")
    with lower_left:
        st.markdown(
            '<div class="gw-panel"><div class="gw-panel__head"><span class="gw-panel__title">Статус компонентов</span>'
            '<span class="gw-panel__meta">READINESS</span></div>'
            + readiness([
                ("Модель", evidence["label_ru"], model_ready),
                ("Снимки", f"В истории: {len(history)}", bool(history)),
                ("Экспертная проверка", "Решения зафиксированы" if has_review else "Нет решений", has_review),
                ("Экспорт", "Отчёты доступны" if history else "После первого анализа", bool(history)),
            ])
            + '</div>',
            unsafe_allow_html=True,
        )
    with lower_right:
        counts = Counter(d.class_name for item in history for d in item.detections)
        st.markdown(
            '<div class="gw-panel__head"><span class="gw-panel__title">Кандидаты по классам</span>'
            f'<span class="gw-panel__meta">{sum(counts.values())} объектов</span></div>',
            unsafe_allow_html=True,
        )
        if counts:
            st.bar_chart(pd.DataFrame({"Кандидаты": counts}).sort_values("Кандидаты", ascending=False), color="#45e6cf")
        else:
            st.markdown('<div class="empty">Распределение появится после первого анализа сцены.</div>', unsafe_allow_html=True)

    if not model_ready:
        st.info("Следующее для конкурсной версии: получить best.pt и metrics_test.json из DOTA4 Colab, затем выполнить scripts/promote_model.py --apply.")
    elif not history:
        st.info("Следующее: загрузьте демонстрационный снимок и проведите экспертную проверку кандидатов.")
    if queue:
        st.markdown('<div class="gw-section-label">Приоритетная очередь</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(queue[:10]).drop(columns=["analysis_id"]), use_container_width=True, hide_index=True)
    st.stop()

if page == "Зоны наблюдения":
    zones = zone_store.list()
    st.caption("Сохранённые области используются для повторного поиска снимков и мониторинга изменений.")
    with st.form("new_watch_zone", clear_on_submit=True):
        name = st.text_input("Название зоны", placeholder="Например: Aktobe industrial area")
        west_col, south_col, east_col, north_col = st.columns(4)
        west = west_col.number_input("West", min_value=-180.0, max_value=180.0, value=57.00, format="%.5f")
        south = south_col.number_input("South", min_value=-90.0, max_value=90.0, value=50.20, format="%.5f")
        east = east_col.number_input("East", min_value=-180.0, max_value=180.0, value=57.30, format="%.5f")
        north = north_col.number_input("North", min_value=-90.0, max_value=90.0, value=50.40, format="%.5f")
        cadence = st.select_slider("Периодичность проверки", options=[1, 3, 5, 7, 14, 30], value=5, format_func=lambda value: f"каждые {value} дн.")
        create_zone = st.form_submit_button("Сохранить зону", type="primary")
    if create_zone:
        try:
            zone_store.create(name, west, south, east, north, cadence)
            st.success("Зона сохранена и готова к мониторингу.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    zones = zone_store.list()
    if zones:
        map_rows = pd.DataFrame([{"lat": (z.south + z.north) / 2, "lon": (z.west + z.east) / 2} for z in zones])
        st.map(map_rows, latitude="lat", longitude="lon", size=22, use_container_width=True)
        st.dataframe(pd.DataFrame([{
            "Зона": z.name, "BBox": f"{z.west:.4f}, {z.south:.4f}, {z.east:.4f}, {z.north:.4f}",
            "Период": f"{z.cadence_days} дн.", "Статус": "активна" if z.enabled else "приостановлена",
            "ID": z.zone_id,
        } for z in zones]), use_container_width=True, hide_index=True)
        selected_zone = st.selectbox("Управление зоной", zones, format_func=lambda item: item.name)
        if st.button("Приостановить" if selected_zone.enabled else "Возобновить"):
            zone_store.set_enabled(selected_zone.zone_id, not selected_zone.enabled)
            st.rerun()
    else:
        st.markdown('<div class="empty">Зон пока нет. Сохраните первую AOI, чтобы использовать её в temporal monitoring.</div>', unsafe_allow_html=True)
    st.stop()

if page == "Оповещения":
    alerts = alert_store.list()
    pending_alerts = [item for item in alerts if item.review_status == "needs_review"]
    high_alerts = [item for item in pending_alerts if item.priority == "high"]
    st.markdown(kpi_grid([
        ("Всего оповещений", str(len(alerts)), "дедуплицированные события", "neutral"),
        ("Ожидают проверки", str(len(pending_alerts)), "human review required", "warn" if pending_alerts else "neutral"),
        ("Высокий приоритет", str(len(high_alerts)), "правило оператора", "warn" if high_alerts else "neutral"),
        ("Активные правила", str(len(alert_rules)), "детерминированные фильтры", "signal"),
    ]), unsafe_allow_html=True)
    st.caption("Triage score используется только для сортировки очереди. Это не вероятность угрозы и не окончательный вывод.")
    if alerts:
        priority_filter = st.multiselect("Приоритет", ["high", "normal", "low"], default=["high", "normal", "low"])
        visible_alerts = [item for item in alerts if item.priority in priority_filter]
        st.dataframe(pd.DataFrame([{
            "Score": item.triage_score, "Приоритет": item.priority, "Событие": item.event_status,
            "Класс": item.class_name, "Confidence": f"{item.confidence:.1%}",
            "Правило": item.rule_name, "Статус": item.review_status, "Alert ID": item.alert_id,
        } for item in visible_alerts]), use_container_width=True, hide_index=True)
        if st.button("Перейти в очередь экспертной проверки", type="primary"):
            navigate("Очередь проверки")
            st.rerun()
    else:
        st.markdown('<div class="empty">Оповещения появятся после temporal comparison, если событие соответствует активному правилу.</div>', unsafe_allow_html=True)
    st.stop()

if page == "Изменения":
    st.caption("Сравнение одной территории в два периода. Все события являются кандидатами и требуют проверки аналитиком.")
    st.markdown(temporal_workbench(), unsafe_allow_html=True)

    source = st.radio("Источник снимков", ["Загрузить два снимка", "Google Earth Engine", "Реальные кейсы", "Демо-сценарий"], horizontal=True, key="scene_source")
    baseline = load_spacenet_baseline_evidence(ROOT)
    with st.expander(f"SpaceNet 7 baseline · {baseline.label}", expanded=baseline.status == "unverified"):
        st.caption(baseline.detail)
        if baseline.metrics:
            st.dataframe(pd.DataFrame([{"Метрика": key, "Значение": value} for key, value in baseline.metrics.items()]), use_container_width=True, hide_index=True)
        if baseline.status != "verified_test":
            st.warning("Эти данные нельзя использовать как подтверждённую точность модели строительства.")
    if source == "Реальные кейсы":
        try:
            real_cases = load_demo_case_catalog(ROOT)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Каталог реальных кейсов повреждён: {exc}")
            st.stop()
        if not real_cases:
            st.info("Каталог ещё не подготовлен. Ожидается data/demo_cases/catalog.json с локальными evidence-изображениями.")
            st.stop()
        selected_case = st.selectbox(
            "Демонстрационный кейс", real_cases,
            format_func=lambda item: f"{item.event_label} · {item.title}", key="real_demo_case",
        )
        evidence_tone = "signal" if selected_case.evidence_status == "ground_truth" else "warn"
        st.markdown(kpi_grid([
            ("Событие", selected_case.event_label, selected_case.title, evidence_tone),
            ("Evidence", selected_case.evidence_label, selected_case.note or "требуется проверка", evidence_tone),
            ("Территория", selected_case.aoi, selected_case.attribution or "источник указан в каталоге", "neutral"),
            ("Интервал", f"{(selected_case.after_date - selected_case.before_date).days} дней", f"{selected_case.before_date:%d.%m.%Y} — {selected_case.after_date:%d.%m.%Y}", "neutral"),
        ]), unsafe_allow_html=True)
        if selected_case.evidence_status != "ground_truth":
            st.warning("Кейс использует реальные снимки, но событие ещё не подтверждено разметкой или аналитиком.")
        else:
            st.success("Событие основано на разметке источника; вывод модели оценивается отдельно.")
        if selected_case.source_url:
            st.markdown(f"Источник: [{selected_case.attribution or selected_case.source_url}]({selected_case.source_url}) · лицензия: {selected_case.license}")
        if not selected_case.is_ready:
            st.error("Evidence-файлы этого кейса отсутствуют локально. Метаданные показаны, но визуальное сравнение недоступно.")
            st.stop()
        try:
            before_case = Image.open(selected_case.before_image).convert("RGB")
            after_case = Image.open(selected_case.after_image).convert("RGB")
        except (OSError, ValueError) as exc:
            st.error(f"Evidence-файл существует, но не читается как изображение: {exc}")
            st.stop()
        left, right = st.columns(2)
        left.image(before_case, caption=f"До · {selected_case.before_date:%d.%m.%Y}", use_container_width=True)
        right.image(after_case, caption=f"После · {selected_case.after_date:%d.%m.%Y}", use_container_width=True)
        st.markdown("#### Карточка изменения")
        if selected_case.event_type == "uncertain":
            st.info("В footprint-разметке нет подтверждённого directional-события. Это сложный негативный кейс, а не доказательство полного отсутствия изменений.")
        else:
            st.info(f"Reference-событие: **{selected_case.event_label}**. Оно используется для демонстрации и последующей проверки вывода модели.")
        st.caption("Реальная пара предназначена для воспроизводимого demo review. Наличие снимков не является доказательством качества ML-модели.")
        st.stop()
    if source == "Демо-сценарий":
        demo = load_demo_temporal_case()
        st.warning(demo.disclaimer)
        st.markdown(kpi_grid([
            ("Сценарий", "UI demo", demo.title, "signal"),
            ("Территория", "A-17", demo.area_name, "neutral"),
            ("Интервал", f"{(demo.after_date - demo.before_date).days} дней", f"{demo.before_date:%d.%m.%Y} — {demo.after_date:%d.%m.%Y}", "neutral"),
            ("Решение", "не принято", "ожидает аналитика", "warn"),
        ]), unsafe_allow_html=True)
        demo_boxes = {event.after_id: event.bbox for event in demo.events if event.after_id}
        render_temporal_workspace(
            before_image=demo.before,
            after_image=demo.after,
            events=demo.events,
            before_boxes={},
            after_boxes=demo_boxes,
            before_label=f"До · {demo.before_date:%d.%m.%Y}",
            after_label=f"После · {demo.after_date:%d.%m.%Y}",
            key_prefix="demo_temporal",
            review_available=False,
        )
        st.caption("Демо-кейс показывает визуальный walkthrough без записи в evidence store. Для полного потока с решением аналитика загрузите совмещённую пару снимков.")
        st.stop()
    if source == "Google Earth Engine":
        adapter = EarthEngineAdapter()
        ee_status = adapter.status()
        st.markdown(kpi_grid([
            ("Источник", "Sentinel-2", ee_status.collection_id, "signal"),
            ("Cloud project", "готов" if ee_status.project_configured else "не задан", ee_status.project_id or "env required", "signal" if ee_status.project_configured else "warn"),
            ("Python adapter", "доступен" if ee_status.dependency_available else "не установлен", "earthengine-api", "signal" if ee_status.dependency_available else "warn"),
            ("Статус", "готов" if ee_status.ready else "ожидает", ee_status.detail, "signal" if ee_status.ready else "neutral"),
        ]), unsafe_allow_html=True)
        zones = [item for item in zone_store.list() if item.enabled]
        if zones:
            zone = st.selectbox("Зона наблюдения", zones, format_func=lambda item: f"{item.name} · каждые {item.cadence_days} дн.")
            west, south, east, north = zone.west, zone.south, zone.east, zone.north
        else:
            st.info("Сначала сохраните AOI в разделе «Зоны наблюдения» либо задайте bbox вручную.")
            bbox_cols = st.columns(4)
            west = bbox_cols[0].number_input("West", value=57.00, format="%.5f", key="ee_west")
            south = bbox_cols[1].number_input("South", value=50.20, format="%.5f", key="ee_south")
            east = bbox_cols[2].number_input("East", value=57.30, format="%.5f", key="ee_east")
            north = bbox_cols[3].number_input("North", value=50.40, format="%.5f", key="ee_north")
        ee_dates = st.columns(2)
        ee_before = ee_dates[0].date_input("Дата до", key="ee_before_date")
        ee_after = ee_dates[1].date_input("Дата после", key="ee_after_date")
        cloud_limit = st.slider("Максимальная облачность", 0, 100, 25, 5, format="%d%%")
        if st.button("Найти пару сцен", type="primary", disabled=not (ee_status.project_configured and ee_status.dependency_available)):
            try:
                adapter.initialize()
                pair = adapter.search_scene_pair(ScenePairRequest(
                    aoi=GeoArea.from_bbox(west, south, east, north), before_date=ee_before,
                    after_date=ee_after, max_cloud_percent=float(cloud_limit),
                ))
                st.session_state.ee_scene_pair = pair
            except (EarthEngineError, ValueError) as exc:
                st.error(str(exc))
        pair = st.session_state.get("ee_scene_pair")
        if pair:
            st.success("Пара сцен найдена. Следующий production шаг — export одинаковых тайлов для YOLO.")
            st.dataframe(pd.DataFrame([
                {"Период": "до", "Scene ID": pair.before.scene_id, "Дата": pair.before.acquired_at, "Облачность": pair.before.cloud_percent},
                {"Период": "после", "Scene ID": pair.after.scene_id, "Дата": pair.after.acquired_at, "Облачность": pair.after.cloud_percent},
            ]), use_container_width=True, hide_index=True)
            for warning in pair.warnings:
                st.warning(warning)
        st.info("Sentinel-2: 10 м/пиксель и повторное покрытие примерно раз в 5 дней. Для ежедневного контроля небольших зданий потребуется коммерческая съёмка высокого разрешения.")
        st.stop()

    before_date, after_date = st.columns(2)
    with before_date:
        st.subheader("До")
        date_before = st.date_input("Дата первого снимка", key="change_before_date")
        before_upload = st.file_uploader("Снимок до", type=["jpg", "jpeg", "png", "tif", "tiff"], key="change_before")
    with after_date:
        st.subheader("После")
        date_after = st.date_input("Дата второго снимка", key="change_after_date")
        after_upload = st.file_uploader("Снимок после", type=["jpg", "jpeg", "png", "tif", "tiff"], key="change_after")

    st.caption("Для корректного сопоставления снимки должны показывать одну и ту же область и быть геометрически совмещены.")
    prepared_pair = None
    comparison_allowed = False
    alignment_report = None
    if before_upload and after_upload:
        try:
            before_data, after_data = before_upload.getvalue(), after_upload.getvalue()
            before = validate_and_prepare(before_data, before_upload.name)
            after = validate_and_prepare(after_data, after_upload.name)
            if before.metadata.format == "GeoTIFF" and after.metadata.format == "GeoTIFF":
                aligned = align_geotiff_pair(before_data, after_data)
            elif not before.metadata.crs and not after.metadata.crs:
                aligned = register_translation_pair(before.array, after.array)
            else:
                aligned = None
            if aligned is not None:
                alignment_report = aligned.report
                if alignment_report.safe_for_change_detection:
                    before.array, after.array = normalize_aligned_pair(aligned.before, aligned.after, aligned.valid_mask)
                    update = {"width": before.array.shape[1], "height": before.array.shape[0]}
                    if alignment_report.common_grid:
                        grid = alignment_report.common_grid
                        update.update({"crs": grid.crs, "bounds": list(grid.bounds), "transform": list(grid.transform)})
                    before.metadata = before.metadata.model_copy(update=update)
                    after.metadata = after.metadata.model_copy(update=update)
            before_quality = assess_image_quality(before.array)
            after_quality = assess_image_quality(after.array)
            pair_report = validate_image_pair(before.metadata, after.metadata)
            prepared_pair = (before, after)
            alignment_safe = aligned is None or aligned.report.safe_for_change_detection
            comparison_allowed = alignment_safe and pair_report.safe_to_compare and before_quality.safe_for_inference and after_quality.safe_for_inference
            st.subheader("Входной контроль")
            st.dataframe(pd.DataFrame([
                {"Проверка": "Снимок до", "Статус": before_quality.status, "Яркость": f"{before_quality.brightness:.2f}", "Контраст": f"{before_quality.contrast:.2f}", "Резкость": f"{before_quality.sharpness:.4f}"},
                {"Проверка": "Снимок после", "Статус": after_quality.status, "Яркость": f"{after_quality.brightness:.2f}", "Контраст": f"{after_quality.contrast:.2f}", "Резкость": f"{after_quality.sharpness:.4f}"},
                {"Проверка": "Совместимость пары", "Статус": pair_report.status, "Яркость": "—", "Контраст": "—", "Резкость": "—"},
                {"Проверка": "Совмещение", "Статус": alignment_report.status if alignment_report else "manual", "Яркость": "—", "Контраст": "—", "Резкость": "—"},
            ]), use_container_width=True, hide_index=True)
            alignment_issues = alignment_report.issues if alignment_report else ("Автоматическое совмещение недоступно для смешанного типа геопривязки.",)
            for issue in (*before_quality.issues, *after_quality.issues, *pair_report.issues, *alignment_issues):
                (st.error if not comparison_allowed else st.warning)(issue)
        except (AlignmentError, ImageValidationError, ValueError) as exc:
            st.error(str(exc))

    if before_upload and after_upload and st.button("Сравнить снимки", type="primary", use_container_width=True, disabled=not comparison_allowed):
        if date_after <= date_before:
            st.error("Дата «после» должна быть позже даты «до».")
        else:
            try:
                with st.spinner("Подготовка → детекция двух периодов → пространственное сопоставление"):
                    before, after = prepared_pair
                    before_result = run_inference(before.array, before.metadata, detector, threshold)
                    after_result = run_inference(after.array, after.metadata, detector, threshold)
                    store.save(before_result)
                    store.save(after_result)
                    image_store.save(before.array, before_result.analysis_id)
                    image_store.save(after.array, after_result.analysis_id)
                    events = compare_results(before_result, after_result)
                    timeline_record = temporal_store.save(
                        before_result, after_result, events,
                        before_date=date_before.isoformat(), after_date=date_after.isoformat(),
                        pair_validation={**pair_report.to_dict(), "alignment": alignment_report.to_dict() if alignment_report else None},
                        quality={"before": before_quality.to_dict(), "after": after_quality.to_dict()},
                    )
                    if alignment_report:
                        persist_alignment_report(alignment_report, settings.runs_dir / "alignments" / f"{timeline_record.comparison_id}.json")
                    alerts = evaluate_alert_rules(timeline_record.comparison_id, events, alert_rules)
                    alert_store.save_many(alerts)
                    st.session_state.change_comparison = (before, after, before_result, after_result, timeline_record.comparison_id, date_before, date_after)
            except (ImageValidationError, ModelUnavailable, OSError, ValueError) as exc:
                st.error(str(exc))

    comparison = st.session_state.get("change_comparison")
    if comparison:
        before, after, before_result, after_result, timeline_id, saved_before_date, saved_after_date = comparison
        timeline_record = temporal_store.load(timeline_id)
        events = list(timeline_record.events)
        appeared = [item for item in events if item.status == "appeared"]
        disappeared = [item for item in events if item.status == "disappeared"]
        stable = [item for item in events if item.status == "stable"]
        st.markdown(kpi_grid([
            ("Новых", str(len(appeared)), "кандидаты после", "signal"),
            ("Исчезнувших", str(len(disappeared)), "кандидаты до", "warn"),
            ("Сохранилось", str(len(stable)), "пространственно сопоставлены", "neutral"),
            ("На проверке", str(len(appeared) + len(disappeared)), "решение аналитика", "warn"),
        ]), unsafe_allow_html=True)
        before_boxes = {
            detection.id: (detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2)
            for detection in before_result.detections
        }
        after_boxes = {
            detection.id: (detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2)
            for detection in after_result.detections
        }
        render_temporal_workspace(
            before_image=before.array,
            after_image=after.array,
            events=events,
            before_boxes=before_boxes,
            after_boxes=after_boxes,
            before_label=f"До · {saved_before_date:%d.%m.%Y}",
            after_label=f"После · {saved_after_date:%d.%m.%Y}",
            key_prefix=f"temporal_{after_result.analysis_id}",
        )
        review_candidates = [item for item in events if item.status != "stable"]
        if review_candidates:
            st.markdown("#### Решение по изменению")
            selected_change = st.selectbox(
                "Событие", review_candidates,
                format_func=lambda item: f"{CHANGE_LABELS.get(item.status, item.status)} · {item.class_name} · {item.event_id[:8]}",
                key=f"temporal_review_event_{timeline_id}",
            )
            decision_col, note_col = st.columns([1, 2])
            decision = decision_col.radio(
                "Решение аналитика", ["needs_review", "confirmed", "rejected"],
                index=["needs_review", "confirmed", "rejected"].index(selected_change.review_status),
                format_func=lambda value: REVIEW_STATUS_LABELS[value], horizontal=True,
                key=f"temporal_decision_{selected_change.event_id}",
            )
            comment = note_col.text_input("Основание", value=selected_change.comment, key=f"temporal_comment_{selected_change.event_id}")
            can_save_review = review_decision_ready(decision, comment)
            st.caption(review_decision_hint(decision, comment))
            if st.button("Сохранить решение по изменению", type="primary", disabled=not can_save_review, key=f"temporal_save_{selected_change.event_id}"):
                temporal_store.review(timeline_id, selected_change.event_id, decision, comment)
                st.success("Решение сохранено в SQLite и журнале temporal-события.")
                st.rerun()
        elif events:
            st.success("Новых или исчезнувших объектов выше порога нет. Сопоставленные объекты показаны как стабильные; при формальном кейсе вывод всё равно проверяет аналитик.")
        rows = [{
            "Событие": {"appeared": "появилось", "disappeared": "исчезло", "stable": "сохранилось"}[item.status],
            "Класс": item.class_name,
            "Confidence": f"{item.confidence:.1%}",
            "До ID": item.before_id or "—",
            "После ID": item.after_id or "—",
            "IoU": f"{item.overlap:.2f}",
            "Статус": item.review_status if item.status != "stable" else "matched",
            "Event ID": item.event_id,
        } for item in events]
        if rows:
            with st.expander("Техническая таблица сопоставления"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Модель не нашла объектов выше выбранного порога ни на одном снимке.")
    elif not (before_upload and after_upload):
        st.markdown('<div class="empty"><b>Добавьте пару снимков одной территории.</b><br>GeoWatch сопоставит объекты по классу и положению и сформирует проверяемую хронологию.</div>', unsafe_allow_html=True)
    st.stop()

if page == "Очередь проверки":
    st.header("Очередь экспертной проверки")
    queue = pending_rows(saved_results())
    temporal_pending = [
        (comparison, event)
        for comparison in temporal_store.list()
        for event in comparison.events
        if event.status != "stable" and event.review_status == "needs_review"
    ]
    if not queue and not temporal_pending:
        st.info("Очередь пуста.")
        st.stop()
    detection_tab, change_tab = st.tabs([f"Объекты · {len(queue)}", f"Изменения · {len(temporal_pending)}"])
    with detection_tab:
        if not queue:
            st.info("Непроверенных объектов нет.")
        else:
            classes = sorted({item["Класс"] for item in queue})
            selected_classes = st.multiselect("Классы", classes, default=classes)
            min_conf = st.slider("Минимальная confidence", 0.0, 1.0, 0.0, 0.05, key="queue_conf")
            visible = [item for item in queue if item["Класс"] in selected_classes and item["Confidence"] >= min_conf]
            st.dataframe(pd.DataFrame(visible).drop(columns=["analysis_id"]), use_container_width=True, hide_index=True)
            if visible:
                current = st.selectbox("Кандидат для решения", visible, format_func=lambda item: f"{item['Класс']} · {item['Confidence']:.1%} · {item['Файл']}")
                decision = st.radio("Решение", ["confirmed", "rejected", "needs_review"], horizontal=True, key="queue_decision")
                reason = st.text_area("Основание решения", max_chars=2000, key="queue_reason")
                if st.button("Сохранить решение очереди", type="primary"):
                    store.review(ReviewRequest(analysis_id=current["analysis_id"], detection_id=current["ID"], status=decision, comment=reason))
                    st.success("Решение сохранено в журнале анализа.")
                    st.rerun()
    with change_tab:
        if not temporal_pending:
            st.info("Непроверенных изменений нет.")
        else:
            st.dataframe(pd.DataFrame([{
                "Событие": CHANGE_LABELS.get(event.status, event.status), "Класс": event.class_name,
                "Confidence": event.confidence, "Дата до": comparison.before_date or "—",
                "Дата после": comparison.after_date or "—", "Event ID": event.event_id,
            } for comparison, event in temporal_pending]), use_container_width=True, hide_index=True)
            selected_pair = st.selectbox(
                "Изменение для решения", temporal_pending,
                format_func=lambda item: f"{CHANGE_LABELS.get(item[1].status, item[1].status)} · {item[1].class_name} · {item[1].event_id[:8]}",
            )
            temporal_decision = st.radio(
                "Решение", ["confirmed", "rejected", "needs_review"], horizontal=True,
                format_func=lambda value: REVIEW_STATUS_LABELS[value], key="queue_temporal_decision",
            )
            temporal_reason = st.text_area("Основание решения", max_chars=2000, key="queue_temporal_reason")
            temporal_can_save = review_decision_ready(temporal_decision, temporal_reason)
            st.caption(review_decision_hint(temporal_decision, temporal_reason))
            if st.button("Сохранить решение по изменению", type="primary", disabled=not temporal_can_save):
                temporal_store.review(selected_pair[0].comparison_id, selected_pair[1].event_id, temporal_decision, temporal_reason)
                st.success("Решение по изменению сохранено.")
                st.rerun()
    st.stop()

if page == "Снимки":
    st.header("Каталог снимков")
    history = saved_results()
    if not history:
        st.info("Каталог заполнится после первого анализа.")
        st.stop()
    formats = sorted({item.metadata.format for item in history})
    chosen_formats = st.multiselect("Форматы", formats, default=formats)
    query = st.text_input("Поиск по имени файла")
    visible = [item for item in history if item.metadata.format in chosen_formats and query.lower() in item.metadata.filename.lower()]
    st.caption("Сравнение показывает два сохранённых запуска и их статистику. Это не change detection.")
    cards = []
    for item in visible:
        cards.append({"Анализ": item.analysis_id[:8], "Файл": item.metadata.filename, "Формат": item.metadata.format, "Размер": f"{item.metadata.width}×{item.metadata.height}", "CRS": item.metadata.crs or "нет", "Кандидатов": len(item.detections), "Время": item.created_at.strftime("%d.%m %H:%M UTC")})
    st.dataframe(pd.DataFrame(cards), use_container_width=True, hide_index=True)
    labels = {f"{item.analysis_id[:8]} · {item.metadata.filename}": item for item in visible}
    selected = st.multiselect("Выберите до двух запусков", list(labels), max_selections=2)
    if len(selected) == 1:
        item = labels[selected[0]]
        asset = image_store.get(item.analysis_id)
        if asset:
            st.image(str(asset), caption=f"{item.metadata.filename} · {item.analysis_id[:8]}", use_container_width=True)
        else:
            st.warning("Для этого старого запуска превью не сохранено. Новые анализы сохраняют его автоматически.")
    elif len(selected) == 2:
        left, right = (labels[label] for label in selected)
        a, b = st.columns(2)
        for column, item in ((a, left), (b, right)):
            asset = image_store.get(item.analysis_id)
            column.subheader(item.metadata.filename)
            if asset:
                column.image(str(asset), caption=f"{item.analysis_id[:8]} · {item.created_at:%d.%m %H:%M UTC}", use_container_width=True)
            else:
                column.info("Превью отсутствует для старого запуска.")
            counts = Counter(d.class_name for d in item.detections)
            column.json({"candidates": len(item.detections), **dict(counts), "model": item.model_version, "threshold": item.confidence_threshold})
    st.stop()

if page == "Пакетный анализ":
    st.header("Пакетный анализ")
    st.caption("Обрабатывает до 10 снимков последовательно, сохраняет каждый запуск в историю и формирует единый ZIP с JSON/CSV. Исходные изображения в ZIP не включаются.")
    batch_uploads = st.file_uploader(
        "Выберите до 10 JPG, PNG или GeoTIFF",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_uploads",
    )
    if len(batch_uploads) > 10:
        st.error("Выберите не более 10 файлов за один запуск.")
    elif batch_uploads and st.button("Запустить пакетный анализ", type="primary"):
        outcomes: list[dict] = []
        completed: list[AnalysisResult] = []
        progress = st.progress(0, text="Подготовка пакетного анализа")
        for index, upload in enumerate(batch_uploads, start=1):
            try:
                prepared = validate_and_prepare(upload.getvalue(), upload.name)
                result = run_inference(prepared.array, prepared.metadata, detector, threshold)
                store.save(result)
                image_store.save(prepared.array, result.analysis_id)
                completed.append(result)
                outcomes.append({
                    "Файл": upload.name,
                    "Статус": "готово",
                    "Анализ": result.analysis_id[:8],
                    "Кандидатов": len(result.detections),
                    "Время, мс": round(result.inference_ms, 1),
                    "Ошибка": "",
                })
            except (ImageValidationError, ModelUnavailable, OSError, ValueError) as exc:
                outcomes.append({"Файл": upload.name, "Статус": "ошибка", "Анализ": "", "Кандидатов": 0, "Время, мс": 0, "Ошибка": str(exc)})
            progress.progress(index / len(batch_uploads), text=f"Обработано {index} из {len(batch_uploads)}")
        st.session_state.batch_outcomes = outcomes
        st.session_state.batch_results = completed

    outcomes = st.session_state.get("batch_outcomes", [])
    completed = st.session_state.get("batch_results", [])
    if outcomes:
        st.subheader("Сводка пакета")
        st.dataframe(pd.DataFrame(outcomes), use_container_width=True, hide_index=True)
        ready = len(completed)
        failed = len(outcomes) - ready
        cards = st.columns(3)
        cards[0].metric("Успешно", ready)
        cards[1].metric("Ошибок", failed)
        cards[2].metric("Кандидатов", sum(len(item.detections) for item in completed))
        if completed:
            st.download_button(
                "Скачать ZIP результатов",
                export_batch_zip(completed),
                "geowatch_batch_results.zip",
                "application/zip",
                type="secondary",
            )
    elif not batch_uploads:
        st.markdown('<div class="empty"><b>Готово к серии снимков.</b><br>Пакетный режим подходит для первичного разбора набора файлов; финальная интерпретация остаётся за аналитиком.</div>', unsafe_allow_html=True)
    st.stop()

if page == "Кейсы":
    st.header("Кейсы")
    st.caption("Кейс объединяет связанные анализы и заметки аналитика; он не является автоматическим выводом системы.")
    with st.form("new_case", clear_on_submit=True):
        title = st.text_input("Название кейса")
        note = st.text_area("Заметка")
        priority = st.selectbox("Приоритет", ["normal", "high", "low"])
        create = st.form_submit_button("Создать кейс", type="primary")
    if create:
        if not title.strip():
            st.error("Укажите название кейса.")
        else:
            case_store.create(CaseCreateRequest(title=title.strip(), note=note, priority=priority))
            st.success("Кейс создан.")
            st.rerun()
    cases = case_store.list()
    history = saved_results()
    if not cases:
        st.info("Кейсов пока нет. Создайте первый, чтобы прикреплять к нему анализы.")
        st.stop()
    for record in cases:
        with st.expander(f"{record.title} · {record.status} · {record.priority}", expanded=False):
            st.write(record.note or "Без заметки.")
            st.caption(f"Прикреплено анализов: {len(record.analysis_ids)}")
            statuses = ["open", "in_review", "closed"]
            new_status = st.selectbox("Статус кейса", statuses, index=statuses.index(record.status), key=f"status_{record.case_id}")
            if new_status != record.status and st.button("Сохранить статус", key=f"status_button_{record.case_id}"):
                case_store.update_status(record.case_id, new_status)
                st.rerun()
            available = {f"{item.analysis_id[:8]} · {item.metadata.filename}": item.analysis_id for item in history if item.analysis_id not in record.analysis_ids}
            if available:
                selected = st.selectbox("Добавить анализ", list(available), key=f"attach_{record.case_id}")
                if st.button("Прикрепить", key=f"attach_button_{record.case_id}"):
                    case_store.attach_analysis(record.case_id, available[selected])
                    st.rerun()
            if record.analysis_ids:
                st.code("\n".join(record.analysis_ids), language=None)
            linked_comparisons = temporal_store.comparisons_for_case(record.case_id)
            available_comparisons = [item for item in temporal_store.list() if item.comparison_id not in linked_comparisons]
            st.caption(f"Temporal-сравнений: {len(linked_comparisons)}")
            if available_comparisons:
                selected_comparison = st.selectbox(
                    "Добавить сравнение", available_comparisons,
                    format_func=lambda item: f"{item.before_date or '—'} → {item.after_date or '—'} · {item.comparison_id[:8]}",
                    key=f"attach_temporal_{record.case_id}",
                )
                if st.button("Прикрепить сравнение", key=f"attach_temporal_button_{record.case_id}"):
                    temporal_store.attach_to_case(record.case_id, selected_comparison.comparison_id)
                    st.rerun()
            try:
                bundle = export_case_bundle(record.case_id, cases=case_store, analyses=store, timelines=temporal_store, images=image_store)
                st.download_button("Скачать кейс с доказательствами", bundle, f"geowatch-case-{record.case_id}.zip", "application/zip", key=f"case_bundle_{record.case_id}")
            except KeyError as exc:
                st.warning(str(exc))
    st.stop()

if page == "Feedback":
    st.header("Feedback для дообучения")
    history = saved_results()
    records = feedback_records(history)
    st.caption("В экспорт попадают только объекты с решением эксперта confirmed или rejected. Непроверенные предсказания не считаются разметкой.")
    cards = st.columns(3)
    cards[0].metric("Проверенных записей", len(records))
    cards[1].metric("Подтверждено", sum(item["expert_status"] == "confirmed" for item in records))
    cards[2].metric("Отклонено", sum(item["expert_status"] == "rejected" for item in records))
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет экспертно проверенных находок. Сначала обработайте снимок и сохраните решение в очереди review.")
    a, b = st.columns(2)
    a.download_button("Скачать feedback CSV", export_feedback_csv(history), "geowatch_feedback.csv", "text/csv")
    b.download_button("Скачать feedback JSON", export_feedback_json(history), "geowatch_feedback.json", "application/json")
    st.stop()

if page == "Оценка качества":
    st.header("Оценка качества")
    st.caption(evidence["label_ru"])
    metrics = evidence["metrics"]
    if evidence["state"] == "smoke":
        st.warning("Эти значения относятся к минимальному DOTA8 smoke-тесту и не должны использоваться в конкурсной презентации.")
    elif evidence["state"] == "unverified":
        st.warning("Метрики финальной DOTA4-модели ещё не установлены. После Colab используйте scripts/promote_model.py.")
    if metrics:
        fields = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("mAP@50", "map50"), ("mAP@50–95", "map50_95")]
        cards = st.columns(5)
        for card, (label, key) in zip(cards, fields):
            value = metrics.get(key)
            card.metric(label, f"{float(value):.3f}" if isinstance(value, (int, float)) else "—")
        st.json(metrics)
    st.stop()

if page == "Ограничения":
    st.header("Ограничения и ответственное использование")
    st.write("Таксономию реального применения определяет заказчик. MVP использует только нейтральные классы DOTA: aircraft, ship, small vehicle, large vehicle.")
    st.write("Точность зависит от GSD, облачности, угла съёмки и доменного сдвига. Система не делает выводов о назначении объектов или событиях.")
    st.info("Демонстрационный протокол: загрузить снимок → проверить кандидаты → зафиксировать решение эксперта → выгрузить проверяемый отчёт.")
    st.stop()

if page == "История запусков":
    st.header("История запусков")
    history = saved_results()
    if not history:
        st.markdown('<div class="empty"><b>История пока пуста.</b><br>Первый завершённый анализ появится здесь автоматически.</div>', unsafe_allow_html=True)
        st.stop()

    st.dataframe(pd.DataFrame([result_row(item) for item in history]), use_container_width=True, hide_index=True)
    options = {f"{item.analysis_id[:8]} · {item.metadata.filename} · {item.created_at:%d.%m %H:%M}": item for item in history}
    selected_labels = st.multiselect("Сравнить до двух запусков", list(options), max_selections=2)
    selected_results = [options[label] for label in selected_labels]
    if len(selected_results) == 1:
        st.caption("Выберите ещё один запуск, чтобы увидеть сравнение.")
    elif len(selected_results) == 2:
        comparison = []
        for item in selected_results:
            counts = Counter(d.class_name for d in item.detections)
            comparison.append({
                "Показатель": f"{item.analysis_id[:8]} · {item.metadata.filename}",
                "Обнаружения": len(item.detections), "aircraft": counts["aircraft"],
                "ship": counts["ship"], "small vehicle": counts["small vehicle"],
                "large vehicle": counts["large vehicle"], "Время, мс": round(item.inference_ms, 1),
                "Порог": item.confidence_threshold,
            })
        st.subheader("Сравнение запусков")
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
    st.stop()

st.header("Анализ снимка")
with st.expander("Сценарий демо", expanded=not st.session_state.get("result")):
    st.markdown("""1. Загрузите JPG, PNG или GeoTIFF.  
2. Запустите анализ и отфильтруйте кандидатов.  
3. Подтвердите или отклоните находки аналитиком.  
4. Выгрузите JSON, CSV или HTML-отчёт.""")

uploaded = st.file_uploader("Загрузите JPG, PNG или GeoTIFF", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded:
    st.markdown('<div class="empty"><b>Готово к анализу.</b><br>Поддерживаются обычные изображения и GeoTIFF. Метаданные CRS сохраняются в отчёте, если они есть в файле.</div>', unsafe_allow_html=True)
else:
    try:
        prepared = validate_and_prepare(uploaded.getvalue(), uploaded.name)
        if st.session_state.get("uploaded_image_id") != prepared.metadata.sha256:
            for key in ("result", "image", "qwen"):
                st.session_state.pop(key, None)
            st.session_state.uploaded_image_id = prepared.metadata.sha256
        metadata = prepared.metadata
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ширина", metadata.width)
        c2.metric("Высота", metadata.height)
        c3.metric("Формат", metadata.format)
        c4.metric("CRS", metadata.crs or "нет")
        quality = assess_image_quality(prepared.array)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Quality gate", quality.status.upper())
        q2.metric("Яркость", f"{quality.brightness:.2f}")
        q3.metric("Контраст", f"{quality.contrast:.2f}")
        q4.metric("Резкость", f"{quality.sharpness:.4f}")
        for issue in quality.issues:
            (st.error if quality.status == "reject" else st.warning)(issue)
        st.image(prepared.array, caption="Исходный снимок", use_container_width=True)
        if st.button("Запустить анализ", type="primary", disabled=not quality.safe_for_inference):
            with st.spinner("Preprocessing → тайлинг → YOLO → class-aware NMS"):
                result = run_inference(prepared.array, metadata, detector, threshold)
                store.save(result)
                image_store.save(prepared.array, result.analysis_id)
                st.session_state.result = result
                st.session_state.image = prepared.array
                st.session_state.qwen = None
    except (ImageValidationError, ModelUnavailable) as exc:
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    reviews = review_counts(result)
    st.success(f"Обработка завершена: {result.inference_ms:.1f} мс · тайлов: {result.tile_count} · обнаружений: {len(result.detections)}")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Всего кандидатов", len(result.detections))
    summary_columns[1].metric("Подтверждено", reviews["confirmed"])
    summary_columns[2].metric("Отклонено", reviews["rejected"])
    summary_columns[3].metric("На проверке", reviews["needs_review"])
    st.image(annotate(st.session_state.image, result), caption="Результат с bounding boxes", use_container_width=True)
    classes = sorted({d.class_name for d in result.detections})
    if not classes:
        st.info("Кандидаты выше выбранного порога не найдены. Попробуйте другой снимок или снизьте порог уверенности.")
    else:
        chosen = st.multiselect("Фильтр классов", classes, default=classes)
        visible = [d for d in result.detections if d.class_name in chosen and d.confidence >= threshold]
        st.dataframe(pd.DataFrame(detection_rows(result, threshold, chosen)), use_container_width=True, hide_index=True)
        if not visible:
            st.info("После фильтрации кандидатов не осталось. Измените классы или порог confidence.")
        else:
            selected = st.selectbox("Карточка обнаружения", visible, format_func=lambda d: f"{d.id} · {d.class_name} · {d.confidence:.1%}")
            x1, y1, x2, y2 = (int(selected.bbox.x1), int(selected.bbox.y1), int(selected.bbox.x2), int(selected.bbox.y2))
            padding = max(24, int(max(x2 - x1, y2 - y1) * 0.25))
            image = st.session_state.image
            crop = image[max(0, y1-padding):min(image.shape[0], y2+padding), max(0, x1-padding):min(image.shape[1], x2+padding)]
            evidence, details = st.columns([1, 2])
            evidence.image(crop, caption=f"Evidence crop · {selected.id}", use_container_width=True)
            details.caption("Карточка доказательства: фрагмент сформирован из исходного снимка; координаты указаны в пикселях изображения.")
            st.json({"ID": selected.id, "Класс": selected.class_name, "Confidence": selected.confidence, "Bounding box": selected.bbox.model_dump(), "Площадь": selected.bbox.area, "Время обработки, мс": selected.processing_ms, "Статус эксперта": selected.review.status if selected.review else "needs_review", "Комментарий": selected.review.comment if selected.review else "", "Версия модели": selected.model_version})
            status_ru = st.radio("Экспертное решение", ["confirmed", "rejected", "needs_review"], horizontal=True, key=selected.id)
            comment = st.text_area("Комментарий эксперта", value=selected.review.comment if selected.review else "", key="c" + selected.id, max_chars=2000)
            if st.button("Сохранить решение", type="secondary"):
                store.review(ReviewRequest(analysis_id=result.analysis_id, detection_id=selected.id, status=status_ru, comment=comment))
                st.session_state.result = store.load_by_analysis(result.analysis_id)
                st.success("Решение сохранено. Оно также попадёт в историю и выгрузки.")

    st.subheader("Qwen-аналитик")
    if not settings.qwen_enabled:
        st.info("Qwen не настроен. Основной анализ YOLO доступен.")
    elif st.button("Сформировать Qwen-сводку"):
        st.session_state.qwen = QwenAnalyst().summarize(result)
    qwen = st.session_state.get("qwen")
    if qwen:
        st.info(qwen.message_ru)
    if qwen and qwen.summary:
        st.json(qwen.summary.model_dump())

    st.subheader("Выгрузки")
    a, b, c, d = st.columns(4)
    a.download_button("JSON", export_json(result), f"{result.image_id}.json", "application/json")
    b.download_button("CSV", export_csv(result), f"{result.image_id}.csv", "text/csv")
    annotated = annotate(st.session_state.image, result)
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    metrics_path = settings.runs_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    except (OSError, json.JSONDecodeError):
        metrics = None
    c.download_button("HTML-отчёт", export_html(result, qwen.summary if qwen else None, buffer.getvalue(), metrics), f"{result.image_id}.html", "text/html")
    try:
        d.download_button("PDF-отчёт", export_pdf(result), f"{result.image_id}.pdf", "application/pdf")
    except RuntimeError as exc:
        d.caption(str(exc))
