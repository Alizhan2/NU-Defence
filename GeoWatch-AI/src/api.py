from __future__ import annotations
from dataclasses import asdict
from datetime import date
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from .alerting import AlertRule, AlertStore, evaluate_alert_rules
from .change_detection import compare_results
from .case_service import case_store
from .case_export import export_case_bundle
from .image_store import image_store
from .image_quality import assess_image_quality
from .job_queue import job_queue
from .geospatial_alignment import AlignmentError, align_geotiff_pair, normalize_aligned_pair, persist_alignment_report, register_translation_pair
from .config import model_evidence, settings
from .inference import ModelUnavailable, YoloDetector, run_inference
from .models import CaseCreateRequest, CaseUpdateRequest, ReviewRequest, SummaryRequest
from .preprocessing import ImageValidationError, validate_and_prepare
from .pair_validation import validate_image_pair
from .qwen_analyst import QwenAnalyst
from .reporting import export_feedback_csv, export_feedback_json
from .review_service import store
from .temporal_store import TemporalReviewStatus, temporal_store
from .earth_engine import EarthEngineAdapter
from .watch_zones import WatchZoneStore

app=FastAPI(title="GeoWatch AI API",version="0.1.0",description="API MVP первичного анализа космических снимков. Результаты требуют экспертной проверки.")
detector=YoloDetector()
alert_store=AlertStore(settings.runs_dir)
zone_store=WatchZoneStore(settings.runs_dir)
earth_engine=EarthEngineAdapter()


class WatchZoneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    cadence_days: int = Field(default=5, ge=1, le=365)


class WatchZoneState(BaseModel):
    enabled: bool


class TemporalReviewRequest(BaseModel):
    status: TemporalReviewStatus
    comment: str = Field(default="", max_length=2000)


DEFAULT_ALERT_RULES = (
    AlertRule("temporal-change", "Изменение объекта", frozenset({"appeared", "disappeared"}), min_confidence=0.35),
    AlertRule("high-confidence", "Высокая уверенность", frozenset({"appeared", "disappeared"}), min_confidence=0.70, priority="high"),
)

@app.get("/api/v1/health")
def health(): return {"status":"ok","detector":detector.status().model_dump(),"model_evidence":model_evidence(),"qwen_enabled":settings.qwen_enabled}

@app.get("/api/v1/sources/earth-engine")
def earth_engine_status():
    return asdict(earth_engine.status())

@app.get("/api/v1/watch-zones")
def watch_zones():
    return [asdict(item) for item in zone_store.list()]

@app.post("/api/v1/watch-zones")
def create_watch_zone(request: WatchZoneCreate):
    try:
        return asdict(zone_store.create(**request.model_dump()))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@app.patch("/api/v1/watch-zones/{zone_id}")
def update_watch_zone(zone_id: str, request: WatchZoneState):
    try:
        return asdict(zone_store.set_enabled(zone_id, request.enabled))
    except KeyError as exc:
        raise HTTPException(404, str(exc))

@app.get("/api/v1/alerts")
def alerts(review_status: str | None = None):
    if review_status not in {None, "needs_review", "confirmed", "dismissed"}:
        raise HTTPException(400, "Unsupported review status")
    return [asdict(item) for item in alert_store.list(review_status=review_status)]

@app.get("/api/v1/jobs")
def jobs(limit: int = Query(100, ge=1, le=500)):
    return [item.to_dict() for item in job_queue.list(limit)]

@app.post("/api/v1/watch-zones/{zone_id}/schedule")
def schedule_watch_zone(zone_id: str, slot: str = Query(..., min_length=1, max_length=64)):
    zone = next((item for item in zone_store.list() if item.zone_id == zone_id), None)
    if zone is None:
        raise HTTPException(404, "Зона наблюдения не найдена.")
    if not zone.enabled:
        raise HTTPException(409, "Зона наблюдения отключена.")
    job = job_queue.enqueue(
        "watch_zone",
        {"zone": asdict(zone), "slot": slot, "requires_earth_engine": True},
        dedupe_key=f"{zone.zone_id}:{slot}",
    )
    return job.to_dict()

@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return job_queue.cancel(job_id).to_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@app.post("/api/v1/inference")
async def inference(file: UploadFile=File(...),confidence:float=Query(settings.confidence,ge=0.0,le=1.0)):
    try: prepared=validate_and_prepare(await file.read(),file.filename or "upload")
    except ImageValidationError as exc: raise HTTPException(400,str(exc))
    quality = assess_image_quality(prepared.array)
    if not quality.safe_for_inference:
        raise HTTPException(422, {"message": "Снимок не прошёл входной контроль качества.", "quality": quality.to_dict()})
    try: result=run_inference(prepared.array,prepared.metadata,detector,confidence)
    except ModelUnavailable as exc: raise HTTPException(503,str(exc))
    store.save(result)
    try:
        image_store.save(prepared.array, result.analysis_id)
    except (AttributeError, OSError, TypeError, ValueError):
        # A preview improves the catalog but must never discard a valid inference result.
        pass
    return result

@app.post("/api/v1/quality")
async def image_quality(file: UploadFile = File(...)):
    """Return deterministic input diagnostics; this is not a cloud classifier."""
    try:
        prepared = validate_and_prepare(await file.read(), file.filename or "upload")
    except ImageValidationError as exc:
        raise HTTPException(400, str(exc))
    return {"metadata": prepared.metadata.model_dump(), "quality": assess_image_quality(prepared.array).to_dict()}

@app.post("/api/v1/changes/compare")
async def compare_changes(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    confidence: float = Query(settings.confidence, ge=0.0, le=1.0),
    match_iou: float = Query(0.25, ge=0.05, le=0.95),
    before_date: date | None = Query(None),
    after_date: date | None = Query(None),
):
    """Compare two co-registered views of one area and return review candidates."""
    if (before_date is None) != (after_date is None):
        raise HTTPException(400, "Укажите обе даты или не указывайте ни одной.")
    if before_date and after_date and after_date <= before_date:
        raise HTTPException(400, "Дата после должна быть позже даты до.")
    before_data, after_data = await before_file.read(), await after_file.read()
    try:
        before = validate_and_prepare(before_data, before_file.filename or "before")
        after = validate_and_prepare(after_data, after_file.filename or "after")
    except ImageValidationError as exc:
        raise HTTPException(400, str(exc))
    try:
        if before.metadata.format == "GeoTIFF" and after.metadata.format == "GeoTIFF":
            aligned = align_geotiff_pair(before_data, after_data)
        elif not before.metadata.crs and not after.metadata.crs:
            aligned = register_translation_pair(before.array, after.array)
        else:
            aligned = None
    except AlignmentError as exc:
        raise HTTPException(422, {"message": "Не удалось безопасно совместить снимки.", "detail": str(exc)})
    alignment_payload = None
    if aligned is not None:
        alignment_payload = aligned.report.to_dict()
        if not aligned.report.safe_for_change_detection:
            raise HTTPException(422, {"message": "Качество совмещения недостаточно.", "alignment": alignment_payload})
        aligned_before, aligned_after = normalize_aligned_pair(aligned.before, aligned.after, aligned.valid_mask)
        before.array, after.array = aligned_before, aligned_after
        metadata_update = {"width": aligned_before.shape[1], "height": aligned_before.shape[0]}
        if aligned.report.common_grid:
            grid = aligned.report.common_grid
            metadata_update.update({"crs": grid.crs, "bounds": list(grid.bounds), "transform": list(grid.transform)})
        before.metadata = before.metadata.model_copy(update=metadata_update)
        after.metadata = after.metadata.model_copy(update=metadata_update)
    pair_report = validate_image_pair(before.metadata, after.metadata)
    before_quality = assess_image_quality(before.array)
    after_quality = assess_image_quality(after.array)
    if not pair_report.safe_to_compare:
        raise HTTPException(422, {"message": "Пара снимков несовместима.", "pair_validation": pair_report.to_dict()})
    if not before_quality.safe_for_inference or not after_quality.safe_for_inference:
        raise HTTPException(422, {
            "message": "Один из снимков не прошёл входной контроль качества.",
            "before_quality": before_quality.to_dict(),
            "after_quality": after_quality.to_dict(),
        })
    try:
        before_result = run_inference(before.array, before.metadata, detector, confidence)
        after_result = run_inference(after.array, after.metadata, detector, confidence)
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc))

    for prepared, result in ((before, before_result), (after, after_result)):
        store.save(result)
        try:
            image_store.save(prepared.array, result.analysis_id)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    changes = compare_results(before_result, after_result, match_iou)
    quality_payload = {"before": before_quality.to_dict(), "after": after_quality.to_dict()}
    timeline = temporal_store.save(
        before_result,
        after_result,
        changes,
        before_date=before_date.isoformat() if before_date else None,
        after_date=after_date.isoformat() if after_date else None,
        match_iou=match_iou,
        pair_validation={**pair_report.to_dict(), "alignment": alignment_payload},
        quality=quality_payload,
    )
    saved_alerts = alert_store.save_many(evaluate_alert_rules(timeline.comparison_id, changes, DEFAULT_ALERT_RULES))
    if aligned is not None:
        persist_alignment_report(aligned.report, settings.runs_dir / "alignments" / f"{timeline.comparison_id}.json")
    needs_review = sum(change.status != "stable" for change in changes)
    warnings = ["Результаты являются кандидатами и требуют подтверждения аналитиком.", *pair_report.issues]
    return {
        "comparison_id": timeline.comparison_id,
        "before_analysis_id": before_result.analysis_id,
        "after_analysis_id": after_result.analysis_id,
        "summary": {
            "appeared": sum(change.status == "appeared" for change in changes),
            "disappeared": sum(change.status == "disappeared" for change in changes),
            "stable": sum(change.status == "stable" for change in changes),
            "needs_review": needs_review,
        },
        "changes": [change.__dict__ for change in timeline.events],
        "alert_ids": [item.alert_id for item in saved_alerts],
        "warnings": warnings,
        "quality": quality_payload,
        "pair_validation": pair_report.to_dict(),
        "alignment": alignment_payload,
    }

@app.get("/api/v1/temporal-comparisons")
def temporal_comparisons(limit: int = Query(100, ge=1, le=500)):
    return [item.to_dict() for item in temporal_store.list(limit)]

@app.get("/api/v1/temporal-comparisons/{comparison_id}")
def temporal_comparison(comparison_id: str):
    try:
        return temporal_store.load(comparison_id).to_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc))

@app.post("/api/v1/temporal-comparisons/{comparison_id}/events/{event_id}/review")
def review_temporal_event(comparison_id: str, event_id: str, request: TemporalReviewRequest):
    try:
        return asdict(temporal_store.review(comparison_id, event_id, request.status, request.comment))
    except KeyError as exc:
        raise HTTPException(404, str(exc))

@app.get("/api/v1/results/{image_id}")
def results(image_id:str):
    try: return store.load(image_id)
    except KeyError as exc: raise HTTPException(404,str(exc))

@app.post("/api/v1/review")
def review(request:ReviewRequest):
    try: return store.review(request)
    except KeyError as exc: raise HTTPException(404,str(exc))

@app.get("/api/v1/cases")
def cases(): return case_store.list()

@app.post("/api/v1/cases")
def create_case(request: CaseCreateRequest): return case_store.create(request)

@app.post("/api/v1/cases/{case_id}/analyses/{analysis_id}")
def attach_analysis(case_id: str, analysis_id: str):
    try:
        store.load_by_analysis(analysis_id)
        return case_store.attach_analysis(case_id, analysis_id)
    except KeyError as exc: raise HTTPException(404, str(exc))

@app.post("/api/v1/cases/{case_id}/temporal-comparisons/{comparison_id}")
def attach_temporal_comparison(case_id: str, comparison_id: str):
    try:
        case_store.load(case_id)
        temporal_store.attach_to_case(case_id, comparison_id)
        return {"case_id": case_id, "comparison_id": comparison_id, "attached": True}
    except KeyError as exc:
        raise HTTPException(404, str(exc))

@app.patch("/api/v1/cases/{case_id}")
def update_case(case_id: str, request: CaseUpdateRequest):
    try: return case_store.update_status(case_id, request.status)
    except KeyError as exc: raise HTTPException(404, str(exc))

@app.get("/api/v1/feedback.json")
def feedback_json():
    return Response(export_feedback_json(store.all_results()), media_type="application/json")

@app.get("/api/v1/feedback.csv")
def feedback_csv():
    return Response(export_feedback_csv(store.all_results()), media_type="text/csv; charset=utf-8", headers={"Content-Disposition":"attachment; filename=geowatch_feedback.csv"})

@app.post("/api/v1/analyst-summary")
def summary(request:SummaryRequest):
    try: result=store.load(request.image_id)
    except KeyError as exc: raise HTTPException(404,str(exc))

@app.get("/api/v1/cases/{case_id}/bundle.zip")
def case_bundle(case_id: str):
    try:
        payload = export_case_bundle(
            case_id, cases=case_store, analyses=store, timelines=temporal_store, images=image_store,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return Response(
        payload,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=geowatch-case-{case_id}.zip"},
    )
    return QwenAnalyst().summarize(result)
