from __future__ import annotations
import base64, csv, html, io, json, zipfile
from .models import AnalysisResult, AnalystSummary

DISCLAIMER="Результаты являются информационно-аналитическими, требуют экспертной проверки и не предназначены для автономного принятия решений."

def export_json(result): return result.model_dump_json(indent=2).encode("utf-8")
def export_csv(result: AnalysisResult):
    out=io.StringIO(); w=csv.writer(out); w.writerow(["id","class","confidence","x1","y1","x2","y2","area_px","expert_status","expert_comment","model_version"])
    for d in result.detections:
        w.writerow([d.id,d.class_name,f"{d.confidence:.6f}",d.bbox.x1,d.bbox.y1,d.bbox.x2,d.bbox.y2,d.bbox.area,d.review.status if d.review else "needs_review",d.review.comment if d.review else "",d.model_version])
    return out.getvalue().encode("utf-8-sig")


def feedback_records(results: list[AnalysisResult]) -> list[dict]:
    """Return only reviewed candidates; model predictions are never labels by themselves."""
    records=[]
    for result in results:
        for detection in result.detections:
            if not detection.review or detection.review.status == "needs_review":
                continue
            records.append({
                "analysis_id": result.analysis_id,
                "image_id": result.image_id,
                "filename": result.metadata.filename,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "x1": detection.bbox.x1,
                "y1": detection.bbox.y1,
                "x2": detection.bbox.x2,
                "y2": detection.bbox.y2,
                "expert_status": detection.review.status,
                "expert_comment": detection.review.comment,
                "reviewed_at": detection.review.timestamp.isoformat(),
                "model_version": result.model_version,
                "confidence_threshold": result.confidence_threshold,
            })
    return records


def export_feedback_csv(results: list[AnalysisResult]) -> bytes:
    rows=feedback_records(results)
    out=io.StringIO()
    writer=csv.DictWriter(out, fieldnames=["analysis_id","image_id","filename","class_name","confidence","x1","y1","x2","y2","expert_status","expert_comment","reviewed_at","model_version","confidence_threshold"])
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode("utf-8-sig")


def export_feedback_json(results: list[AnalysisResult]) -> bytes:
    return json.dumps({"schema":"geowatch-feedback-v1","human_review_required":True,"records":feedback_records(results)}, ensure_ascii=False, indent=2).encode("utf-8")


def export_operational_brief(results: list[AnalysisResult], cases: list, events: list, evidence: dict) -> bytes:
    """Portable handoff summary; it records candidate findings, never autonomous conclusions."""
    reviews = feedback_records(results)
    payload = {
        "schema": "geowatch-operational-brief-v1",
        "human_review_required": True,
        "model_evidence": evidence,
        "summary": {
            "analyses": len(results),
            "candidates": sum(len(item.detections) for item in results),
            "expert_reviewed_candidates": len(reviews),
            "open_cases": sum(item.status != "closed" for item in cases),
        },
        "cases": [item.model_dump(mode="json") for item in cases],
        "recent_review_events": [item.model_dump(mode="json") for item in events],
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_batch_zip(results: list[AnalysisResult]) -> bytes:
    """Bundle per-image machine-readable reports, without retaining source imagery."""
    output = io.BytesIO()
    manifest = {
        "schema": "geowatch-batch-export-v1",
        "human_review_required": True,
        "source_images_included": False,
        "analyses": [
            {
                "analysis_id": item.analysis_id,
                "filename": item.metadata.filename,
                "detections": len(item.detections),
                "model_version": item.model_version,
            }
            for item in results
        ],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for result in results:
            prefix = f"analyses/{result.analysis_id}"
            archive.writestr(f"{prefix}.json", export_json(result))
            archive.writestr(f"{prefix}.csv", export_csv(result))
    return output.getvalue()
def export_html(result: AnalysisResult, summary: AnalystSummary|None=None, annotated_png: bytes|None=None, metrics: dict|None=None):
    rows="".join(f"<tr><td>{html.escape(d.id)}</td><td>{html.escape(d.class_name)}</td><td>{d.confidence:.1%}</td><td>{d.bbox.x1:.0f}, {d.bbox.y1:.0f}, {d.bbox.x2:.0f}, {d.bbox.y2:.0f}</td><td>{html.escape(d.review.status if d.review else 'needs_review')}</td><td>{html.escape(d.review.comment if d.review else '')}</td></tr>" for d in result.detections)
    findings="".join(f"<li>{html.escape(f.detection_id)}: {html.escape(f.description_ru)} — {html.escape(f.evidence)} — {html.escape(f.risk_note)}</li>" for f in summary.findings) if summary else ""
    qwen=f"<h2>Qwen-сводка</h2><p>{html.escape(summary.summary_ru)}</p><ul>{findings}</ul>" if summary else "<p>Qwen-сводка отсутствует.</p>"
    image_html=f'<img alt="Annotated image" style="max-width:100%" src="data:image/png;base64,{base64.b64encode(annotated_png).decode()}">' if annotated_png else "<p>Annotated image не приложен.</p>"
    metrics_html=f"<pre>{html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))}</pre>" if metrics else "<p>Метрики не рассчитаны; требуется scripts/evaluate.py.</p>"
    metadata=html.escape(json.dumps(result.metadata.model_dump(),ensure_ascii=False,indent=2))
    limitations="".join(f"<li>{html.escape(x)}</li>" for x in result.limitations)
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><style>body{{font:14px Arial;color:#18232c;max-width:1000px;margin:40px}}h1{{color:#087f73}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6dd;padding:7px}}.note{{background:#eef8f6;padding:12px}}pre{{white-space:pre-wrap}}</style><h1>GeoWatch AI</h1><p>ID анализа: {result.analysis_id}<br>Timestamp: {result.created_at.isoformat()}<br>Модель: {html.escape(result.model_version)} · threshold {result.confidence_threshold:.2f}<br>Inference: {result.inference_ms:.1f} ms</p><h2>Метаданные</h2><pre>{metadata}</pre><h2>Снимок с bounding boxes</h2>{image_html}<h2>Обнаружения</h2><table><tr><th>ID</th><th>Класс</th><th>Confidence</th><th>BBox</th><th>Статус</th><th>Комментарий</th></tr>{rows}</table>{qwen}<h2>Метрики</h2>{metrics_html}<h2>Ограничения</h2><ul>{limitations}</ul><p class="note">{DISCLAIMER}</p></html>'''.encode("utf-8")
def export_pdf(result: AnalysisResult):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc: raise RuntimeError("Для PDF установите reportlab.") from exc
    out=io.BytesIO(); c=canvas.Canvas(out,pagesize=A4); y=800
    c.setFont("Helvetica-Bold",18); c.drawString(45,y,"GeoWatch AI"); y-=28
    c.setFont("Helvetica",9)
    for line in [f"Analysis: {result.analysis_id}",f"Image: {result.metadata.filename} ({result.metadata.width}x{result.metadata.height})",f"Model: {result.model_version}; threshold: {result.confidence_threshold:.2f}",f"Inference: {result.inference_ms:.1f} ms; detections: {len(result.detections)}"]:
        c.drawString(45,y,line); y-=15
    y-=8
    for d in result.detections:
        if y<70: c.showPage(); y=800; c.setFont("Helvetica",9)
        c.drawString(45,y,f"{d.id} | {d.class_name} | {d.confidence:.1%} | status: {d.review.status if d.review else 'needs_review'}"); y-=13
    c.setFont("Helvetica-Oblique",8); c.drawString(45,45,"AI results require expert verification and are not for autonomous decisions.")
    c.save(); return out.getvalue()
