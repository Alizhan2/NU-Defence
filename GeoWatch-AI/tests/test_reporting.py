from src.models import AnalysisResult,ImageMetadata
from src.models import BoundingBox, Detection, ReviewRecord
import io, zipfile
from src.reporting import DISCLAIMER,export_batch_zip,export_csv,export_feedback_csv,export_feedback_json,export_html,export_json,export_operational_brief,feedback_records

def sample(): return AnalysisResult(analysis_id="a",image_id="i",metadata=ImageMetadata(filename="x.png",format="PNG",width=10,height=10,sha256="0"*64),detections=[],model_version="m",confidence_threshold=.2,inference_ms=1,tile_count=1)
def test_exports():
    result=sample(); assert AnalysisResult.model_validate_json(export_json(result)); assert b"model_version" in export_csv(result); assert DISCLAIMER.encode() in export_html(result)


def test_feedback_export_excludes_unreviewed_predictions():
    result=sample().model_copy(update={"detections":[
        Detection(id="d1",class_name="ship",confidence=.8,bbox=BoundingBox(x1=1,y1=2,x2=3,y2=4),model_version="m",review=ReviewRecord(detection_id="d1",status="confirmed",comment="verified")),
        Detection(id="d2",class_name="ship",confidence=.7,bbox=BoundingBox(x1=1,y1=2,x2=3,y2=4),model_version="m"),
    ]})
    rows=feedback_records([result])
    assert len(rows)==1 and rows[0]["expert_status"]=="confirmed"
    assert b"verified" in export_feedback_csv([result])
    assert b"geowatch-feedback-v1" in export_feedback_json([result])


def test_batch_export_contains_manifest_and_per_analysis_reports():
    archive=zipfile.ZipFile(io.BytesIO(export_batch_zip([sample()])))
    assert set(archive.namelist()) == {"manifest.json", "analyses/a.json", "analyses/a.csv"}
    assert b"source_images_included" in archive.read("manifest.json")


def test_operational_brief_keeps_human_review_boundary():
    brief = export_operational_brief([sample()], [], [], {"state": "unverified"})
    assert b"human_review_required" in brief
    assert DISCLAIMER.encode("utf-8") in brief
