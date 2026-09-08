import json
from src.models import AnalysisResult,BoundingBox,Detection,ImageMetadata
from src.qwen_analyst import QwenAnalyst,safe_payload

def result(): return AnalysisResult(analysis_id="a",image_id="i",metadata=ImageMetadata(filename="secret.png",format="PNG",width=10,height=10,sha256="0"*64),detections=[Detection(id="det_1",class_name="ship",confidence=.8,bbox=BoundingBox(x1=1,y1=1,x2=2,y2=2),model_version="m")],model_version="m",confidence_threshold=.2,inference_ms=1,tile_count=1)
def test_disabled_does_not_call():
    called=[]; response=QwenAnalyst(False,lambda *_:called.append(1)).summarize(result()); assert not response.available and not called
def test_validated_response():
    raw=json.dumps({"summary_ru":"Сводка","findings":[{"detection_id":"det_1","description_ru":"обнаружение модели","evidence":"bbox и confidence","risk_note":"требуется подтверждение экспертом"}],"limitations_ru":["только structured results"]},ensure_ascii=False)
    assert QwenAnalyst(True,lambda *_:raw).summarize(result()).available
def test_failure_is_nonfatal():
    assert not QwenAnalyst(True,lambda *_:(_ for _ in ()).throw(RuntimeError("offline"))).summarize(result()).available
def test_payload_allowlist():
    payload=json.dumps(safe_payload(result())); assert "filename" not in payload and "sha256" not in payload
