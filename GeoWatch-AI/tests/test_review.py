from src.models import AnalysisResult,BoundingBox,Detection,ImageMetadata,ReviewRequest
from src.review_service import ReviewStore

def sample(): return AnalysisResult(analysis_id="a",image_id="0"*16,metadata=ImageMetadata(filename="x.png",format="PNG",width=10,height=10,sha256="0"*64),detections=[Detection(id="det_1",class_name="ship",confidence=.8,bbox=BoundingBox(x1=1,y1=1,x2=2,y2=2),model_version="m")],model_version="m",confidence_threshold=.2,inference_ms=1,tile_count=1)
def test_review_persists(tmp_path):
    store=ReviewStore(tmp_path); store.save(sample()); store.review(ReviewRequest(analysis_id="a",detection_id="det_1",status="confirmed",comment="ok"))
    assert store.load("0"*16).detections[0].review.comment=="ok"
    events=store.events()
    assert events[0].status=="confirmed"
    assert events[0].detection_id=="det_1"
