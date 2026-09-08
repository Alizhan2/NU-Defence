from __future__ import annotations
import time, uuid
from typing import Any
import numpy as np
from .config import active_model_version, settings
from .models import AnalysisResult, BoundingBox, Detection, ImageMetadata, ModelStatus
from .postprocessing import class_aware_nms, clip_box
from .tiling import make_tiles

CLASS_ALIASES={"plane":"aircraft","airplane":"aircraft","aircraft":"aircraft","ship":"ship","small vehicle":"small vehicle","large vehicle":"large vehicle"}

class ModelUnavailable(RuntimeError): pass

class YoloDetector:
    def __init__(self, model_path=None, model_version=None):
        self.model_path = model_path or settings.model_path
        self.model_version = model_version or active_model_version()
        self._model = None; self._error = None
    def status(self):
        if not self.model_path.exists():
            return ModelStatus(state="missing", model_version=self.model_version, message_ru=f"Веса не найдены: {self.model_path}. Поместите обученный best.pt или задайте GEOWATCH_MODEL_PATH.")
        if self._error: return ModelStatus(state="load_error", model_version=self.model_version, message_ru=self._error)
        return ModelStatus(state="ready", model_version=self.model_version, message_ru="Модель настроена и готова к загрузке.")
    def _load(self):
        if not self.model_path.exists(): raise ModelUnavailable(self.status().message_ru)
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.model_path))
            except Exception as exc:
                self._error = f"Не удалось загрузить модель: {exc}"
                raise ModelUnavailable(self._error) from exc
        return self._model
    def detect_tile(self, image: np.ndarray, confidence: float) -> list[dict[str, Any]]:
        result = self._load().predict(image, conf=confidence, verbose=False)[0]
        output = []
        predictions = result.obb if getattr(result,"obb",None) is not None else result.boxes
        for box in predictions:
            raw_name=str(result.names[int(box.cls.item())]).lower().replace("_"," ").replace("-"," ").strip()
            name=CLASS_ALIASES.get(raw_name)
            if not name: continue
            if getattr(box,"xyxy",None) is not None: coords=box.xyxy[0].tolist()
            else:
                points=box.xyxyxyxy[0].cpu().numpy(); coords=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
            output.append({"class_name":name,"confidence":float(box.conf.item()),"bbox":coords})
        return output

def run_inference(image, metadata: ImageMetadata, detector=None, confidence=None, nms_iou=None):
    detector = detector or YoloDetector(); confidence = settings.confidence if confidence is None else confidence
    nms_iou = settings.nms_iou if nms_iou is None else nms_iou
    tiles = make_tiles(image, settings.tile_size, settings.overlap); started = time.perf_counter(); found=[]; serial=1
    for tile in tiles:
        ts=time.perf_counter(); raw=detector.detect_tile(tile.image, confidence); tile_ms=(time.perf_counter()-ts)*1000
        for item in raw:
            x1,y1,x2,y2=item["bbox"]
            bbox=clip_box(BoundingBox(x1=x1+tile.x,y1=y1+tile.y,x2=x2+tile.x,y2=y2+tile.y),metadata.width,metadata.height)
            found.append(Detection(id=f"det_{serial:06d}",class_name=item["class_name"],confidence=item["confidence"],bbox=bbox,processing_ms=tile_ms,model_version=detector.model_version)); serial+=1
    elapsed=(time.perf_counter()-started)*1000
    return AnalysisResult(analysis_id=str(uuid.uuid4()),image_id=metadata.sha256[:16],metadata=metadata,
        detections=class_aware_nms([d for d in found if d.confidence>=confidence],nms_iou),model_version=detector.model_version,
        confidence_threshold=confidence,inference_ms=elapsed,tile_count=len(tiles),limitations=["Текущая таксономия MVP не является окончательной таксономией заказчика.","Качество зависит от разрешения, облачности, освещения и угла съёмки.","Результаты AI требуют подтверждения аналитиком."])
