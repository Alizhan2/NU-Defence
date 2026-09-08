from __future__ import annotations
import json
from .config import settings
from .models import AnalysisResult, AnalystResponse, AnalystSummary

SYSTEM_PROMPT='''Ты — аналитический помощник GeoWatch AI.
Работай только с предоставленными metadata, изображением и JSON-результатами детектора.
Не добавляй объекты, события, координаты, назначения или выводы, которых нет во входных данных.
Не называй объект угрозой, военной целью или подтверждённым событием.
Используй формулировки: «обнаружение модели», «кандидат на проверку», «требуется подтверждение экспертом».
Отвечай только валидным JSON согласно схеме и на русском языке.'''

def safe_payload(result: AnalysisResult):
    return {"metadata":{"format":result.metadata.format,"width":result.metadata.width,"height":result.metadata.height,"crs":result.metadata.crs},"detections":[{"id":d.id,"class_name":d.class_name,"confidence":d.confidence,"bbox":d.bbox.model_dump(),"expert_status":d.review.status if d.review else "needs_review","expert_comment":d.review.comment if d.review else ""} for d in result.detections]}

class QwenAnalyst:
    def __init__(self,enabled=None,client=None): self.enabled=settings.qwen_enabled if enabled is None else enabled; self.client=client
    def summarize(self,result: AnalysisResult):
        if not self.enabled: return AnalystResponse(available=False,message_ru="Qwen не настроен. Основной анализ YOLO доступен")
        try:
            raw=self.client(SYSTEM_PROMPT,json.dumps(safe_payload(result),ensure_ascii=False)) if self.client else self._local(safe_payload(result))
            parsed=AnalystSummary.model_validate_json(raw)
            ids={d.id for d in result.detections}
            if any(f.detection_id not in ids for f in parsed.findings): raise ValueError("Qwen сослался на неизвестное обнаружение.")
            return AnalystResponse(available=True,message_ru="Сводка сформирована.",summary=parsed)
        except Exception as exc:
            return AnalystResponse(available=False,message_ru=f"Qwen временно недоступен; основной pipeline не затронут: {exc}")
    def _local(self,payload):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model=AutoModelForImageTextToText.from_pretrained(settings.qwen_model_id,device_map=settings.qwen_device,trust_remote_code=True)
        processor=AutoProcessor.from_pretrained(settings.qwen_model_id,trust_remote_code=True)
        prompt=SYSTEM_PROMPT+"\nВходные данные:\n"+json.dumps(payload,ensure_ascii=False)
        inputs=processor(text=prompt,return_tensors="pt").to(model.device)
        output=model.generate(**inputs,max_new_tokens=settings.qwen_max_new_tokens)
        return processor.decode(output[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
