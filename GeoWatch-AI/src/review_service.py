from __future__ import annotations
import json
import re
from pathlib import Path
from .config import settings
from .models import AnalysisResult, ReviewEvent, ReviewRecord, ReviewRequest

class ReviewStore:
    def __init__(self, root: Path | None=None): self.root=root or settings.runs_dir; self.root.mkdir(parents=True,exist_ok=True)
    def path(self, analysis_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", analysis_id): raise KeyError("Некорректный ID анализа.")
        return self.root/f"analysis_{analysis_id}.json"
    def save(self,result: AnalysisResult):
        target=self.path(result.analysis_id); tmp=target.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=2),encoding="utf-8"); tmp.replace(target)
    def load(self,image_id):
        if not re.fullmatch(r"[a-f0-9]{16,64}",image_id): raise KeyError("Некорректный ID изображения.")
        matches=[]
        for target in self.root.glob("analysis_*.json"):
            try: result=AnalysisResult.model_validate_json(target.read_text(encoding="utf-8"))
            except Exception: continue
            if result.image_id==image_id: matches.append(result)
        if not matches: raise KeyError("Результат анализа не найден.")
        return max(matches,key=lambda item:item.created_at)
    def load_by_analysis(self,analysis_id):
        try:
            target=self.path(analysis_id)
            if target.exists(): return AnalysisResult.model_validate_json(target.read_text(encoding="utf-8"))
        except KeyError: pass
        for target in self.root.glob("analysis_*.json"):
            try: result=AnalysisResult.model_validate_json(target.read_text(encoding="utf-8"))
            except Exception: continue
            if result.analysis_id==analysis_id: return result
        raise KeyError("Анализ не найден.")
    def all_results(self):
        records=[]
        for target in self.root.glob("analysis_*.json"):
            try: records.append(AnalysisResult.model_validate_json(target.read_text(encoding="utf-8")))
            except Exception: continue
        return sorted(records,key=lambda item:item.created_at,reverse=True)
    def review(self,request: ReviewRequest):
        result=self.load_by_analysis(request.analysis_id)
        for det in result.detections:
            if det.id==request.detection_id:
                det.review=ReviewRecord(detection_id=det.id,status=request.status,comment=request.comment)
                self.save(result)
                event=ReviewEvent(analysis_id=result.analysis_id,image_id=result.image_id,detection_id=det.id,status=request.status,comment=request.comment,timestamp=det.review.timestamp)
                with (self.root / "review_events.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(event.model_dump_json()+"\n")
                return det.review
        raise KeyError("Обнаружение не найдено.")

    def events(self, limit: int=50) -> list[ReviewEvent]:
        target=self.root / "review_events.jsonl"
        if not target.exists(): return []
        records=[]
        for line in target.read_text(encoding="utf-8").splitlines():
            try: records.append(ReviewEvent.model_validate_json(line))
            except (ValueError, json.JSONDecodeError): continue
        return list(reversed(records[-limit:]))

store=ReviewStore()
