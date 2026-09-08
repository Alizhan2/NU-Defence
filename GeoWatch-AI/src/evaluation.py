from __future__ import annotations
from .models import BoundingBox, EvaluationMetrics
from .postprocessing import iou

def evaluate_records(predictions, ground_truth, model_version="unknown", threshold=.25, average_ms=0.0):
    matched=set(); tp=fp=correct=0
    for pred in sorted(predictions,key=lambda x:-x["confidence"]):
        best=None; best_iou=0
        for idx,truth in enumerate(ground_truth):
            score=iou(BoundingBox(**pred["bbox"]),BoundingBox(**truth["bbox"]))
            if idx not in matched and score>best_iou: best,best_iou=idx,score
        if best is not None and best_iou>=.5:
            tp+=1; matched.add(best); correct+=int(pred["class_name"]==ground_truth[best]["class_name"])
        else: fp+=1
    fn=len(ground_truth)-len(matched); precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    return EvaluationMetrics(precision=precision,recall=recall,f1_score=2*precision*recall/max(precision+recall,1e-12),map50=None,
        classification_accuracy=correct/max(tp,1),false_positives=fp,average_processing_ms=average_ms,test_images=1,
        model_version=model_version,confidence_threshold=threshold)
