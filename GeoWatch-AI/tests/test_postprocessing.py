from src.models import BoundingBox,Detection
from src.postprocessing import class_aware_nms

def det(id,cls,conf): return Detection(id=id,class_name=cls,confidence=conf,bbox=BoundingBox(x1=0,y1=0,x2=100,y2=100),model_version="test")
def test_nms_is_class_aware():
    assert [d.id for d in class_aware_nms([det("1","ship",.9),det("2","ship",.8),det("3","aircraft",.7)],.5)]==["1","3"]
