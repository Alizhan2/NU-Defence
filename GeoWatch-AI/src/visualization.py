from PIL import Image, ImageDraw
from .models import AnalysisResult

COLORS={"aircraft":"#55d6be","ship":"#4ea8de","small vehicle":"#f7b267","large vehicle":"#e76f51"}
def annotate(image, result: AnalysisResult):
    out=Image.fromarray(image).convert("RGB"); draw=ImageDraw.Draw(out)
    for d in result.detections:
        b=d.bbox; color=COLORS.get(d.class_name,"#55d6be")
        draw.rectangle((b.x1,b.y1,b.x2,b.y2),outline=color,width=max(2,out.width//700))
        draw.text((b.x1+3,max(0,b.y1-14)),f"{d.class_name} {d.confidence:.2f}",fill=color)
    return out
