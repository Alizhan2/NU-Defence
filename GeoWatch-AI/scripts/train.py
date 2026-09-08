import argparse
from pathlib import Path

from ultralytics import YOLO

p=argparse.ArgumentParser()
p.add_argument("--data",required=True)
p.add_argument("--base",default="yolov8n-obb.pt")
p.add_argument("--epochs",type=int,default=80)
p.add_argument("--imgsz",type=int,default=1024)
p.add_argument("--batch",type=int,default=4)
p.add_argument("--device",default="0")
p.add_argument("--seed",type=int,default=42)
p.add_argument("--name",default="dota4_baseline")
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
model=YOLO(args.base)
model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,
            seed=args.seed,deterministic=True,project=str(root/"models"),name=args.name,exist_ok=True)
print(f"Веса: {root/'models'/args.name/'weights'/'best.pt'}")
