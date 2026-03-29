import os 
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE=os.path.join(BASE_DIR,"data","raw","sample_videos","test.mp4")
UNET_WEIGHTS=os.path.join(BASE_DIR,"models","unet_lane.pth")
YOLO_WEIGHTS=os.path.join(BASE_DIR,"models","yolov8n.pt")

OUTPUT_VIDEO=os.path.join(BASE_DIR,"outputs","output.avi")
OUTPUT_FRAMES=os.path.join(BASE_DIR,"outputs","annotated_frames")

import torch
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

LANE_INPUT_HEIGHT=256
LANE_INPUT_WIDTH=512
LANE_THRESHOLD=0.5

YOLO_CONFIDENCE=0.01
YOLO_CLASSES=[0,2,3,5,7]

KNOWN_VEHICLE_WIDTH=1.8
FOCAL_LENGTH=700

TTC_SAFE=5.0
TTC_CAUTION=3.0
TTC_HIGH_RISK=1.5
 
LANE_COLOR=(0,255,0) #GREEN
BOX_COLOR=(255,0,0)  #BLUE
SAFE_COLOR=(0,255,0) #GREEN
CAUTION_COLOR=(0,255,255) #YELLOW
RISK_COLOR=(0,0,255)      #RED
FONT_COLOR=0.6
FONT_THICKNESS=2