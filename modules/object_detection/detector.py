from ultralytics import YOLO
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class Detector:
  def __init__(self):
    print(f"Loading YOLO model on{config.DEVICE}...")
    self.model=YOLO(config.YOLO_WEIGHTS)
    self.model.to(config.DEVICE)
    print("Model Loaded Successfully")

  def detect(self,frame):
    results=self.model.track(
      frame,
      persist=True,
      tracker="bytetrack.yaml",
      conf=config.YOLO_CONFIDENCE,
      classes=config.YOLO_CLASSES,
      verbose=False
    )
    return self._parse_results(results)
  
  def _parse_results(self,results):
    detections=[]

    for result in results:
      boxes=result.boxes

      if boxes is None:
        return detections
      
      for box in boxes:

        track_id=int(box.id.item()) if box.id is not None else -1

        class_id=int(box.cls.item())
        class_name=self.model.names[class_id]

        confidence=float(box.conf.item())

        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())

        detections.append({
          "id":  track_id,
          "class_id":class_id,
          "class_name":class_name,
          "confidence":confidence,
          "bbox":[x1,y1,x2,y2]
        })
    return detections
  
if __name__ == "__main__":
    import cv2

    detector = Detector()
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    cv2.namedWindow("ADAS - Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f'{d["class_name"]} #{d["id"]} {d["confidence"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # resize frame for display
        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow("ADAS - Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()