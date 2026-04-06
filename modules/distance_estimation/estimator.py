import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class Estimator:
  def __init__(self):
    self.focal_length=config.FOCAL_LENGTH
    self.known_vehicle_width=config.KNOWN_VEHICLE_WIDTH

  def estimate(self,detections):
      results=[]

      for d in detections:
        x1,y1,x2,y2=d["bbox"]
        pixel_width=x2-x1

        if pixel_width <=0:
          continue
        distance=(self.known_vehicle_width*self.focal_length)/pixel_width

        results.append({
          "id":d["id"],
          "class_name":d["class_name"],
          "bbox":d["bbox"],
          "distance":round(distance,2)
        })

      return results
  
if __name__ == "__main__":
    import cv2
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from modules.object_detection.detector import Detector

    detector  = Detector()
    estimator = Estimator()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cv2.namedWindow("Distance Estimation Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        estimates  = estimator.estimate(detections)

        for e in estimates:
            x1, y1, x2, y2 = e["bbox"]
            label = f'{e["class_name"]} #{e["id"]} | {e["distance"]}m'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Distance Estimation Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()