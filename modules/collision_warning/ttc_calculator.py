import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class TTCCalculator:
  def __init__(self, fps=30, smoothing=15):
    self.fps                 = fps
    self.smoothing           = smoothing
    self.previous_distances  = {}
    self.distance_history    = {}  # stores last N distances per car

  def calculate(self,estimates):
    results=[]

    for e in estimates:
      track_id=e["id"]
      distance=e["distance"]

      if track_id not in self.distance_history:
        self.distance_history[track_id]=[]

      self.distance_history[track_id].append(distance)
        
      if len(self.distance_history[track_id])>self.smoothing:
        self.distance_history[track_id].pop(0)

      smoothed_distance= sum(self.distance_history[track_id])/len(self.distance_history[track_id])
      
      if track_id in self.previous_distances:
          prev_distance = self.previous_distances[track_id]
          distance_change = prev_distance - smoothed_distance  
          relative_speed = distance_change * self.fps

          if relative_speed>0:
            ttc=smoothed_distance/relative_speed
          else:
            ttc=float('inf')
      else:
        ttc=float('inf') 

      self.previous_distances[track_id]=smoothed_distance
      risk=self._classify_risk(ttc)

      results.append({
        "id":track_id,
        "class_name":e["class_name"],
        "bbox":e["bbox"],
        "distance":round(smoothed_distance,1),
        "ttc":round(ttc,2) if ttc!=float('inf') else None,
        "risk":risk
      })

    return results
  
  def _classify_risk(self,ttc):
    if ttc is None or ttc ==float('inf'):
      return "SAFE"
    elif ttc<=config.TTC_HIGH_RISK:
      return "HIGH RISK"
    elif ttc<=config.TTC_CAUTION:
      return "CAUTION"
    elif ttc <=config.TTC_SAFE:
      return "SAFE"
    else:
      return "SAFE"
    
    
if __name__ == "__main__":
    import cv2
    from modules.object_detection.detector import Detector
    from modules.distance_estimation.estimator import Estimator

    detector   = Detector()
    estimator  = Estimator()
    calculator = TTCCalculator(fps=24)

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cv2.namedWindow("TTC Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections  = detector.detect(frame)
        estimates   = estimator.estimate(detections)
        ttc_results = calculator.calculate(estimates)

        for t in ttc_results:
            x1, y1, x2, y2 = t["bbox"]
            label = f'{t["class_name"]} #{t["id"]} | {t["distance"]}m | {t["ttc"]}s | {t["risk"]}'

            if t["risk"] == "HIGH RISK":
                color = config.RISK_COLOR
            elif t["risk"] == "CAUTION":
                color = config.CAUTION_COLOR
            else:
                color = config.SAFE_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("TTC Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()