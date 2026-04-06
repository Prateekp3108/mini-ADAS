import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class Display:
  def __init__(self):
    self.font=cv2.FONT_HERSHEY_SIMPLEX
    self.font_scale=config.FONT_SCALE
    self.thickness=config.FONT_THICKNESS

  def draw_detections(self,frame,ttc_results):
    for t in ttc_results:
      x1,y1,x2,y2=t["bbox"]

      if t["risk"]=="HIGH RISK":
        color=config.RISK_COLOR
      elif t["risk"]=="CAUTION":
        color=config.CAUTION_COLOR
      else:
        color=config.SAFE_COLOR

      cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

      ttc_text=f'{t["ttc"]}s' if t["ttc"] is not None else "N/A"
      label=f'{t["class_name"]}#{t["id"]} | {t["distance"]}m | {ttc_text} | {t["risk"]}'

      (w,h),_=cv2.getTextSize(label,self.font,self.font_scale,self.thickness)
      cv2.rectangle(frame,(x1,y1-h-10),(x1+w,y1),color,-1)

      cv2.putText(frame,label,(x1,y1-5),self.font,self.font_scale,(0,0,0),self.thickness)

    return frame
    
  def draw_warning_banner(self,frame,ttc_results):

    risks=[t["risk"] for t in ttc_results]

    if "HIGH RISK" in risks:
      banner_color=config.RISK_COLOR
      banner_text="FORWARD COLLISION WARNING"
    elif "CAUTION" in risks:
      banner_color=config.CAUTION_COLOR
      banner_text="CAUTION-SLOW DOWN"
    else:
      return frame
    
    cv2.rectangle(frame,(0,0),(frame.shape[1],60),banner_color,-1)
    cv2.putText(frame,banner_text,(frame.shape[1]//2-300,40),self.font,1.2,(255,255,255),3)

    return frame
  
  def draw(self,frame,ttc_results,lane_mask=None):

    if lane_mask is not None:
      frame=self.draw_lane(frame,lane_mask)
    
    frame=self.draw_detections(frame,ttc_results)
    frame=self.draw_warning_banner(frame,ttc_results)

    return frame
  
  def draw_lane(self,frame,lane_mask):
    colored_mask=cv2.cvtColor(lane_mask,cv2.COLOR_GRAY2BGR)
    colored_mask[lane_mask>0]=config.LANE_COLOR

    frame=cv2.addWeighted(frame,1.0,colored_mask,0.4,0)
    return frame

if __name__ == "__main__":
    import cv2
    from modules.object_detection.detector import Detector
    from modules.distance_estimation.estimator import Estimator
    from modules.collision_warning.ttc_calculator import TTCCalculator

    detector   = Detector()
    estimator  = Estimator()
    calculator = TTCCalculator(fps=24)
    display    = Display()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cv2.namedWindow("ADAS Display Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections  = detector.detect(frame)
        estimates   = estimator.estimate(detections)
        ttc_results = calculator.calculate(estimates)
        frame       = display.draw(frame, ttc_results)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("ADAS Display Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()