import cv2
import sys
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from modules.object_detection.detector import Detector
from modules.distance_estimation.estimator import Estimator
from modules.collision_warning.ttc_calculator import TTCCalculator
from modules.visualization.display import Display
from modules.lane_detection.inference import LaneDetector

def main():

  detector     = Detector()
  estimator    = Estimator()
  calculator   = TTCCalculator(fps=24)
  display      = Display()
  lane_detector = LaneDetector()

  cap=cv2.VideoCapture(config.VIDEO_SOURCE)
  cv2.namedWindow("Mini ADAS",cv2.WINDOW_NORMAL)

  print("pipeline running... press q to quit")
  print(f"Cap opened: {cap.isOpened()}") 

  while True:
    ret,frame=cap.read()
    print(f"ret: {ret}")
    if not ret:
      break

    detections  = detector.detect(frame)
    estimates   = estimator.estimate(detections)
    ttc_results = calculator.calculate(estimates)
    lane_mask   = lane_detector.detect(frame)

    frame=display.draw(frame,ttc_results,lane_mask)

    frame=cv2.resize(frame,(1280,720))
    cv2.imshow("Mini ADAS",frame)

    if cv2.waitKey(1)& 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()
  print("Pipeline stopped.")

if __name__=="__main__":
  main()