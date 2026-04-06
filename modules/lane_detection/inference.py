import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class LaneDetector:
    def __init__(self):
        self.input_height = config.LANE_INPUT_HEIGHT
        self.input_width  = config.LANE_INPUT_WIDTH

    def _preprocess(self, frame):
        # focus only on bottom half of frame (where lanes are)
        height = frame.shape[0]
        roi = frame[height//2: int(height*0.85), :]
        
        # convert to grayscale and blur
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # edge detection
        edges = cv2.Canny(blurred, 50, 150)

        return edges, height//2
    
    def _detect_lines(self, edges):
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )
        return lines
    
    def _separate_lanes(self, lines, width):
        left_lines  = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 == x1:
                continue  # skip vertical lines
            
            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.3:
                left_lines.append(line[0])
            elif slope > 0.3:
                right_lines.append(line[0])

        return left_lines, right_lines
    
    def _average_line(self, lines, height, roi_offset):
        if not lines:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        poly    = np.polyfit(y_coords, x_coords, 1)
        y_start = height - roi_offset
        y_end   = int(height * 0.05)

        x_start = int(np.polyval(poly, y_start))
        x_end   = int(np.polyval(poly, y_end))

        return (x_start, y_start + roi_offset,
                x_end,   y_end   + roi_offset)
    
    def detect(self, frame):
        edges, roi_offset = self._preprocess(frame)
        lines             = self._detect_lines(edges)
        left_lines, right_lines = self._separate_lanes(lines, frame.shape[1])

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        left_lane  = self._average_line(left_lines,  frame.shape[0], roi_offset)
        right_lane = self._average_line(right_lines, frame.shape[0], roi_offset)

        for lane in [left_lane, right_lane]:
            if lane is not None:
                x1, y1, x2, y2 = lane
                cv2.line(mask, (x1, y1), (x2, y2), 255, 8)

        return mask
    
if __name__ == "__main__":
  detector = LaneDetector()
  cap = cv2.VideoCapture(config.VIDEO_SOURCE)
  cv2.namedWindow("Lane Test", cv2.WINDOW_NORMAL)

  while True:
      ret, frame = cap.read()
      if not ret:
          break

      mask   = detector.detect(frame)
      overlay = frame.copy()
      overlay[mask > 0] = [0, 255, 0]
      result = cv2.addWeighted(frame, 0.8, overlay, 0.4, 0)

      result = cv2.resize(result, (1280, 720))
      cv2.imshow("Lane Test", result)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()