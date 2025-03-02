import cv2 as cv
import numpy as np

class Vision():
    def __init__(self, debug_mode=False):
        
        """Initiates the Vision System for the Droid
        
        Args:
            Debug_mode (bool): If True, displays visualisation windows for debugging
        
        """
        
        self.debug_mode = debug_mode
        self.obstacle:bool = False
        # The following HSV values will need to be tested
        self.hsv_ranges = {
            'blue': {'lower': np.array([100, 150, 100]), 'upper': np.array([140, 255, 255])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([40, 255, 255])},
            'purple': {'lower': np.array([130, 100, 100]), 'upper': np.array([170, 255, 255])},
            'red1': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'red2': {'lower': np.array([170, 100, 100]), 'upper': np.array([180, 255, 255])}
        }
        
        
    def capture_frame(self, source=0):
        """
        Capture a frame from the camera or video source.
        
        Args:
            source: Camera index or video file path
            
        Returns:
            frame: Captured frame or None if capture failed
        """
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            exit()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            cv.imshow("frame", frame)
            if cv.waitKey(1000 // 100) == ord('q'):  # Adjust delay based on desired FPS
                break 
        cap.release()
        cv.destroyAllWindows()

# Example usage of the Vision system
Vision(debug_mode=True).capture_frame()