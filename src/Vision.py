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
            'blue': {'lower': np.array([21,40,50]), 'upper': np.array([170,0,255])},
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
            self.detect_colour(frame, "yellow")
            if cv.waitKey(1000 // 100) == ord('q'):  # Adjust delay based on desired FPS
                break 
        cap.release()
        cv.destroyAllWindows()
        
        
    def detect_colour(self, hsv_frame, colour_key):
        """Due to the weak pigmentation of the tape 
        Morphological oeprations may help enhancing the track for better
        detection.
        
        Args:
            hsv_frame: Input HSV frame
            color_key: Color to detect ('blue', 'yellow', 'purple', 'red')
        
        Returns:
            mask: Binary mask of the detected color
        """
        
        colour_range = self.hsv_ranges[colour_key]
        mask = cv.inRange(hsv_frame, colour_range["lower"], colour_range["upper"])
        
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv.erode(mask, kernel, iterations=1)
        # mask = cv.dilate(mask, kernel, iterations=2)
        
        if self.debug_mode:
            cv.imshow(f"{colour_key} Mask", mask)
            
        return mask

# Example usage of the Vision system
Vision(debug_mode=True).capture_frame()