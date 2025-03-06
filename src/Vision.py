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
        
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame
           
        
    def preprocess_frame(self, frame):
        """
        Preprocess the frame for better detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            hsv_frame: Converted HSV frame
            blurred_frame: Gaussian blurred frame
        """
        # Convert to HSV color space
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred_frame = cv.GaussianBlur(hsv_frame, (5,5), 0)
        
        if self.debug_mode:
            while True:
                yellow_mask_frame = self.detect_colour(frame, "blue")
                yellow_mask_frame_hsv = self.detect_colour(hsv_frame, "blue")
                cv.imshow("yellow_mask_frame", yellow_mask_frame)
                cv.imshow("yellow_mask_frame_hsv", yellow_mask_frame_hsv)
                
                if cv.waitKey(1000 // 100) == ord('q'):  # Adjust delay based on desired FPS
                    break 
            cv.destroyAllWindows()
            
        return hsv_frame, blurred_frame
    
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
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)
        
        if self.debug_mode:
            cv.imshow(f"{colour_key} Mask", mask)
            
        return mask
    
    def find_contours(self, mask):
        """Finds contour in a binary mask

        Args:
            mask (binary): binary mask for now
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_centroid(self, contour) -> tuple[int, int]: 
        M = cv.moments(contour)
        # print(type(M["m10"]))
        # The zeroth moment, which represents the area of the contour.
        if M["m00"] != 0:
            # Typecasting is crucial as the moments are in default float.
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return (cX, cY)

# Example usage of the Vision system
model = Vision(debug_mode=True)
frame = model.capture_frame()
model.preprocess_frame(frame)