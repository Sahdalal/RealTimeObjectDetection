import torch
import cv2
import numpy as np
import time

class YOLOv5Detector:
    def __init__(self, model_name='yolov5s', confidence=0.45, iou=0.45):
        """
        Initialize YOLOv5 detector
        Args:
            model_name: YOLOv5 model variant to use
            confidence: Detection confidence threshold
            iou: NMS IOU threshold
        """
        try:
            print(f"Loading {model_name} model...")
            # Check if CUDA (GPU) is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Load the model
            self.model = torch.hub.load('ultralytics/yolov5', model_name)
            # Set model parameters
            self.model.conf = confidence
            self.model.iou = iou
            # Move model to GPU if available
            self.model.to(self.device)
            print(f"Model Loaded Successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_frame(self, frame):
        """
        Process a single frame for object detection
        """
        # Convert BGR to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.model(frame_rgb)
        return results

    def draw_detections(self, frame, results):
        """
        Draw Detection boxes and labels on frame
        """
        output_frame = frame.copy()
        
        detections = results.xyxy[0].cpu().numpy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            class_name = results.names[int(cls)]
            
            label = f'{class_name} {conf:.2f}'
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(output_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         (0, 255, 0),
                         -1)
            
            cv2.putText(output_frame,
                       label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 0),
                       2)
        
        return output_frame

    def run_detection(self):
        """
        Run Realtime detection using webcam
        """
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Video Properties: {frame_width}x{frame_height} @ {fps}fps")
            print("Press 'q' to quit, 's' to save a screenshot")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                start_time = time.time()
                results = self.process_frame(frame)
                
                output_frame = self.draw_detections(frame, results)
                
                fps = 1 / (time.time() - start_time)
                cv2.putText(output_frame,
                           f'FPS: {fps:.1f}',
                           (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 255, 0),
                           2)
                
                cv2.imshow("YOLOv5 Detection", output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'detection_{timestamp}.jpg', output_frame)
                    print(f"Screenshot saved as detection_{timestamp}.jpg")
                    
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        detector = YOLOv5Detector(model_name="yolov5x")
        detector.run_detection()
    except Exception as e:
        print(f"Program Failed: {str(e)}")

if __name__ == "__main__":
    main()