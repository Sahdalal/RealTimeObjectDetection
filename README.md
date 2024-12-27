YOLOv5 Real-Time Object Detection
This repository contains a Python implementation of real-time object detection using YOLOv5. The program utilizes your system's webcam to detect and classify objects in real-time, drawing bounding boxes and labels on the video feed.

Features
Real-time object detection using a YOLOv5 model.
GPU support for faster detection (if CUDA is available).
Customizable detection thresholds for confidence and IOU.
Ability to take screenshots of detected frames.
Easy to extend with custom-trained models for additional object classes.
Installation
Prerequisites
Python 3.8 or higher
pip (Python package manager)
Required Libraries
Install the dependencies using the following command:

bash
Copy code
pip install torch torchvision opencv-python numpy
Usage
Clone the Repository:

bash
Copy code
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Run the Program: Use the following command to start the real-time detection:

bash
Copy code
python your_script_name.py
Key Commands:

Press q to quit the application.
Press s to save a screenshot of the current frame.
Customization
Modify Detection Thresholds
You can change the detection confidence or IOU thresholds by editing the YOLOv5Detector class in the script:

python
Copy code
detector = YOLOv5Detector(confidence=0.5, iou=0.5)
Use a Custom Model
To use a custom-trained YOLOv5 model, update the model_path in the YOLOv5Detector class:

python
Copy code
detector = YOLOv5Detector(model_path='path/to/your/custom_model.pt')
How It Works
Model Initialization:

Loads the YOLOv5 model (default: yolov5s) from the Ultralytics Hub.
Supports GPU acceleration if available.
Frame Processing:

Captures video frames from the webcam.
Converts frames to the expected RGB format and performs inference using YOLOv5.
Detection:

Draws bounding boxes, labels, and confidence scores on detected objects in the frame.
Display and Interaction:

Displays the processed video feed in a GUI window.
Allows users to take screenshots or quit the application with key presses.
Example Output

Bounding Boxes: Green rectangles around detected objects.
Labels: Object class and confidence score displayed above the bounding boxes.
FPS Display: Frame rate is displayed in the top-left corner.
Training Your Own Model
To detect custom objects, train YOLOv5 on your dataset:

Prepare a dataset and format it in YOLO format.
Train YOLOv5 using the following command:
bash
Copy code
python train.py --data custom_dataset.yaml --cfg yolov5s.yaml --weights yolov5s.pt --epochs 50
Replace model_path in the script with the path to your trained weights (best.pt).
Troubleshooting
Webcam Not Detected:

Ensure no other application is using the webcam.
Try changing the camera index in the script:
python
Copy code
cap = cv2.VideoCapture(1)
CUDA Not Found:

If you don’t have a GPU, the program will automatically use the CPU.
Install CUDA for GPU support if available.
Low Detection Accuracy:

Use a larger model (e.g., yolov5m or yolov5l).
Train a custom model on a more relevant dataset.
