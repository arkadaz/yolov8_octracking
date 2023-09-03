from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import ocsort
import numpy as np
model = YOLO('yolov8n.pt')
model.export(format='onnx')
