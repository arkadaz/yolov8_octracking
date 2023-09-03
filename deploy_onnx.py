import onnxruntime as rt
from PIL import Image
import numpy as np
import cv2
import ocsort

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2-x1)*(y2-y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2)/union(box1, box2)


def discard(predict, iou_th, conf_th):
    bboxs = []
    for all_preds in predict[0].transpose():
        if all_preds[4:].max() > conf_th and all_preds[4:].argmax() == 0:
            xc, yc, w, h = all_preds[:4]
            x1 = (xc-w/2)/640*680
            y1 = (yc-h/2)/640*480
            x2 = (xc+w/2)/640*680
            y2 = (yc+h/2)/640*480
            class_id = all_preds[4:].argmax()
            bboxs.append(
                [x1, y1, x2, y2, all_preds[4:].max(), class_id])
    bboxs.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(bboxs) > 0:
        result.append(bboxs[0])
        bboxs = [box for box in bboxs if iou(box, bboxs[0]) < iou_th]
    return result


EP_list = ['CUDAExecutionProvider']
sess = rt.InferenceSession("yolov8n.onnx", providers=EP_list)
output_name = sess.get_outputs()[0].name
input_name = sess.get_inputs()[0].name
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tracker = ocsort.OCSort(det_thresh=0.30, max_age=10, min_hits=2)
while True:
    r, frame = cap.read()
    image_data = (cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640), interpolation=cv2.INTER_AREA).transpose(
        2, 0, 1).reshape(1, 3, 640, 640)/255.0).astype(np.float32)
    detections = sess.run([output_name], {input_name: image_data})[0]
    result = discard(detections, 0.7, 0.5)
    # print(len(bboxs))
    for box in result:
        cv2.rectangle(frame, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (255, 0, 0), 2)
    try:
        xyxyc = np.array(result)[:, :5]
    except:
        xyxyc = None
    tracks = tracker.update(xyxyc, (480, 680), (480, 680))
    for track in tracker.trackers:
        track_id = track.id
        hits = track.hits
        color = (0, 0, 255)
        x1, y1, x2, y2 = np.round(track.get_state()).astype(int).squeeze()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"track{track_id}-{hits}",
                    (x1+10, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA)
    cv2.imshow("bbox", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
