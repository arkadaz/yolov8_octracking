from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import ocsort
import numpy as np
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
img_info = (480, 680)
tracker = ocsort.OCSort(det_thresh=0.30, max_age=10, min_hits=2)

while True:
    _, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img, iou=0.3, conf=0.5)

    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes
        xyxyc = np.hstack((boxes.xyxy,
                           np.c_[boxes.conf]))
        print(xyxyc)
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        tracks = tracker.update(xyxyc, img_info, img_info)

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

    frame = annotator.result()
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
