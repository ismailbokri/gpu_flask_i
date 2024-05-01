from flask import Flask, render_template, request,Response
import cv2,imutils,time
import pyshine as ps
from IPython import display
import supervision as sv
from IPython import display
import ultralytics
from ultralytics import YOLO
import supervision as sv
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def index():
   return render_template('index.html')



def model(params):
    MODEL_pollen = "./weights/best_pollen.pt "
    model_p = YOLO(MODEL_pollen)
    CAMERA = False
    if CAMERA:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('./videos/video.mp4')
    selected_classes = [1]
    LINE_START = sv.Point(50, 1500)
    LINE_END = sv.Point(3840 - 50, 1500)
    SOURCE_VIDEO_PATH = f"./videos/video.mp4"
    TARGET_VIDEO_PATH = f"./videos/video_m.mp4"
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    # create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # create LineZone instance, it is previously called LineCounter class
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=3, text_scale=2)

    # create instance of TraceAnnotator
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=52)

    # create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=3, text_scale=1)

    # define call back function to be used in video processing
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # model prediction on single frame and conversion to supervision Detections
        results = model_p(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # only consider class id from selected_classes define above
        detections = detections[np.isin(detections.class_id, selected_classes)]
        # tracking detections
        detections = byte_tracker.update_with_detections(detections)
        labels = [
            f"{tracker_id}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        annotated_frame = trace_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

        # update line counter
        line_zone.trigger(detections)
        # return frame with box and line annotated result
        return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # process the whole video
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )


def pyshine_process():
    MODEL_pollen = "./weights/best_pollen.pt"
    model_p = YOLO(MODEL_pollen)

    CAMERA = False
    if CAMERA:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('./videos/video.mp4')
    print('FUNCTION DONE')
    # Read until video is completed
    fps = 0
    st = 0
    frames_to_count = 20
    cnt = 0

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if cnt == frames_to_count:
                try:  # To avoid divide by 0 we put it in try except
                    fps = round(frames_to_count / (time.time() - st))
                    st = time.time()
                    cnt = 0
                except ZeroDivisionError:
                    pass
            cnt += 1

            # Perform object detection with YOLOv8
            results = model_p.predict(img)
            for obj in results:
                print(obj.names)
                for box in obj.boxes:
                    conf = box.data[:, 4]
                    class_d = box.data[:, 5]
                    if obj.names[int(class_d)] == "pollen" and conf > 0.75:
                        text = 'Pollen detected'
                        img = ps.putBText(img, text, text_offset_x=20, text_offset_y=30, background_RGB=(10, 20, 222))

            frame = cv2.imencode('.JPEG', img, [cv2.IMWRITE_JPEG_QUALITY, 20])[1].tobytes()

            cv2.waitKey(1)  # Wait for a key press
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

    cap.release()  # Release the video capture


@app.route('/res',methods = ['POST','GET'])
def res():
	global result
	if request.method == 'POST':
		fichier_mp4 = request.files["videoFile"]
		fichier_mp4.save("./videos/video.mp4")
		return render_template("results.html")

@app.route('/results')
def video_feed():
	return Response(pyshine_process(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)

