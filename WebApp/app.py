from flask import Flask, Response, render_template
import cv2
import depthai as dai

# Create a pipeline
pipeline = dai.Pipeline()
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
# Create a color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 360)  # Set the preview size
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)  # OpenCV expects BGR format

# Create an XLink output node for the video stream
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam_rgb.preview.link(xout_video.input)

app = Flask(__name__)


def selectObjectToTrack():
        
    success, frame = cap.read()
    if not success:
        print("Failed to capture video")
        return

    # Use selectROI for the initial bounding box
    bbox = cv2.selectROI("Object Tracking", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    ok = tracker.init(frame, bbox)
    return bbox



def generate_frames():
    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:

        # Output queue will be used to get the rgb frames from the output defined above
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        bbox_original = selectObjectToTrack()

        while True:
            in_video = q_video.get()  # Get the frame from the video output queue
            frame = in_video.getCvFrame()  # Convert the frame to an OpenCV-compatible format
            track_ok, bbox = tracker.update(frame)
            if track_ok:
            # Tracking success: draw a rectangle around the tracked object
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            else:
                # Tracking failure detected, reinitialize tracker with the same ROI
                tracker = cv2.TrackerCSRT_create()  # Reinitialize the tracker
                ok = tracker.init(frame, bbox)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
