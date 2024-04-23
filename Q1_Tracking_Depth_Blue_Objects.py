import depthai as dai
import cv2
import numpy as np

def create_pipeline():
    pipeline = dai.Pipeline()

    # Create RGB camera node
    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    # Create StereoDepth node
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo.setConfidenceThreshold(255)
    
    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # XLinkOut
    xoutRgb = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    xoutDepth.setStreamName("depth")
    camRgb.video.link(xoutRgb.input)
    stereo.depth.link(xoutDepth.input)

    return pipeline

def detect_blue_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Clean up mask and find contours
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    return frame, mask, contours

pipeline = create_pipeline()

with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        frameRgb = videoQueue.tryGet()
        frameDepth = depthQueue.tryGet()

        if frameRgb is not None:
            image = frameRgb.getCvFrame()
            detected_frame, mask, contours = detect_blue_objects(image)

            cv2.imshow("Detected Frame", detected_frame)

        if frameDepth is not None and contours:
            depthFrame = frameDepth.getFrame()

            for cnt in contours:
                if cv2.contourArea(cnt) > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    depth_roi = depthFrame[y:y + h, x:x + w]
                    depth_value = np.mean(depth_roi[depth_roi > 0])  # Average depth in mm, ignoring zeros
                    cv2.putText(image, f"{depth_value:.2f} mm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
            cv2.imshow("RGB with Depth", image)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
