#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from collections import deque
from scipy.spatial import distance

class FeatureTrackerDrawer:
    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    trackedFeaturesPathLength = 10
    proximityThreshold = 50  # Distance threshold to consider features as 'close'

    def __init__(self, trackbarName, windowName):
        self.trackbarName = trackbarName
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar(trackbarName, windowName, self.trackedFeaturesPathLength, self.maxTrackedFeaturesPathLength, self.onTrackBar)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()

    def onTrackBar(self, val):
        self.trackedFeaturesPathLength = val

    def trackFeaturePath(self, features):
        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)
            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()
            path = self.trackedFeaturesPath[currentID]
            path.append(currentFeature.position)
            while len(path) > max(1, self.trackedFeaturesPathLength):
                path.popleft()
            self.trackedFeaturesPath[currentID] = path
        featuresToRemove = self.trackedIDs - newTrackedIDs
        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)
        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img):
        feature_positions = []
        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath
            for j in range(len(path) - 1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j + 1].x), int(path[j + 1].y))
                cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA)
            feature_positions.append((int(path[-1].x), int(path[-1].y)))
        self.drawBoundingBoxes(feature_positions, img)

    def drawBoundingBoxes(self, feature_positions, img):
        if len(feature_positions) < 15:
            return  # Only draw if there are enough features
        clusters = self.findClusters(feature_positions)
        for cluster in clusters:
            if len(cluster) >= 15:
                x_min = min([pos[0] for pos in cluster])
                y_min = min([pos[1] for pos in cluster])
                x_max = max([pos[0] for pos in cluster])
                y_max = max([pos[1] for pos in cluster])
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def findClusters(self, feature_positions):
        clusters = []
        for pos in feature_positions:
            found_cluster = False
            for cluster in clusters:
                if any(distance.euclidean(pos, cpos) < self.proximityThreshold for cpos in cluster):
                    cluster.append(pos)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([pos])
        return clusters
    
    def calculate_depth(self, img, bbox):
        depth_frame = self.depthQueue.get().getCvFrame()  # Get the latest depth frame
        depth_crop = depth_frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        depth_value = np.nanmean(depth_crop)  # Calculate average depth within the box
        print(f"Depth: {depth_value:.2f} mm")

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
colorCam = pipeline.create(dai.node.ColorCamera)

featureTrackerColor = pipeline.create(dai.node.FeatureTracker)

xoutPassthroughFrameColor = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesColor = pipeline.create(dai.node.XLinkOut)
xinTrackedFeaturesConfig = pipeline.create(dai.node.XLinkIn)

xoutPassthroughFrameColor.setStreamName("passthroughFrameColor")
xoutTrackedFeaturesColor.setStreamName("trackedFeaturesColor")
xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# Properties
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

if 1:
    colorCam.setIspScale(2, 3)
    colorCam.video.link(featureTrackerColor.inputImage)
else:
    colorCam.isp.link(featureTrackerColor.inputImage)

# Linking
featureTrackerColor.passthroughInputImage.link(xoutPassthroughFrameColor.input)
featureTrackerColor.outputFeatures.link(xoutTrackedFeaturesColor.input)
xinTrackedFeaturesConfig.out.link(featureTrackerColor.inputConfig)

# By default the least amount of resources are allocated
# increasing it improves performance
numShaves = 2
numMemorySlices = 2
featureTrackerColor.setHardwareResources(numShaves, numMemorySlices)
featureTrackerConfig = featureTrackerColor.initialConfig.get()

print("Press 's' to switch between Lucas-Kanade optical flow and hardware accelerated motion estimation!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues used to receive the results
    passthroughImageColorQueue = device.getOutputQueue("passthroughFrameColor", 8, False)
    outputFeaturesColorQueue = device.getOutputQueue("trackedFeaturesColor", 8, False)
    inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    colorWindowName = "color"
    colorFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", colorWindowName)

    while True:
        inPassthroughFrameColor = passthroughImageColorQueue.get()
        passthroughFrameColor = inPassthroughFrameColor.getCvFrame()
        colorFrame = passthroughFrameColor


        trackedFeaturesColor = outputFeaturesColorQueue.get().trackedFeatures
        colorFeatureDrawer.trackFeaturePath(trackedFeaturesColor)
        colorFeatureDrawer.drawFeatures(colorFrame)

        # Show the frame
        cv2.imshow(colorWindowName, colorFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if featureTrackerConfig.motionEstimator.type == dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.HW_MOTION_ESTIMATION
                print("Switching to hardware accelerated motion estimation")
            else:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW
                print("Switching to Lucas-Kanade optical flow")

            cfg = dai.FeatureTrackerConfig()
            cfg.set(featureTrackerConfig)
            inputFeatureTrackerConfigQueue.send(cfg)
