import json
import cv2
# import cv2.cv2 as cv2
import depthai as dai
import argparse

import numpy as np
import requests
import os
from depthai_sdk import Previews, PreviewManager, PipelineManager, frameNorm

directory = r"C:\Users\ckirc\Pictures\plant detection"
blobdir = r'C:\Users\ckirc\Documents\hslu\semester_5\PREN\OpenCV\PREN_AVehicle_Plant_id\PREN_Plant_Id\detect\detect.blob'
api_key = '2b10glUixSPZOunMJ952kc5Pe'
url = f'https://my-api.plantnet.org/v2/identify/all?api-key={api_key}'
nr_imgs = 3
os.chdir(directory)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--video_detection', action='store_true', help="debug only, displays video out and detects qr codes, works with -pi")
parser.add_argument('-r', '--raw', action='store_true', help="use nn to detect qr code")
parser.add_argument('-pi', '--picam', action='store_true', help="use default sys cam")
parser.add_argument('-v', '--video', action='store_true', help="output video")
args = parser.parse_args()

def detect_plants(use_pi, video_out):

    if args.pi or use_pi:
        # setup resources
        cap = cv2.VideoCapture(0)
        detector = cv2.QRCodeDetector()
        take_images(cap, detector, use_pi, video_out)

        #debug
        # show_images()
        send_imgs_to_API()

        # release all ressources
        cap.release()
        cv2.destroyAllWindows()
    else:
        pipeline = dai.Pipeline()

        # Define source and outputs
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_video = pipeline.create(dai.node.XLinkOut)

        xout_video.setStreamName("video")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setVideoSize(1920, 1080)
        cam_rgb.setInterleaved(False)

        xout_video.input.setBlocking(False)
        xout_video.input.setQueueSize(1)

        # Linking
        cam_rgb.video.link(xout_video.input)

        # qr detector
        detector = cv2.QRCodeDetector()

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            video = device.getOutputQueue(xout_video.getStreamName(), maxSize=1, blocking=False)
            take_images(video, detector, use_pi, video_out)
            show_images()
            send_imgs_to_API()


def run_detection(cap, detector, use_pi, video_out):
    while True:
        if use_pi:
            _, img = cap.read()
        else:
            img = cap.get().getCvFrame()
        data, bbox, _ = detector.detectAndDecode(img)
        if video_out:
            cv2.imshow("code detector", img)
            cv2.waitKey(1)
        if bbox is not None and data:
            if data:
                print("data found: ", data)
                break
    cv2.destroyAllWindows()
    return data, img


def run_detection_with_video_noexit(use_pi):

    if use_pi:
        # set up camera object
        cap = cv2.VideoCapture(0)

        # QR code detection object
        detector = cv2.QRCodeDetector()

        while True:
            # get the image
            _, img = cap.read()
            # get bounding box coords and data
            img = video_detection(img, detector)
            # display the image preview
            cv2.imshow("code detector", img)
            if cv2.waitKey(1) == ord("q"):
                break
        # free camera object and exit
        cap.release()
        cv2.destroyAllWindows()
    else:
        pipeline = dai.Pipeline()

        # Define source and outputs
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_video = pipeline.create(dai.node.XLinkOut)

        xout_video.setStreamName("video")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setVideoSize(1920, 1080)
        cam_rgb.setInterleaved(False)

        xout_video.input.setBlocking(False)
        xout_video.input.setQueueSize(1)

        # Linking
        cam_rgb.video.link(xout_video.input)

        # qr detector
        detector = cv2.QRCodeDetector()

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            video = device.getOutputQueue(xout_video.getStreamName(), maxSize=1, blocking=False)

            while True:
                frame = video.get().getCvFrame()
                frame = video_detection(frame, detector)
                cv2.imshow(xout_video.getStreamName(), frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            cv2.destroyAllWindows()


def video_detection(img, detector):
    data, bbox, _ = detector.detectAndDecode(img)

    # if there is a bounding box, draw one, along with the data
    # (len(data) > 0)
    if bbox is not None and data:
        points = bbox[0]
        for i in range(len(points)):
            pt1 = [int(val) for val in points[i]]
            pt2 = [int(val) for val in points[(i + 1) % 4]]
            cv2.line(img, pt1, pt2, color=(255, 0, 0), thickness=3)
        cv2.putText(img, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        if data:
            print("data found: ", data)
    return img


def show_images():
    x = 0
    while x < nr_imgs:
        name_img = f'{directory}\\plant_pic_{x}.jpg'
        img = cv2.imread(name_img)
        x += 1
        if img.data:
            cv2.imshow(name_img, img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        return

#debug
def plant_qr_detector(cap, detector):
    qr_found = False
    while not qr_found:
        _, img = cap.read()
        data, bbox, _ = detector.detectAndDecode(img)
        cv2.imshow("code detector", img)
        cv2.waitKey(1)
        if bbox is not None and data:
            qr_found = True
            print("data found: ", data)
            stop_vehicle()
    cap.release()
    cv2.destroyAllWindows()
    return data, img


def take_images(cap, detector, use_pi=True, video_out=False):
    x = 0
    attempt = 0
    while x < nr_imgs and attempt < 10:
        # _, img = plant_qr_detector()
        _, img = run_detection(cap, detector, use_pi, video_out)
        attempt += 1
        if img.any():
            cv2.imwrite(f'{directory}\\plant_pic_{x}.jpg', img)
            x += 1
            move_a_little_forward()
            print("img taken")
        else:
            move_a_little_backward()


def send_imgs_to_API():
    x = 0
    while x < nr_imgs:
        name_img = f'{directory}\\plant_pic_{x}.jpg'
        img = open(name_img, 'rb')
        if img:
            files = [('images', (name_img, img))]
            data = {'organs': ['leaf']}
            # , 'multipart/form-data', {'Expires': '0'}
            req = requests.Request('POST', url=url, files=files, data=data)
            prep = req.prepare()
            s = requests.Session()
            r = s.send(prep)
            print(json.loads(r.text))
            # with requests.Session() as s:
            #     r = s.post(url, files=files, data=data)
            #     print(json.loads(r.text))
        x += 1


def stop_vehicle():
    return


def move_a_little_forward():
    return


def move_a_little_backward():
    return


def continue_vehicle():
    return


def event_to_server(message):
    print(message)
    # send message to server


def raw_detection():
    classes = ["nothing", "QR_CODE"]
    pipeline = dai.Pipeline()
    pm = PipelineManager()
    cam = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.NeuralNetwork)
    xout_preview = pipeline.create(dai.node.XLinkOut)

    cam.setPreviewSize(384, 384)
    cam.setInterleaved(False)
    nn.setBlobPath(blobdir)

    xout_preview.setStreamName("preview")
    cam.preview.link(nn.input)
    cam.preview.link(xout_preview.input)

    # Send NN out to the host via XLink
    nnXout = pipeline.create(dai.node.XLinkOut)
    nnXout.setStreamName("nn")
    nn.out.link(nnXout.input)

    with dai.Device(pipeline) as device:
        qNn = device.getOutputQueue(nnXout.getStreamName())
        pv = device.getOutputQueue(xout_preview.getStreamName(), maxSize=1, blocking=False)
        while True:
            nnData = qNn.get()  # Blocking
            frame = pv.get().getCvFrame()
            bboxes = np.array(nnData.getLayerFp16("detection_output"))
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.5]
            labels = bboxes[:, 1].astype(int)
            confidences = bboxes[:, 2]
            bboxes = bboxes[:, 3:7]
            for label, conf, raw_bbox in zip(labels, confidences, bboxes):
                bbox = frameNorm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, classes[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            255)
            cv2.imshow(xout_preview.getStreamName(), frame)
            if cv2.waitKey(1) == ord('q'):
                break

            print(labels, confidences)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    if args.video_detection:
        run_detection_with_video_noexit(args.picam)
    elif args.raw:
        raw_detection()
    else:
        detect_plants(args.picam, args.video)

