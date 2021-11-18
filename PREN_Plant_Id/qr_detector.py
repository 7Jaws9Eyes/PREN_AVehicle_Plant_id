import json

import cv2
import cv2.cv2 as cv2
import depthai as dai
import argparse
import requests
import os

directory = r"C:\Users\ckirc\Pictures\plant detection"
api_key = '2b10glUixSPZOunMJ952kc5Pe'
url = f'https://my-api.plantnet.org/v2/identify/all?api-key={api_key}'
nr_imgs = 3
os.chdir(directory)


# main method
def detect_plants(use_pi):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pi', nargs='?', help="use default sys cam")
    args = parser.parse_args()

    if args.pi or use_pi:
        # setup resources
        cap = cv2.VideoCapture(0)
        detector = cv2.QRCodeDetector()

        # take images
        take_images(cap, detector)

        # show images for debugging
        show_images()

        # send images to API
        send_imgs_to_API()

        # release all ressources
        cap.release()
        cv2.destroyAllWindows()
    else:
        # setup resources
        pipeline = dai.Pipeline()

        # Define source and outputs
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_video = pipeline.create(dai.node.XLinkOut)

        xout_video.setStreamName("video")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(True)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Linking
        cam_rgb.video.link(xout_video.input)

        # qr detector
        detector = cv2.QRCodeDetector()

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            video = device.getOutputQueue('video')
            take_images(video, detector, True)
            show_images()
            send_imgs_to_API()


def run_detection(cap, detector, use_oak):
    while True:
        if use_oak:
            img = cap.get().getCvFrame()
        else:
            _, img = cap.read()
        data, bbox, _ = detector.detectAndDecode(img)
        if bbox is not None and data:
            if data:
                print("data found: ", data)
                break
        # debug
        cv2.imshow("code detector", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return data, img


def run_detection_with_video_noexit():
    # set up camera object
    cap = cv2.VideoCapture(0)

    # QR code detection object
    detector = cv2.QRCodeDetector()

    while True:
        # get the image
        _, img = cap.read()
        # get bounding box coords and data
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
        # display the image preview
        cv2.imshow("code detector", img)
        if cv2.waitKey(1) == ord("q"):
            break
    # free camera object and exit
    cap.release()
    cv2.destroyAllWindows()
    return data, img


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


def take_images(cap, detector, use_oak=False):
    x = 0
    attempt = 0
    while x < nr_imgs and attempt < 10:
        # _, img = plant_qr_detector()
        _, img = run_detection(cap, detector, use_oak)
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


def det_dai():
    pipeline = dai.Pipeline()

    # Define source and outputs
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_video = pipeline.create(dai.node.XLinkOut)

    xout_video.setStreamName("video")

    # Properties
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(True)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Linking
    cam_rgb.video.link(xout_video.input)

    # qr detector
    detector = cv2.QRCodeDetector()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        video = device.getOutputQueue('video')

        while True:
            video_frame = video.get()

            data, bbox, _ = detector.detectAndDecode(video_frame.getCvFrame())
            if bbox is not None and data:
                print("data found: ", data)
                break

            if cv2.waitKey(1) == ord('q'):
                break

