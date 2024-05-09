import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util_img import set_background, write_csv
import uuid
import os
import util_video
from sort.sort import *
from util_video import get_car, read_license_plate, write_csv

# Set background image
set_background("./imgs/background.png")

# Define paths and models
folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
#################################################################################
results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
frame_nmr = -1
ret = True


###################################################################################
reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2, 3, 5, 7]  # coco model's id numbers. Car: 2 Motorbike: 3 Bus: 5 Truck: 7

# Initialize YOLO models
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

state = "Uploader"
if "state" not in st.session_state :
    st.session_state["state"] = "Uploader"

# Functions for license plate detection and recognition
def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())

            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}

                results[license_numbers][license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                      'text': license_plate_text,
                                      'bbox_score': score,
                                      'text_score': license_plate_text_score}}
                license_numbers += 1

        write_csv(results, f"./csv_detections/detection_results.csv")

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return [img_wth_box, licenses_texts, license_plate_crops_total]

    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

# Function to change state to Image Uploader
def change_state_image_uploader():
    st.session_state["state"] = "Image Uploader"

# Function to change state to Video Uploader
def change_state_video_uploader():
    st.session_state["state"] = "Video Uploader"

# Streamlit app layout
with st.container():
    # Header section
    st.title("Automatic License plate Recognition ")
    st.subheader(" ")

    _, colb1, colb2, _ = st.columns([0.2, 0.9, 0.7, 0.1])
    colb1.image("./imgs/license_detection.gif", width=600)
    st.write(" ")

# The rest of your Streamlit app layout goes here...


with st.container():
    _, col1, _, = st.columns([0.1, 1, 0.2])
    col1.subheader(" ")

    _, colb1, colb2, _ = st.columns([0.2, 0.7, 0.7, 0.2])
    if colb1.button("Upload an Image", on_click=change_state_image_uploader):
        pass
    elif colb2.button("Upload a Video", on_click=change_state_video_uploader):
        pass

    if st.session_state["state"] == "Image Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Video Uploader":
        video = st.file_uploader("Upload a Car Video: ", type=["mp4"])

    _, col2, _ = st.columns([0.3, 1, 0.2])

    _, col5, _ = st.columns([0.8, 1, 0.2])

    if st.session_state["state"] == "Image Uploader":
        if img is not None:
            image = np.array(Image.open(img))
            col2.image(image, width=400)

            if col5.button("Apply Recognition"):
                results = model_prediction(image)

                if len(results) == 3:
                    prediction, texts, license_plate_crop = results[0], results[1], results[2]

                    texts = [i for i in texts if i is not None]

                    if len(texts) == 1 and len(license_plate_crop):
                        _, col3, _ = st.columns([0.4, 1, 0.2])
                        col3.header("Recognition Results ✅:")

                        _, col4, _ = st.columns([0.1, 1, 0.1])
                        col4.image(prediction)

                        _, col9, _ = st.columns([0.4, 1, 0.2])
                        col9.header("License Cropped ✅:")

                        _, col10, _ = st.columns([0.3, 1, 0.1])
                        col10.image(license_plate_crop[0], width=350)

                        _, col11, _ = st.columns([0.45, 1, 0.55])
                        col11.success(f"License Number: {texts[0]}")

                        df = pd.read_csv(f"./csv_detections/detection_results.csv")
                        st.dataframe(df)
                    elif len(texts) > 1 and len(license_plate_crop) > 1:
                        _, col3, _ = st.columns([0.4, 1, 0.2])
                        col3.header("Detection Results ✅:")

                        _, col4, _ = st.columns([0.1, 1, 0.1])
                        col4.image(prediction)

                        _, col9, _ = st.columns([0.4, 1, 0.2])
                        col9.header("License Cropped ✅:")

                        _, col10, _ = st.columns([0.3, 1, 0.1])

                        _, col11, _ = st.columns([0.45, 1, 0.55])

                        col7, col8 = st.columns([1, 1])
                        for i in range(0, len(license_plate_crop)):
                            col10.image(license_plate_crop[i], width=350)
                            col11.success(f"License Number {i}: {texts[i]}")

                        df = pd.read_csv(f"./csv_detections/detection_results.csv")
                        st.dataframe(df)
                else:
                    prediction = results[0]
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.3, 1, 0.1])
                    col4.image(prediction)

#########################################################################################
    elif st.session_state["state"] == "Video Uploader":
        _, col2, _ = st.columns([0.3, 1, 0.2])
        _, col5, _ = st.columns([0.8, 1, 0.2])

        if video is not None:
            # Save the uploaded video to a temporary file
            with st.spinner('Processing video...'):
                temp_video_path = 'temp_video.mp4'

                with open(temp_video_path, 'wb') as temp_file:
                    temp_file.write(video.read())

            # Display video
            video_file = cv2.VideoCapture(temp_video_path)
            # Button to trigger detection
            if col5.button("Apply Recognition"):
                while video_file.isOpened():
                    ret, frame = video_file.read()

                    if not ret:
                        break

                    while ret:
                        frame_nmr += 1
                        results[frame_nmr] = {}
                        # detect vehicles
                        detections = coco_model(frame)[0]
                        detections_ = []
                        for detection in detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            if int(class_id) in vehicles:
                                detections_.append([x1, y1, x2, y2, score])

                        # track vehicles
                        track_ids = mot_tracker.update(np.asarray(detections_))

                        # detect license plates
                        license_plates = license_plate_detector(frame)[0]
                        for license_plate in license_plates.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate

                            # assign license plate to car
                            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                            if car_id != -1:

                                # crop license plate
                                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                                # process license plate
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                             cv2.THRESH_BINARY_INV)

                                # read license plate number
                                license_plate_text, license_plate_text_score = read_license_plate(
                                    license_plate_crop_thresh)

                                if license_plate_text is not None:
                                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                    'text': license_plate_text,
                                                                                    'bbox_score': score,
                                                                                    'text_score': license_plate_text_score}}

                        # write results
                    write_csv(results, './test.csv')

                    video_file.release()

                    st.success("Video processing complete!")
                    st.balloons()

                    if st.success:
                        _, colE1, colV1, colV2 = st.columns([0.2, 0.7, 0.7, 0.2])

                        # Button for Interpolate
                        if colE1.button("Interpolate"):
                            # Run Interpolate script
                            subprocess.run(['python', 'add_missing_data.py'])
                            st.success("Interpolate script completed!")

                        # Button for Visualize
                        if colV1.button("Visualize"):
                            # Run Visualize script
                            subprocess.run(['python', 'visualize.py'])
                            st.success("Visualize script completed!")

                        # Button for Visualize
                        if colV2.button("view"):
                            cap = cv2.VideoCapture('./out.mp4')

                    # st.button("Clear Cache", on_click=clear_cache)

                # ... (other code)


