
import os
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from pathlib import Path
from PIL import Image
import layoutparser as lp
from paddleocr import PaddleOCR

# Ensure necessary packages are installed
def install_packages():
    try:
        import fitz  # PyMuPDF
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle", "paddleocr"])

    subprocess.check_call([sys.executable, "-m", "pip", "install", "layoutparser"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.0"])

install_packages()

# Define the main processing function
def process_image(image_path):
    # Load OCR model
    ocr = PaddleOCR(lang='en')

    # Load Layout model
    model = lp.PaddleDetectionLayoutModel(
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
        threshold=0.5,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=False,
        enable_mkldnn=True
    )

    # Read image
    image_cv = cv2.imread(image_path)
    image_height, image_width = image_cv.shape[:2]

    # Detect layout
    layout = model.detect(image_cv)
    x_1, y_1, x_2, y_2 = 0, 0, 0, 0
    for l in layout:
        if l.type == 'Table':
            x_1, y_1 = int(l.block.x_1), int(l.block.y_1)
            x_2, y_2 = int(l.block.x_2), int(l.block.y_2)
            break

    # Perform OCR
    output = ocr.ocr(image_path)[0]
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    # Prepare image for visualization
    image_boxes = image_cv.copy()
    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1)
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)

    # Prepare for non-max suppression
    horiz_boxes, vert_boxes = [], []
    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        cv2.rectangle(image_boxes, (x_h, y_h), (x_h + width_h, y_h + height_h), (0, 0, 255), 1)
        cv2.rectangle(image_boxes, (x_v, y_v), (x_v + width_v, y_v + height_v), (0, 255, 0), 1)

    # Apply non-max suppression
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
    )
    vert_out = tf.image.non_max_suppression(
        vert_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
    )

    horiz_lines = np.sort(np.array(horiz_out))
    vert_lines = np.sort(np.array(vert_out))

    # Extract text and table data
    out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter = max(x_2 - x_1, 0) * max(y_2 - y_1, 0)
        if inter == 0:
            return 0

        box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
        box_2_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

        return inter / float(box_1_area + box_2_area - inter)

    unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
    ordered_boxes = np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if iou(resultant, the_box) > 0.1:
                    out_array[i][j] = texts[b]

    return image_boxes, out_array

# Streamlit app
def main():
    st.title("Image Processing and OCR with Streamlit")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read image
        image_path = "/tmp/uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process image
        image_boxes, out_array = process_image(image_path)

        # Display results
        st.image(image_boxes, caption="Processed Image with Detections", use_column_width=True)

        # Display results in a table
        st.write("Extracted Table Data:")
        df = pd.DataFrame(out_array)
        st.dataframe(df)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="output.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
