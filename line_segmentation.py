
import cv2
import numpy as np

def segment_lines(image_path):
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image_path

    h, w, _ = img.shape

    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def thresholding(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
        return thresh

    thresh_img = thresholding(img)

    # Dilation
    kernel = np.ones((3, 85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    # Line Segmentation
    (contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y, w, h)

    segmented_lines = []
    for ctr in sorted_contours_lines:
        if cv2.contourArea(ctr)<700:
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        segmented_lines.append(img[y:y+h, x:x+w])

    return segmented_lines

