import os
import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def CCA_Analysis(orig_image, predict_image, erode_iteration, open_iteration):
    kernel1 = (np.ones((5, 5), dtype=np.float32))
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = predict_image
    image2 = orig_image
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1, iterations=open_iteration)
    image = cv2.filter2D(image, -1, kernel_sharpening)
    image = cv2.erode(image, kernel1, iterations=erode_iteration)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    labels = cv2.connectedComponents(thresh, connectivity=8)[1]
    a = np.unique(labels)
    count2 = 0
    for label in a:
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts, hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        c_area = cv2.contourArea(cnts)
        if c_area < 1000:
            continue
        if c_area > 5000:
            count2 += 1

        (x, y), radius = cv2.minEnclosingCircle(cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        color1 = (list(np.random.choice(range(150), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        cv2.drawContours(image2, [box.astype("int")], 0, color, 4)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        pixelsPerMetric = 1
    teeth_count = count2
    return image2, teeth_count


dir_path = "/home/gaurav/Projects/medical/TeethU2Net/test_data"
output = './Result'
os.makedirs(output, exist_ok=True)


def convert_one_channel(img):
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img


for file_name in os.listdir(os.path.join(dir_path, 'Images')):
    mask_path = os.path.join(dir_path, 'Masks', file_name).replace('.jpg', '.jpeg')
    image_path = os.path.join(dir_path, 'Images', file_name)
    img = cv2.imread(image_path)
    predicted = cv2.imread(mask_path)
    predicted = cv2.resize(predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    cca_result, teeth_count = CCA_Analysis(img, predicted, 3, 2)
    print(os.path.join(output, file_name))
    cv2.imwrite(os.path.join(output, file_name), cca_result)
