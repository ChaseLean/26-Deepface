import cv2
import os
import numpy as np
from tqdm import tqdm

from retinaface import RetinaFace

# Find the Euclidean distance between two 2D vectors
def euclidean(u, v):
    return np.sqrt(sum([(i - j) ** 2 for i, j in zip(u, v)]))

def detect_faces(input_path, output_path, resolution, align=True, scale=1):

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)

    accepted_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    labels = [fn for fn in os.listdir(input_path) if fn.split(".")[-1] in accepted_extensions]

    for label in tqdm(labels):

        img = cv2.imread(os.path.join(input_path, label))
        # Turns the BGR image to RGB
        img = img[..., ::-1]

        # Detect the faces using RetinaFace
        face_detector = RetinaFace.build_model()
        objs = RetinaFace.detect_faces(img, model=face_detector, threshold=0.9)

        if isinstance(objs, dict):
            for index, identity in enumerate(list(objs.values())):
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                confidence = identity["score"]

                landmarks = identity["landmarks"]
                face_img = align_face(img, landmarks, resolution, align, scale)

                cv2.imwrite(os.path.join(output_path, f"{label.split('.')[0]}_{index}.jpg"), face_img[..., ::-1])

# Align the face based on the position of the eyes before extracting the face
# Scale represents how closely the face is cropped
def align_face(img, landmarks, resolution, align=True, scale=1):

    # Define the width of the left eye to the right eye, and the height of the eyes
    eye_pos_w, eye_pos_h = 0.3 * scale, 0.35 * scale
    width, height = resolution[0], resolution[1]

    # Find the location of the eyes and their centre
    l_e = landmarks['left_eye']
    r_e = landmarks['right_eye']
    center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

    # Find the distance between the eyes
    dx = (r_e[0] - l_e[0])
    dy = (r_e[1] - l_e[1])
    dist = euclidean(l_e, r_e)

    # Find the angle between the eyes
    angle = np.degrees(np.arctan2(dy, dx)) + 180 if align else 0
    scale = width * (1 - (2 * eye_pos_w)) / dist

    # Find the x and y translations needed to center the image
    tx = width * 0.5
    ty = height * eye_pos_h

    # Get the affine matrix needed to perform the transformation
    m = cv2.getRotationMatrix2D(center, angle, scale)

    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])
    
    face_align = cv2.warpAffine(img, m, (width, height))

    return face_align

detect_faces(
    input_path="./inputs", 
    output_path="./extracted", 
    resolution=(240, 240), 
    align=True, 
    scale=1.2
)