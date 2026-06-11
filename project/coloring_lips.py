import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, send_file 
import io
import coloring_face       
import restoration_face as restoration_face   

app = Flask(__name__)

def get_lips_binary_mask(image):
    mp_face_mesh = mp.solutions.face_mesh
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
                LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                
                def get_points(points_arr):
                    points = []
                    for point in points_arr:
                        landmark = face_landmarks.landmark[point]
                        x = int(landmark.x * w) 
                        y = int(landmark.y * h)
                        points.append((x, y))
                    return np.array(points, dtype=np.int32)

                cv2.fillPoly(mask, [get_points(UPPER_LIP)], 255)
                cv2.fillPoly(mask, [get_points(LOWER_LIP)], 255)
                
                mask = cv2.GaussianBlur(mask, (3,3), 0)
    return mask

def coloring_lips(mask, image):
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] > 0:
                L = image[i, j, 0]
                L_norm = (L / 255.0) ** 0.5
                x = mask[i][j] / 255.0
                image[i, j, 1] += 25 * L_norm * x   
                image[i, j, 2] += 10 * L_norm * x   
    return np.clip(image, 0, 255).astype(np.uint8)


def process_lips_coloring(image_bgr):
    """
    פונקציית מעטפת שמקבלת תמונת BGR, הופכת ל-LAB,
    צובעת את השפתיים ומחזירה חזרה תמונת BGR צבועה.
    """
    mask = get_lips_binary_mask(image_bgr)
    if mask is None or np.sum(mask) == 0:
        return None  # לא זוהו שפתיים
        
    lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = coloring_lips(mask, lab_img)
    lips_colored_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return lips_colored_bgr

