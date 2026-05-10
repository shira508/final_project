import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import splprep, splev

def get_smooth_lips_full_indices(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    
    # אלו כל הנקודות שמרכיבות את התיחום החיצוני המלא של השפתיים במדיה-פייפ
    # הן מסודרות לפי סדר שמאפשר יצירת לולאה סגורה
    LIPS_OUTER_CIRCLE = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, # שפה תחתונה
        409, 270, 269, 267, 0, 37, 39, 40, 185, 61      # שפה עליונה
    ]

    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        image = cv2.imread(image_path)
        if image is None: return
        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # 1. שליפת הנקודות לפי הסדר
            points = []
            for idx in LIPS_OUTER_CIRCLE:
                lm = landmarks.landmark[idx]
                points.append((lm.x * w, lm.y * h))
            
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]

            # 2. שימוש ב-splprep עבור עקומות סגורות (Parametric Spline)
            # זו הדרך הנכונה לעשות ספליין למסלול מעגלי/סגור
            tck, u = splprep([x, y], s=0, per=True) # per=True אומר שזו עקומה סגורה
            
            # יצירת 200 נקודות חדשות לאורך המסלול להחלקה מקסימלית
            u_new = np.linspace(0, 1, 200)
            x_new, y_new = splev(u_new, tck)
            
            # 3. הכנה לציור
            res_pts = np.vstack((x_new, y_new)).T.astype(np.int32)
            
            # ציור הקו החלק
            cv2.polylines(image, [res_pts], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("Full Spline Lips", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

get_smooth_lips_full_indices("image.png")