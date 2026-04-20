import cv2
import mediapipe as mp
import numpy as np

def get_lip_landmarks(image_path, target_w=None, target_h=None):
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        image = cv2.imread(image_path)
        if image is None: return None
        
        # שינוי גודל במידה והוגדר
        if target_w and target_h:
            image = cv2.resize(image, (target_w, target_h))
            
        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # אינדקסים לשפתיים
            LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            landmarks_list = []
            for idx in LIPS_INDICES:
                lm = face_landmarks.landmark[idx]
                landmarks_list.append((int(lm.x * w), int(lm.y * h)))
            return landmarks_list
        return None