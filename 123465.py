import cv2
import mediapipe as mp
import numpy as np

def get_EYEBROWS_binary_mask(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    if image is None:
        print("לא ניתן לטעון את התמונה.")
        return None
    
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
                # הפרדה בין הגבות כדי למנוע חיבור ביניהן
                LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
                
                for eyebrow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
                    points = []
                    for idx in eyebrow_indices:
                        landmark = face_landmarks.landmark[idx]
                        points.append((int(landmark.x * w), int(landmark.y * h)))
                    
                    points_arr = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points_arr], 255)

                # הרחבה קלה של המסיכה לכיסוי שערות בורחות
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

# הגדרות גוון חום (מחוץ לפונקציה)
A_max, A_min = 135, 130
B_max, B_min = 145, 135

def coloring_eyebrows(mask, image, A_max, A_min, B_max, B_min):
    image = image.astype(np.float32)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if mask[i][j] == 255:
                L = image[i, j, 0]
                L_norm = L / 255.0
                
                # שימוש במשתני ה-A וה-B שהגדרנו (חום)
                new_A = A_min + (A_max - A_min) * (1 - L_norm) 
                new_B = B_min + (B_max - B_min) * (1 - L_norm)
                
                # שילוב (Blending) - 50% צבע חדש, 50% מקורי כדי לשמור על טקסטורת שיער
                image[i, j, 1] = new_A * 0.5 + image[i, j, 1] * 0.5
                image[i, j, 2] = new_B * 0.5 + image[i, j, 2] * 0.5
                
                # החשכה קלה של הגבות (אופציונלי)
                image[i, j, 0] = L * 0.9 

    return np.clip(image, 0, 255).astype(np.uint8)

# הרצה
img_path = '11.jpg'
mask = get_EYEBROWS_binary_mask(img_path)

if mask is not None:
    original_img = cv2.imread(img_path)
    lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # שימוש בפונקציה המעודכנת לגבות
    result_lab = coloring_eyebrows(mask, lab_img, A_max, A_min, B_max, B_min)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('Eyebrows Result', result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()