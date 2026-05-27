import cv2
import mediapipe as mp
import numpy as np

def get_lips_binary_mask(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    if image is None: return None
    
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
                LIPS_INDICES = [
                    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 
                    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
                ]
                points = [(int(face_landmarks.landmark[idx].x * w), 
                           int(face_landmarks.landmark[idx].y * h)) for idx in LIPS_INDICES]
                
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def apply_color_vectorized(image_lab, mask, A_max, A_min, B_max, B_min):
    # יצירת עותק לעבודה
    result_lab = image_lab.copy()
    
    # חילוץ ערוץ ה-L רק איפה שהמסכה קיימת
    L = result_lab[:, :, 0]
    
    # נורמליזציה של L לטווח 0-1 (OpenCV uint8 LAB: L is 0-255)
    L_norm = L / 255.0
    
    # חישוב וקטורי של A ו-B לכל התמונה (לפי הנוסחה שלך)
    # הערה: הפכתי את ה-1-(L/100) לחישוב מבוסס L_norm
    new_A = A_min + (A_max - A_min) * (1 - L_norm) + 128
    new_B = B_min + (B_max - B_min) * (1 - L_norm) + 128
    
    # החלת השינויים רק באזורי המסכה
    result_lab[mask == 255, 1] = new_A[mask == 255]
    result_lab[mask == 255, 2] = new_B[mask == 255]
    
    return result_lab

# הגדרות צבע
A_max, A_min = 90, 40
B_max, B_min = 40, 20

mask = get_lips_binary_mask(r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\img_3.jpg')

if mask is not None:
    original_img = cv2.imread(r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\img_3.jpg')
    # עבודה עם float32 לדיוק בחישובים
    lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    result_lab = apply_color_vectorized(lab_img, mask, A_max, A_min, B_max, B_min)
    
    # המרה חזרה ל-BGR
    result_bgr = cv2.cvtColor(result_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    cv2.imshow('Result', result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()