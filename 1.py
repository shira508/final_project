import cv2
import mediapipe as mp
import numpy as np
import os

# פונקציה לחישוב ערכי ה-A וה-B במרחב LAB
def Color_conversion(l, a_max, a_min, b_max, b_min):
    # שימוש בשורש (0.5) מחזק את נוכחות הצבע גם באזורים בהירים
    factor = (1 - (l / 100))**0.5
    a = a_min + (a_max - a_min) * factor
    b = b_min + (b_max - b_min) * factor
    return a, b

def get_lips_mask(image):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # המרה ל-RGB עבור המודל
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # אינדקסים המקיפים את כל השפתיים (עליונה ותחתונה)
                LIPS_INDICES = [
                    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                    185, 40, 39, 37, 0, 267, 269, 270, 409, 291
                ]
                
                points = []
                for idx in LIPS_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    points.append((int(landmark.x * w), int(landmark.y * h)))
                
                # ציור המסיכה
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    
    # ריכוך המסיכה למניעת קצוות חדים מדי
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask

def coloring_lips(image_path):
    # טעינת התמונה
    img = cv2.imread(image_path)
    if img is None:
        print("לא ניתן למצוא את הקובץ.")
        return

    # יצירת המסיכה
    mask = get_lips_mask(img)
    
    # המרה ל-LAB ועבודה עם float32 לדיוק מתמטי
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # הגדרת טווחים לצבע אדום חי
    A_MAX, A_MIN = 80, 30
    B_MAX, B_MIN = 30, 25
    
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            alpha = mask[i, j] / 255.0 # עוצמת השקיפות מהמסיכה
            
            if alpha > 0:
                L_raw = lab_img[i, j, 0]
                L_norm = (L_raw / 255.0) * 100 # נרמול ל-100 לצורך הנוסחה
                
                # מניעת צביעה של שיניים (אם הפיקסל בהיר מאוד)
                if L_norm > 85:
                    alpha *= 0.1

                new_a, new_b = Color_conversion(L_norm, A_MAX, A_MIN, B_MAX, B_MIN)
                
                # השמה עם הוספת 128 (קריטי למניעת צבע כחול) וערבוב (Blending)
                lab_img[i, j, 1] = lab_img[i, j, 1] * (1 - alpha) + (new_a + 128) * alpha
                lab_img[i, j, 2] = lab_img[i, j, 2] * (1 - alpha) + (new_b + 128) * alpha

    # המרה חזרה לפורמט תצוגה
    final_res = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # שמירת התוצאה כקובץ חדש
    cv2.imwrite("final_colored_lips.png", final_res)
    cv2.imshow("Result", final_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# הרצה
if __name__ == "__main__":
    coloring_lips("image.png")