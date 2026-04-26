import cv2
import numpy as np
import os
import random

def apply_realistic_tear_to_batch(input_faces_folder, textures_folder, output_folder):
    """
    מריץ עיבוד אצווה על תיקיית פנים ומשלב קרעים ריאליסטיים עם שוליים לבנים.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # טעינת כל נתיבי טקסטורות הקרעים (חובה PNG עם שקיפות)
    texture_paths = [os.path.join(textures_folder, f) for f in os.listdir(textures_folder) if f.endswith('.png')]
    if not texture_paths:
        print("שגיאה: לא נמצאו טקסטורות קרעים (PNG) בתיקייה.")
        return

    # מעבר על תמונות הפנים
    for filename in os.listdir(input_faces_folder):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 1. טעינת תמונת הפנים והמרה לשחור-לבן
        face_path = os.path.join(input_faces_folder, filename)
        face_img = cv2.imread(face_path)
        if face_img is None: continue
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = face_gray.shape

        # 2. בחירת טקסטורת קרע רנדומלית
        texture_path = random.choice(texture_paths)
        # טעינה כולל ערוץ אלפא (IMREAD_UNCHANGED)
        tear_texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        if tear_texture is None: continue

        # 3. התאמת גודל הקרע לתמונה
        tear_texture = cv2.resize(tear_texture, (w, h))

        # 4. הפרדת ערוץ האלפא (השקיפות) של הקרע
        # ערוץ 3 הוא האלפא ב-BGRA
        tear_alpha = tear_texture[:, :, 3] 

        # 5. יצירת רקע לבן חלק באותו גודל
        white_background = np.ones((h, w), dtype=np.uint8) * 255

        # 6. ערבוב (Blending): המפתח לריאליזם
        # היכן שהאלפא של הקרע הוא לבן (255), נשתמש בתמונה המקורית.
        # היכן שהוא שקוף (0), נשתמש ברקע הלבן.
        
        # המרה למטריצות פלוט לחישוב מדויק
        alpha_factor = tear_alpha.astype(float) / 255.0
        face_factor = face_gray.astype(float)
        bg_factor = white_background.astype(float)

        # נוסחת הערבוב
        final_img = (face_factor * alpha_factor) + (bg_factor * (1.0 - alpha_factor))
        
        # המרה חזרה ל-uint8
        final_img = final_img.astype(np.uint8)

        # 7. שמירה
        output_path = os.path.join(output_folder, f"torn_{filename}")
        cv2.imwrite(output_path, final_img)
        print(f"עובדה תמונה: torn_{filename}")

# --- דוגמה להרצה ---
# עליך לעדכן את הנתיבים האלה במחשב שלך:
# 1. תיקייה עם תמונות פנים שלמות
# 2. תיקייה עם טקסטורות קרעים (PNG עם שקיפות)
# 3. תיקיית יעד ריקה
apply_realistic_tear_to_batch(
    r'C:\Projects\FinalProject\NewProject\woman', 
    r'C:\Projects\FinalProject\NewProject\textures', 
    r'C:\Projects\FinalProject\NewProject\woman_torn'
)