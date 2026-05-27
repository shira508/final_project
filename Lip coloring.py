import cv2
import mediapipe as mp
import numpy as np

def get_accurate_lips_mask(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    if image is None:
        print("לא ניתן לטעון את התמונה.")
        return None
    
    h, w, _ = image.shape
    # יצירת מסיכה ריקה
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # אינדקסים נפרדים לשפה עליונה ותחתונה ליצירת מבנה אנטומי מדויק
            UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
            LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
            
            def get_pts(indices):
                pts = []
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
                return np.array(pts, dtype=np.int32)

            # ציור שני חלקי השפתיים על המסיכה
            cv2.fillPoly(mask, [get_pts(UPPER_LIP)], 255)
            cv2.fillPoly(mask, [get_pts(LOWER_LIP)], 255)
            
            #ריכוך של השוליים של השפתיים למראה טבעי
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask

def color_lips_fast(image_path, mask):
    img_bgr = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # יצירת שכבת צבע LAB חדשה באותו גודל
    colored_lab = img_lab.copy()
    
    # חילוץ ערוץ הבהירות L הנוכחי כדי להתאים את עוצמת הצבע אליו
    L = img_lab[:, :, 0]
    L_norm = L / 255.0
    
    # חישוב מטריציוני מהיר עבור ערכי A ו-B המבוקשים (בדומה לנוסחה שלך)
    A_channel = 145 + (185 - 145) * (1.0 - L_norm)
    B_channel = 125 + (145 - 125) * (1.0 - L_norm)
    
    colored_lab[:, :, 1] = A_channel
    colored_lab[:, :, 2] = B_channel
    
    # המרה חזרה ל-BGR של השכבה הצבועה
    colored_bgr = cv2.cvtColor(colored_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # נרמול המסיכה המטושטשת לטווח שבין 0.0 ל-1.0 כדי שתשמש כמשקל לשילוב
    mask_blur_norm = mask.astype(np.float32) / 255.0
    mask_blur_norm = np.expand_dims(mask_blur_norm, axis=2) # התאמה ל-3 ערוצים
    
    # שילוב (Blending) ליניארי בין התמונה המקורית לצבועה לפי משקלי המסיכה המרוככת
    result = (img_bgr * (1.0 - mask_blur_norm) + colored_bgr * mask_blur_norm).astype(np.uint8)
    
    return result

# הרצה
img_name = r'C:\Projects\FinalProject\NewProject\face_w (522).jpg' # ודאי שזה שם הקובץ הנכון אצלך
mask = get_accurate_lips_mask(img_name)

if mask is not None:
    img_name = r'C:\Projects\FinalProject\NewProject\face_w (522).jpg'
    final_result = color_lips_fast(img_name, mask)
    
    # שמירת התוצאה המקורית
    cv2.imwrite('final_result_perfect.png', final_result)
    
    # --- חלק ההגדלה והתצוגה החדש ---
    
    # 1. הגדרת חלון שניתן לשנות את הגודל שלו באופן חופשי
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    
    # 2. הגדלת התמונה פי 3 (או כל גורם אחר שתרצי, למשל 2 או 4)
    # משתמשים באינטרפולציית CUBIC השומרת על איכות גבוהה בהגדלה
    scale_percent = 300 # הגדלה ל-300% מהגודל המקורי
    width = int(final_result.shape[1] * scale_percent / 100)
    height = int(final_result.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_result = cv2.resize(final_result, dim, interpolation=cv2.INTER_CUBIC)
    
    # 3. הצגת התמונה המוגדלת בחלון
    cv2.imshow('Result', resized_result)
    
    # 4. התאמת גודל החלון הפיזי בדיוק לממדים החדשים של התמונה
    cv2.resizeWindow('Result', width, height)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()