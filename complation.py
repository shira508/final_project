
#קוד להרצת מודל להשלמת תמונה קרועה
#!!!!!!!!!!!!!!!לא המודל שאני בניתי

import os
import sys
import ssl

# =====================================================================
# 1. פתרון חסימות ותאימות גרסאות
# =====================================================================
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# =====================================================================
# 2. ייבוא ספריות
# =====================================================================
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

def main():
    # הגדרות נתיבים - שאי שם הקובץ החדש שלך מעודכן כאן
    model_path = r'GFPGANv1.4.pth'  
    input_path = r'C:\Projects\FinalProject\NewProject\dataset\damaged\face_w (111).jpg' # שם התמונה הנוכחית עם הנקודות
    output_dir = r'results'          
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"שגיאה: הקובץ '{input_path}' לא נמצא!")
        return

    print("טוען את מודל GFPGAN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,           
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    # קריאת התמונה
    input_img = cv2.imread(input_path)

    # =====================================================================
    # 3. קדם-עיבוד מתקדם (מתאים לנקודות מפוזרות / רעש מלח-פלפל)
    # =====================================================================
    print("מנקה את הנקודות המפוזרות מהפנים...")
    
    # שלב א': נמיר לשחור-לבן
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # שלב ב': נזהה את הנקודות הלבנות הקיצוניות
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # שלב ג': הרחבת המסכה (Dilation) כדי לוודא שאנחנו תופסים גם את השוליים של כל נקודה
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # שלב ד': טשטוש חצי-חכם (Median Blur) רק על האזורים הנגועים כדי לרכך את המעברים
    # זה מוחק את הנקודות המבודדות ומחליף אותן בגוון הפנים האמיתי שמסביבן
    cleaned_base = cv2.medianBlur(input_img, 5)
    
    # נשריש את התיקון רק איפה שהמסכה סימנה שיש נקודות, ונשמור על שאר התמונה המקורית
    pre_processed_img = np.where(mask[:, :, None] == 255, cleaned_base, input_img)

    # לשם ביטחון, נריץ אינפיינטינג קל על התוצאה הממוזגת
    pre_processed_img = cv2.inpaint(pre_processed_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # =====================================================================
    # 4. הפעלת מודל ה-GAN לשחזור ויצירת הפנים מחדש
    # =====================================================================
    print("מפעיל את מודל השחזור על הפנים הנקיות...")
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        pre_processed_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True       
    )

    # שמירת התוצאה
    save_path = os.path.join(output_dir, 'restored_face_no_noise.png')
    cv2.imwrite(save_path, restored_img)
    
    print("-" * 50)
    print(f"התהליך הסתיים! התמונה נשמרה ב: {save_path}")
    print("-" * 50)

if __name__ == '__main__':
    main()