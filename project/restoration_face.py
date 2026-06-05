import os
import sys
import ssl
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

# פתרון חסימות ותאימות גרסאות
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# טעינת המודל מראש פעם אחת בלבד
model_path = r'GFPGANv1.4.pth'  # ודאי שהקובץ הזה נמצא בתיקיית הפרויקט שלך
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("טוען את מודל GFPGAN עבור השרת...")
restorer = GFPGANer(
    model_path=model_path,
    upscale=1,           
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def restore_face_image(input_img):
    if input_img is None:
        return None

    # קדם-עיבוד וניקוי נקודות לבנות
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    cleaned_base = cv2.medianBlur(input_img, 5)
    pre_processed_img = np.where(mask[:, :, None] == 255, cleaned_base, input_img)
    pre_processed_img = cv2.inpaint(pre_processed_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # הפעלת השחזור
    try:
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            pre_processed_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True       
        )
        return restored_img
    except Exception as e:
        print(f"שגיאה בתהליך השחזור: {e}")
        return input_img  # החזרת המקור במקרה של תקלה למניעת קריסה