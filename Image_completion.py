from mp_LandMarks import get_lip_landmarks
import numpy as np
import cv2

# הגדרת גודל עבודה
W, H = 500, 500

# קריאת הנקודות עם העברת הגודל
points_raw = get_lip_landmarks('4.jpg', target_w=W, target_h=H)

if points_raw:
    # הכנת המסיכה והנקודות
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array(points_raw, dtype=np.int32)
    
    # ציור המסיכה
    cv2.fillPoly(mask, [points], (255, 255, 255))
    
    # תצוגה
    cv2.imshow("Lip Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("שגיאה: לא זוהו שפתיים.")