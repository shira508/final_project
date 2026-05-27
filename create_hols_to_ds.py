import cv2
import numpy as np
import os
import random

def create_realistic_tear(img):
    """מצייר קרעים דקים ומשוננים למראה טבעי יותר"""
    h, w = img.shape[:2]
    # כמות קרעים: נשמור על 1-2 כדי לא להרוס את התמונה לגמרי
    num_tears = random.randint(1, 2)
    
    for _ in range(num_tears):
        # נקודת התחלה
        current_x = random.randint(5, w - 5)
        current_y = random.randint(5, h - 5)
        
        # עובי דק משמעותית (1 עד 2 פיקסלים בלבד לרזולוציית 64)
        thickness = random.randint(1, 2)
        
        # אורך הקרע (כמות מקטעים)
        steps = random.randint(15, 30)
        
        for i in range(steps):
            # תנועה קטנה מאוד בכל שלב ליצירת מראה משונן
            # אנחנו נותנים העדפה לכיוון מסוים כדי שהקרע יתקדם ולא יסתובב סביב עצמו
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            
            next_x = np.clip(current_x + dx, 0, w - 1)
            next_y = np.clip(current_y + dy, 0, h - 1)
            
            # ציור הקטע הקטן
            cv2.line(img, (current_x, current_y), (next_x, next_y), (255, 255, 255), thickness)
            
            current_x, current_y = next_x, next_y
            
            # מדי פעם נשנה את העובי בפיקסל אחד באמצע הקרע למראה לא אחיד
            if i % 5 == 0:
                thickness = random.randint(1, 2)
                
    return img

def process_dataset_only_tears(source_dir, torn_dir, size=64):
    if not os.path.exists(torn_dir):
        os.makedirs(torn_dir)

    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"מעבד {len(filenames)} תמונות ברמת פירוט גבוהה...")

    for fname in filenames:
        img = cv2.imread(os.path.join(source_dir, fname))
        if img is None: continue
        
        img = cv2.resize(img, (size, size))
        
        # הפעלת הפונקציה המשופרת
        torn_img = create_realistic_tear(img)
        
        cv2.imwrite(os.path.join(torn_dir, fname), torn_img)

    print("הסתיים! הקרעים עכשיו דקים וטבעיים יותר.")

# נתיבים (מומלץ להשתמש ב-Raw strings בגלל ה-backslashes של Windows)
source_path = r'C:\Projects\FinalProject\NewProject\Full'
torn_path = r'C:\Projects\FinalProject\NewProject\Torn'

#process_dataset_only_tears(source_path, torn_path)

create_realistic_tear("face_386.jpg")