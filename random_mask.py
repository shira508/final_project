import cv2
import numpy as np
import os
import random

def create_tear_on_image(img):
    """מצייר קרעים לבנים ישירות על התמונה"""
    image_size = img.shape[0]
    # בחירת כמות ה"קרעים" בתמונה אחת
    num_tears = random.randint(1, 3)
    
    for _ in range(num_tears):
        # נקודת התחלה אקראית
        start_x = random.randint(10, image_size - 10)
        start_y = random.randint(10, image_size - 10)
        
        # עובי הקרע
        thickness = random.randint(3, 10)
        
        # יצירת מסלול של קרע (סדרת קווים קצרים)
        current_x, current_y = start_x, start_y
        for _ in range(random.randint(5, 12)):
            next_x = np.clip(current_x + random.randint(-15, 15), 0, image_size - 1)
            next_y = np.clip(current_y + random.randint(-15, 15), 0, image_size - 1)
            
            # ציור קו לבן (255, 255, 255)
            cv2.line(img, (current_x, current_y), (next_x, next_y), (255, 255, 255), thickness)
            current_x, current_y = next_x, next_y
            
    return img

def process_dataset_only_tears(source_dir, torn_dir, size=64):
    """עובר על התיקייה ומייצר רק תמונות קרועות עם קרע לבן"""
    if not os.path.exists(torn_dir):
        os.makedirs(torn_dir)

    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"מתחיל לעבד {len(filenames)} תמונות...")

    for fname in filenames:
        # טעינת התמונה המקורית
        img = cv2.imread(os.path.join(source_dir, fname))
        if img is None: 
            continue
        
        # שינוי גודל ל-64x64
        img = cv2.resize(img, (size, size))
        
        # יצירת הקרעים הלבנים על התמונה
        torn_img = create_tear_on_image(img)
        
        # שמירת התמונה הקרועה בלבד
        cv2.imwrite(os.path.join(torn_dir, fname), torn_img)

    print("הסתיים בהצלחה! התמונות עם הקרעים הלבנים מוכנות בתיקייה.")

# הגדרת נתיבים
source_path = r'C:\Projects\FinalProject\NewProject\woman'
torn_path = r'C:\Projects\FinalProject\NewProject\woman_torn'

# הרצה
process_dataset_only_tears(source_path, torn_path)