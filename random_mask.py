import cv2
import numpy as np
import os
import random

def create_tear_on_image(img):
    """מצייר קרעים לבנים ברורים אך בגודל הגיוני ישירות על התמונה"""
    image_size = img.shape[0] # זה יהיה 512
    
    # קרע אחד ברור בכל תמונה
    num_tears = 1 
    
    for _ in range(num_tears):
        # נקודת התחלה אקראית באזור הפנים (לא בקצוות התמונה)
        start_x = random.randint(int(image_size * 0.25), int(image_size * 0.75))
        start_y = random.randint(int(image_size * 0.25), int(image_size * 0.75))
        
        # 1. עובי מאוזן: 4 עד 6 פיקסלים (רואים את זה מצוין ב-512x512)
        thickness = random.randint(4, 6)
        
        current_x, current_y = start_x, start_y
        
        # 2. אורך הקרע: 4 עד 7 מקטעים
        for _ in range(random.randint(4, 7)):
            # 3. קפיצות גדולות יותר: 15 עד 30 פיקסלים כדי שהקרע יקבל אורך וצורה של שריטה/קרע
            # אבל הגבלנו את המקטעים כדי שהוא לא יהיה ענקי
            next_x = np.clip(current_x + random.randint(-25, 25), 0, image_size - 1)
            next_y = np.clip(current_y + random.randint(-25, 25), 0, image_size - 1)
            
            # ציור קו לבן (255, 255, 255)
            cv2.line(img, (current_x, current_y), (next_x, next_y), (255, 255, 255), thickness)
            current_x, current_y = next_x, next_y
            
    return img

def process_dataset_only_tears(source_dir, torn_dir, size=512):
    """עובר על התיקייה ומייצר תמונות קרועות עם קרע לבן ברור ומאוזן"""
    if not os.path.exists(torn_dir):
        os.makedirs(torn_dir)

    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"מתחיל לעבד {len(filenames)} תמונות ברזולוציה של {size}x{size}...")

    for fname in filenames:
        img = cv2.imread(os.path.join(source_dir, fname))
        if img is None: 
            continue
        
        # שינוי גודל ל-512x512 (תואם לאימון)
        img = cv2.resize(img, (size, size))
        
        # יצירת הקרעים
        torn_img = create_tear_on_image(img)
        
        # שמירה
        cv2.imwrite(os.path.join(torn_dir, fname), torn_img)

    print("הסתיים בהצלחה! עכשיו הקרעים ברורים ורואים אותם מעולה.")

# הגדרת נתיבים
source_path = r'C:\Projects\FinalProject\NewProject\dataset_color\images'
torn_path = r'C:\Projects\FinalProject\NewProject\dataset_color\damaged'

# הרצה
process_dataset_only_tears(source_path, torn_path, size=512)