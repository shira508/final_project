#קוד ליצירת מסיכה בינארית של החורים בתמונה
import numpy as np
import cv2
import os

#קריאת הפרמטרים מהקובץ החיצוני
with open("params.txt") as f:
    for line in f:
        name, value = line.strip().split('=')
        globals()[name.strip()] = int(value.strip())

#תיקיית מקור ותקיית יעד
#הקוד עובר על כך התמונות הקרועות הנמצאות בתיקיית המקור ומייצר לכל תמונה מסיכה בינארית
#ושומר לכל תמונה את המסיכה שלה בתיקיית היעד
input_folder = r'C:\Projects\FinalProject\NewProject\dataset_color\damaged'
output_folder = r'C:\Projects\FinalProject\NewProject\dataset_color\masks'

#אפ התיקייה לא קיימת יוצר אותה חדשה
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#".png", ".jpg", ".jpeg" עובר על כל הקבצים בתיקייה בעלי הסיומות 
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error loading: {filename}")
            continue
        
        #שינוי גודל התמונה לגודל אחיד לכ התמונות
        image = cv2.resize(image, (450, 450))
        rows, cols, _ = image.shape

        #יצירת מטריצת פיקסלים ריקה בשביל המסיכה
        mat = np.zeros((rows, cols, 3), dtype=np.uint8)

        #מעבר על הפיקסלים בתמונה ובדיקה האם הפיקסל נקרא חור:
        # לבן או דומה ללבן בטווה הערכים שנקבע
        #וכן בודק עם 4 לפחות מתוך 8 השכנים של הפיקסל נקראים גם חור 
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                pixel = image[i][j]
                if pixel[0] >= 255 - COLOR_RANGE:
                    is_hole = 0
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i,   j-1),           (i,   j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                    for ni, nj in neighbors:
                        if image[ni][nj][0] >= 255 - COLOR_RANGE:
                            is_hole += 1
                            if is_hole == NUM_OF_NEIGHBORS:
                                break
                    
                    if is_hole == NUM_OF_NEIGHBORS: 
                        mat[i][j] = [255, 255, 255]
                    else:
                        mat[i][j] = [0, 0, 0]
                else:
                    mat[i][j] = [0, 0, 0]
                    
        #שמירה בתיקייה חדשה
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, mat)
        print(f"Processed and saved: {filename}")




