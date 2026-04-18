#קוד ליצירת מסכה לתמונה
#ייבוא של ספריות
import numpy as np
import cv2
from PIL import Image

#הגדרת הטווח של ערך של פיקסל שעדיין נקרא חור
#Color_range=0
#Num_of_neighbors=0

with open("params.txt") as f:
     for line in f:
         name , value = line.strip().split('=')
         globals()[name.strip()] = int(value.strip())

#קריאת התמונה
image = cv2.imread("man_3.png")

#שינוי גודל התמונה ל450*450:
image=cv2.resize(image,(450,450))

# openCV כל הקוד שלי בשורה אחת של זימון פונקצייה של
#_, mat = cv2.threshold(image[:,:,0], 255 - Color_range, 255, cv2.THRESH_BINARY)
#מס שורות ועמודות של מטריצת התמונה

rows=image.shape[0]
cols=image.shape[1]

i=0
j=0

mat = np.zeros_like(image)

for i in range(1,rows-1):
    for j in range(1,cols-1):
        pixel=image[i][j]     
        if pixel[0] >= 255 - Color_range and pixel[0] <= 255:
            is_hole=0
            neighbors = [ (i-1, j-1), (i-1, j), (i-1, j+1),  # שורה עליונה
                        (i,   j-1), (i,   j+1),  # צדדים
                        (i+1, j-1), (i+1, j), (i+1, j+1)]   # שורה תחתונה
            for ni ,nj in neighbors:
                if image[ni][nj][0] >= 255 - Color_range:
                    is_hole+=1
                    if is_hole==Num_of_neighbors:
                        break       
            if is_hole==Num_of_neighbors: 
                mat[i][j]=[255,255,255]
            else:
                mat[i][j]=[0,0,0]
        else:
            mat[i][j]=[0,0,0]

#כתיבה לתמונה
cv2.imwrite("man_3_mask.png",mat)

#הצגת התמונה
cv2.imshow("man_3_mask.png",mat)
cv2.waitKey(0)
cv2.destroyAllWindows()



