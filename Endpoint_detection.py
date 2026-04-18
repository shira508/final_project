#קוד שכתבתי בעצמי לזיהוי חורים ע"י הפיכת התמונה לשחור לבן
#לא רלוונטי כל כך
import cv2
import numpy as np

#קריאת התמונה
img = cv2.imread("1.jpg")

# HSV המרה של התמונה לייצוג 
#HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#חישוב הבהירות של כל פיקסל

img = np.array(img)
print(img)

for i in range(len(img)):
    for j in range(len(img[0])):
            B=img[i][j][0]
            G=img[i][j][1]
            R=img[i][j][2]
            gray=0.299*R+0.587*G+0.114*B 
            img[i][j]=[gray,gray,gray]


print("----------------------------------------------------------------")
print(img)

#הפיכת התמונה לשחור לבן
for i in range(len(img)): 
    for j in range(len(img[0])):
         if img[i][j][0]<127:
              img[i][j]=[0,0,0]
         if img[i][j][0]>=127:
              img[i][j]=[255,255,255]

#mat=[]
#for i in range(len(img)): 
#    for j in range(len(img[0])):
#          if list(img[i][j])==[0,0,0]:
#               continue
#          elif list(img[i][j])==[255,255,255]:
#             mat.append((i, j))               
     
#print(mat)  


cv2.imshow('IMAGE1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


rows=len(img)
cols=len(img[0])
visited = np.full((len(img), len(img[0])), False) #מטריצת המתארת האם בדקתי את הפיקסל
all_contours = [] #רשימת קווי המתאר
all_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

for i in range(len(img)): 
    for j in range(len(img[0])):
         #אם הפיקסל לבן ועוד לא ביקרתי בו
         if img[i][j]==[0,0,0] and visited[i][j]==False:
              current_line = [] #יצירת רשימה של קו חדש
              while(all_neighbors):
                    visited[i][j]==True
                    for k in range(len(all_neighbors)):
                         x=all_neighbors[k][0]
                         z=all_neighbors[k][1]
                         if i+x <0 or j+z <0 or i+x > len(img) or i+x >len(img[0]):
                              if img[i+x][j+z]==[0,0,0]:
                                   if visited[i+x][j+z]==False:
                                        visited[i+x][j+z]==True
                                        current_line.append(img[i+x][j+z])
                    all_contours.append(current_line)    
