import cv2
import mediapipe as mp
import numpy as np

#פונקציה ליצירת מסיכה בינארית למיקום השפתיים
#עוברים על כך נקודות השפתיים של המדיה פיפ 
#ממירים למיקום פיקסלים וצובעים את הפוליגון ביניהם בלבן
def get_lips_binary_mask(image_path):

    mp_face_mesh = mp.solutions.face_mesh
    
    image = cv2.imread(image_path)
    if image is None:
        print("לא ניתן לטעון את התמונה.")
        return
    
    h, w, _ = image.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
                LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                
                def get_points(points_arr):
                    points = []
                    for point in points_arr:
                        landmark = face_landmarks.landmark[point]
                        x = int(landmark.x * w) 
                        y = int(landmark.y * h)
                        points.append((x, y))
                    return np.array(points, dtype=np.int32)

                cv2.fillPoly(mask, [get_points(UPPER_LIP)], 255)
                cv2.fillPoly(mask, [get_points(LOWER_LIP)], 255)

                #מפעילים פונקציה גואסונית למעבר מדורג לקצוות של השפתיים
                mask = cv2.GaussianBlur(mask, (3,3),0)

    return mask

mask = get_lips_binary_mask(r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\face_w (647).jpg')

cv2.imwrite('mask.png', mask)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 800, 600)    
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread(r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\face_w (647).jpg')
if img is None:
    print("Image not found")
    exit()
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

A[mask>0] += 20
B[mask>0] += 10

A = np.clip(A, 0, 255).astype(np.uint8)
B = np.clip(B, 0, 255).astype(np.uint8)

colored_lab = cv2.merge([L, A, B])

result = cv2.cvtColor(colored_lab, cv2.COLOR_Lab2BGR)

cv2.imwrite('result.png', result)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 800, 600)    
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()