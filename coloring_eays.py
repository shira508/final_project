#קוד לצביעת העיניים
#L צריך למתוח פרבולות בים הנקודות של העיניים ולצבוע בשחור עם בהירות 

import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


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
                
                right = [469, 470, 471, 472]
                points=[]
                p=[]
                i=0
                for point in right:
                        landmark = face_landmarks.landmark[point]
                        x = int(landmark.x * w) 
                        y = int(landmark.y * h)
                        point_curr = (x,y)
                        points.append(point_curr)
                        if point % 2==0:
                            p.append(point_curr)
                c= abs(p[0][1]-p[1][1])/2        
                #print(abs(c/2))
                #print(p)                          

                #  468,  473
                a = face_landmarks.landmark[468]
                x = int(a.x * w) 
                y = int(a.y * h)
                center = (x,y)
                cv2.circle(mask,center, int(c),255,-1)
                arr=np.array(points, dtype=np.int32)

                cv2.fillPoly(mask, [arr], 255)
                #cv2.fillPoly(mask, [get_points(right)], 255)
                #cv2.fillPoly(mask, [get_points(left)], 255)

                mask = cv2.GaussianBlur(mask, (3,3),0)

    return mask

mask = get_lips_binary_mask('img_4.jpg')


def coloring_eays(mask, image):
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if mask[i][j] > 0:
                L = image[i, j, 0]
                #כדי שיהיה נח לעבוד איתו L נרמול של ערך מרחב
                L_norm = L / 255.0
                
                x = mask[i][j]/255.0

                image[i, j, 1] = x + ((1.0 - x) * image[i, j, 1])
                image[i, j, 2] = x + ((1.0 - x) * image[i, j, 2])

    return np.clip(image, 0, 255).astype(np.uint8)

if mask is not None:
    original_img = cv2.imread('img_4.jpg')

    lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = coloring_eays(mask, lab_img)
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    cv2.imwrite('final_result.png', result_bgr)
    cv2.imshow('Result', result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("המסיכה לא נוצרה, ודא שהתמונה תקינה ושיש פנים בתמונה.")