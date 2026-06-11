import cv2
import mediapipe as mp
import numpy as np

#פונקציה של צביעת הפנים ע"י מעבר למרחב LAB והוספת צבע על הבהירות של הפיקסל
def coloring_skin(mask, image):
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] > 0:
                L = image[i, j, 0]
                L_norm = (L / 255.0) ** 0.5
                x = mask[i][j] / 255.0
                
                image[i, j, 1] += 15 * L_norm * x   
                image[i, j, 2] += 18 * L_norm * x   
                
    return np.clip(image, 0, 255).astype(np.uint8)


def process_face_coloring(img):
    if img is None:
        return None

    h, w, _ = img.shape

    #הפעלת מודל מדיה פיפ
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        #פונקציה הממירה את נקודות המדיה פיפ לנקודות X,Y 
        def get_pixel_points(indices):
            pts = []
            for idx in indices:
                pt = landmarks[idx]
                px = int(pt.x * w)
                py = int(pt.y * h)
                pts.append([max(0, min(w-1, px)), max(0, min(h-1, py))])
            return np.array(pts, dtype=np.int32)

        face_oval_connections = list(mp_face_mesh.FACEMESH_FACE_OVAL)
        adjacency_list = {start: end for start, end in face_oval_connections}
        
        ordered_indices = []
        current_node = face_oval_connections[0][0]
        for _ in range(len(face_oval_connections)):
            ordered_indices.append(current_node)
            current_node = adjacency_list.get(current_node, None)
            if current_node is None: break
                
        all_y = [int(landmarks[idx].y * h) for idx in ordered_indices]
        center_y = sum(all_y) // len(all_y)
        
        face_points = []
        for idx in ordered_indices:
            pt = landmarks[idx]
            px = int(pt.x * w)
            py = int(pt.y * h)
            if py < center_y:
                py = max(0, py - int(h * 0.12))  
            face_points.append([px, py])
        face_points = np.array(face_points, dtype=np.int32)

        #אינדקסים של איברי הפנים להוריד מהמסיכה של כל הפנים
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        lips_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

        #חילוץ הנקודות כ X,Y 
        left_eye_pts = get_pixel_points(left_eye_indices)
        right_eye_pts = get_pixel_points(right_eye_indices)
        left_eyebrow_pts = get_pixel_points(left_eyebrow_indices)
        right_eyebrow_pts = get_pixel_points(right_eyebrow_indices)
        lips_pts = get_pixel_points(lips_indices)

        #מילוי הפנים בלבן וכל האיברים (עיניים, פה,גבות) בשחור
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        cv2.fillPoly(mask, [left_eye_pts], 0)
        cv2.fillPoly(mask, [right_eye_pts], 0)
        cv2.fillPoly(mask, [left_eyebrow_pts], 0)
        cv2.fillPoly(mask, [right_eyebrow_pts], 0)
        cv2.fillPoly(mask, [lips_pts], 0)

        
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_lab = coloring_skin(mask_blurred, lab_img)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    else:
        print("לא נמצאו פנים בתמונה.")
        return img  