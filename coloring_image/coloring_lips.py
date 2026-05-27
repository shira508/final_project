import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, send_file 
import io

app = Flask(__name__)

def get_lips_binary_mask(image):
    mp_face_mesh = mp.solutions.face_mesh
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
                mask = cv2.GaussianBlur(mask, (3,3), 0)
    return mask

def coloring_lips(mask, image):
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] > 0:
                L = image[i, j, 0]
                L_norm = (L / 255.0) ** 0.5
                x = mask[i][j] / 255.0
                image[i, j, 1] += 25 * L_norm * x   
                image[i, j, 2] += 10 * L_norm * x   
    return np.clip(image, 0, 255).astype(np.uint8)


# 🌟 הנתיב החדש ששרת ה-#C פונה אליו
@app.route('/api/color-lips', methods=['POST'])
def color_lips_endpoint():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400
        
    file = request.files['image']
    
    # 1. קריאת קובץ הבייטס מהרשת
    file_bytes = file.read()
    if not file_bytes:
        return {"error": "Empty image data"}, 400

    # 2. המרה למערך נומרי של numpy
    in_memory_file = np.frombuffer(file_bytes, np.uint8)
    
    # 3. 🌟 התיקון המרכזי: שימוש ב-IMREAD_UNCHANGED כדי לשמור על פורמט התמונה המקורי והמדויק שעבד לך!
    original_img = cv2.imdecode(in_memory_file, cv2.IMREAD_UNCHANGED)

    if original_img is None:
        return {"error": "Failed to decode image"}, 400

    # 4. וידוא שלתמונה יש 3 ערוצי צבע (במידה והועלתה תמונת שחור-לבן/אפור)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)

    # 5. הרצת המסיכה על התמונה המיושרת והנכונה
    mask = get_lips_binary_mask(original_img)
    
    if mask is not None:
        lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_lab = coloring_lips(mask, lab_img)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # קידוד התמונה המשוחזרת חזרה לפורמט PNG ושליחה ישירה ב-HTTP
        _, img_encoded = cv2.imencode('.png', result_bgr)
        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')
    else:
        return {"error": "Could not detect face/lips"}, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)