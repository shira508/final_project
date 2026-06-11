import cv2
import numpy as np
from flask import Flask, request, send_file 
import io

# ייבוא המודולים החיצוניים שלך
import restoration_face              # השלמת חורים / שחזור פנים
import coloring_lips        # צביעת שפתיים (הקובץ החדש שיצרנו למעלה)
import coloring_face       # צביעת פנים

app = Flask(__name__)

@app.route('/api/color-lips', methods=['POST'])
def color_lips_endpoint():
    # 1. בדיקת תקינות הקובץ המועלה
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400
        
    file = request.files['image']
    file_bytes = file.read()
    if not file_bytes:
        return {"error": "Empty image data"}, 400

    # 2. פענוח התמונה
    in_memory_file = np.frombuffer(file_bytes, np.uint8)
    original_img = cv2.imdecode(in_memory_file, cv2.IMREAD_UNCHANGED)

    if original_img is None:
        return {"error": "Failed to decode image"}, 400

    # התאמת ערוצי צבע (מערך דו-מימדי או RGBA ל-BGR)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)

    # 3. שלב א': שחזור פנים / השלמת חורים
    restored_img = restoration_face.restore_face_image(original_img)
    if restored_img is None:
        restored_img = original_img

    # 4. שלב ב': צביעת שפתיים (מתוך המודול המופרד)
    lips_colored_img = coloring_lips.process_lips_coloring(restored_img)
    
    # אם זיהוי השפתיים נכשל, נמשיך עם התמונה המשוחזרת כבסיס לשלב הבא
    base_for_face = lips_colored_img if lips_colored_img is not None else restored_img

    # 5. שלב ג': צביעת פנים
    final_bgr = coloring_face.process_face_coloring(base_for_face)
    if final_bgr is None:
        final_bgr = base_for_face

    # 6. החזרת התמונה המעובדת הסופית למשתמש
    _, img_encoded = cv2.imencode('.png', final_bgr)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)