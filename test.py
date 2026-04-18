import cv2
import mediapipe as mp

image_path = '4.jpg'
# הגדרות קיצור דרך
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# שימוש ב-with מבטיח סגירה תקינה של המודל
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3  # הורדנו מעט ל-0.3 כדי לעזור בזיהוי בתמונות קשות/קרועות
) as face_mesh:
    image = cv2.imread(image_path)
    if image is None:
        print(f"שגיאה: לא נמצאה תמונה בנתיב: {image_path}")
    else:
        # המרה ל-RGB
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            print(f"נמצאו פנים! מצייר נקודות...")
            for face_landmarks in results.multi_face_landmarks:
                # ציור רשת (Tesselation)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                # ציור קווי מתאר
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())                
            # שמירה והצגה
            cv2.imwrite('man_4_mp.png', image)
            # התאמת גודל חלון התצוגה למסך
            cv2.namedWindow('man_4_mp.png', cv2.WINDOW_NORMAL)
            cv2.imshow('man_4_mp.png', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("לצערי, המודל לא הצליח לזהות פנים בתמונה הזו.")


