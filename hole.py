import cv2
import numpy as np
import random
import math

def create_small_random_hole(image_path, output_path, base_radius=40, irregularity=0.3, num_points=20):
    """
    מייצר חור לבן אקראי וקטן יותר על גבי התמונה.
    
    :param base_radius: שונה ל-40 כדי שהחור יהיה קטן משמעותית
    :param irregularity: רמת העיוות של החור
    :param num_points: מספר הנקודות (הורדנו ל-20 שיתאים לגודל הקטן)
    """
    image = cv2.imread(image_path)
    if image is None:
        print("שגיאה: לא ניתן לטעון את התמונה.")
        return

    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    points = []
    
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        
        # חישוב רדיוס משתנה קטן
        random_factor = random.uniform(-irregularity, irregularity)
        current_radius = base_radius * (1 + random_factor)
        
        x = int(center_x + current_radius * math.cos(angle))
        y = int(center_y + current_radius * math.sin(angle))
        
        points.append([x, y])

    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    white_color = (255, 255, 255)

    # ציור החור הקטן
    cv2.fillPoly(image, [points_array], white_color)

    cv2.imwrite(output_path, image)
    print(f"החור הקטן נוצר בהצלחה ונשמר ב: {output_path}")

# הפעלה עם רדיוס קטן של 40 פיקסלים בלבד
create_small_random_hole(r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\face_w (647).jpg', base_radius=10)