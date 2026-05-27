import cv2
import mediapipe as mp
import numpy as np

# =========================
# LOAD IMAGE
# =========================

image_path = r'C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\image.png'

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("Image not found")
    exit()

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# =========================
# FACE DETECTION
# =========================

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
) as face_detection:

    results = face_detection.process(image_rgb)

if results.detections is None:
    print("No face detected")
    exit()

face = results.detections[0]

# =========================
# GET FACE BOX
# =========================

h, w, _ = image_rgb.shape

bbox = face.location_data.relative_bounding_box

x1 = int(bbox.xmin * w)
y1 = int(bbox.ymin * h)

x2 = int((bbox.xmin + bbox.width) * w)
y2 = int((bbox.ymin + bbox.height) * h)

# Make square crop
size = max(x2 - x1, y2 - y1)

center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2

x1_square = center_x - size // 2
y1_square = center_y - size // 2

x2_square = x1_square + size
y2_square = y1_square + size

# Prevent out of bounds
x1_square = max(0, x1_square)
y1_square = max(0, y1_square)

x2_square = min(w, x2_square)
y2_square = min(h, y2_square)

# =========================
# CROP FACE
# =========================

square_face_region = image_bgr[y1_square:y2_square, x1_square:x2_square]

resized_image = cv2.resize(square_face_region, (480, 480))

cv2.imwrite("resized_image.jpg", resized_image)

# =========================
# FACE MESH
# =========================

mp_face_mesh = mp.solutions.face_mesh

upper_new = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78
]

lower_new = [
   61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]


image = resized_image.copy()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(image_rgb)

if results.multi_face_landmarks is None:
    print("No face mesh detected")
    exit()

# =========================
# CREATE MASKS
# =========================

mask_upper = np.zeros(image.shape[:2], dtype=np.uint8)
mask_lower = np.zeros(image.shape[:2], dtype=np.uint8)

for face_landmarks in results.multi_face_landmarks:

    # Upper lip
    points_upper = []

    for i in upper_new:
        landmark = face_landmarks.landmark[i]

        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        points_upper.append((x, y))

    cv2.fillConvexPoly(mask_upper, np.array(points_upper, dtype=np.int32), 255)

    # Lower lip
    points_lower = []

    for i in lower_new:
        landmark = face_landmarks.landmark[i]

        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        points_lower.append((x, y))

    cv2.fillPoly(mask_lower, [np.array(points_lower, dtype=np.int32)], 255)

# =========================
# FINAL LIPS MASK
# =========================

mask_diff = cv2.subtract(mask_upper, mask_lower)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN, kernel)
mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_CLOSE, kernel)

# =========================
# SHOW MASK
# =========================

cv2.imshow("Lip Mask", mask_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========================
# CREATE COLOR MASK
# =========================

def create_colored_mask(hex_color, shape):

    hex_color = hex_color.replace("#", "")

    rgb_color = tuple(
        int(hex_color[i:i+2], 16)
        for i in (0, 2, 4)
    )

    colored_mask = np.zeros(shape, dtype=np.uint8)

    # BGR
    colored_mask[:, :, 0] = rgb_color[2]
    colored_mask[:, :, 1] = rgb_color[1]
    colored_mask[:, :, 2] = rgb_color[0]

    return colored_mask

hex_color = input(
    "Enter hex color (example FF0000): "
)

mask_diff_3ch = cv2.cvtColor(mask_diff, cv2.COLOR_GRAY2BGR)

colored_mask = create_colored_mask(
    hex_color,
    mask_diff_3ch.shape
)

masked_color = cv2.bitwise_and(
    colored_mask,
    colored_mask,
    mask=mask_diff
)

# =========================
# COMBINE WITH IMAGE
# =========================

final_image = cv2.addWeighted(
    image,
    1.0,
    masked_color,
    0.6,
    0
)

# =========================
# SHOW FINAL IMAGE
# =========================

cv2.imshow("Final Image", final_image)

cv2.imwrite("final_result.jpg", final_image)

print("Saved as final_result.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()