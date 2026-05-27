import cv2
import numpy as np
import skimage.exposure

# specify desired bgr color for lips and make into array
desired_color = (170, 130, 255)
desired_color = np.asarray(desired_color, dtype=np.float64)

# create swatch
swatch = np.full((200,200,3), desired_color, dtype=np.uint8)

# read image
img = cv2.imread(r"C:\Projects\FinalProject\NewProject\coloring_image\images_black_white\images_black_white\img_2.jpg")

# threshold on lip color
lower = (0,0,200)
upper = (140,140,255)
mask = cv2.inRange(img, lower, upper)

# apply morphology open and close
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# get average bgr color of lips
ave_color = cv2.mean(img, mask=mask)[:3]
print(ave_color)

# compute difference colors and make into an image the same size as input
diff_color = desired_color - ave_color
diff_color = np.full_like(img, diff_color, dtype=np.uint8)

# shift input image color
# cv2.add clips automatically
new_img = cv2.add(img, diff_color)

# antialias mask, convert to float in range 0 to 1 and make 3-channels
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
mask = skimage.exposure.rescale_intensity(mask, in_range=(128,255), out_range=(0,1)).astype(np.float32)
mask = cv2.merge([mask,mask,mask])

# combine img and new_img using mask
result = (img * (1 - mask) + new_img * mask)
result = result.clip(0,255).astype(np.uint8)

# save result
cv2.imwrite('lady_swatch.png', swatch)
cv2.imwrite('lady_mask.png', (255*mask).clip(0,255).astype(np.uint8))
cv2.imwrite('lady_recolor.jpg', result)

cv2.imshow('swatch', swatch)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
