import numpy as np
import cv2
import os

with open("params.txt") as f:
    for line in f:
        name, value = line.strip().split('=')
        globals()[name.strip()] = int(value.strip())

input_folder = r'C:\Projects\FinalProject\NewProject\woman_torn'
output_folder = r'C:\Projects\FinalProject\NewProject\woman_masks'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error loading: {filename}")
            continue
            
        image = cv2.resize(image, (450, 450))
        rows, cols, _ = image.shape

        mat = np.zeros((rows, cols, 3), dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                pixel = image[i][j]
                if pixel[0] >= 255 - Color_range:
                    is_hole = 0
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i,   j-1),           (i,   j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                    for ni, nj in neighbors:
                        if image[ni][nj][0] >= 255 - Color_range:
                            is_hole += 1
                            if is_hole == Num_of_neighbors:
                                break
                    
                    if is_hole == Num_of_neighbors: 
                        mat[i][j] = [255, 255, 255]
                    else:
                        mat[i][j] = [0, 0, 0]
                else:
                    mat[i][j] = [0, 0, 0]

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, mat)
        print(f"Processed and saved: {filename}")

print("---סיום")