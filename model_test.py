import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# ========================================================
# 1. הגדרות ומשתני נתיבים (התאימי לפי התיקיות אצלך במחשב)
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"עובד על מכשיר: {device}")

# נתיבים מקומיים - מניח שהתיקייה dataset נמצאת באותו מקום של הקוד
CLEAN_DIR = "./dataset/clean"
DAMAGED_DIR = "./dataset/damaged"
MASK_DIR = "./dataset/mask"
CHECKPOINT_PATH = "./unet_inpainting_checkpoint.pth"
TEST_IMAGE_PATH = "./dataset/damaged/face_w (800).jpg" # נתיב לתמונה בודדת לבדיקה בסוף

# ========================================================
# 2. הגדרת ה-Dataset (מחלקת טעינת הנתונים)
# ========================================================
class ColorInpaintingDataset(Dataset):
    def __init__(self, clean_dir, damaged_dir, mask_dir, img_size=(256, 256)):
        self.clean_dir = clean_dir
        self.damaged_dir = damaged_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        # טעינת קבצים רק אם התיקייה קיימת
        self.filenames = sorted(os.listdir(clean_dir)) if os.path.exists(clean_dir) else []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        clean_img = cv2.imread(os.path.join(self.clean_dir, filename), cv2.IMREAD_COLOR)
        damaged_img = cv2.imread(os.path.join(self.damaged_dir, filename), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        clean_img = cv2.resize(clean_img, self.img_size)
        damaged_img = cv2.resize(damaged_img, self.img_size)
        mask_img = cv2.resize(mask_img, self.img_size)
        
        clean_img = clean_img.astype(np.float32) / 255.0
        damaged_img = damaged_img.astype(np.float32) / 255.0
        mask_img = mask_img.astype(np.float32) / 255.0
        mask_img = np.where(mask_img > 0.5, 1.0, 0.0).astype(np.float32)
        
        clean_img = np.transpose(clean_img, (2, 0, 1))
        damaged_img = np.transpose(damaged_img, (2, 0, 1))
        
        mask_tensor = torch.from_numpy(mask_img).float().unsqueeze(0)
        
        return torch.from_numpy(damaged_img), torch.from_numpy(clean_img), mask_tensor

# ========================================================
# 3. ארכיטקטורת רשת ה-U-Net (עם 4 ערוצי קלט)
# ========================================================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet4ChannelsInpainting(nn.Module):
    def __init__(self):
        super(UNet4ChannelsInpainting, self).__init__()
        
        self.enc1 = conv_block(4, 64) # קלט של 4 ערוצים (תמונה + מסכה)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = conv_block(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        d3 = torch.cat((d3, s3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, s2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, s1), dim=1)
        d1 = self.dec1(d1)
        
        return self.sigmoid(self.final_conv(d1))

# ========================================================
# 4. פונקציית ה-Perceptual Loss (VGG16)
# ========================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1: y = y.repeat(1, 3, 1, 1)
        return torch.mean(torch.abs(self.slice1(x) - self.slice1(y)))

# ========================================================
# 5. פונקציית האימון הראשי (Train Function)
# ========================================================
def run_training():
    print("\n--- מתחיל שלב האימון ---")
    if not (os.path.exists(CLEAN_DIR) and os.path.exists(DAMAGED_DIR) and os.path.exists(MASK_DIR)):
        print("שגיאה: תיקיות הדאטה סט לא נמצאו! ודאי שיצרת תיקיית dataset עם clean, damaged ו-mask באותו מיקום.")
        return

    model = UNet4ChannelsInpainting().to(device)
    perceptual_criterion = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataset = ColorInpaintingDataset(CLEAN_DIR, DAMAGED_DIR, MASK_DIR)
    # ב-VS Code מקומי, באץ' של 16 או 32 הוא מעולה
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    num_epochs = 20 # אפשר להקטין/להגדיל לפי כוח המחשב שלך
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for damaged_batch, clean_batch, mask_batch in dataloader:
            damaged_batch = damaged_batch.to(device)
            clean_batch = clean_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # שירשור ל-4 ערוצי קלט
            model_input = torch.cat([damaged_batch, mask_batch], dim=1)
            outputs = model(model_input)
            
            valid_pixels = (1 - mask_batch)
            loss_valid = torch.sum(torch.abs(outputs - clean_batch) * valid_pixels) / torch.sum(valid_pixels).clamp(min=1)
            loss_hole = torch.sum(torch.abs(outputs - clean_batch) * mask_batch) / torch.sum(mask_batch).clamp(min=1)
            loss_perceptual = perceptual_criterion(outputs * mask_batch, clean_batch * mask_batch)
            
            total_loss = loss_valid + (5.0 * loss_hole) + (10.0 * loss_perceptual)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] -> Total Loss: {running_loss/len(dataloader):.4f}")
        
        # שמירת המשקולות בסוף כל אפוק מקומית במחשב
        torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
        
    print("האימון הסתיים בהצלחה! הקובץ נשמר מקומית.")

# ========================================================
# 6. פונקציית הטסט והצגת התוצאה (Test/Inference Function)
# ========================================================
def run_test():
    print("\n--- מתחיל שלב הטסט ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print("שגיאה: לא נמצא קובץ משקולות מאומן. יש להריץ אימון קודם.")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"שגיאה: תמונת הטסט לא נמצאה בנתיב: {TEST_IMAGE_PATH}")
        return

    # בניית המודל וטעינת המשקולות מהקובץ המקומי
    model = UNet4ChannelsInpainting().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    img_bgr = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_bgr, (256, 256))
    gray_temp = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # יצירת מסכה נקייה (ערך 254 לסינון הקרע הלבן)
    mask_np = np.where(gray_temp > 254, 1.0, 0.0).astype(np.float32)
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    img_filled = cv2.inpaint(img_resized, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    img_normalized = img_filled.astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # שירשור ל-4 ערוצים
    test_input = torch.cat([img_tensor, mask_tensor], dim=1)
    
    with torch.no_grad():
        output_tensor = model(test_input)
        
    output_img = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_uint8 = (output_img * 255).astype(np.uint8)
    output_gray = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2GRAY)
    output_bw_3d = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2RGB)
    
    final_output = img_resized.copy()
    final_output[mask_np == 1.0] = output_bw_3d[mask_np == 1.0]
    
    input_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
    
    # הצגת גרף התוצאות
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_rgb)
    axes[0].set_title("1. Input (Original)")
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("2. Cleaned Mask")
    axes[1].axis('off')
    
    axes[2].imshow(final_rgb)
    axes[2].set_title("3. Output (Perfect Fixed!)")
    axes[2].axis('off')
    
    print("מציג את התוצאות הויזואליות...")
    plt.show()

# ========================================================
# 7. נקודת ההרצה הראשית (Main Execution)
# ========================================================
if __name__ == "__main__":
    # אם את רוצה רק לאמן: שחררי את השורה הראשונה ותמחקי/תעשי הערה לשנייה
    # אם כבר אימנת ואת רוצה רק לבדוק טסט: תעשי הערה לשורה הראשונה ותריצי רק את השנייה
    
    run_training() # מריץ את האימון מהתחלה
    run_test()     # מריץ את בדיקת התמונה ומציג את הגרף בסוף