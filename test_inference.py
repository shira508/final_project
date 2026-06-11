import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ==========================================
# 1. הגדרת הארכיטקטורה של המודל
# ==========================================

class UpConv(torch.nn.Module):
    def __init__(self, inc, outc):
        super(UpConv, self).__init__()
        self.conv = torch.nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))

class AOTBlock(torch.nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(rate),
                    torch.nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    torch.nn.ReLU(True)
                ),
            )
        self.fuse = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)

        feat = self.gate(x)
        mean = feat.mean((2, 3), keepdim=True)
        std = feat.std((2, 3), keepdim=True) + 1e-9
        mask = torch.sigmoid(5 * (2 * (feat - mean) / std - 1))

        return x * (1 - mask) + out * mask

class InpaintGenerator(torch.nn.Module):
    def __init__(self, args):
        super(InpaintGenerator, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(2, 64, 7), 
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.ReLU(True),
        )
        self.middle = torch.nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])
        self.decoder = torch.nn.Sequential(
            UpConv(256, 128), torch.nn.ReLU(True),
            UpConv(128, 64), torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.tanh(x)


# ==========================================
# 2. הגדרת נתיבים מקומיים (עדכני לפי המחשב שלך)
# ==========================================

# שימוש בנתיבים יחסיים או מלאים במחשב שלך
MODEL_PATH = r"C:\Projects\FinalProject\NewProject\project\checkpoints\G0010000.pt"      
IMAGE_PATH = r"C:\Projects\FinalProject\NewProject\project\inputs\face_m (147).jpg"  
MASK_PATH  = r"C:\Projects\FinalProject\NewProject\project\inputs\face_m_mask (147).jpg" 

IMAGE_SIZE = 512 

# בדיקה אוטומטית אם יש כרטיס מסך (GPU), ואם לא - שימוש ב-CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" usando dispositivo: {device}")


# ==========================================
# 3. טעינה והכנת הקלט (Preprocessing)
# ==========================================

print("⏳ מכין את הקלטים לטסט...")

if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
    raise FileNotFoundError("❌ שגיאה: לא נמצאה תמונת הקלט או המסכה בנתיב המבוקש!")

# טעינת התמונה והמסיכה והמרתן לשחור-לבן ("L")
img_pil = Image.open(IMAGE_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
mask_pil = Image.open(MASK_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)

# המרה למערכי numpy
img_np = np.array(img_pil, dtype=np.float32) / 255.0
mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
mask_np = (mask_np > 0.5).astype(np.float32)

# נירמול התמונה המקורית לטווח [1-, 1]
img_norm = img_np * 2.0 - 1.0

# יצירת חור לבן מוחלט (ערך 1.0) בדיוק כמו באימון
img_masked_norm = np.where(mask_np > 0.5, 1.0, img_norm)

# המרה לטנסורים והעברה למכשיר המתאים (GPU או CPU)
img_tensor = torch.from_numpy(img_masked_norm).unsqueeze(0).unsqueeze(0).to(device)
mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


# ==========================================
# 4. בניית המודל וטעינת משקולות
# ==========================================

print("⏳ בונה את הגנרטור וטוען משקולות...")
args = SimpleNamespace(rates=[1, 2, 4, 8], block_num=8)
generator = InpaintGenerator(args).to(device)

if os.path.exists(MODEL_PATH):
    # טעינת המשקולות (עם התאמה ל-CPU במידת הצורך)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    generator.load_state_dict(state_dict, strict=True)
    print(f"✅ המשקולות נטענו בהצלחה מ-{MODEL_PATH}!")
else:
    raise FileNotFoundError(f"❌ שגיאה: לא נמצא קובץ משקולות בנתיב: {MODEL_PATH}")

generator.eval()


# ==========================================
# 5. הרצת המודל והצגת תוצאות
# ==========================================

print("⏳ מריץ את התמונה דרך המודל...")
with torch.no_grad():
    # הרשת מייצרת את הפיקסלים שבתוך החור
    pred_img = generator(img_tensor, mask_tensor)
    
    # שילוב הניחוש של המודל בתוך אזור המסכה
    comp_tensor = img_tensor * (1.0 - mask_tensor) + pred_img * mask_tensor

# החזרת התוצאה ל-Numpy
comp_np = comp_tensor.squeeze(0).squeeze(0).cpu().numpy()

# המרה חזרה מטווח נירמול לטווח [0, 255]
comp_final = ((comp_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_masked_final = ((img_masked_norm + 1.0) / 2.0 * 255.0).astype(np.uint8)

print("✅ הריצה הסתיימה בהצלחה! מציג את התוצאות...")

# הצגת התמונות להשוואה
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original (Torn Image)")
plt.imshow(np.array(img_pil), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Input to Model (Masked)")
plt.imshow(img_masked_final, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("AOT-GAN Output (Restored)")
plt.imshow(comp_final, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()