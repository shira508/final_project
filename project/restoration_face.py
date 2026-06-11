import os
import torch
import numpy as np
import cv2
from types import SimpleNamespace

# ==========================================
# ארכיטקטורת המודל (חובה להשאיר קבוע)
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
# אתחול המודל פעם אחת בלבד בעת טעינת השרת
# ==========================================
MODEL_PATH = r"C:\Projects\FinalProject\NewProject\project\checkpoints\G0010000.pt"  # 📌 נתיב לקובץ המשקולות שלך במחשב/שרת
IMAGE_SIZE = 512 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"טוען את מודל AOT-GAN עבור השרת במכשיר: {device}...")

args = SimpleNamespace(rates=[1, 2, 4, 8], block_num=8)
generator = InpaintGenerator(args).to(device)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    print("✅ מודל AOT-GAN נטען בהצלחה ומארח בקשות!")
else:
    print(f"⚠️ אזהרה: לא נמצא קובץ משקולות ב-{MODEL_PATH}. הפונקציה תחזיר את תמונת המקור.")
    generator = None


# ==========================================
# פונקציית השירות הראשי עבור השרת
# ==========================================
def restore_face_image(input_img):
    """
    מקבלת תמונת OpenCV (BGR), משחזרת אותה באמצעות AOT-GAN ומחזירה תמונת OpenCV (BGR)
    """
    if input_img is None or generator is None:
        return input_img

    try:
        # שמירת המימדים המקוריים של התמונה כדי להחזיר אותה לגודלה המקורי בסוף
        orig_h, orig_w = input_img.shape[:2]

        # 1. המרה לשחור-לבן ושינוי גודל ל-512x512
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # 2. הכנת הנתונים וסילום לטווח [0, 1]
        img_np = resized_img.astype(np.float32) / 255.0

        # 3. יצירת מסכה אוטומטית (פיקסלים לבנים מוחלטים)
        #mask_np = (img_np > 0.95).astype(np.float32)

        # 3. יצירת מסכה משופרת לקרע ידני אמיתי
        mask_raw = (img_np > 0.90).astype(np.uint8)  # סף קצת יותר נמוך
        kernel = np.ones((7, 7), np.uint8)
        mask_dilated = cv2.dilate(mask_raw, kernel, iterations=3)  # הרחבה לכיסוי קצוות
        mask_blurred = cv2.GaussianBlur(mask_dilated.astype(np.float32), (5, 5), 0)
        mask_np = (mask_blurred > 0.5).astype(np.float32)  # חזרה לבינארי נקי

        # 4. נירמול התמונה לטווח של [1-, 1] ווידוא חור לבן
        img_norm = img_np * 2.0 - 1.0
        img_masked_norm = np.where(mask_np > 0.5, 1.0, img_norm)

        # 5. המרה לטנסורים והעברה ל-GPU/CPU
        img_tensor = torch.from_numpy(img_masked_norm).unsqueeze(0).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        # 6. הרצת המודל
        with torch.no_grad():
            pred_img = generator(img_tensor, mask_tensor)
            comp_tensor = img_tensor * (1.0 - mask_tensor) + pred_img * mask_tensor

        # 7. החזרה ל-Numpy וביטול נירמול לטווח [0, 255]
        comp_np = comp_tensor.squeeze(0).squeeze(0).cpu().numpy()
        comp_final = ((comp_np + 1.0) / 2.0 * 255.0).astype(np.uint8)

        # 8. החזרת התמונה לגודלה המקורי
        restored_gray = cv2.resize(comp_final, (orig_w, orig_h))

        # 9. המרה חזרה ל-BGR צבעוני (3 ערוצים) כדי לא לשבור את קוד השרת
        restored_bgr = cv2.cvtColor(restored_gray, cv2.COLOR_GRAY2BGR)
        
        return restored_bgr

    except Exception as e:
        print(f"❌ שגיאה במהלך הרצת מודל AOT-GAN: {e}")
        return input_img