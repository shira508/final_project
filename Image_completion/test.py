import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import InpaintingGenerator

# 1. הגדרת מכשיר הריצה (מכיוון שזו רק תמונה אחת, נשתמש ב-CPU וזה יהיה מהיר)
device = torch.device("cpu")

# 2. אתחול המחולל וטעינת ה"מוח" (המשקולות) שלו
generator = InpaintingGenerator().to(device)

# --- שימי לב: תשני את השם של הקובץ כאן לקובץ האחרון שנשמר לך בתיקייה ---
checkpoint_path = r"C:\Projects\FinalProject\NewProject\checkpoints\generator_epoch_25.pth" 

if os.path.exists(checkpoint_path):
    # טעינת המשקולות לתוך הארכיטקטורה של המחולל
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval() # מעביר את המודל למצב בדיקה (מכבה שכבות אימון כמו BatchNorm)
    print(f"המודל נטען בהצלחה מתוך: {checkpoint_path}")
else:
    print(f"שגיאה: לא נמצא קובץ מודל בנתיב {checkpoint_path}")
    exit()

# 3. הגדרת הטרנספורמציות (בדיוק באותו גודל שבו המודל התאמן)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 4. טעינת תמונת בדיקה והמסכה שלה מהמחשב שלך
# (תבחרי תמונה ומסכה מתוך התיקיות שלך לצורך הבדיקה)
test_image_path = r"C:\Projects\FinalProject\NewProject\dataset\damaged\face_m (82).jpg"
test_mask_path = r"C:\Projects\FinalProject\NewProject\dataset\masks\face_m (82).jpg"

try:
    corrupted_img = Image.open(test_image_path).convert('RGB')
    mask_img = Image.open(test_mask_path).convert('L')
    
    # הפיכה ל-Tensors והוספת מימד ה-Batch (צריך להוסיף מימד 1 בהתחלה כי המודל מצפה לקבוצה)
    corrupted_tensor = transform(corrupted_img).unsqueeze(0).to(device)
    mask_tensor = transform(mask_img).unsqueeze(0).to(device)
    
    # 5. הרצת התמונה בתוך המחולל!
    with torch.no_grad(): # אומרים למחשב לא לחשב נגזרות (חוסך זיכרון, אנחנו לא באימון)
        generated_output = generator(corrupted_tensor, mask_tensor)
    
    # 6. המרה חזרה מ-Tensor לקובץ תמונה רגיל ושמירה
    # מורידים את מימד ה-Batch שהוספנו ומחזירים את ערכי הפיקסלים לטווח 0-1
    output_image_tensor = generated_output.squeeze(0).clamp(-1, 1)
    output_image_tensor = (output_image_tensor + 1) / 2.0 # העברה מטווח של [-1,1] ל-[0,1]
    
    # הפיכה לתמונת PIL ושמירה בדיסק
    output_pil = transforms.ToPILImage()(output_image_tensor)
    output_pil.save("inpainting_result.jpg")
    print("התמונה המתוקנת נשמרה בהצלחה בשם: inpainting_result.jpg")

except Exception as e:
    print(f"קרתה שגיאה בזמן טעינת או עיבוד התמונות: {e}")