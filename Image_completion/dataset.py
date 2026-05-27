import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class InpaintingDataset(Dataset):
    def __init__(self, corrupted_dir, mask_dir, original_dir=None):
        """
        כאן אנחנו רק מגדירים למחשב איפה התיקיות נמצאות ומכינים את רשימת הקבצים.
        """
        self.corrupted_dir = corrupted_dir
        self.mask_dir = mask_dir
        self.original_dir = original_dir  # תיקייה אופציונלית לתמונות המקוריות השלמות
        
        # אנחנו לוקחים את כל שמות הקבצים בתיקייה וממיינים אותם כדי שהכל יתאים
        self.filenames = sorted(os.listdir(corrupted_dir))
        
        # הגדרת טרנספורמציה - הופכת את התמונה ל-Tensor (מטריצה של מספרים בין 0 ל-1)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()          
            ])
    def __len__(self):
        """
        הפונקציה הזו פשוט אומרת למחשב כמה תמונות סך הכל יש לנו בתיקייה.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        הלב של ה-Dataset. לפי אינדקס (מספר סידורי), הפונקציה הזו טוענת את הקבצים.
        """
        # קבלת שם הקובץ הנוכחי
        filename = self.filenames[idx]
        
        # יצירת נתיב מלא לכל קובץ
        corrupted_path = os.path.join(self.corrupted_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        # פתיחת התמונות מהדיסק
        # תמונה קרועה נפתחת כ-RGB (3 ערוצים)
        corrupted_img = Image.open(corrupted_path).convert('RGB')
        # מסכה נפתחת כ-L (שחור/לבן - ערוץ 1 בלבד)
        mask_img = Image.open(mask_path).convert('L')
        
        # הפיכת התמונות ל-Tensors של PyTorch
        corrupted_tensor = self.transform(corrupted_img)
        mask_tensor = self.transform(mask_img)
        
        # אם יש לנו גם את התיקייה של התמונות המקוריות השלמות, נטען גם אותה
        if self.original_dir:
            original_path = os.path.join(self.original_dir, filename)
            original_img = Image.open(original_path).convert('RGB')
            original_tensor = self.transform(original_img)
            return corrupted_tensor, mask_tensor, original_tensor
            
        # אם אין תמונות מקוריות, נחזיר רק את התמונה הקרועה והמסכה
        return corrupted_tensor, mask_tensor
    

# הגדרת הנתיבים לתיקיות שלך במחשב
# (תשני את הנתיבים האלו למיקום האמיתי של התיקיות אצלך)
my_damaged_folder = r"C:\Projects\FinalProject\NewProject\dataset\damaged"
my_masks_folder = r"C:\Projects\FinalProject\NewProject\dataset\masks"
my_images_folder = r"C:\Projects\FinalProject\NewProject\dataset\images"

# יצירת ה-Dataset (ה"ספרן" שלנו)
train_dataset = InpaintingDataset(
    corrupted_dir=my_damaged_folder,
    mask_dir=my_masks_folder,
    original_dir=my_images_folder
)

# בדיקה קטנה שהכל עובד: נבקש ממנו את הדוגמה הראשונה (אינדקס 0) ונבדוק גדלים
corrupted_sample, mask_sample, original_sample = train_dataset[0]

print("Corrupted image shape:", corrupted_sample.shape) # צריך להדפיס משהו כמו: [3, H, W]
print("Mask shape:", mask_sample.shape)           # צריך להדפיס משהו כמו: [1, H, W]

from torch.utils.data import DataLoader

# הגדרת ה-DataLoader שיחלק את הנתונים ל-Batches
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)