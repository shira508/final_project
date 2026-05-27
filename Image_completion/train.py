import os
import torch
import torch.nn as nn
import torch.optim as optim

# ייבוא המודלים והנתונים מהדפים האחרים
from Image_completion.model import InpaintingGenerator, InpaintingDiscriminator
from Image_completion.dataset import train_dataloader

# ---------------------------------------------------------
# אתחול המודלים, האופטימייזרים והגדרות ריצה
# ---------------------------------------------------------

# הגדרת מכשיר הריצה (GPU אם זמין, אחרת CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# יצירת מופעים של המודלים והעברתם למכשיר הריצה
generator = InpaintingGenerator().to(device)
discriminator = InpaintingDiscriminator().to(device)

# הגדרת האופטימייזרים
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# הגדרת פונקציית ההפסד (Loss)
criterion = nn.BCELoss()

# הגדרות משתני הלולאה - מעודכן ל-40 סיבובים
num_epochs = 40  
start_epoch = 0

# הגדרת נתיבים ישירים לקבצים של סיבוב 10
gen_checkpoint = "checkpoints/generator_epoch_19.pth"
disc_checkpoint = "checkpoints/discriminator_epoch_19.pth"

# מנגנון טעינה אוטומטית מסיבוב 10 של היום
if os.path.exists(gen_checkpoint) and os.path.exists(disc_checkpoint):
    print(f"נמצאו קבצי סיבוב 10! טוען משקולות...")
    
    # טעינת המשקולות ישירות לרשתות
    generator.load_state_dict(torch.load(gen_checkpoint, map_location=device, weights_only=False))
    discriminator.load_state_dict(torch.load(disc_checkpoint, map_location=device, weights_only=False))
    
    # אנחנו יודעים בוודאות שהגענו ל-10, אז נמשיך מ-11
    start_epoch = 20
    print(f"הטעינה הצליחה! האימון יתחדש מ-Epoch מספר {start_epoch}")
else:
    print("לא נמצאו קבצי סיבוב 10 בנתיב המבוקש. מתחילים מאפס.")


# ---------------------------------------------------------
# לולאת האימון הראשית
# ---------------------------------------------------------

for epoch in range(start_epoch, num_epochs):
    print(f"\n=== מתחיל Epoch {epoch}/{num_epochs} ===")
    
    # הגדרת המודלים למצב אימון
    generator.train()
    discriminator.train()
    
    # לולאת ה-Batch החוזרת על ה-Dataset שלך
    for i, batch in enumerate(train_dataloader):
        # שליפת התמונות והמסכות והעברתן ל-GPU/CPU
        corrupted_images, masks = batch[0].to(device), batch[1].to(device)
        
        # ---------------------------------------------------------
        # 1. אימון ה-Discriminator (המבחין)
        # ---------------------------------------------------------
        optimizer_D.zero_grad()
        
        # ייצור תמונות מתוקנות (מזויפות) ע"י הגנרטור
        fake_images = generator(corrupted_images, masks)
        
        # ריצה של הדיסקרימינטור על האמיתי ועל המזויף
        outputs_real = discriminator(corrupted_images, masks)
        outputs_fake = discriminator(fake_images.detach(), masks)
        
        # יצירת תוויות (Labels) של 1 (לאמיתי) ו-0 (למזויף) בהתאמה לגודל הפלט
        labels_real = torch.ones_like(outputs_real).to(device)
        labels_fake = torch.zeros_like(outputs_fake).to(device)
        
        # חישוב ה-Loss של הדיסקרימינטור ועדכון משקולות
        loss_D = criterion(outputs_real, labels_real) + criterion(outputs_fake, labels_fake)
        loss_D.backward()
        optimizer_D.step()
        
        # ---------------------------------------------------------
        # 2. אימון ה-Generator (היוצר)
        # ---------------------------------------------------------
        optimizer_G.zero_grad()
        
        # בדיקה מחדש של התמונות המזויפות מול הדיסקרימינטור
        outputs_fake_for_G = discriminator(fake_images, masks)
        
        # הגנרטור רוצה שהדיסקרימינטור יטעה ויחשוב שהן אמיתיות
        loss_G = criterion(outputs_fake_for_G, labels_real)
        loss_G.backward()
        optimizer_G.step()
        
        # הדפסת התקדמות קטנה מדי פעם בתוך ה-Epoch
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_dataloader)} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")
            
    # יצירת תיקיית checkpoints אם היא לא קיימת (מחוץ ללולאת ה-batch, בסוף ה-Epoch)
    os.makedirs("checkpoints", exist_ok=True)

    # שמירה בנפרד של כל מודל בסיום ה-Epoch הנוכחי
    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch}.pth")
    
    print(f"סיום Epoch {epoch}! המשקולות נשמרו בתיקיית checkpoints.")

print("\nהאימון הסתיים במלואו!")