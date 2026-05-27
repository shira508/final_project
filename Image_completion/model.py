import torch
import torch.nn as nn

class InpaintingGenerator(nn.Module):
    def __init__(self):
        super(InpaintingGenerator, self).__init__()
        
        # -----------------------------------------
        # חלק 1: ENCODER (כיווץ והבנת המבנה)
        # -----------------------------------------
        self.down1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # -----------------------------------------
        # חלק 2: DECODER (שחזור והגדלה חזרה)
        # -----------------------------------------
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # מחזיר ערכים בין 1- ל-1
        )

    def forward(self, corrupted_image, mask):
        # שרשור התמונה והמסכה לאורך ציר הערוצים
        x = torch.cat([corrupted_image, mask], dim=1)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.up1(x2)
        output = self.final(x3)
        return output


class InpaintingDiscriminator(nn.Module):
    def __init__(self):
        super(InpaintingDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # שכבה 1: קלט של 4 ערוצים (תמונה + מסכה)
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # שכבה 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # שכבה 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # שכבה סופית: מוציאה מטריצת ציונים
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # ערכים בין 0 ל-1
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.model(x)