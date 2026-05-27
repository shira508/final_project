import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid

image_size=64
batch_size=128
state = (0.5,0.5,0.5),(0.5,0.5,0.5)


#צריך להגדיר כאן את הקלט של הדאטה סט
#(data_dir)

train_ds=ImageFolder(data_dir, transform=T.Compose([
T.Resize(image_size),
T.CenterCrop(image_size),
T.ToTensor(),
T.Normalize(*state)]))

train_dl= DataLoader(train_ds,
                     batch_size,
                     shuffle=True,
                     num_workers=3,
                     pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


#בניית המבחין - discriminator
discriminator = nn.Sequential(
    nn.Conv2d(4,64,kernel_size=4,stride=2,padding=1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(512,1,kernel_size=4,stride=1,padding=0,bias=False),
    
    nn.Flatten(),
    nn.Sigmoid()
)

discriminator = to_device(discriminator,device)




class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)










#בניית המחולל - generator 
latent_size=128
generator=nn.Sequential(

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

generator=to_device(generator, device)

xb=torch.randn(batch_size, latent_size, 1, 1)
fake_images=generator(xb)
print(fake_images.shape)
show_images(fake_images)