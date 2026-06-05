from dataset import InpaintingDataset
from model import InpaintingGenerator, InpaintingDiscriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Dataset
# =========================

train_dataset = InpaintingDataset(
    corrupted_dir=r"C:\Projects\FinalProject\NewProject\dataset\damaged",
    mask_dir=r"C:\Projects\FinalProject\NewProject\dataset\images",
    original_dir=r"C:\Projects\FinalProject\NewProject\dataset\masks_2"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# =========================
# Models
# =========================

generator = InpaintingGenerator().to(device)
discriminator = InpaintingDiscriminator().to(device)

# =========================
# Optimizers
# =========================

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# =========================
# Losses
# =========================

adv_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# =========================
# Training
# =========================

num_epochs = 40

for epoch in range(num_epochs):

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for i, batch in enumerate(train_dataloader):

        corrupted = batch[0].to(device)
        mask = batch[1].to(device)
        original = batch[2].to(device)

        # ===================================
        # Generator output
        # ===================================

        fake_hole = generator(corrupted, mask)

        # שילוב עם התמונה המקורית
        completed = corrupted * (1 - mask) + fake_hole * mask

        # ===================================
        # Train Discriminator
        # ===================================

        optimizer_D.zero_grad()

        real_out = discriminator(original, mask)
        fake_out = discriminator(completed.detach(), mask)

        real_labels = torch.ones_like(real_out)
        fake_labels = torch.zeros_like(fake_out)

        loss_real = adv_loss(real_out, real_labels)
        loss_fake = adv_loss(fake_out, fake_labels)

        loss_D = (loss_real + loss_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        # ===================================
        # Train Generator
        # ===================================

        optimizer_G.zero_grad()

        fake_out = discriminator(completed, mask)

        adv = adv_loss(fake_out, real_labels)

        # reconstruction רק על החור
        recon = l1_loss(
            fake_hole * mask,
            original * mask
        )

        loss_G = adv + 100 * recon

        loss_G.backward()

        optimizer_G.step()

        # ===================================
        # Print
        # ===================================

        if i % 10 == 0:
            print(
                f"Batch {i} | "
                f"D Loss: {loss_D.item():.4f} | "
                f"G Loss: {loss_G.item():.4f}"
            )

    # ===================================
    # Save checkpoint
    # ===================================

    torch.save(
    generator.state_dict(),
    f"checkpoints/generator_epoch_{epoch}.pth"
    )

    print("Checkpoint saved")