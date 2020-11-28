import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

print('Cuda:', torch.cuda.is_available())
torch.manual_seed(0)
#torch.set_deterministic(True)
np.random.seed(0)

INDIR = '../../input'
OUTDIR = '../../output'
N_IN_CHANNELS = 3 # RGB
N_CLASSES = 2 # binary classification
N_LATENT = 100
N_EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 96
CROP_SIZE = 64
N_ROW_IMG = 4 # show 4x4 grid of generated img

LABELS = torch.LongTensor(range(N_CLASSES)).repeat_interleave(N_ROW_IMG * N_ROW_IMG).cuda()
LABELS_ONEHOT = F.one_hot(LABELS, N_CLASSES)

def imshow(x):
    img = x.data.cpu().permute(1, 2, 0).numpy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

class TumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.id[index] + '.tif'
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        y_label = torch.tensor(self.annotations.label[index])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

class ConditionalConvVAE(nn.Module):
    def __init__(self, latent_dim, n_in_channels, n_classes):
        super(ConditionalConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        n_channels = 16 # tuneable hyperparam
        self.n_channels = n_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(n_in_channels + n_classes, n_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1),
            nn.Flatten()
        )
        self.flat_dim = n_channels * 8 * 4 * 4

        self.mu = nn.Linear(self.flat_dim, latent_dim)
        self.logvar = nn.Linear(self.flat_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim + n_classes, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 2, n_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels, n_in_channels, 4, 2, 1),
        )

    def encode(self, input):
        # implementation goes here
        x = self.encoder(input)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        # implementation goes here
        epsilon = torch.normal(0., 1., size=mu.size()).cuda()
        std = torch.exp(logvar * 0.5)
        z = epsilon * std + mu
        return z

    def decode(self, input):
        # implementation goes here
        out = self.decoder_fc(input)
        out = out.reshape(-1, self.n_channels * 8, 4, 4)
        out = self.decoder(out)
        return out

    def forward(self, x, y):
        """
        y must be one-hot
        """
        # add n_classes as additional channels
        # num_per_batch x n_classes x 1 x 1
        channels = y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, channels), dim=1)

        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        z = torch.cat((z, y), dim=1)

        out = self.decode(z)
        return mu, logvar, out

    def generate(self, n, y):
        """
        y must be one-hot and be of length n
        """
        z = torch.randn(n, self.latent_dim).cuda()
        z = torch.cat((z, y), dim=1)
        samples = self.decode(z)
        return samples

def vae_loss(x, out, mu, logvar, beta=1):
    # implementation goes here
    recons_loss = ((out - x) * (out - x)).sum()
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recons_loss + beta * kld_loss
    return recons_loss, kld_loss, loss

class ConditionalConvGenerator(nn.Module):
    def __init__(self, latent_dim, n_in_channels, n_classes, img_size):
        """
        assume img has same height and width
        """
        super(ConditionalConvGenerator, self).__init__()
        self.latent_dim = latent_dim

        n_channels = 16 # tuneable hyperparam
        self.n_channels = n_channels
        self.emb_size = 128
        self.flat_dim = n_channels * 8 * 4 * 4

        # to embed noise
        self.emb = nn.Embedding(n_classes, self.emb_size)
        self.decoder_fc = nn.Linear(latent_dim + self.emb_size, self.flat_dim)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 2, n_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels, n_in_channels, 4, 2, 1)
        )

    def decode(self, input):
        # implementation goes here
        out = self.decoder_fc(input)
        out = out.reshape(-1, self.n_channels * 8, 4, 4)
        out = self.network(out)
        return out

    def forward(self, n, y):
        """
        y must be scalar labels
        """
        z = torch.randn(n, self.latent_dim).cuda()
        embed = self.emb(y)
        z = torch.cat((z, embed), dim=1)
        samples = self.decode(z)
        return samples

class ConditionalConvDiscriminator(nn.Module):
    def __init__(self, latent_dim, n_in_channels, n_classes, img_size):
        super(ConditionalConvDiscriminator, self).__init__()
        self.latent_dim = latent_dim

        n_channels = 16
        self.n_channels = n_channels
        self.flat_dim = n_channels * 8 * 4 * 4
        self.n_in_channels = n_in_channels

        # to embed class labels
        self.emb = nn.Embedding(n_classes, img_size * img_size)
        self.network = nn.Sequential(
            # one more channel from label
            nn.Conv2d(n_in_channels + 1, n_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(self.flat_dim, 1) # scalar output
            # no need for sigmoid as we are using BCEWithLogitsLoss
        )

    def forward(self, x, y):
        # implementation goes here
        embed = self.emb(y).view(y.shape[0], 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, embed), dim=1)
        out = self.network(x)
        return out

def create_classifier(n_in_channels, n_channels=16):
    flat_dim = n_channels * 8 * 4 * 4
    model = nn.Sequential(
            nn.Conv2d(n_in_channels, n_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(flat_dim, 1),
            nn.Sigmoid() # scalar output, use BCELoss
        )
    return model.cuda()

def train_classifier(epoch, model, opt, criterion, train_loader, dev_loader, writer):
    model.train()
    epoch_loss = 0
    for data, labels in train_loader:
        data = data.cuda()
        labels = labels.cuda()
        preds = model(data)
        loss = criterion(preds.squeeze(), labels.float())

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    # end of epoch, eval on dev and record stats
    model.eval()
    with torch.no_grad():
        # a single batch
        for data, labels in dev_loader:
            x = data.cuda()
            y = labels.cuda()
            preds = model(x).squeeze()
            loss = criterion(preds, y.float())

    dev_auc = roc_auc_score(labels, preds.cpu())
    writer.add_scalars('loss',
                       {'train': epoch_loss, 'dev': loss.item()},
                       epoch)
    writer.add_scalar('AUC/dev', dev_auc, epoch)
    # save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()
            },
        os.path.join(writer.log_dir, 'model_{}.pth'.format(epoch)))
    return dev_auc # save model with best dev auc

def train_cvae(epoch, model, opt, loader, writer):
    model.train()
    epoch_recons_loss = 0
    epoch_kld_loss = 0
    epoch_loss = 0
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        # y: one-hot labels
        y = F.one_hot(y, N_CLASSES).cuda()

        mu, logvar, out = model(x, y)
        recons_loss, kld_loss, loss = vae_loss(x, out, mu, logvar)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # record batch stats
        with torch.no_grad():
            epoch_recons_loss += recons_loss.item()
            epoch_kld_loss += kld_loss.item()
            epoch_loss += loss.item()

            if i == 0: # first batch, generate fakes
                model.eval()
                data = model.generate(LABELS_ONEHOT.shape[0], LABELS_ONEHOT)
                grid_img = torchvision.utils.make_grid(data, nrow=N_ROW_IMG, normalize=True)
                writer.add_image('generated image', grid_img, epoch)

    writer.add_scalar('reconstruction loss', epoch_recons_loss, epoch)
    writer.add_scalar('KL-Divergence loss', epoch_kld_loss, epoch)
    writer.add_scalar('total loss', epoch_loss, epoch)
    # save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()
            },
        os.path.join(writer.log_dir, 'model_{}.pth'.format(epoch)))
    return epoch_kld_loss

def train_cgan(epoch, generator, discriminator, gopt, dopt, critierion, loader, writer):
    model.train()
    epoch_gloss = 0
    epoch_dloss = 0
    for i, (x, y) in enumerate(loader):
        batch_size = x.shape[0]

        real_x = x.cuda()
        real_y = y.cuda()

        fake_x = generator(batch_size, real_y).cuda()
        fake_y = torch.LongTensor(torch.randint(0, N_CLASSES, size=(BATCH_SIZE,))).cuda()

        disc_labels_real = torch.ones((batch_size,), dtype=torch.float).cuda()
        disc_labels_fake = torch.zeros((batch_size,), dtype=torch.float).cuda()

        # Train G
        # tell discriminator these are real data
        preds_fake = discriminator(fake_x, real_y).squeeze().cuda()
        gloss = criterion(preds_fake, disc_labels_real)
        gopt.zero_grad()
        gloss.backward()
        gopt.step()

        # Train D to tell if the labeled images are fake
        # real
        preds_real = discriminator(real_x, real_y).squeeze().cuda()
        dloss = criterion(preds_real, disc_labels_real)
        # fake, detach so grad doesn't go into generator
        preds_fake = discriminator(fake_x.detach(), fake_y).squeeze().cuda()
        dloss += criterion(preds_fake, disc_labels_fake)
        dopt.zero_grad()
        dloss.backward()
        dopt.step()

        # record batch stats
        with torch.no_grad():
            epoch_gloss += gloss.item()
            epoch_dloss += dloss.item()

            if i == 0: # first batch, generate fakes
                generator.eval()
                data = generator(LABELS.shape[0], LABELS)
                grid_img = torchvision.utils.make_grid(data, nrow=N_ROW_IMG, normalize=True)
                writer.add_image('generated image', grid_img, epoch)

    writer.add_scalar('generator loss', epoch_gloss, epoch)
    writer.add_scalar('discriminator loss', epoch_dloss, epoch)
    torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'gopt_state_dict': gopt.state_dict(),
            'dopt_state_dict': dopt.state_dict()
            },
        os.path.join(writer.log_dir, 'model_{}.pth'.format(epoch)))
    return epoch_gloss
