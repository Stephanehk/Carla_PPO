#NOT MY CODE:
#TAKEN FROM HERE:
#https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import cv2
import os

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        out= input.view(input.size(0), size, 1, 1)
        return out

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu.to(device) + std.to(device) * esp.to(device)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x).to(device)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def load_images_from_folder(folder):
    images = []
    filenames = []
    i = 0
    for filename in os.listdir(folder):
        i+=1
        if i > 10000:
            return images,filenames
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images,filenames


def format_frame(frame):
    frame = cv2.resize(frame,(64,64))
    frame = torch.FloatTensor(frame).to(device)
    h,w,c = frame.shape
    frame = frame.unsqueeze(0).view(1, c, h, w)
    return frame

def train(epochs):
    print ("loading images...")
    X, filenames = load_images_from_folder("sample_frames")
    print("DONE\n")
    print ("training...")

    for epoch in range(epochs):
        train_loss = 0
        for data, filename in zip(X,filenames):
            #data = data.to(device)
            data = format_frame(data)
            recon_images, mu, logvar = model(data)
            loss, bce, kld = loss_fn(recon_images, data, mu, logvar)
            train_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(X)))
    print ("DONE\n")
    torch.save(model.state_dict(), "dim=64VAE_state_dictionary.pt")


train (1000)
