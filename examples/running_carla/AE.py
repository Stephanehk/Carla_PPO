from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import cv2
import os

device = torch.device("cpu")

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.sigmoid(self.t_conv5(x))

        return x

model = AE()
bce = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def load_train_images_from_folder(folder):
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

def load_test_images_from_folder(folder):
    images = []
    filenames = []
    i = 0
    for filename in os.listdir(folder):
        i+=1
        if i > 10000:
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
        if i > 15000:
            return images,filenames
    return images,filenames


def format_frame(frame):
    #frame = cv2.resize(frame,(516,516))
    frame = torch.FloatTensor(frame).to(device)
    h,w,c = frame.shape
    frame = frame.unsqueeze(0).view(1, c, h, w)
    return frame

def train(epochs):
    print ("loading images...")
    X, filenames = load_train_images_from_folder("sample_frames")
    print("DONE\n")
    print ("training...")

    for epoch in range(epochs):
        train_loss = 0
        for data, filename in zip(X,filenames):
            #data = data.to(device)
            data = format_frame(data)
            recon_images= model(data)

            loss = bce(recon_images,data)
            train_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(X)))
        if epoch%50 == 0:
            torch.save(model.state_dict(), "dim=orig-VAE_state_dictionary.pt")
    print ("DONE\n")
    torch.save(model.state_dict(), "dim=orig-VAE_state_dictionary.pt")

def test():
    model = AE()
    model.load_state_dict(torch.load("/Users/stephanehatgiskessell/Desktop/Carla_PPO/examples/running_carla/dim=516VAE_state_dictionary.pt",map_location=torch.device('cpu')))
    model.eval()

    img_ = cv2.imread("/Users/stephanehatgiskessell/Desktop/Carla_PPO/examples/running_carla/sample_frames/frame0.0077115736319527395.png")
    img = format_frame(img_)

    encoded_img = model.forward(img).squeeze().view(480,640,3).detach().numpy()
    print (img_.shape)
    print (encoded_img.shape)
    cv2.imshow("original",img_)
    cv2.imshow("encoded",encoded_img)
    cv2.waitKey(0)
train(1000)
