# -*- coding: utf-8 -*-
"""


@author: kajal
"""
#GAN Code
#PyTorch libraries for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
#Used to load image datasets like CIFAR-10 and apply image transformations
import torchvision
from torchvision import datasets, transforms
#Used to display generated images
import matplotlib.pyplot as plt
import numpy as np

#If GPU is available, training runs on GPU. Otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Defining Image Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Loading the CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data',\
              train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, \
                                batch_size=32, shuffle=True)
    
#Defining GAN Hyperparameters
latent_dim = 100  #Random noise vector size. Generator starts from this noise
lr = 0.0002 #Learning rate. Controls how fast weights are updated.
beta1 = 0.5 #Adam optimizer parameters
beta2 = 0.999 #optimizer parameters
num_epochs = 10 #The model will go through the full dataset 10 times.

#Building the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim): #Constructor. It receives noise size.
        super(Generator, self).__init__() #Initializes PyTorch neural network class.

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8), #convert noise into a large feature vector.
            nn.ReLU(),#activation function
            nn.Unflatten(1, (128, 8, 8)), #converts flat vector into image-like feature map
            nn.Upsample(scale_factor=2), #Increases image size from 8×8 to 16×16
            nn.Conv2d(128, 128, kernel_size=3, padding=1),#Applies convolution to learn image features.
            nn.BatchNorm2d(128, momentum=0.78), #Stabilizes training
            nn.ReLU(), #add non-linearity again
            nn.Upsample(scale_factor=2), #Increases image size from 8×8 to 16×16
            nn.Conv2d(128, 64, kernel_size=3, padding=1), #Reduces channels from 128 to 64.
            nn.BatchNorm2d(64, momentum=0.78), #Again stabilizes learning
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1), #Converts 64 channels into 3 channels:
            nn.Tanh() #Final output image pixel values between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        return img
    
#Building the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),#stride=2 reduces image size
        nn.LeakyReLU(0.2),#Activation function
        nn.Dropout(0.25),#Randomly drops some neurons to avoid overfitting
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #Extracts deeper features
        nn.ZeroPad2d((0, 1, 0, 1)), #Adds padding to adjust image size
        nn.BatchNorm2d(64, momentum=0.82),#Normalizes activations
        nn.LeakyReLU(0.25),#Adds non-linearity
        nn.Dropout(0.25),#Regularization
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),#Learns more complex image features
        nn.BatchNorm2d(128, momentum=0.82),#stabilizes training
        nn.LeakyReLU(0.2),#Activation
        nn.Dropout(0.25),#Prevents overfitting
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),#Learns high-level image features
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(256 * 5 * 5, 1), #Final classification layer
        nn.Sigmoid() #Converts output into probability between 0 and 1
    )

    def forward(self, img):
        validity = self.model(img)
        return validity
    
#Initializing GAN Components
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

#Updates Generator weights
optimizer_G = optim.Adam(generator.parameters()\
                         , lr=lr, betas=(beta1, beta2)) 
#Updates Discriminator weight
optimizer_D = optim.Adam(discriminator.parameters()\
                         , lr=lr, betas=(beta1, beta2))
    
#Training the GAN
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
       
        real_images = batch[0].to(device) 
       
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)
       
        real_images = real_images.to(device)

#train discriminator
        optimizer_D.zero_grad()
       
        z = torch.randn(real_images.size(0), latent_dim, device=device) #noise
      
        fake_images = generator(z) #generator creates fake images

        real_loss = adversarial_loss(discriminator\
                                     (real_images), valid) #discriminator checks real images, expected answer =1
        fake_loss = adversarial_loss(discriminator\
                                     (fake_images.detach()), fake) #discriminator checks fake images, expected answer =0
            #.detach() means Generator is not updated here
        d_loss = (real_loss + fake_loss) / 2
    
        d_loss.backward() #backpropagation calculates gradients
        optimizer_D.step() #updates discriminator weights

#train generator
        optimizer_G.zero_grad() #class generator gradients
      
        gen_images = generator(z) #generate fake images
        
        g_loss = adversarial_loss(discriminator(gen_images), valid) #Generator wants Discriminator 
        #to think fake images are real.So fake images are compared with label valid = 1.
        g_loss.backward() #Backpropagation for Generator
        optimizer_G.step() #updates generator weights
       
        #after every 100 batches, print loss
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\
                        Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
                #after 10 epochs, generate sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu() #Generator creates 16 fake images.
            grid = torchvision.utils.make_grid(generated,\
                                        nrow=4, normalize=True) #arranges images in 4x4 grid
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()