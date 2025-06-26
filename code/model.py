#This is just a modification on the coding style. Please refers to the original paper https://arxiv.org/pdf/2311.15719v1 and author Benjamin Keel
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.optim import RMSprop,Adam,SGD
from torch.distributions.dirichlet import Dirichlet
from torch.autograd import Variable
import numpy as np
import math
class LoadImages(Dataset):
    def __init__(self, main_dir, files_list, HU_Upper, HU_Lower):
        # Set the loading directory
        self.main_dir = main_dir

        # Transforms
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Get list of all image file names
        self.all_imgs = files_list
        
        # Get HU limits
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower
                  
        
    def __len__(self):
        # Return the previously computed number of images
        return len(self.all_imgs)    

    def __getitem__(self, index):
        # Get image location
        img_loc = self.main_dir + self.all_imgs[index]
        # Represent image as a tensor
        img = np.load(img_loc)
        # scaling idea from https://www.mdpi.com/2076-3417/10/21/7837
        # set all air (<-1000) to 0 and all bone (>400) to 1, scale all other numbers to between [0,1] 
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower)/(self.HU_Upper - self.HU_Lower), img)
        img[img<self.HU_Lower] = 0
        img[img>self.HU_Upper] = 1
        img = self.transform(img) 
        return img


class DIR_VAE(nn.Module):
    def __init__(self, base, latent_size, alpha_fill_value, annealing, beta, alpha_scalar, ssim_indicator, ssim_scalar):
        super(DIR_VAE, self).__init__()
        self.base = base
        self.beta = beta
        self.latent_size = latent_size
        self.alpha_fill_value = alpha_fill_value
        self.alpha_scalar = alpha_scalar
        self.ssim_indicator = ssim_indicator
        self.ssim_scalar = ssim_scalar
        self.annealing = annealing
        # output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),        # (64 - 3 + 2)/1 + 1  = 64
            Conv(base, 2*base, 3, stride=1, padding=1),   # 64
            Conv(2*base, 2*base, 3, stride=2, padding=1), # (64 - 3 + 2)/2 + 1 = 32
            Conv(2*base, 2*base, 3, stride=1, padding=1), # 32
            Conv(2*base, 4*base, 3, stride=2, padding=1), # (32 - 3 + 2)/2 + 1 = 16
            Conv(4*base, 4*base, 3, stride=1, padding=1), # 16
            Conv(4*base, 4*base, 3, stride=2, padding=1), # (16 - 3 + 2)/2 + 1 = 8
            nn.Conv2d(4*base, 32*base, 8),                # (8 - 8 + 0)/1 + 1 = 1
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(in_features= 32*base, out_features=latent_size*base, bias=False),
            nn.BatchNorm1d(num_features=latent_size*base, momentum=0.9),
            nn.GELU()
        )
        
        self.alpha_fc = nn.Linear(in_features=latent_size*base, out_features=latent_size*base)
        
        # Conv2d_output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        # ConvTranspose_output_width =(input_width −1)*stride − 2*in_padding + dilation*(kernel_width−1) + out_padding +1
        self.decoder = nn.Sequential(
             nn.Linear(in_features=latent_size*base, out_features=32*base, bias=False),
             nn.BatchNorm1d(num_features=32*base, momentum=0.9),
             nn.GELU(),
             nn.Unflatten(1,(32*base,1,1)),
             nn.Conv2d(32*base, 32*base, 1),                       # (1 - 1)/1 + 1 = 1                              ## 32 64
             ConvTranspose(32*base, 4*base, 8),                    # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4         
             Conv(4*base, 4*base, 3, padding=1),                   # (8 - 3 + 2)/1 + 1 = 8                          ## 4 4 
             ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4         
             Conv(4*base, 2*base, 3, padding=1),                   # (16 - 3 + 2)/1 + 1 = 16                        ## 4 2 
             ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2         
             Conv(2*base, base, 3, padding=1),                     # 32                                             ## 2 1  
             ConvUpsampling(base, base, 4, stride=2, padding=1),    # (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64   ## 1 1 
             nn.Conv2d(base, 1, 3, padding=1),                     # 64                                             ## 1 1
             nn.Sigmoid() #nn.Tanh()
        )
    def encode(self, x):
        return self.encoder(x)    
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = self.encode(x)
        batch_size = x.shape[0]
        #print(x.shape[0])
        alpha = self.alpha_fc(x)
        resampler = ResampleDir(self.latent_size, batch_size, self.alpha_fill_value)
        dirichlet_sample = resampler.sample(alpha) # This variable that follows a Dirichlet distribution
        recon_x = self.decoder(dirichlet_sample)   # can be interpreted as a probability that the sum is 1)
        return recon_x, alpha, dirichlet_sample
    

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, alpha, epoch):
        
        batch_size = x.shape[0]
        scale_factor = 1/(batch_size*self.base)
        # linear annealing: reduce the effect of KL divergence over time
        def linear_annealing(init, fin, step, annealing_steps):
            """Linear annealing of a parameter."""
            if annealing_steps == 0:
                return fin
            assert fin > init
            delta = fin - init
            annealed = min(init + delta * step / annealing_steps, fin)
            return annealed

        if self.annealing == 1:
            C = (linear_annealing(0, 1, epoch, 1))
        if self.annealing == 0:
            C = 0
            
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        resampler = ResampleDir(self.latent_size*self.base, batch_size, self.alpha_fill_value) 

        # 0.5 * sum(1 + log(logvar^2) - mu^2 - logvar^2)
        #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * scale_factor
        kld = resampler.prior_forward(alpha)
        kld = torch.sum(kld) 
        
        l1_loss = nn.L1Loss(reduction='sum') 
        recon_loss = l1_loss(recon_x, x) * scale_factor
        #print("recon scale factor", scale_factor)

        if self.ssim_scalar == 2:
            ssim_scalar = batch_size

        if self.ssim_indicator == 0:
            recon_mix = recon_loss  

        # idea from https://arxiv.org/pdf/1511.08861.pdf     alpha*l1_loss + (1-aplha)*ssim_loss  *guassian kernal (not included!)
        if self.ssim_indicator == 1:
            # https://github.com/VainF/pytorch-msssim
            ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
            recon_mix = self.alpha_scalar*recon_loss + (1-self.alpha_scalar)*ssim_loss*self.ssim_scalar  

        if self.ssim_indicator == 2:
            # https://github.com/VainF/pytorch-msssim
            ssim_loss = 1 - ms_ssim(x, recon_x, data_range=1, win_size=3)
            recon_mix = self.alpha_scalar*recon_loss + (1-self.alpha_scalar)*ssim_loss*self.ssim_scalar  


        # idea from beta vae: https://openreview.net/pdf?id=Sy2fzU9gl
        beta_norm = (10*self.beta*self.latent_size)/(64*64*batch_size)
        #print("KLD scale factor", beta_norm)
        beta_vae_loss = recon_mix + beta_norm*(kld - C).abs()
        pure_loss = recon_loss + kld

        ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
        ms_ssim_score = ms_ssim(x, recon_x, data_range=1, win_size=3)
        if epoch%100==1 or (epoch < 3):
            print('recon loss: {:.4f}'.format(recon_loss.item()),'recon mix: {:.4f}'.format(recon_mix), 
                  'kld loss: {:.4f}'.format(kld.item()), 'kld loss scaled: {:.4f}'.format((kld - C).abs().item()),
                  'SSIM score: {:.4f}'.format(ssim_score.item()), 'MS-SSIM: {:.4f}'.format(ms_ssim_score.item()))
        return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss
    
class ResampleDir(nn.Module):
    def __init__(self, latent_dim, batch_size, alpha_fill_value):
        super(ResampleDir, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.alpha_fill_value = alpha_fill_value
        self.alpha_target = torch.full((batch_size, latent_dim), fill_value=alpha_fill_value, dtype=torch.float, device="cuda")#'cuda') #0.5
        
    def concentrations_from_logits(self, logits):
        alpha_c = torch.exp(logits)
        alpha_c = torch.clamp(alpha_c, min=1e-10, max=1e10)
        alpha_c = torch.log(1.+alpha_c)
        return alpha_c

    def dirichlet_kl_divergence(self, logits, eps=1e-10):
        alpha_c_pred = self.concentrations_from_logits(logits)

        alpha_0_target = torch.sum(self.alpha_target, axis=-1, keepdims=True)
        alpha_0_pred = torch.sum(alpha_c_pred, axis=-1, keepdims=True)

        term1 = torch.lgamma(alpha_0_target) - torch.lgamma(alpha_0_pred)
        term2 = torch.lgamma(alpha_c_pred + eps) - torch.lgamma(self.alpha_target + eps)

        term3_tmp = torch.digamma(self.alpha_target + eps) - torch.digamma(alpha_0_target + eps)
        term3 = (self.alpha_target - alpha_c_pred) * term3_tmp

        result = torch.squeeze(term1 + torch.sum(term2 + term3, keepdims=True, axis=-1))
        return result
 
    def prior_forward(self, logits): # analytical kld loss
        latent_vector = self.dirichlet_kl_divergence(logits)
        return latent_vector

    def sample(self, logits):
        alpha_pred = self.concentrations_from_logits(logits)   
        # rsample creates a differentiable sample
        dir_sample = torch.squeeze(Dirichlet(alpha_pred).rsample()) #1 # output to decoder 
        return dir_sample

   # def direct_kld_loss(self, alpha_pred):
   #     dir1 = Dirichlet(self.alpha_target)
   #     dir2 = Dirichlet(alpha)

   #     kld = dir2.log_prob(alpha_pred) - dir1.log_prob(alpha_pred)
   #     return kld
############################################################


# Concolutional Block
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Convolutional Transpose Block for upsampling (decoder)
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Convolutional Bilinear Upsampling Block    
#https://distill.pub/2016/deconv-checkerboard/
#  the checkerboard could be reduced by replacing transpose convolutions with bilinear upsampling
class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)

class MLP(nn.Module):
    def __init__(self, latent_size, base, fchead, dropout):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size * base, fchead[0]),
            nn.GELU(),
            nn.BatchNorm1d(fchead[0], eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(fchead[0],fchead[1]),
            nn.GELU(),
            nn.BatchNorm1d(fchead[1], eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(fchead[1],fchead[2]),
            nn.GELU(),
            nn.BatchNorm1d(fchead[2], eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(fchead[2],1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    #model = DIR_VAE(18, 8, 0.6, 1, 1, 0.5, 1, 2)
    #    def __init__(self, base, latent_size, alpha_fill_value, annealing, beta, alpha_scalar, ssim_indicator, ssim_scalar):

    #summary(model, inputsize = (1,1,64,64))
    model = MLP(32,128,[256,128,64],0.2)
    summary(model, inputsize = (1,1,32*128))

    #train_images = [x for x in glob.glob("./train/images/*.npy")]
    #train_images = LoadImages(main_dir="", files_list=train_images, HU_Upper=150, HU_Lower=-1350)
    #train_loader = DataLoader(train_images, 32, shuffle=True)
    #optimiser = optim.Adam(model.parameters(), lr=1e-4) #, weight_decay=1e-4)
    #device = "cuda"
    #model.to(device)
    #train(model, 1, optimiser, train_loader)
