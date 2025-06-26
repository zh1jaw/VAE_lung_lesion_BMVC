#This is just a modification on the coding style. Please refers to the original paper https://arxiv.org/pdf/2311.15719v1 and author Benjamin Keel
import glob
from model import DIR_VAE
from torch import optim
import torch
from torchvision import transforms
import math
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision.utils import make_grid, save_image
import argparse
import os
import time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--base", type = int, default = 128)
parser.add_argument("--latent_size", type = int, default = 32)
parser.add_argument("--alpha_fill_value", type = float, default = 0.5)
parser.add_argument("--annealing", type = int, default = 0)
parser.add_argument("--beta", type = float, default = 0.5)
parser.add_argument("--alpha_scalar", type = float, default = 0.3)
parser.add_argument("--ssim_indicator", type = int, default = 2)
parser.add_argument("--ssim_scalar", type = int, default = 2)
parser.add_argument("--resultfolder", type = str, default = "./Dirchlet_VAE_RandomSearch")
parser.add_argument("--HU_low", type = int, default = -500)
parser.add_argument("--HU_high", type = int, default = 500)
parser.add_argument("--epochs", type = int, default = 400)

args = parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=500, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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

def train_model(model, ResultFolder, args, train_loader, test_loader):
    train_losses, test_losses, train_ssim_score_list ,test_ssim_score_list = [], [], [], []
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) #, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                           threshold=0.001, threshold_mode='abs')
    
    counter = 0
    es = EarlyStopping(patience=300, verbose=False, delta=0, path=os.path.join(ResultFolder, 'best.pt'))
    for epoch in range(1, args.epochs + 1):
        train_loss, ssim_score, kld = train(model, epoch, optimiser, train_loader)
        test_loss, test_ssim = test(model, epoch, test_loader, ResultFolder, args)
        scheduler.step(train_loss)
        if math.isnan(train_loss):
            print('Training stopped due to infinite loss')
            break
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ssim_score_list.append(ssim_score)
        test_ssim_score_list.append(test_ssim)
        print(ResultFolder)
        if (epoch+1) % 25 == 1:
            torch.save({"state_dict": model.state_dict(),"optim": optimiser},\
                       ResultFolder + f"/{epoch:04}check.pt")
        es(test_ssim, model)
        if es.early_stop:
            print('Training stopped due early stop')
            break

    torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "test_losses": test_losses, 'trainssim': ssim_score, 'test_ssim': test_ssim}, ResultFolder + "/last.pt") 
    return test_loss, test_ssim

def test(model, epoch, test_loader, ResultFolder, args):
    batch_size = args.batch_size
    model.eval()
    test_loss, beta_test_loss = 0, 0
    ssim_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            recon_batch, alpha, dirichlet_sample = model(data)
            testloss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch)
            test_loss += pure_loss.item()
            beta_test_loss += testloss.item()
            ssim_list.append(ssim_score.item())
            if math.isnan(testloss):
                break
            if (epoch%20 == 1) or epoch == args.epochs - 1:
                if i < 10 and data.shape[0] == batch_size:
                    n = min(data.size(0), 16)
                    comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               ResultFolder + '/test_' + str(epoch) + f'_{i:03}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    beta_test_loss /= len(test_loader.dataset)
    print('====> Pure Test Loss: {:.4f}'.format(test_loss))
    print('====> Beta Test Loss: {:.4f}'.format(beta_test_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Test SSIM: {:.4f}'.format(ssim_mean))
    return test_loss, ssim_mean


def train(model, epoch, optimiser, train_loader):
    model.train()
    train_loss, beta_train_loss = 0, 0
    ssim_list = []
    device = "cuda"
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        optimiser.zero_grad()
        batch_size = data.shape[0]
        recon_batch, alpha, dirichlet_sample = model(data)
        loss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch)
        ssim_list.append(ssim_score.item())
        loss.backward()
        train_loss += pure_loss.item()
        beta_train_loss += loss.item()
        optimiser.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                pure_loss.item(), loss.item()))
        if math.isnan(loss):
            break
    
    train_loss /= len(train_loader.dataset)
    beta_train_loss /= len(train_loader.dataset)
    print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
    print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))

    return train_loss, ssim_mean, kld

if __name__ == "__main__":
    model = DIR_VAE(base = args.base, latent_size = args.latent_size, alpha_fill_value = args.alpha_fill_value,\
                    annealing = args.annealing, beta = args.beta, alpha_scalar = args.alpha_scalar,\
                    ssim_indicator = args.ssim_indicator, ssim_scalar = args.ssim_scalar)
    #    def __init__(self, base, latent_size, alpha_fill_value, annealing, beta, alpha_scalar, ssim_indicator, ssim_scalar):
    ResultFolder = os.path.join(args.resultfolder, str(time.strftime("%Y%m%d%H%M")))
    outp = Path(ResultFolder)
    outp.mkdir(exist_ok = True)
    with open(os.path.join(ResultFolder, "Args.txt"), "w") as fp:
        for arg in vars(args):
            fp.write(f"{arg}: {getattr(args, arg)}\n")
    train_images = [x for x in glob.glob("./train/images/*.npy")]
    train_images = LoadImages(main_dir="", files_list=train_images, HU_Upper=500, HU_Lower=-1000)
    valid_images = [x for x in glob.glob("./valid/images/*.npy")]
    valid_images = LoadImages(main_dir="", files_list=valid_images, HU_Upper=500, HU_Lower=-1000)
    train_loader = DataLoader(train_images, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, args.batch_size, shuffle=False)
    device = "cuda"
    model.to(device)    
    test_loss, ssim_score = train_model(model, ResultFolder, args, train_loader=train_loader, test_loader=valid_loader) 
