import glob
from model import DIR_VAE, MLP, VAE
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
import pandas as pd
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torcheval.metrics import BinaryConfusionMatrix


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 5e-5)
parser.add_argument("--base", type = int, default = 128)
parser.add_argument("--latent_size", type = int, default = 32)
parser.add_argument("--alpha_fill_value", type = float, default = 0.85)
parser.add_argument("--annealing", type = int, default = 1)
parser.add_argument("--beta", type = int, default = 1)
parser.add_argument("--alpha_scalar", type = float, default = 0.5)
parser.add_argument("--ssim_indicator", type = int, default = 2)
parser.add_argument("--ssim_scalar", type = int, default = 2)
parser.add_argument("--resultfolder", type = str, default = "./JointVAEMLP")
parser.add_argument("--HU_low", type = int, default = -500)
parser.add_argument("--HU_high", type = int, default = 1000)
parser.add_argument("--epochs", type = int, default = 1200)
parser.add_argument("--Folder", type = str)
parser.add_argument("--vaename", type = str)
parser.add_argument("--DirVAE", action='store_true')
parser.add_argument("--frzmlp", action='store_true')
parser.add_argument("--Startover", action='store_true')
parser.add_argument("--mlp_scalar", type = float, default = 0.025)
args = parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0, path='checkpoint.pt'):
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
        self.all_imgs = files_list[0]
        self.label = files_list[1]
        # Get HU limits
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower
                  
        
    def __len__(self):
        # Return the previously computed number of images
        return len(self.all_imgs)    

    def __getitem__(self, index):
        # Get image location
        img_loc = self.main_dir + self.all_imgs[index] + ".npy"
        # Represent image as a tensor
        img = np.load(img_loc)
        # scaling idea from https://www.mdpi.com/2076-3417/10/21/7837
        # set all air (<-1000) to 0 and all bone (>400) to 1, scale all other numbers to between [0,1] 
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower)/(self.HU_Upper - self.HU_Lower), img)
        img[img<self.HU_Lower] = 0
        img[img>self.HU_Upper] = 1
        img = self.transform(img) , self.label[index]
        return img

def parse_text_to_dict(lines):
    def convert_value(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    result_dict = {}
    for line in lines:
        if ': ' in line:
            key, value = line.strip().split(': ', 1)
            result_dict[key] = convert_value(value)

    return result_dict

def trainMLPModel(model, vaemodel, device, train_loader, test_loader, args, ResultFolder):
    model.to(device)
    vaemodel.to(device)
    vaemodel.eval()
    lossFn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    es = EarlyStopping(patience=100, verbose=False, delta=0, path=os.path.join(ResultFolder, 'best.pt'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.001, threshold_mode='abs')
    History = {
        'train_loss':[],
        'test_loss':[],
        'state_dict':model.state_dict(),
        'train_acc':[],
        'test_acc':[]
    }
    
    for epoch in range(1,args.epochs+1):  # loop over the dataset multiple times
        train_loss, trainacc = train(model, vaemodel, device, train_loader, lossFn, optimizer, scheduler)
        print('Train Epoch: {}, acc {:.2f} loss {:.6f}'.format(epoch, trainacc, train_loss))

        test_loss, testacc = test(model, vaemodel, device, test_loader, lossFn)
        print('Test Epoch: {}, acc {:.2f} loss {:.6f}'.format(epoch, testacc, test_loss))

        History['train_loss'].append(train_loss)
        History['test_loss'].append(test_loss)
        History['train_acc'].append(trainacc)
        History['test_acc'].append(testacc)
        es(test_loss, model)
        if epoch % 25 == 1 or epoch == args.epochs:
            History['state_dict'] = model.state_dict()
            torch.save(History, ResultFolder + f"/{epoch:03}check.pt")

def trainVaewithMlpFrz(mlpmodel, vaemodel, device, train_loader, test_loader, ResultFolder, args):
    train_losses, test_losses, train_ssim_score_list ,test_ssim_score_list = [], [], [], []
    optimiser = optim.AdamW(vaemodel.parameters(), lr=args.lr, weight_decay=1e-5) #, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                           threshold=0.001, threshold_mode='abs')
    lossFn = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([5256./3450.]).to(device))

    counter = 0
    es = EarlyStopping(patience=200, verbose=False, delta=0, path=os.path.join(ResultFolder, 'best.pt'))
    
    for epoch in range(1, args.epochs + 1):
        train_loss, ssim_score = trainfrz(mlpmodel, vaemodel, epoch, device, optimiser, train_loader, lossFn, args)
        test_loss, test_ssim = test(mlpmodel, vaemodel, epoch, device, test_loader, ResultFolder, lossFn, args)
        scheduler.step(train_loss)
        if math.isnan(train_loss):
            print('Training stopped due to infinite loss')
            break
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ssim_score_list.append(ssim_score)
        test_ssim_score_list.append(test_ssim)
        print(f"trained: {epoch}|{args.epochs}, save @ {ResultFolder}")
        
        if (epoch+1) % 25 == 1:
            torch.save({"state_dict": vaemodel.state_dict(),"optim": optimiser},\
                       ResultFolder + f"/{epoch:04}check.pt")
        es(test_ssim, vaemodel)
        if es.early_stop:
            print('Training stopped due early stop')
            break

    torch.save({"state_dict": vaemodel.state_dict(), "train_losses": train_losses, "test_losses": test_losses, \
                'trainssim': train_ssim_score_list, 'test_ssim': test_ssim_score_list, "mlpmodel": mlpmodel.state_dict()}, ResultFolder + "/last.pt") 

    return test_loss, test_ssim

def trainVaewithMlp(mlpmodel, vaemodel, device, train_loader, test_loader, ResultFolder, args):
    train_losses, test_losses, train_ssim_score_list ,test_ssim_score_list = [], [], [], []
    optimiser = optim.AdamW(list(vaemodel.parameters()) + list(mlpmodel.parameters()), lr=args.lr, weight_decay=1e-5) #, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                           threshold=0.001, threshold_mode='abs')
    lossFn = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([5256./3450.]).to(device))

    counter = 0
    es = EarlyStopping(patience=200, verbose=False, delta=0, path=os.path.join(ResultFolder, 'best.pt'))
    for epoch in range(1, args.epochs + 1):
        train_loss, ssim_score = train(mlpmodel, vaemodel, epoch, device, optimiser, train_loader, lossFn, args)
        test_loss, test_ssim = test(mlpmodel, vaemodel, epoch, device, test_loader, ResultFolder, lossFn, args)
        scheduler.step(train_loss)
        if math.isnan(train_loss):
            print('Training stopped due to infinite loss')
            break
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ssim_score_list.append(ssim_score)
        test_ssim_score_list.append(test_ssim)
        print(f"trained: {epoch}|{args.epochs}, save @ {ResultFolder}")
        
        if (epoch+1) % 25 == 1:
            torch.save({"state_dict": vaemodel.state_dict(),"optim": optimiser},\
                       ResultFolder + f"/{epoch:04}check.pt")
        es(test_ssim, vaemodel)
        if es.early_stop:
            print('Training stopped due early stop')
            break

    torch.save({"state_dict": vaemodel.state_dict(), "train_losses": train_losses, "test_losses": test_losses, \
                'trainssim': train_ssim_score_list, 'test_ssim': test_ssim_score_list, "mlpmodel": mlpmodel.state_dict()}, ResultFolder + "/last.pt") 

    return test_loss, test_ssim


def trainfrz(mlpmodel, vaemodel, epoch, device, optimiser, train_loader, lossFn, args):
    vaemodel.train()
    mlpmodel.eval()
    train_loss, beta_train_loss = 0, 0
    correct = 0          # number of examples predicted correctly (for accuracy)
    total = 0            # number of examples

    ssim_list = []
    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img = img.float().to(device)
        label = label.float().to(device)
        optimiser.zero_grad()
        batch_size = img.shape[0]
        recon_batch, mu, logvar = vaemodel(img)
        
        vaeloss, recon_loss, kld, ssim_score, pure_loss = vaemodel.loss_function(recon_batch, img, mu, logvar, epoch, args.epochs)
        ssim_list.append(ssim_score.item())
        train_loss += pure_loss.item()
        beta_train_loss += vaeloss.item()
        
        output = mlpmodel(mu.squeeze().squeeze()).squeeze()
        mlploss = lossFn(output, label)# Step3 Combines mlploss while training the VAE
        
        loss = vaeloss + mlploss 
        loss.backward()
        
        train_loss += pure_loss.item()
        beta_train_loss += loss.item()
        optimiser.step()
        
        pred = (torch.sigmoid(output) > 0.5).float() # get the index of the max log-probability
        total += label.size(0)    # add in the number of labels in this minibatch
        correct += (pred == label).sum().item()  # add in the number of correct labels

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                pure_loss.item(), loss.item()))
        if math.isnan(loss):
            break
        if (epoch%20 == 1) or epoch == args.epochs - 1:
            if batch_idx < 2 and img.shape[0] == batch_size:
                n = min(img.size(0), 16)
                comparison = torch.cat([img[:n], recon_batch.view(batch_size, 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                           ResultFolder + '/train_' + str(epoch) + f'_{batch_idx:03}.png', nrow=n)

    
    train_loss /= len(train_loader.dataset)
    beta_train_loss /= len(train_loader.dataset)
    print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
    print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))
    print('====> Average Train MLP Loss: {:.4f}'.format(mlploss))
    return train_loss, ssim_mean

def train(mlpmodel, vaemodel, epoch, device, optimiser, train_loader, lossFn, args):
    vaemodel.train()
    mlpmodel.train()
    train_loss, beta_train_loss = 0, 0
    correct = 0          # number of examples predicted correctly (for accuracy)
    total = 0            # number of examples
    running_loss = 0

    ssim_list = []
    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img = img.float().to(device)
        label = label.float().to(device)
        optimiser.zero_grad()
        batch_size = img.shape[0]
        recon_batch, mu, logvar = vaemodel(img)
        
        vaeloss, recon_loss, kld, ssim_score, pure_loss = vaemodel.loss_function(recon_batch, img, mu, logvar, epoch, args.epochs)
        ssim_list.append(ssim_score.item())
        train_loss += pure_loss.item()
        beta_train_loss += vaeloss.item()
        
        output = mlpmodel(mu.squeeze().squeeze()).squeeze()
        mlploss = lossFn(output, label)# Step3 Combines mlploss while training the VAE
        mlpscalar = epoch * args.mlp_scalar
        mlpscalar = min(1.0, mlpscalar)
        loss = vaeloss + mlploss * mlpscalar
        loss.backward()
        running_loss += mlploss.item()*label.size(0)

        train_loss += pure_loss.item()
        beta_train_loss += loss.item()
        optimiser.step()
        
        pred = (torch.sigmoid(output) > 0.5).float() # get the index of the max log-probability
        total += label.size(0)    # add in the number of labels in this minibatch
        correct += (pred == label).sum().item()  # add in the number of correct labels

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                pure_loss.item(), loss.item()))
        if math.isnan(loss):
            break
        if (epoch%20 == 1) or epoch == args.epochs - 1:
            if batch_idx < 2 and img.shape[0] == batch_size:
                n = min(img.size(0), 16)
                comparison = torch.cat([img[:n], recon_batch.view(batch_size, 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                           ResultFolder + '/train_' + str(epoch) + f'_{batch_idx:03}.png', nrow=n)

    average_loss = running_loss / len(train_loader)

    train_loss /= len(train_loader.dataset)
    beta_train_loss /= len(train_loader.dataset)
    print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
    print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))
    print('====> Average Train MLP Loss: {:.4f}'.format(mlploss))
    print('====> Train Acc{:.3f}'.format(correct / total))
    return train_loss, ssim_mean

                             
def test(mlpmodel, vaemodel, epoch, device, test_loader, ResultFolder, lossFn, args):
    metric = BinaryConfusionMatrix()

    mlpmodel.eval()
    vaemodel.eval()

    correct = 0          # number of examples predicted correctly (for accuracy)
    total = 0            # number of examples
    running_loss = 0
    test_loss, beta_test_loss = 0, 0
    ssim_list = []
    with torch.no_grad():                                               
        for batch_idx, data in enumerate(test_loader):
            img, label = data
            img = img.float().to(device)
            label = label.float().to(device)
            recon_batch, mu, logvar = vaemodel(img)
            testloss, recon_loss, kld, ssim_score, pure_loss = vaemodel.loss_function(recon_batch, img, mu, logvar, epoch, args.epochs)
            ssim_list.append(ssim_score.item())
            test_loss += pure_loss.item()
            beta_test_loss += testloss.item()

            output = mlpmodel(mu.squeeze().squeeze()).squeeze()
            loss = lossFn(output, label)
            pred = (torch.sigmoid(output) > 0.5).float() # get the index of the max log-probability

            total += label.size(0)    # add in the number of labels in this minibatch
            correct += (pred == label).sum().item()  # add in the number of correct labels
            running_loss += loss.item()*label.size(0)
            if math.isnan(testloss):
                break
            if (epoch%10 == 1) or epoch == args.epochs - 1:
                metric.update(pred.int(), label.int())

                if batch_idx < 10 and img.shape[0] == args.batch_size:
                    n = min(img.size(0), 16)
                    comparison = torch.cat([img[:n], recon_batch.view(args.batch_size, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               ResultFolder + '/test_' + str(epoch) + f'_{batch_idx:03}.png', nrow=n)
    if (epoch % 10 == 1) or epoch == args.epochs-1:
        confusion_matrix = metric.compute()
        print(confusion_matrix)
        fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix.detach().numpy(), figsize=(8, 8), cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(ResultFolder + f"/{epoch:03}_confusion_matrix.png")
        plt.close("fig") 
    
    average_loss = running_loss / len(test_loader)
    test_loss /= len(test_loader.dataset)
    beta_test_loss /= len(test_loader.dataset)
    
    print('====> Pure Test Loss: {:.4f}'.format(test_loss))
    print('====> Beta Test Loss: {:.4f}'.format(beta_test_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Test SSIM: {:.4f}'.format(ssim_mean))
    print('====> Average Test MLP Loss: {:.4f}'.format(loss))
    print('====> Test Acc{:.3f}'.format(correct / total))

    return test_loss, ssim_mean

if __name__ == "__main__":
    ResultFolder = os.path.join(args.resultfolder, str(time.strftime("%Y%m%d%H%M")))
    outp = Path(ResultFolder)
    outp.mkdir(exist_ok = True)

    with open(os.path.join(ResultFolder, "Args.txt"), "w") as fp:
        for arg in vars(args):
            fp.write(f"{arg}: {getattr(args, arg)}\n")       

    ## DirVae loader ##
    if not args.Startover:
        if args.DirVAE:
            pretrainroot = "./Dirchlet_VAE_RandomSearch/" + args.Folder
            with open(pretrainroot + "/Args.txt", "r") as fp:
                vaeargs = parse_text_to_dict(fp.readlines())

            vaemodel = DIR_VAE(base = vaeargs['base'], latent_size = vaeargs['latent_size'], alpha_fill_value = vaeargs['alpha_fill_value'],\
                        annealing = vaeargs['annealing'], beta = vaeargs['beta'], alpha_scalar = vaeargs['alpha_scalar'],\
                        ssim_indicator = vaeargs['ssim_indicator'], ssim_scalar = vaeargs['ssim_scalar'])

        else:
            pretrainroot = "./VAE_RandomSearch/" + args.Folder
            with open(pretrainroot + "/Args.txt", "r") as fp:
                vaeargs = parse_text_to_dict(fp.readlines())

            vaemodel = VAE(base = vaeargs['base'], latent_size = vaeargs['latent_size'], alpha_fill_value = vaeargs['alpha_fill_value'],\
                        annealing = vaeargs['annealing'], beta = vaeargs['beta'], alpha_scalar = vaeargs['alpha_scalar'],\
                        ssim_indicator = vaeargs['ssim_indicator'], ssim_scalar = vaeargs['ssim_scalar'])
        pretrainvae = torch.load(os.path.join(pretrainroot, args.vaename)) # should use best.pt in the future

        ## Vae loader ##
        vaemodel.load_state_dict(pretrainvae['state_dict']) 

    ## MLP loader ##
        pretrainedmlp = torch.load('/media/data/NoduleClassfication/Dirchlet_VAE_RandomSearch/MLP/202506270148/best.pt')
        mlpmodel = MLP(vaeargs['latent_size'], vaeargs['base'], [1024,256,128], 0.5)
        mlpmodel.load_state_dict(pretrainedmlp)
    
    ## MLP loader ##
    else:
        if args.DirVAE:
            vaemodel = DIR_VAE(base = args.base, latent_size = args.latent_size, alpha_fill_value = args.alpha_fill_value,\
                    annealing = args.annealing, beta = args.beta, alpha_scalar = args.alpha_scalar,\
                    ssim_indicator = args.ssim_indicator, ssim_scalar = args.ssim_scalar)
        else:
            vaemodel = VAE(base = args.base, latent_size = args.latent_size, alpha_fill_value = args.alpha_fill_value,\
                    annealing = args.annealing, beta = args.beta, alpha_scalar = args.alpha_scalar,\
                    ssim_indicator = args.ssim_indicator, ssim_scalar = args.ssim_scalar)
        mlpmodel = MLP(args.latent_size, args.base, [2048,2048,1024], 0.5)

    ## Data loader ##       
    csv_file = pd.read_csv('./test.csv')
    train_datalist = csv_file[csv_file['data_split'] == 'train']
    train_label = (train_datalist.iloc[:,5]).tolist()
    train_datalist = (train_datalist.iloc[:,3]).tolist()
    val_datalist = csv_file[csv_file['data_split'] == 'valid']
    val_label = (val_datalist.iloc[:,5]).tolist()
    val_datalist = (val_datalist.iloc[:,3]).tolist()
    if not args.Startover:
        train_images = LoadImages(main_dir="./train/images/", files_list=[train_datalist, train_label], HU_Upper=vaeargs['HU_high'], HU_Lower=vaeargs['HU_low'])
        valid_images = LoadImages(main_dir="./valid/images/", files_list=[val_datalist, val_label], HU_Upper=vaeargs['HU_high'], HU_Lower=vaeargs['HU_low'])
    else:
        train_images = LoadImages(main_dir="./train/images/", files_list=[train_datalist, train_label], HU_Upper=args.HU_high, HU_Lower=args.HU_low)
        valid_images = LoadImages(main_dir="./valid/images/", files_list=[val_datalist, val_label], HU_Upper=args.HU_high, HU_Lower=args.HU_low)

    train_loader = DataLoader(train_images, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, args.batch_size, shuffle=False)
    
    ## Data loader ##
    device = "cuda"
    vaemodel.to(device)
    mlpmodel.to(device)
    if args.frzmlp:
        trainVaewithMlpFrz(mlpmodel, vaemodel, device, train_loader, valid_loader, ResultFolder, args)
    else:
        trainVaewithMlp(mlpmodel, vaemodel, device, train_loader, valid_loader, ResultFolder, args)
    #model.to(device)    
    #test_loss, ssim_score = train_model(vaemodel, model, ResultFolder, args, train_loader=train_loader, test_loader=valid_loader) 
