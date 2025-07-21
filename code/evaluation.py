import torch
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from model import VAE, MLP
from mlxtend.plotting import plot_confusion_matrix
from torcheval.metrics import BinaryConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import argparse
from collections import OrderedDict
from torchvision.utils import make_grid, save_image

new_state_dict = OrderedDict()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--lr", type = float, default = 1e-6)
parser.add_argument("--base", type = int, default = 32)
parser.add_argument("--latent_size", type = int, default =8)
# parser.add_argument("--alpha_fill_value", type = float, default = 0.85)
# parser.add_argument("--annealing", type = int, default = 1)
# parser.add_argument("--beta", type = int, default = 1)
# parser.add_argument("--alpha_scalar", type = float, default = 0.5)
# parser.add_argument("--ssim_indicator", type = int, default = 2)
# parser.add_argument("--ssim_scalar", type = int, default = 2)
parser.add_argument("--resultfolder", type = str, default = "./MLP")
parser.add_argument("--HU_low", type = int, default = -500)
parser.add_argument("--HU_high", type = int, default = 400)
parser.add_argument("--epochs", type = int, default = 400)
parser.add_argument("--useposwgt", action = "store_true")
# parser.add_argument("--Folder", type = str)
# parser.add_argument("--vaename", type = str)
parser.add_argument("--l2reg", type = float, default = 1e-4)
args = parser.parse_args()




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
        
        #img_loc = self.main_dir + self.all_imgs[index] + ".npy"

        img_loc = self.main_dir + self.all_imgs[index]
        img = np.load(img_loc)
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower)/(self.HU_Upper - self.HU_Lower), img)
        img[img<self.HU_Lower] = 0
        img[img>self.HU_Upper] = 1
        img = self.transform(img) , self.label[index]
        return img
    
def visualize_features_tsne(aemodel, loader, device):
    """Generates a t-SNE plot of the autoencoder's latent vectors."""
    aemodel.to(device)
    aemodel.eval()

    all_features = []
    all_labels = []

    print("Extracting features from the autoencoder...")
    with torch.no_grad():
        for img, label in loader:
            img = img.float().to(device)
            _, features, _ = aemodel(img)
            all_features.append(features.squeeze().cpu().numpy())
            all_labels.append(label.numpy())

    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Running t-SNE... (this may take a minute)")
    for p in range(5,50,5):
        tsne = TSNE(n_components=2, perplexity=p, random_state=42)
        features_2d = tsne.fit_transform(all_features)

        print("Creating plot...")
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
        plt.title('t-SNE Visualization of Autoencoder Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=['Benign (0)', 'Malignant (1)'])
        plt.grid(True)
        plt.savefig(f"mal_nonmal_{p}tsne_feature_visualization.png")
        print("t-SNE plot saved as tsne_feature_visualization.png")

def visualize_performance(history_file_path):
    if not os.path.exists(history_file_path):
        print(f"Error: File not found at {history_file_path}")
        return

    history = torch.load(history_file_path, map_location=torch.device('cpu'))
    print("Successfully loaded history file.")

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Model Performance Over Epochs', fontsize=16)

    # --- Plot 1: Training and Validation Loss ---
    ax1.plot(epochs, history['train_loss'], 'g', label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'b', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Performance Metrics ---
    ax2.plot(epochs, history['train_acc'], 'r', label='Training Accuracy')
    ax2.plot(epochs, history['test_precision'], 'c', label='Validation Precision')
    ax2.plot(epochs, history['test_recall'], 'm', label='Validation Recall')
    ax2.set_title('Performance Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = 'performance_plots.png'
    plt.savefig(output_filename)
    print(f"Visualization saved as {output_filename}")

def test(model, aemodel, device, test_loader, lossFn, epoch, is_test):
    model.eval()
    aemodel.eval()
    correct = 0          # number of examples predicted correctly (for accuracy)
    total = 0            # number of examples
    running_loss = 0
    metric = BinaryConfusionMatrix()
    output_list = []
    label_list = []
    pred_list = []
    with torch.no_grad():       
        for batch_idx, data in enumerate(test_loader):
            img, label = data
            img = img.float().to(device)
            label = label.float().to(device)
            recon_batch, _,transformed_vector= aemodel(img)
            output = model(transformed_vector.squeeze().squeeze()).squeeze()
            loss = lossFn(output, label)
            pred = (torch.sigmoid(output) > 0.5).float() # get the index of the max log-probability
            total += label.size(0)    # add in the number of labels in this minibatch
            correct += (pred == label).sum().item()  # add in the number of correct labels
            running_loss += loss.item()*label.size(0)
            metric.update(pred.int(), label.int())
            output_list.extend(torch.sigmoid(output).cpu().numpy())
            label_list.extend(label.cpu().numpy())
            pred_list.extend(pred.cpu().numpy())
            
            if (epoch%20 == 1) or epoch == args.epochs - 1:
                if batch_idx < 10 and img.shape[0] == 128:
                    n = min(img.size(0), 16)
                    comparison = torch.cat([img[:n], recon_batch.view(128, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               './testLIDC_VAE' + str(epoch) + f'_{batch_idx:03}.png', nrow=n)
    
    cm = confusion_matrix(label_list, pred_list)
    np.save('LIDCInferencelabel.npy', label_list)
    np.save('LIDCInferenceOut.npy', output_list)
    print("Confusion Matrix:")
    aucsklean = metrics.roc_auc_score(label_list, output_list)
    torch_confusion_matrix = metric.compute()
    if (epoch % 10 == 1) or epoch == args.epochs-1:
        print(torch_confusion_matrix)
        print("At epoch {}, we got auc {}".format(epoch,aucsklean))
        fig, ax = plot_confusion_matrix(conf_mat=torch_confusion_matrix.detach().numpy(), figsize=(8, 8), cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if is_test:
            plt.savefig(f"./FinalResult_confusion_matrix.png")
        else:
            plt.savefig(f"./{epoch:03}_confusion_matrix.png")
        plt.close(fig)
        
    average_loss = running_loss / len(test_loader)
    tn, fp, fn, tp = torch_confusion_matrix.view(-1)
    if (tp + fp) == 0:
        precision = 0.0 # Or float('nan') if you prefer
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0.0 # Or float('nan')
    else:
        recall = tp / (tp + fn)
    print(f"Precision: {precision}; Recall: {recall}")

    return average_loss, precision, recall, aucsklean
        
if __name__ == '__main__':
    history_file_path = './MLP/202507080957/best.pt' 
    #visualize_performance(history_file_path)

    FeatureExtracter = VAE(args)
    retrainmodel = torch.load("VAE_params_gaussian1.pt", map_location = torch.device('cpu'))
    retrainmodel = retrainmodel['state_dict']
    new_state_dict = OrderedDict()

    for old_key, value in retrainmodel.items():
        # Check if this key belongs to a batch norm layer that needs renaming
        if '.conv.1.' in old_key:
            # Replace the '.conv.1.' part with '.conv.2.'
            new_key = old_key.replace('.conv.1.', '.conv.2.')
            new_state_dict[new_key] = value
        else:
            # If the key doesn't need changing, copy it as is
            new_state_dict[old_key] = value
            
    FeatureExtracter.load_state_dict(new_state_dict)
    mlpmodel = MLP(32,16, [2048,512,256], 0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testlossFn = nn.BCEWithLogitsLoss()
    mlphistory = torch.load(history_file_path, map_location=torch.device('cpu'))
    mlpmodel.load_state_dict(mlphistory)
    mlpmodel.to(device)
    FeatureExtracter.to(device)
    FeatureExtracter.eval()
    mlpmodel.eval()

    csv_file = pd.read_csv('./meta_mal_ben.csv')
    csv_file['is_cancer'].replace('True', 1, inplace=True)
    #csv_file['is_cancer'].replace('Ambiguous', 0, inplace=True)
    csv_file['is_cancer'].replace('False', 0, inplace=True)
    test_datalist = csv_file[csv_file['data_split'] == 'Test']
    test_label = (test_datalist.iloc[:,8]).tolist()
    test_datalist = (test_datalist.iloc[:,5]).tolist()
    test_label = list(map(int, test_label))
    print(len(test_label))


    test_images = LoadImages(main_dir="../Images/", files_list=[test_datalist, test_label], HU_Upper=600, HU_Lower=-1000)
    test_loader = DataLoader(test_images, 128, shuffle=False)
    test(mlpmodel, FeatureExtracter, device, test_loader,testlossFn,1, True)
    