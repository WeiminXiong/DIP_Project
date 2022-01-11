from numpy.lib.function_base import delete
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import flatten
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from torch import optim
import cv2
import numpy as np
from tqdm import tqdm



class dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, tag = self.data[index]
        if tag>='A' and tag<='Z':
            tag = ord(tag)-ord('A')
        elif tag>='0' and tag<='9':
            tag = int(tag)-int('0')
        return image, tag

def load_data(dir_path):
    '''
    载入训练数据
    '''
    datalist = []
    for i in list_dir:
        directory_path = os.path.join(dir_path, i)
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            image = cv2.imread(file_path)
            datalist.append((image, i))
    return datalist


# a = iter(train_dataloader)
# b = a.next()
# print(len(b))
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.reshape(N, -1)  # "flatten" the C * H * W values into a single vector per image

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_class):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc = nn.Linear(channel_2*64*64, num_class)

    def forward(self, x):
        scores = self.fc(flatten(F.relu(self.conv2(F.relu(self.conv1(x))))))
        return scores

def check_accuracy_part(loader, model):   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.float()
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            x= x.permute(0, 3, 1, 2)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

def train_part(model, optimizer, train_dataloader, valid_dataloader, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    max_score = 0
    for e in tqdm(range(epochs)):
        for t, (x, y) in enumerate(train_dataloader):
            model.train()  # put model to training mode
            # print(y)
            x = x.float()
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            x= x.permute(0, 3, 1, 2)
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            
        print('Iteration %d, loss = %.4f' % (t, loss.item()))
        acc=check_accuracy_part(valid_dataloader, model)
        if acc>max_score:
            delete_path = '../model_' + str(max_score) +'.pth'
            if os.path.exists(delete_path):
                os.remove(delete_path)
            max_score =acc
            save_path = '../model_'+str(max_score)+'.pth'
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    trainset_path = '../dataset/number/train'
    validset_path = '../dataset/number/valid'
    list_dir = os.listdir(trainset_path)
    learning_rate = 1e-5
    channel_1 = 32
    channel_2 = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = load_data(trainset_path)
    valid_data = load_data(validset_path)
    train_dataset = dataset(train_data)
    valid_dataset = dataset(valid_data)
    train_dataloader = DataLoader(train_dataset, 32, True)
    valid_dataloader = DataLoader(valid_dataset, 32, True)

    model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    train_part(model, optimizer, train_dataloader, valid_dataloader,epochs=20)