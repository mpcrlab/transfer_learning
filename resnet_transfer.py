import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import pickle
import datetime
import argparse
parser = argparse.ArgumentParser(description='Do transfer learning!')
parser.add_argument('-f', '--data_pickle', default='data.pkl', help='Data Pickle file to process. Must be list of [X, Y], where X and Y are lists of data as numpy arrays.')
parser.add_argument('-d', '--device', default='cpu', help='Device to process on: cuda:0, cuda:1, cpu.')
parser.add_argument('-m', '--model_version', default='18', help='Resnet version to use: 18, 34, 50, 101, 152.')
parser.add_argument('-e', '--epochs', default='500', help='Number of epochs to train.')
args = parser.parse_args()
def random_crop(np_array, size=224):
    starting_point = (np.random.randint(np_array.shape[0]-size), np.random.randint(np_array.shape[1]-size))
    patch = np_array[starting_point[0]:starting_point[0]+size, starting_point[1]:starting_point[1]+size, ...]
    return patch
def img_preprocess(np_array):
    np_array = np.array(list(map(random_crop,np_array)))
    np_array = np.moveaxis(np_array, -1, 1)
    return np_array
with open(args.data, 'rb') as f:
    content = pickle.load(f)
X, Y = content
content_zip = np.array(list(zip(X, Y)))
np.random.shuffle(content_zip)
X, Y = zip(*content_zip)
X, Y = np.array(list(map(img_preprocess, np.split(np.array(X), len(X))))), np.array(np.split(np.array(content[1]), len(Y)))
device = torch.device(args.device)
def train_model(model, Xx, Yy, criterion, optimizer, scheduler, num_epochs=25, run_id='train001'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999.0
    logfile = open(run_id + '.log', 'w')
    logfile.close()
    for epoch in range(num_epochs):
        logfile = open(run_id + '.log', 'a')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=logfile)
        print('-' * 10, file=logfile)
        logfile.close()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in zip(Xx, Yy):
                inputs = torch.FloatTensor(inputs).to(device)
                float_labels = torch.FloatTensor(labels).to(device)
                long_labels = torch.LongTensor(labels).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model.stride = 1
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, float_labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == long_labels.data)
            epoch_loss = running_loss / Xx.shape[0]
            epoch_acc = running_corrects.double() / Xx.shape[0]
            logfile = open(run_id + '.log', 'a')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc), file=logfile)
            logfile.close()
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        logfile = open(run_id + '.log', 'a')
        print('',file=logfile)
        logfile.close()
    logfile = open(run_id + '.log', 'a')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=logfile)
    print('Best val Loss: {:4f}'.format(best_loss), file=logfile)
    logfile.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model_ft = exec("models.resnet{}(pretrained=True)".format(args.model_version))
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, Y.shape[2])
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.8)
r_id = str(datetime.datetime.now())[:-7]
print(r_id)
model_ft = train_model(model_ft, X, Y, criterion, optimizer_ft, exp_lr_scheduler,
                          num_epochs=args.epochs, run_id=r_id)
torch.save(model_ft, r_id+'_resnet{}.ml'.format(args.model_version))
print("Training Complete!")