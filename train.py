import tqdm
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets, models
from Dataloader.WSASL_Load import MyCustomDataset
from Model.CNN_Frame_classification import Net
from torch.utils.data import random_split
import csv
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
dataset = MyCustomDataset(category='labels_100',json_file_path="/home/marius/Documents/Projects/WLASL_v0.3.json", frame_location="/home/marius/Documents/Projects/Processed_data")

#dataset = MyCustomDataset("labels_2000")
dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.2))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
dataloader_train = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
dataloader_val = DataLoader(validset, batch_size=10, shuffle=True, num_workers=1)


net = Net()
#num_ftrs = net.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#net.fc = nn.Linear(num_ftrs, 2000)

net = net.to(device)
print(net)
#net = Net()
#net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = criterion.to(device)
scheduler = StepLR(optimizer, step_size = 10, gamma=0.1)
def accuracy(ys, ts):
    y = torch.argmax(ys, dim = 1)
    x = ts
    correct = 0
    for i in range(len(y)):
        if y[i] == x[i]:
            correct += 1
    return correct/len(y)
now = datetime.now()
filename = "{}".format(now.strftime("%H:%M:%S") + ".csv")
title = ['{}'.format(net)]
headers = ['ID', 'Type','Epoch','Loss','Accuracy']
with open(filename,'w') as csvfile:
    Identification = 1
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)
    for epoch in range(30):  # loop over the dataset multiple times

        net.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            #print(inputs.shape)
            #ins = inputs[0]
            inputs = inputs.view(-1,3,224,224)
            #print(ins.shape)
            #ins = inputs.view(-1,224,224,3)
            #img = ins
            #img = img[0]
            #imgplot = plt.imshow(img)
            #plt.show()
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = torch.LongTensor(labels)
            labels = torch.LongTensor(labels).to(device)


            optimizer.zero_grad()

            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            correct = (predicted == labels).sum().item()
            #print(labels.size(0))
            running_acc += (correct/(labels.size(0)))
            loss = criterion(outputs, (labels))
            loss.backward()
            #print(correct)
            training_loss += loss.item()
            #print(loss.item())
            optimizer.step()
            if i%30 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Training"),'{}'.format(epoch),'{}'.format(training_loss/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1

        net.eval()
        valError = 0
        running_acc = 0
        for i, (inputs,labels) in enumerate(dataloader_val):
            with torch.no_grad():
                inputs = inputs.view(-1,3,224,224)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = torch.LongTensor(labels)
                labels = labels.to(device)
                outputs = net(inputs)
                _,predicted = torch.max(outputs.data,1)
                correct = (predicted == labels).sum().item()
                running_acc += (correct/(labels.size(0)))
                loss = criterion(outputs, labels)
                #print("preds: ",torch.argmax(outputs, dim=1))
                #print("labs:", labels)
                valError += loss.item()
            if i%10 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Validation"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
        scheduler.step()
    csvwriter.writerow(title)
print('Finished Training')
