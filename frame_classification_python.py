#!/usr/bin/env python
# coding: utf-8

# In[26]:


import tqdm
from preprocessing import exctract_json_data, define_categories
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
data_directory = "/Users/mjo/Desktop/WLASL/Processed_data/100"


count_dictionary, video_id_dictionary = exctract_json_data()
labels_100, _, _, _, _ = define_categories(count_dictionary)

labels_iterated = {}
counter = 0
for label in labels_100:
    labels_iterated[label] = counter
    counter += 1

inv_video_id_dictionary = {}
for k, v in video_id_dictionary.items(): 
    for video in v:
        inv_video_id_dictionary[video] = k
        
def make_training_data(labels_x, video_id_dictionary, labels_iterated):
    training_data = []
    num_labels = len(labels_x)
    for label in (labels_x):
        for video in video_id_dictionary[label]:
            path = os.path.join(data_directory, video)
            for file in (os.listdir(path)):
                if "jpg" in file:
                    try:
                        path = os.path.join(data_directory,video, file)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        training_data.append([np.array(img),labels_iterated[label]])
                    except Exception as e:
                        print(e)
                        pass
    return training_data

training_data = make_training_data(labels_100, video_id_dictionary, labels_iterated)


# In[27]:


np.random.shuffle(training_data)


# In[28]:


#import torch
#trainloader = torch.utils.data.DataLoader(training_data, batch_size=20,
#                                          shuffle=True, num_workers=2)
X = [None] * len(training_data)
y = [None] * len(X)
counter = 0
for i in training_data:
    X[counter] = torch.Tensor(i[0])
    y[counter] = i[1]
    counter += 1


# In[30]:


training_inputs = torch.stack(X)
training_labels = (np.array(y))


# In[31]:


training_labels = torch.LongTensor(y)


# In[33]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[34]:



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 5)
        self.conv3 = nn.Conv2d(256, 512, 5)
        self.conv4 = nn.Conv2d(512, 256, 5)
        self.fc3 = nn.Linear(16384, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 16384)
        x = self.fc3(x)
        return x


net = Net()
net = net.to(device)


# In[35]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = criterion.to(device)


# In[36]:


def accuracy(ys, ts):
    print("ts:", ts.shape)
    print("ys:", ys.shape)
    y = torch.argmax(ys, dim = 1)
    x = torch.argmax(ts, dim = 1)
    correct = 0
    for i in range(len(y)):
        if y[i] == x[i]:
            correct += 1
    return correct/len(y)


# In[37]:


BATCH_SIZE = 100
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0,len(training_inputs), BATCH_SIZE):
        # get the inputs; data is a list of [inputs, labels]
        inputs = training_inputs[i:i+BATCH_SIZE].view(-1, 1, 256, 256)
        labels = training_labels[i:i+BATCH_SIZE]
        #inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.view(-1,1,256,256)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        
        outputs = net(inputs)
        value, index = (torch.max(outputs,0))
        value, index = (torch.max(labels,0))
        #print(accuracy(outputs,labels))
        preds = torch.max(outputs, 1)[1]
        loss = criterion(outputs, torch.LongTensor(labels))
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}. Loss: {loss}")

print('Finished Training')


# In[ ]:




