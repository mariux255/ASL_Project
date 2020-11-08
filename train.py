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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


dataloader = DataLoader(VideoDataset(('lables_100'), batch_size=20, shuffle=True, num_workers=2))



# net = Net()
# net = net.to(device)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# criterion = criterion.to(device)

# def accuracy(ys, ts):
#     print("ts:", ts.shape)
#     print("ys:", ys.shape)
#     y = torch.argmax(ys, dim = 1)
#     x = torch.argmax(ts, dim = 1)
#     correct = 0
#     for i in range(len(y)):
#         if y[i] == x[i]:
#             correct += 1
#     return correct/len(y)



# BATCH_SIZE = 100
# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i in range(0,len(training_inputs), BATCH_SIZE):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs = training_inputs[i:i+BATCH_SIZE].view(-1, 1, 256, 256)
#         labels = training_labels[i:i+BATCH_SIZE]
#         #inputs, labels = batch
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         inputs = inputs.view(-1,1,256,256)
#         # zero the parameter gradients
#         optimizer.zero_grad()
        
#         # forward + backward + optimize
        
#         outputs = net(inputs)
#         value, index = (torch.max(outputs,0))
#         value, index = (torch.max(labels,0))
#         #print(accuracy(outputs,labels))
#         preds = torch.max(outputs, 1)[1]
#         loss = criterion(outputs, torch.LongTensor(labels))
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch: {epoch}. Loss: {loss}")

# print('Finished Training')
