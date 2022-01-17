import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train = datasets.MNIST("", train=True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test =  datasets.MNIST("", train=False, download = True, transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #make the layers
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        #pass whatever comes in thru the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()


optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 3  #go thru entire dataset 3 times

for epoch in range(EPOCHS):
    for data in trainset:
        X,y = data
        net.zero_grad()
        #get the output
        output = net(X.view(-1,28*28))
        #calculate loss
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)



