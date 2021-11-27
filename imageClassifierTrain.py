import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Defining a simple convoluational neural network
class Net(nn.Module):
    def __init__(self):
        # super constructor is here to inherit attributes from nn.Module
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    

# Viualize dataset

def imgshow(img):
    # unormalize
    img = img / 2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def main():
    print("Number of threads " + str(torch.get_num_threads()))

    # The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
    # This will first transform from a PIL image to a tensor
    # Then this will normalize each chanel to have a mean of 0.5 and standard deviation of 0.5
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainset = torchvision.datasets.CIFAR10(root='./data/data_cifar_train', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)
    dataiter = iter(trainloader)
    images,labels = dataiter.next()

    imgshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss() #the loss function
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    # Train the nn
    for epoch in range(50):
        # loop over the dataset multiple times (twice)
        running_loss = 0
        for num, data in enumerate(trainloader, 0):

            inputs,labels = data

            optimizer.zero_grad() #zero the parameter gradients, otherwise they will add up from previous passes
            
            outputs = net.forward(inputs) # forward pass of current image
            
            loss = criterion(outputs, labels)

            loss.backward() # calculate partial derivatives 

            optimizer.step() # update weights of the network

            running_loss += loss.item() # to get a more meaningful idea of how our loss is improving just check the sum every n steps

            if num % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, num + 1, running_loss / 2000))
                running_loss = 0

    #save the model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    main()
