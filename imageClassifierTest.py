from imageClassifierTrain import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_threads', action='store', dest='num_threads', type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(args.num_threads)
print("Number of threads " + str(torch.get_num_threads()))

PATH = './cifar_net.pth'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
net.load_state_dict(torch.load(PATH))
net.eval() # set dropout and batch normalization layers to evaluation mode before testing

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=args.num_threads)

# test on testset
correct = 0
total = 0
with torch.no_grad():
    start = time.time()
    for data in testloader:

        images, labels = data
        
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

runtime = time.time() - start
with open('results.txt', 'a') as f:
    point = str(args.num_threads) + ',' + str(runtime) + '\n'
    f.write(point)
#print('Accuracy of the network on the 10000 test images: %d %%' % (
#    100 * correct / total))