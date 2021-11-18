import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data_cifar_train', train=True,
                                            download=True, transform=transform)
                                            
testset = torchvision.datasets.CIFAR10(root='./data_cifar_test', train=False,
                                           download=True, transform=transform)
