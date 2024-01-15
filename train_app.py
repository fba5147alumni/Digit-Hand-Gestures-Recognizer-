import torch, torchvision
from model import NeuralNetwork, ModelActions

input_size = 784
num_classes = 10 
num_epochs = 10

num_correctTest = 0
num_correctTrain = 0
totalTest = 0
totalTrain = 0

net = NeuralNetwork(input_size,  num_classes)
net_actions = ModelActions(net)

net_actions.load_model()

MNIST_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
splits=['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']

#Stochastic Gradient Descendent
#Training Data
trainsetSGD = torchvision.datasets.MNIST(root='./data', train= True, download=True, transform = MNIST_transform)
trainloaderSGD = torch.utils.data.DataLoader(trainsetSGD, batch_size = 1, shuffle=True)

#Testing Data
testsetSGD = torchvision.datasets.MNIST(root='./data', train= False, download=True, transform = MNIST_transform)
testloaderSGD = torch.utils.data.DataLoader(testsetSGD, batch_size = 1, shuffle=False)

#Stochastic Gradient Descent
for epoch in range(num_epochs):
    trainingAccuray, totalTrain, num_correctTrain = net_actions.SGD(trainloaderSGD, totalTrain, num_correctTrain)
    testAccuracy, totalTest, num_correctTest = net_actions.SGD(testloaderSGD, totalTest, num_correctTest)
    print('Stochastic gradient descent, Epoch {} , Training accuracy: {}, Test accuracy: {}' .format(epoch+1, trainingAccuray ,testAccuracy))
#net_actions.save_model()
