import torch, os
import torch.nn as nn 
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self,input_size, num_classes):
        super().__init__() 
        self.fc1 = nn.Linear(input_size,num_classes)

    def model(self,x):
        x = x.reshape(x.size(0), 28*28)
        x = self.fc1(x)
        return(x)

class ModelActions():
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr = 0.0001)
        
    def SGD(self,loader,total, num_correct):
        for i, (images, labels) in enumerate(loader):
            outputs = self.net.model(images)
            loss = self.criterion(outputs, labels) 
            self.optimizer.zero_grad()
            loss.backward()  
            self.optimizer.step()
            predicted = torch.max(outputs, 1)[1] 
            total += labels.size(0)
            num_correct += (predicted == labels).sum()
        return(float(num_correct)/total, total, num_correct)
    
    def predict(self, image):
        outputs = self.net.model(image)
        predicted = torch.max(outputs, 1)[1]
        return(predicted)

    def save_model(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.net.state_dict(), file_name)

    def load_model(self, file_name = 'model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.net.load_state_dict(torch.load(file_name))
        self.net.eval()