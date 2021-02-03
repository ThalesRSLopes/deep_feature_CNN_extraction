import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import *
from model.alexnet_Model import alexnet

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder('DataSet', transform=transform)

# Realizando o split do dataset
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(valid_idx)

# Criando os loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, sampler=train_sampler,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=4, sampler=test_sampler,
                                          num_workers=0)

#Definindo a rede
AlexNet_model = alexnet(pretrained=True)

#Atualizando os classificadores
AlexNet_model.classifier[4] = nn.Linear(4096,1024)
AlexNet_model.classifier[6] = nn.Linear(1024,2)

#Definindo device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Colocando a rede para operar com cuda
AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)

#Treino
valid_loss_min = np.Inf # Calcula as mudanças no valor da nossa loss function

epochs = 1 #Numero de epochs
for epoch in range(1, epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    #Treinando modelo

    AlexNet_model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = AlexNet_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #Calculando a train loss
        train_loss += loss.item() * data.size(0)

    #Testando o modelo
    AlexNet_model.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        # Se CUDA estiver disponivel, move os tensors para a GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = AlexNet_model(data)
        # Calcula o loss
        loss = criterion(output, target)
        # Atualiza a media do loss
        valid_loss += loss.item() * data.size(0)

        # Valores retornados pela rede (0 = Covid, 1 = Healthy)
        _, predicted = torch.max(output.data, dim=1)

    # Calcula a media do loss
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    # printando as estatísticas
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # Salva o modelo se obtivermos uma loss menor
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(AlexNet_model.state_dict(), 'AlexNetCovid.pt')
        valid_loss_min = valid_loss