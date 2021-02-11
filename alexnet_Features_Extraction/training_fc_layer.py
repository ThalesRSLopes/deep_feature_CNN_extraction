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
from write import *

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, sampler=train_sampler,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=1, sampler=test_sampler,
                                          num_workers=0)

#Definindo a rede
AlexNet_model = alexnet(pretrained=True)

#Congelando as camadas de features
for params in AlexNet_model.parameters():
    params.requires_grad = False

#Atualizando os classificadores para trabalhar com 2 classes
AlexNet_model.classifier[6] = nn.Linear(4096,2)

#Definindo device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Colocando a rede para operar com cuda
AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)

# Treino
valid_loss_min = np.Inf # Calcula as mudanças no valor da nossa loss function

epochs = 50
for epoch in range(1, epochs + 1):
    print('Epoch {}. \n\tTreinando o modelo...'.format(epoch))
    train_loss = 0.0
    valid_loss = 0.0

    # Treinando modelo
    AlexNet_model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item() * data.size(0)

    # Testando o modelo
    AlexNet_model.eval()
    # limpando as listas de tensors
    print('\tTestando o modelo...')
    for batch_idx, (data, target) in enumerate(test_loader):
        # Se CUDA estiver disponivel, move os tensors para a GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # Foward pass
        output = AlexNet_model(data)
        # Calcula o loss
        loss = criterion(output, target)
        # Atualiza a media do loss
        valid_loss += loss.item() * data.size(0)

    # Calcula a media do loss
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # Salva o modelo se obtivermos uma loss menor
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(AlexNet_model.state_dict(), 'AlexNet_covid.pt')
        valid_loss_min = valid_loss