import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import PIL
from PIL import Image
import numpy as np
import csv
from write import *
import torchvision.models as models
from model.alexnet_Model import alexnet

#Definindo a rede
model = alexnet(pretrained=True)

#Atualizando os classificadores
model.classifier[6] = nn.Linear(4096,2)

#Carregando os pesos
model.load_state_dict(torch.load('AlexNet_covid.pt'))

#Definindo device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Colocando a rede para operar com cuda
model.to(device)

# Criando o dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder('DataSet', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=0)

# Criando os hooks das layers desejadas
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output
    return hook

model.features[12].register_forward_hook(get_features('features_alexnet'))

#criando as listas para cada camada
max_pooling_f_list = []

#criando as listas para as classes
output_class = []

i = 0
# Avaliando a rede
with torch.no_grad():
    # Deixando a rede em modo de avaliação
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)

        i = i + 1

        if target[0] == 0:
            print("{} - Covid19".format(i))
            output_class.append("Covid19")
        else:
            print("{} - Healthy".format(i))
            output_class.append("Healthy")

        aux_array = features['features_alexnet'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        max_pooling_f_list.append(aux_array)

writeARFF("features_alexnet", max_pooling_f_list, output_class)