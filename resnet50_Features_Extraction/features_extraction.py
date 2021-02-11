import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import PIL
from PIL import Image
from model.ResnetModels import resnet50
import numpy as np
import fine_tuned_models as ftm
import hyper_parameters as hp
from write import *

# Criando o modelo e carregando os pesos
model = ftm.custom_resnet50(2, True, False)
model.load_state_dict(torch.load('resnet50_covid.pt'))

# Criando o dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder('DataSet', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=hp.BATCH_SIZE, num_workers=0)

# Criando os hooks das layers desejadas
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output
    return hook

model.maxpool.register_forward_hook(get_features('max_pooling2d_1'))
model.layer1[0].relu.register_forward_hook(get_features('activation_4_relu'))
model.layer4[2].bn2.register_forward_hook(get_features('activation_48_relu'))
model.layer4[2].relu.register_forward_hook(get_features('activation_49_relu'))
model.avgpool.register_forward_hook(get_features('avg_pool'))

#criando as listas para cada camada
max_pooling2d_1_list = []
activation_4_relu_list = []
activation_48_relu_list = []
activation_49_relu_list = []
avg_pool_list = []

#criando as listas para as classes
output_class = []

i = 1
# Avaliando a rede
with torch.no_grad():
    # Deixando a rede em modo de avaliação
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)

        if target[0] == 0:
            print("{} - Covid19".format(i))
            output_class.append("Covid19")
        else:
            print("{} - Healthy".format(i))
            output_class.append("Healthy")

        i = i+1

        aux_array = features['max_pooling2d_1'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        max_pooling2d_1_list.append(aux_array)

        aux_array = features['activation_4_relu'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        activation_4_relu_list.append(aux_array)

        features['activation_48_relu'] = model.relu(features['activation_48_relu'])

        aux_array = features['activation_48_relu'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        activation_48_relu_list.append(aux_array)

        aux_array = features['activation_49_relu'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        activation_49_relu_list.append(aux_array)

        aux_array = features['avg_pool'].cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        avg_pool_list.append(aux_array)


writeARFF("max_pooling2d_1", max_pooling2d_1_list, output_class)
writeARFF("activation_4_relu", activation_4_relu_list, output_class)
writeARFF("activation_48_relu", activation_48_relu_list, output_class)
writeARFF("activation_49_relu", activation_49_relu_list, output_class)
writeARFF("avg_pool", avg_pool_list, output_class)