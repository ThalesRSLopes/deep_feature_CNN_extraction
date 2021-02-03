import torch
import torch.nn as nn
from model.ResnetModels import resnet50


def __freeze_trained_layers(model):
    for params in model.parameters():
        params.requires_grad = False
    return model


def custom_resnet50(n_classes: int, try_to_cuda: bool = False, train_just_fc: bool = False):
    model = resnet50(pretrained=True)

    if train_just_fc:
        model = __freeze_trained_layers(model)

    model.fc = nn.Linear(2048, n_classes)

    if try_to_cuda:
        if torch.cuda.is_available():
            print("CUDA disponivel. Modelo otimizado para uso de GPU")
            model = model.cuda()
        else:
            print("CUDA indisponível. Modelo otimizado para uso de CPU")
    else:
        print("Modelo otimizado para uso de GPU")

    return model
