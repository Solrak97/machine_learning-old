import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
import pickle


class Classifier:

    def __init__(self, path):

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        #    Selección de dispositivo
        self.classes = ('Margarita', 'Dientes de león', 'Rosa', 'Girasol', 'Tulipan')            
        
        self.model = self.load_model(path)
        self.model.eval()
        

        # Transformador de datos
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])
        pass


    def load_model(self, path):

        # Load in checkpoint
        state = torch.load(path, map_location=self.device)

        # Non trainable model
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        # Load in the state dict
        model.classifier = state['classifier']
        model.load_state_dict(state['state_dict'])

        return model.to(self.device)


    def get_metrics():
        pass

    # Clasificador
    # Recive una imagen PIL
    # Retorna (etiqueta, probabilidad)

    def classify(self, img):
        img = self.transform(img)
        img = img.to(self.device).unsqueeze(0)

        pred = F.softmax(self.model(img)[0], dim = 0) * 100

        print(self.model(img)[0])
        print(pred)
        print(torch.argmax(pred))


        class_label = self.classes[torch.argmax(pred)]
        prob = pred.detach().numpy()[torch.argmax(pred)]

        return {'class': class_label, 'accuracy': int(prob)}
