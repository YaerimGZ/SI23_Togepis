import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim, kernel_size=5, stride=2, padding=2)

        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2)
        self.dropout1 = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.dropout2 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256 * n_classes * n_classes, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_classes)

        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout3(x)

        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)

        logits = self.fc2(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path, map_location=self.device))