import torch.nn as nn
import torch.nn.functional as F
import Goban
import torch
import numpy as np

def name_to_coord(s):
    assert s != "PASS"
    assert len(s) ==2 
    indexLetters = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}

    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return col, lin

class WithConv(nn.Module):
    def __init__(self):
        super(WithConv, self).__init__()

        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dense1 = nn.Linear(6 * 8 * 8 , 10)
        self.dense2 = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        out = self.softmax(x)
        return out
        
    
    def predict(self, x):
        # Mettre le modèle en mode évaluation
        self.eval()
        with torch.no_grad():
            # Passer les données à travers le modèle
            predictions = self.forward(x)
        return predictions
    
BLACK = 0
WHITE = 1

def goban_to_matrix(board):
    matrix = np.zeros((2,8, 8), dtype=int)

    for color in [BLACK,WHITE]:
        for x in range(8):
            for y in range(8):
                # Conversion des indices de matrice en indices du Goban
                flat_position = board.flatten((x, y))
            
                if board[flat_position] == color:
                    matrix[color,x, y] = 1
                
    return matrix

def flip_data(boardMatrix):
    toret = boardMatrix.copy()
    toret = np.moveaxis(toret, 0, -1)
    toret = np.flipud(toret)
    toret = np.moveaxis(toret, -1, 0)
    return toret

def heuristic(board, color_ami):
    # Recréer l'instance du modèle
    loaded_model = WithConv()

    # Charger le dictionnaire d'état
    loaded_model.load_state_dict(torch.load('./model.pth'))

    boardMatrix = goban_to_matrix(board)

    if color_ami != Goban.Board._BLACK:
        boardMatrix = flip_data(boardMatrix).copy()

    
    input = torch.tensor(boardMatrix, dtype=torch.float32).unsqueeze(0)
    
    prediction = loaded_model.predict(input)

    p = prediction.cpu().detach().numpy()

    return p[0][0]*100