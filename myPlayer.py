# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import Goban 
from random import choice
from playerInterface import *
import torch
import heuristic
import alphabetamethod

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''
 

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def __heuristic(self,color_ami,board):
        # Recréer l'instance du modèle
        loaded_model = heuristic.WithConv()

        # Charger le dictionnaire d'état
        loaded_model.load_state_dict(torch.load('./model.pth'))

        boardMatrix = heuristic.goban_to_matrix(board)

        if color_ami != Goban.Board._BLACK:
            boardMatrix = heuristic.flip_data(boardMatrix).copy()

        
        input = torch.tensor(boardMatrix, dtype=torch.float32).unsqueeze(0)
        
        prediction = loaded_model.predict(input)

        p = prediction.cpu().detach().numpy()

        return p[0][0]*100
# 
    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS" 
        
        moves = self._board.legal_moves() # Dont use weak_legal_moves() here!
        move = choice(moves) 
        self._board.push(move)


        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move) 

    def playOpponentMove(self, move):
        print("Opponent played ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")



