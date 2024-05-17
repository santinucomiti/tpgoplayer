# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import Goban 
from random import *
from playerInterface import *
import alphabetamethod
import torch
import heuristic
# import opening_move

# Utils
PLACES_HEAT_INVERSE = {
            4: [19, 20, 26, 27, 28, 29, 34, 35, 36, 37, 43, 44],
            3: [11, 12, 18, 21, 25, 30, 33, 38, 42, 45, 51, 52],
            2.5: [1, 6, 8, 15, 48, 55, 57, 62],
            2: [3, 4, 10, 13, 17, 22, 24, 31, 32, 39, 41, 46, 50, 53, 59, 60],
            1: [2, 5, 9, 14, 16, 23, 40, 47, 49, 54, 58, 61],
            0: [0, 7, 56, 63]
} 

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''
    
    _FUSEKI_DEPTH = 10
    _BEST_OPENING_MOVE = [18, 19, 20, 21, 26, 34, 42, 43, 44, 45, 29, 37]
    _PLACES_HEAT = {index: heat_level for heat_level, indices in PLACES_HEAT_INVERSE.items() for index in indices}

    ##########################################################
    ##########################################################

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._turn = 0
#
#    def __heuristic(self,color_ami,board):
#        # Recréer l'instance du modèle
#        loaded_model = heuristic.WithConv()
#
#        # Charger le dictionnaire d'état
#        loaded_model.load_state_dict(torch.load('./model.pth'))
#
#        boardMatrix = heuristic.goban_to_matrix(board)
#
#        if color_ami != Goban.Board._BLACK:
#            boardMatrix = heuristic.flip_data(boardMatrix).copy()
#
#        
#        input = torch.tensor(boardMatrix, dtype=torch.float32).unsqueeze(0)
#        
#        prediction = loaded_model.predict(input)
#
#        p = prediction.cpu().detach().numpy()
#
#        return p[0][0]*100
# 
    def getPlayerName(self):
        return "Bruno Superette (alpha beta player)"

    def getPlayerMove(self):
        self._turn +=1
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS" 
        
        # Opening move if depth lower than _FUSEKI_DEPTH
        if (self._turn < myPlayer._FUSEKI_DEPTH):
            move = self._getOpeningMove()
        else:
            moves = self._board.legal_moves() # Dont use weak_legal_moves() here!
            move = choice(moves) 

        self._board.push(move)

        myPlayer.heatMap(self._board, self._mycolor)

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

    ''' Internal functions only'''

    ###################### HEATMAP ################### 

    @staticmethod
    def heatMap(board, color):
        heat_black = 0
        heat_white = 0

        for i in range(len(board._board)):
            if (board._board[i] == board._BLACK):
                heat_black += myPlayer._PLACES_HEAT[i] 
            if (board._board[i] == board._WHITE):
                heat_white += myPlayer._PLACES_HEAT[i] 
        
        heat_tot = heat_black - heat_white if color == board._BLACK else heat_white - heat_black

        return heat_tot
        
        
    ###################### OPENING ################### 

    def _getOpeningMove(self):

        moves = self._board.legal_moves()
    
        best_moves = []
        for best_move in self._BEST_OPENING_MOVE:
            if best_move in moves:
                best_moves.append(best_move) 

        return choice(best_moves) if best_moves else choice(moves)


