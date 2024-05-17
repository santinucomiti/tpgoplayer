# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import Goban 
from random import *
from playerInterface import *
from opening import *

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''
    
    _FUSEKI_DEPTH = 10
    _BEST_OPENING_MOVE = [18, 19, 20, 21, 26, 34, 42, 43, 44, 45, 29, 37]
    _PLACES = {
        10: 
    } 

    ##########################################################
    ##########################################################

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._depth = 0

    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS" 
        
        # Opening move if depth lower than _FUSEKI_DEPTH
        if (self._depth < myPlayer._FUSEKI_DEPTH):
            move = self._getOpeningMove()
        else:
            moves = self._board.legal_moves() # Dont use weak_legal_moves() here!
            move = choice(moves) 

        self._board.push(move)
        self._depth += 1

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

    def _getOpeningMove(self):

        moves = self._board.legal_moves()
    
        best_moves = []
        for best_move in self._BEST_OPENING_MOVE:
            if best_move in moves:
                best_moves.append(best_move) 

        return choice(best_moves) if best_moves else choice(moves)



