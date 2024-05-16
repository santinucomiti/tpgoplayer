import Goban
import math
import heuristic

from random import *

def minmaxLeaf(board, IA,joueur):
    return heuristic.heuristic(board,IA,joueur)


def minmaxEnemy(board, IA,depth):
    if board.is_game_over() or depth == 0: 
        return minmaxLeaf(board, IA,-1)
    else : 
        moves = board.generate_legal_moves()
        grade = []
        for move in moves:
            board.push(move)
            grade.append(minmaxMe(board,IA,depth-1))
            board.pop()
        return min(grade)

def minmaxMe(board, IA,depth):
    if board.is_game_over() or depth == 0: 
        return minmaxLeaf(board, IA,1)
    else:
        moves = board.generate_legal_moves()
        grade = []
        for move in moves:
            board.push(move)
            grade.append(minmaxEnemy(board,IA,depth-1))
            board.pop()
        return max(grade)
     

def minmax(board, IA,depth):
    if board.is_game_over() or depth == 0: 
        return minmaxLeaf(board, IA)
    else:
        moves = [m for m in board.generate_legal_moves()]
        max = - math.inf
        grades = []
        i=0
        for move in moves:
            board.push(move)
            grade = minmaxEnemy(board,IA, depth-1)
            grades.append(grade)
            if grade > max:
                max = grade
                bestmove = move
                k=i
            elif grade == max:
                if randint(0,1)==0:
                    bestmove = move
                    k=i
            board.pop()
            i+=1
        print(len(moves),k, max, grades)

        return bestmove

