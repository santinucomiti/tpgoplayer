import Goban
import math
import heuristic



def alphabetaLeaf(board, IA, joueur):
    return heuristic.heuristic(board,IA, joueur)

def alphabetaEnemy(board, IA, max_value, min_value,depth):
    if board.is_game_over() or depth == 0: 
        return alphabetaLeaf(board, IA,-1)
    else : 
        moves = [m for m in board.generate_legal_moves()]
        grades = []
        for move in moves:
            board.push(move)
            grade = alphabetaMe(board,IA, max_value, min_value,depth-1)
            grades.append(grade)
            board.pop()

            if grade < max_value:
                max_value = grade
            if max_value <= min_value : 
                #print(grades)
                return max_value
        #print("ENEMY", max_value, min_value)
        #print(grades)
        return min(grades)

def alphabetaMe(board, IA, max_value, min_value,depth):
    if board.is_game_over() or depth == 0: 
        return alphabetaLeaf(board, IA,1)
    else:
        moves = [m for m in board.generate_legal_moves()]
        grades = []
        for move in moves:
            board.push(move)
            grade = alphabetaEnemy(board,IA, max_value, min_value,depth-1)
            grades.append(grade)
            board.pop()
            if grade > min_value : 
                min_value = grade
            if min_value >= max_value :
                #print(grades)
                return min_value
        #print("MOI", max_value, min_value)
        #print(grades)
        return max(grades)
    

def alphabeta(board, IA, depth):
    if board.is_game_over() or depth == 0: 
        return alphabetaLeaf(board, IA)
    else:
        moves = [m for m in board.generate_legal_moves()]
        max_value = - math.inf
        grades = []
        for move in moves:
            board.push(move)
            grade = alphabetaEnemy(board,IA, math.inf, - math.inf, depth-1)
            grades.append(grade)
            if grade > max_value:
                max_value = grade
                bestmove = move
            board.pop()
        print(max_value, grades)

        return bestmove

