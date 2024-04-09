from exceptions import AgentException
from connect4 import Connect4
from copy import deepcopy

class AlphaBetaAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token
        
    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.alphabeta(connect4, 3, True, -float('inf'), float('inf'))[1]
    
    def heuristic(self, connect4):
        fours_count = 0
        score = 0
        for four in connect4.iter_fours():
            fours_count += 1
            my_tokens = four.count(self.my_token)
            opponent_tokens = 4 - my_tokens - four.count('_')
            if opponent_tokens > 0:
                continue
            score += [1,2,4,8][my_tokens]
        return score / fours_count
    
    def alphabeta(self, connect4, depth, maximizing_player, alpha, beta):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1, -1
            elif connect4.wins == None:
                return 0, -1
            else:
                return -1, -1
        if depth == 0:
            return self.heuristic(connect4), -1
        
        best_move = -1
        best_score = -1 if maximizing_player else 1
        for n_column in connect4.possible_drops():
            connect4_copy = deepcopy(connect4)
            connect4_copy.drop_token(n_column)
            score, _ = self.alphabeta(connect4_copy, depth - 1, not maximizing_player, alpha, beta)
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = n_column
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = n_column
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return [best_score, best_move]