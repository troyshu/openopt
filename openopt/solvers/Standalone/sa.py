import random
import math
#import logging
import numpy as np

def P(prev_score,next_score,temperature):
    if next_score > prev_score:
        return 1.0
    else:
        return math.exp( -abs(next_score-prev_score)/temperature )

class ObjectiveFunction:
    '''class to wrap an objective function and 
    keep track of the best solution evaluated'''
    def __init__(self,objective_function):
        self.objective_function=objective_function
        self.best=None
        self.best_score=None
    
    def __call__(self,solution):
        score=self.objective_function(solution)
        if self.best is None or score > self.best_score:
            self.best_score=score
            self.best=solution
#            logging.info('new best score: %f',self.best_score)
        return score

def kirkpatrick_cooling(start_temp,alpha):
    T=start_temp
    while True:
        yield T
        T=alpha*T

def anneal(init_function,move_operator,objective_function,max_evaluations,start_temp,alpha, prob = None):
    
    # wrap the objective function (so we record the best)
    objective_function=ObjectiveFunction(objective_function)
    
    current=init_function()
    current_score=objective_function(current)
    num_evaluations=1
    
    cooling_schedule=kirkpatrick_cooling(start_temp,alpha)
    
#    logging.info('anneal started: score=%f',current_score)
    
    for temperature in cooling_schedule:
        done = False
        # examine moves around our current position
        for next in move_operator(current):
            if num_evaluations >= max_evaluations:
                done=True
                break
            
            next_score=objective_function(next)
            num_evaluations+=1
            
            # probablistically accept this solution
            # always accepting better solutions
            p=P(current_score,next_score,temperature)
            
            if not num_evaluations % 64 and prob is not None:
                prob.iterfcn(np.array(current), -current_score)
                if prob.istop != 0:
                    return (num_evaluations, objective_function.best_score, objective_function.best)            

            if random.random() < p:
                current=next
                current_score=next_score
                break
            
           
        # see if completely finished
        if done: break
    
    best_score=objective_function.best_score
    best=objective_function.best
#    logging.info('final temperature: %f',temperature)
#    logging.info('anneal finished: num_evaluations=%d, best_score=%f',num_evaluations,best_score)
    return (num_evaluations,best_score,best)
