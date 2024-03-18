import pandas as pd
import os

def get_small():
    knapsack = pd.read_csv(f'{os.path.dirname(__file__)}/knapsack-small.csv')
    return knapsack, 10

def get_big():
    knapsack = pd.read_csv(f'{os.path.dirname(__file__)}/knapsack-big.csv')
    return knapsack, 6404180
