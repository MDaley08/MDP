from mdp import Markov_Model
from collections.abc import Sequence
import numpy as np
from typing import Optional


class gridworld(Markov_Model):
    ACTIONS = ('L', 'R', 'U', 'D')
    MOVES = {action: i for i, action in enumerate(ACTIONS)}
    #TODO complete definigion of gridworld and implemetn init
    def __init__(self,
                 size:Sequence[row:int,col:int],
                 terminal_states: Optional[Sequence[states]] = None, #if left blank right and bottom most tile becomes goal and terminal state
                 slip: float = 0, #probability of sliding to another tile 

                 )