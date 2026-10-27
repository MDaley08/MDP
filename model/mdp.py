from abc import ABC
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TypeVar, NamedTuple,Optional

S = TypeVar('S') #state type(e.g, str, int, custom object)
A = TypeVar('A') #action type(e.g, str or int)

@dataclass
class TransitionOutcome(NamedTuple):
    probability: float
    next_state: S
    reward: float
    is_terminal: bool

#reward input types 
#1. state-action pair: R[s][a] -> float
#2. state-only       : R[S] -> float
#3. action-only      : R[a] -> float   
RewardMatrix = Mapping[S, Mapping[A,float]] | Mapping[A,float] | Mapping[S,float] | None
TransitonMatrix = Mapping[S, Mapping[A, Sequence[TransitionOutcome]]]


class Markov_Model(ABC):
    '''
    abstract base class that will be used to define custome MDP environments, expanding as i learn

    '''
    def __init__(self,
                 state_space: Sequence[S],
                 state_transition: TransitonMatrix[S,A],
                 reward_func: Optional[RewardMatrix[S,A]] = None,
                 action_space: Optional[Sequence[A]] = None,
                 gamma: float = 0.9
                 ):
        
        self.state_space = state_space
        self.action_space = action_space
        self.P = state_transition
        self.R = reward_func
        self.gamma = gamma

    def get_transition_outcomes(self, state:S, action:A) -> Sequence[TransitionOutcome]:
        """
        retrieves state_transion P for a target state and action.

        Parameters:
        State: S(str, int,custom obj).
        Actions: A(str,int,).

        Retruns:
        sequence: probability, next_state, Reward, terminal.
        """
        return self.P[state][action]

    
    def get_reward(self, state:S =None, action:A=None) -> float | None:
        """
        retrieves retrieves reward for an action, state or state-action pair

        Parameters:
        State: S(str, int,custom obj).
        Actions: A(str,int,).

        Retruns:
        float: reward mapped to state, action or state-action pair
        """
        #check if state,action present
        s_present = state is not None
        a_present = action is not None
        if s_present and a_present: #both state and action presented
            return self.R[state][action]
        elif s_present and not a_present: #state present, but no action
            return self.R[state] 
        elif a_present and (not s_present): #action presen, but no state
            return self.R[action]
        
        return None