'''
UE20CS302 (D Section)
Machine Intelligence
Week 8 - Hidden Markov Models

Mitul Joby
PES2UG20CS199
'''

import numpy as np

class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()


    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))


    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Probability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        T = len(seq)
        nu = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        nu[0] = self.pi * self.B[:, self.emissions_dict[seq[0]]]
        for t in range(1, T):
            for j in range(self.N):
                nu[t, j] = np.max(nu[t-1] * self.A[:, j]) * self.B[j, self.emissions_dict[seq[t]]]
                psi[t, j] = np.argmax(nu[t-1] * self.A[:, j])

        hidden_states_sequence = [None] * T
        hidden_states_sequence[T-1] = np.argmax(nu[T-1])
        for t in range(T-2, -1, -1):
            hidden_states_sequence[t] = psi[t+1, hidden_states_sequence[t+1]]

        # return nu, hidden_states_sequence

        return [self.states[i] for i in hidden_states_sequence]
