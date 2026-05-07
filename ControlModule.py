# Import required dependencies
import numpy as np
import json
import mdptoolbox
import mdptoolbox.example

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(reactor_file) -> np.array:
        """ Function that generates the probabilities (transition) matrix """

        N = 100 #Number of states

        with open(reactor_file, "r") as file:
            reactor_data = json.load(file)
        probs = reactor_data["probabilities"]
        for a in [probs["decrease"], probs["maintain"], probs["increase"]]:
            if len(a) != 3:
                raise ValueError("Invalid probabilities")
            total = 0
            for p in a:
                total += p
            if not np.isclose(1, total):
                raise ValueError("Invalid probabilities")
        d2, d1, d0 = probs["decrease"]
        m_1, m0, m1 =  probs["maintain"]
        i0, i1, i2 = probs["increase"]

        Pd = np.zeros((N, N))
        Pm = np.zeros((N, N))
        Pi = np.zeros((N, N))
        #rows = current state
        #columns = next state

        for s in range(N):
            #In Pd we modify (s, s-2), (s, s-1) and (s, s)
            if s == 0:  #Bound the probabilities so that we never go below zero 
                Pd[0][0] = 1
            elif s == 1: 
                Pd[1][0] = d2 + d1
                Pd[1][1] = d0
            else:
                Pd[s][s-2] = d2
                Pd[s][s-1] = d1
                Pd[s][s] = d0

            #In Pm we modify (s, s-1), (s, s) and (s, s+1)
            if s == 0:
                Pm[0][0] = m_1 + m0
                Pm[0][1] = m1
            elif s == 99:
                Pm[99][98] = m_1
                Pm[99][99] = m0 + m1
            else:
                Pm[s][s-1] = m_1
                Pm[s][s] = m0
                Pm[s][s+1] = m1

            #In Pi we modify (s, s), (s, s+1), (s, s+2)
            if s == 98:
                Pi[98][98] = i0
                Pi[98][99] = i1 + i2
            elif s == 99:
                Pi[99][99] = 1
            else:
                Pi[s][s] = i0
                Pi[s][s+1] = i1
                Pi[s][s+2] = i2
        return np.array([Pd, Pm, Pi]) #Array of matrices


    @staticmethod
    def generate_R() -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def control_iteration() -> np.int32:
        """ Function that computes one control-iteration """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def control_loop(demand: np.ndarray, 
                    probs: np.ndarray,
                    n_states: np.int32, 
                    n_actions: np.int32,
                    gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        ### TO BE COMPLETED BY THE STUDENTS ###

        ### DUMMY BEHAVIOUR TO PREVENT CRASHING (MUST BE DELETED AFTER THE FULL IMPLEMENTATION) ###
        return np.zeros_like(a=demand, dtype=np.float64)
        ### ###

