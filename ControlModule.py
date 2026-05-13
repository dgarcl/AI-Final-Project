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
    def generate_R(demand_point: np.float64, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        N=n_states #Number of states

        #compute base distance for all possible destination states
        distances = np.zeros(N, dtype=np.float64)
        for s in range(N):
            level_power = s / 100.0 #lower bound of interval
            distances[s] = abs(demand_point - level_power ) 
        
        #build R matrix
        R= np.zeros ((n_actions ,N, N), dtype=np.float64)

        for s in range(N): #original state
            for s_next in range(N): #destination state
                #how far is s_next from demand
                base_cost = distances[s_next] 
                #how far is s from demand
                origin_distance=distances[s] 

                for a in range(n_actions):
                    #penalize x2 if destination is further from demand than origin
                    if base_cost > origin_distance:
                        cost = 2*base_cost
                    else:
                        cost = base_cost
                    
                    #we use negative beacuse mdptoolbox maximizes reward
                    R[a,s,s_next]=-cost 

                    



    @staticmethod
    def control_iteration(demand_point: np.float64, current_state: np.int32, P: np.ndarray, n_states: np.int32, n_actions: np.int32, gamma: np.float64) -> np.int32:
        """ Function that computes one control-iteration """
        #generate reward matrix for demand point 
        R=ControlModule.generate_R(demand_point=demand_point, n_states=n_states, n_actions=n_actions)
        #Build and sol mdp with value iteration (mdptoolbox)
        mdp=mdptoolbox.mdp.ValueIteration(transitions=P, rewards=R, discount=gamma)

        mdp.run()

        #mdp.policy is the optimal action for each state, we return the action for the current state
        optimal_action = mdp.policy[current_state]

        return np.int32(optimal_action)

    @staticmethod
    def control_loop(demand: np.ndarray,
                     probs: np.ndarray,
                     n_states: np.int32,
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        response = np.zeros_like(a=demand, dtype=np.float64)

        for i in range(len(demand)):
            action = ControlModule.control_iteration() #Control iteration devuelve un int pero por ahora me da igual

            if action == "d":
                outcomes = np.array([-2, -1, 0], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[0])
            elif action == "m":
                outcomes = np.array([-1, 0, 1], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[1])
            elif action == "i":
                outcomes = np.array([0, 1, 2], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[2])
            else:
                raise ValueError(f"Unknown action: {action}")

            result /= n_states

            if response[i - 1] + result >= 0 and response[i - 1] + result <= 1:
                response[i] = response[i - 1] + result
            elif response[i - 1] + result < 0:
                response[i] = 0        
            elif response[i - 1] + result > 1:
                response[i] = 1
        
        return response
