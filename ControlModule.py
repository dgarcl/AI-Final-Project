# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(probabilities: np.array) -> np.array:
        """ Function that generates the probabilities (transition) matrix """

        N = 100 #Number of states

        for action in probabilities:
            if len(action) != 3:
                raise ValueError("Invalid probabilities")
            total = 0
            for prob in action:
                total += prob
                if (prob > 1) or (prob < 0):
                    raise ValueError("Invalid probabilities")
            if not np.isclose(1, total):
                raise ValueError("Invalid probabilities")
            
        d2, d1, d0 = probabilities[0]
        m_1, m0, m1 =  probabilities[1]
        i0, i1, i2 = probabilities[2]


        Pd = np.zeros((N, N))
        Pm = np.zeros((N, N))
        Pi = np.zeros((N, N))
        #rows = current state
        #columns = next state

        for row in range(N):
            #In Pd we modify (s, s-2), (s, s-1) and (s, s)
            if row == 0:  #Bound the probabilities so that we never go below zero 
                Pd[0][0] = 1
            elif row == 1: 
                Pd[1][0] = d2 + d1
                Pd[1][1] = d0
            else:
                Pd[row][row-2] = d2
                Pd[row][row-1] = d1
                Pd[row][row] = d0

            #In Pm we modify (s, s-1), (s, s) and (s, s+1)
            if row == 0:
                Pm[0][0] = m_1 + m0
                Pm[0][1] = m1
            elif row == 99:
                Pm[99][98] = m_1
                Pm[99][99] = m0 + m1
            else:
                Pm[row][row-1] = m_1
                Pm[row][row] = m0
                Pm[row][row+1] = m1

            #In Pi we modify (s, s), (s, s+1), (s, s+2)
            if row == 98:
                Pi[98][98] = i0
                Pi[98][99] = i1 + i2
            elif row == 99:
                Pi[99][99] = 1
            else:
                Pi[row][row] = i0
                Pi[row][row+1] = i1
                Pi[row][row+2] = i2
        return np.array([Pd, Pm, Pi]) #Array of matrices

    @staticmethod
    def generate_R(demand_point: np.float64, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        N=n_states #Number of states

        #compute base distance for all possible destination states
        distances = np.zeros(N, dtype=np.float64)
        for s in range(N):
            level_power = s / N #lower bound of interval
            distances[s] = abs(demand_point - level_power) 
        
        #build R matrix
        R = np.zeros ((n_actions ,N, N), dtype=np.float64)

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

        return R

    @staticmethod
    def control_iteration(demand_point: np.float64, current_state: np.float64, P: np.ndarray, n_states: np.int32, n_actions: np.int32, gamma: np.float64) -> np.int32:
        """ Function that computes one control-iteration """
        #generate reward matrix for demand point 
        R=ControlModule.generate_R(demand_point=demand_point, n_states=n_states, n_actions=n_actions)

        #Build and sol mdp with value iteration (mdptoolbox)
        mdp=mdptoolbox.mdp.ValueIteration(transitions=P, reward=R, discount=gamma)

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
        response = np.zeros_like(a=demand, dtype=np.float64) #Generate a vector (as demand) to contain the output

        try: #If probs are valid probabilities, we generate the probability matrices
            P = ControlModule.generate_P(probs)
        except Exception as e:
            raise ValueError(f"Failed to generate transition matrix: {e}")

        response[0] = demand[0]  #Handle the initial state of the MDP

        for i in range(1, len(demand)): #Loop through the demand points in the array
            #Get the demand point and the current state of the MDP
            demand_point = demand[i]

            current_state = int(response[i - 1] * n_states)

            #We handle one control iteration inside the function
            action = ControlModule.control_iteration(demand_point, current_state, P, n_states, n_actions, gamma)

            #Choose one of the outcomes probabilistically
            if action == 0: #Decrease action ("d")
                outcomes = np.array([-2, -1, 0], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[0])
            elif action == 1: #Mantain action ("m")
                outcomes = np.array([-1, 0, 1], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[1])
            elif action == 2: #Increase action ("i")
                outcomes = np.array([0, 1, 2], dtype=np.int32)
                result = np.random.choice(outcomes, p=probs[2])
            else:
                raise ValueError(f"Unknown action: {action}")

            result /= n_states #Normalise the result

            #Modify the output according to the result obtained
            if response[i - 1] + result >= 0 and response[i - 1] + result <= 0.99:
                response[i] = response[i - 1] + result
            elif response[i - 1] + result < 0: #Force 0 as the lower bound
                response[i] = 0
            elif response[i - 1] + result > 0.99: #Force 0.99 as the upper bound
                response[i] = 0.99

        return response
