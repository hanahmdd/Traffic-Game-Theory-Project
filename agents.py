import numpy as np


class IntelligentAgent:

    def __init__(
            self,
            name,
            learning_rate=0.1,
            discount_factor=0.9):

        """
        Intelligent Agent with:
        - Stateful Q-Learning
        - Belief Modeling
        - Mixed Strategies
        - Best Response Learning
        """

        self.name = name

        self.lr = learning_rate
        self.gamma = discount_factor

   
        # STATES
        #
        # 0 -> Low Congestion
        # 1 -> Medium Congestion
        # 2 -> High Congestion
   

        self.q_table = np.zeros((3, 2))

        self.opponent_history = []

        self.beliefs = np.array([0.5, 0.5])

  
        # ADAPTIVE EPSILON
     

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.995

    # BELIEF MODELING


    def update_beliefs(self, opponent_action):

        self.opponent_history.append(opponent_action)

        counts = np.bincount(
            self.opponent_history,
            minlength=2
        )

        self.beliefs = counts / len(self.opponent_history)

  
    # TRUE Q-LEARNING
   

    def learn(
            self,
            state,
            action,
            reward,
            next_state):

        best_future_q = np.max(
            self.q_table[next_state]
        )

        td_target = (
            reward +
            self.gamma * best_future_q
        )

        td_error = (
            td_target -
            self.q_table[state, action]
        )

        self.q_table[state, action] += (
            self.lr * td_error
        )


    # EPSILON DECAY
 

    def decay_epsilon(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay_rate
        )

  
    # ACTION SELECTION
 

    def select_action(
            self,
            game_model,
            state,
            observed_action=None,
            mode="normal",
            player="A"):

  
        # EXTENSIVE FORM
      

        if (
                mode == "extensive"
                and observed_action is not None
        ):

            payoffs_given_A = (
                game_model.payoff_B[
                    observed_action, :
                ]
            )

            return np.argmax(payoffs_given_A)

       
        # ADAPTIVE EPSILON GREEDY
    

        if np.random.random() < self.epsilon:

            return np.random.choice([0, 1])

      
        # EXPECTED UTILITY USING BELIEFS
       

        expected_utilities = []

        for my_action in [0, 1]:

            expected_payoff = 0

            for opponent_action in [0, 1]:

                belief_probability = (
                    self.beliefs[opponent_action]
                )

                if player == "A":

                    payoff_estimate = (
                        game_model.payoff_A[
                            my_action,
                            opponent_action
                        ]
                    )

                else:

                    payoff_estimate = (
                        game_model.payoff_B[
                            opponent_action,
                            my_action
                        ]
                    )

                expected_payoff += (
                    belief_probability *
                    payoff_estimate
                )

            total_expected_utility = (
                    expected_payoff +
                    self.q_table[state, my_action]
            )

            expected_utilities.append(
                total_expected_utility
            )

        return np.argmax(expected_utilities)


    # MIXED STRATEGIES
  

    def select_mixed_strategy_action(
            self,
            probability_long=0.5):

        return np.random.choice(
            [0, 1],
            p=[
                1 - probability_long,
                probability_long
            ]
        )

   
    # BELIEF STATUS
  

    def get_current_belief_status(self):

        return (
            f"{self.name} believes opponent "
            f"plays Long with probability "
            f"{self.beliefs[1]:.2%}"
        )