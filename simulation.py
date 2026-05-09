import pandas as pd
import numpy as np

from game_theory import TrafficGame
from agents import IntelligentAgent


def calculate_waiting_time(
        congestion_level,
        total_efficiency):

    if congestion_level == 2:
        base_wait = 30

    elif congestion_level == 1:
        base_wait = 20

    else:
        base_wait = 10

    efficiency_reduction = total_efficiency * 0.7

    waiting_time = max(
        1,
        base_wait - efficiency_reduction
    )

    return round(waiting_time, 2)


def start_simulation(iterations=100):

    agent_a = IntelligentAgent("North_Light")
    agent_b = IntelligentAgent("East_Light")

    logs = []

    print(f"Running simulation for {iterations} steps...")

    total_reward_a = 0
    total_reward_b = 0

    for i in range(iterations):

        # ENVIRONMENT STATES
      

        rand = np.random.random()

        if rand < 0.4:

            state = 0
            multiplier = 1.0

        elif rand < 0.75:

            state = 1
            multiplier = 1.5

        else:

            state = 2
            multiplier = 2.0

        game = TrafficGame(
            congestion_multiplier=multiplier
        )

  
        # AGENT A
   

        if np.random.random() < 0.2:

            act_a = (
                agent_a
                .select_mixed_strategy_action(
                    probability_long=
                    agent_a.beliefs[1]
                )
            )

        else:

            act_a = agent_a.select_action(
                game_model=game,
                state=state,
                mode="normal",
                player="A"
            )

        # IMPERFECT INFORMATION
   

        observation_noise = 0.15

        if np.random.random() < observation_noise:

            observed_action = 1 - act_a

        else:

            observed_action = act_a

      
        # AGENT B
       

        act_b = agent_b.select_action(
            game_model=game,
            state=state,
            observed_action=observed_action,
            mode="extensive",
            player="B"
        )

     
        # PAYOFFS
      

        reward_a = game.payoff_A[
            act_a,
            act_b
        ]

        reward_b = game.payoff_B[
            act_a,
            act_b
        ]

        total_reward_a += reward_a
        total_reward_b += reward_b

        total_efficiency = (
                reward_a + reward_b
        )

        
        # BASELINES
     

        random_a = np.random.choice([0, 1])
        random_b = np.random.choice([0, 1])

        random_efficiency = (
                game.payoff_A[random_a, random_b]
                +
                game.payoff_B[random_a, random_b]
        )

        fixed_efficiency = (
                game.payoff_A[0, 0]
                +
                game.payoff_B[0, 0]
        )

    
        # NEXT STATE
       

        next_rand = np.random.random()

        if next_rand < 0.4:
            next_state = 0

        elif next_rand < 0.75:
            next_state = 1

        else:
            next_state = 2

      
        # BELIEFS
       

        agent_a.update_beliefs(act_b)
        agent_b.update_beliefs(act_a)

  
        # LEARNING
     

        agent_a.learn(
            state,
            act_a,
            reward_a,
            next_state
        )

        agent_b.learn(
            state,
            act_b,
            reward_b,
            next_state
        )

      
        # EPSILON DECAY
   

        agent_a.decay_epsilon()
        agent_b.decay_epsilon()

       
        # WAITING TIME
        

        waiting_time = calculate_waiting_time(
            state,
            total_efficiency
        )

  
        # LOGS
     

        logs.append({

            "Step": i,

            "Environment":
                ["Low", "Medium", "High"][state],

            "A_Action":
                "Long"
                if act_a == 1
                else "Short",

            "B_Action":
                "Long"
                if act_b == 1
                else "Short",

            "Observed_Action_By_B":
                "Long"
                if observed_action == 1
                else "Short",

            "System_Efficiency":
                total_efficiency,

            "Random_System_Efficiency":
                random_efficiency,

            "Fixed_System_Efficiency":
                fixed_efficiency,

            "A_Payoff":
                reward_a,

            "B_Payoff":
                reward_b,

            "Cumulative_A_Reward":
                total_reward_a,

            "Cumulative_B_Reward":
                total_reward_b,

            "A_Belief_Opponent_Long":
                agent_a.beliefs[1],

            "A_Q_Short":
                agent_a.q_table[state, 0],

            "A_Q_Long":
                agent_a.q_table[state, 1],

            "B_Q_Short":
                agent_b.q_table[state, 0],

            "B_Q_Long":
                agent_b.q_table[state, 1],

            "Epsilon":
                agent_a.epsilon,

            "Waiting_Time_Index":
                waiting_time
        })

    return pd.DataFrame(logs)