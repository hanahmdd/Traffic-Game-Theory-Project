import matplotlib.pyplot as plt


def generate_report_visuals(df):

    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8)
    )

  
    # PERFORMANCE COMPARISON
    

    ax1.plot(
        df['Step'],
        df['System_Efficiency']
        .rolling(window=5)
        .mean(),
        linewidth=2,
        label="Q-Learning System"
    )

    ax1.plot(
        df['Step'],
        df['Random_System_Efficiency']
        .rolling(window=5)
        .mean(),
        linestyle='--',
        label="Random Strategy"
    )

    ax1.plot(
        df['Step'],
        df['Fixed_System_Efficiency']
        .rolling(window=5)
        .mean(),
        linestyle=':',
        label="Fixed Timing"
    )

    ax1.set_title(
        "Traffic Flow Optimization",
        fontsize=14,
        fontweight='bold'
    )

    ax1.set_ylabel("Total Payoff")

    ax1.legend()

  
    # BELIEF CONVERGENCE
  

    ax2.plot(
        df['Step'],
        df['A_Belief_Opponent_Long'],
        label="Belief Convergence"
    )

    ax2.set_title(
        "Belief Evolution"
    )

    ax2.set_xlabel("Rounds")

    ax2.set_ylabel("Probability")

    ax2.legend()

    plt.tight_layout()

    plt.savefig(
        'traffic_analysis_report.png'
    )

    print(
        "[Visuals] traffic_analysis_report.png saved."
    )


def generate_q_learning_analysis(df):

    plt.figure(figsize=(12, 6))

  
    # REWARD CONVERGENCE
  

    plt.plot(
        df['Step'],
        df['Cumulative_A_Reward'],
        label='Agent A Reward'
    )

    plt.plot(
        df['Step'],
        df['Cumulative_B_Reward'],
        label='Agent B Reward'
    )

    plt.title(
        "Reward Convergence"
    )

    plt.xlabel("Rounds")

    plt.ylabel("Cumulative Reward")

    plt.legend()

    plt.savefig(
        'reward_convergence.png'
    )

    print(
        "[Visuals] reward_convergence.png saved."
    )

  
    # Q VALUES
 

    plt.figure(figsize=(12, 6))

    plt.plot(
        df['Step'],
        df['A_Q_Short'],
        label='A Q(Short)'
    )

    plt.plot(
        df['Step'],
        df['A_Q_Long'],
        label='A Q(Long)'
    )

    plt.plot(
        df['Step'],
        df['B_Q_Short'],
        label='B Q(Short)'
    )

    plt.plot(
        df['Step'],
        df['B_Q_Long'],
        label='B Q(Long)'
    )

    plt.title(
        "Q-Learning Convergence"
    )

    plt.xlabel("Rounds")

    plt.ylabel("Q Values")

    plt.legend()

    plt.savefig(
        'q_learning_analysis.png'
    )

    print(
        "[Visuals] q_learning_analysis.png saved."
    )

  
    # EPSILON DECAY
  

    plt.figure(figsize=(10, 5))

    plt.plot(
        df['Step'],
        df['Epsilon']
    )

    plt.title(
        "Adaptive Epsilon Decay"
    )

    plt.xlabel("Rounds")

    plt.ylabel("Exploration Rate")

    plt.savefig(
        'epsilon_decay.png'
    )

    print(
        "[Visuals] epsilon_decay.png saved."
    )

    # PERFORMANCE METRICS
 

    learning_avg = (
        df['System_Efficiency'].mean()
    )

    random_avg = (
        df['Random_System_Efficiency'].mean()
    )

    fixed_avg = (
        df['Fixed_System_Efficiency'].mean()
    )

    improvement_random = (
        (
            learning_avg - random_avg
        ) / random_avg
    ) * 100

    improvement_fixed = (
        (
            learning_avg - fixed_avg
        ) / fixed_avg
    ) * 100

    print("\n========== PERFORMANCE METRICS ==========")

    print(
        f"Average Learning Efficiency: "
        f"{learning_avg:.2f}"
    )

    print(
        f"Average Random Efficiency: "
        f"{random_avg:.2f}"
    )

    print(
        f"Average Fixed Efficiency: "
        f"{fixed_avg:.2f}"
    )

    print(
        f"Improvement over Random: "
        f"{improvement_random:.2f}%"
    )

    print(
        f"Improvement over Fixed: "
        f"{improvement_fixed:.2f}%"
    )