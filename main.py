
from simulation import start_simulation
from analysis import (
    generate_report_visuals,
    generate_q_learning_analysis
)
from game_theory import TrafficGame
from extensive_form_tree import generate_game_tree

import numpy as np


def run_project():

    print("\n" + "=" * 50)
    print("🚦 INTELLIGENT TRAFFIC MULTI-AGENT SYSTEM")
    print("=" * 50)


    # 1. GAME THEORY ANALYSIS


    gt = TrafficGame()

    print(f"\n[1] Nash Equilibria Analysis:")

    equilibria = gt.find_equilibria()

    for i, eq in enumerate(equilibria):

        # Determine equilibrium type
        is_pure = all(
            np.all(np.isin(strategy, [0, 1]))
            for strategy in eq
        )

        eq_type = (
            "Pure Nash Equilibrium"
            if is_pure
            else "Mixed Strategy Equilibrium"
        )

        print(
            f"\n -> Result {i + 1}: "
            f"Strategy A {eq[0]}, "
            f"Strategy B {eq[1]} "
            f"({eq_type})"
        )

       
        # Human-readable interpretation
   

        actions = ["Short Green", "Long Green"]

        if is_pure:

            a_action = actions[np.argmax(eq[0])]
            b_action = actions[np.argmax(eq[1])]

            print(
                f"    -> Interpretation: "
                f"Agent A chooses {a_action}, "
                f"Agent B chooses {b_action}"
            )

        else:

            print(
                "    -> Interpretation: "
                "Agents use probabilistic "
                "mixed strategies."
            )

            print(
                f"       Agent A probabilities: "
                f"{eq[0]}"
            )

            print(
                f"       Agent B probabilities: "
                f"{eq[1]}"
            )


    # 2. DOMINANCE ANALYSIS


    print(f"\n[2] Weak Dominance Analysis:")

    weak_results = gt.check_weak_dominance()

    if weak_results:

        for result in weak_results:
            print(f" -> {result}")

    else:
        print(" -> No weak dominance found.")

   
    # 3. IESDS ANALYSIS


    print(
        f"\n[3] Iterated Elimination of Strictly Dominated Strategies (IESDS):"
    )

    iesds_results = gt.run_iesds()

    for result in iesds_results:
        print(f" -> {result}")

    # 4. EXTENSIVE FORM VISUALIZATION
  
    print(f"\n[4] Generating Extensive Form Visualization...")

    generate_game_tree()

    # 5. MULTI-AGENT LEARNING SIMULATION
  

    print(f"\n[5] Starting Multi-Agent Learning Simulation...")

    results_df = start_simulation(150)


    # 6. PERFORMANCE REPORTS
 

    print(f"\n[6] Generating Performance and Learning Reports...")

    generate_report_visuals(results_df)

    generate_q_learning_analysis(results_df)

  
    # 7. PERFORMANCE SUMMARY
   

    avg_efficiency = results_df['System_Efficiency'].mean()

    print(f"\n--- PERFORMANCE SUMMARY ---")

    print(f"Average System Throughput: {avg_efficiency:.2f}")

    print(
        "\nLearning Outcome:"
    )

    print(
        "Agents successfully:"
    )

    print(" -> Modeled beliefs")
    print(" -> Learned using True Q-Learning")
    print(" -> Applied Bellman Equation updates")
    print(" -> Adapted strategies dynamically")
    print(" -> Used mixed probabilistic strategies")
    print(" -> Learned optimal best responses")



    results_df.to_csv(
        "traffic_simulation_results.csv",
        index=False
    )

    print(
        "\n[Complete] All files and visualizations generated successfully."
    )


if __name__ == "__main__":
    run_project()