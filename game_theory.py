import numpy as np
import nashpy as nash


class TrafficGame:

    def __init__(self,
                 congestion_multiplier=1.0):

        # NORMAL FORM PAYOFF MATRICES
        

        self.payoff_A = np.array([
            [5, 3],
            [8, 2]
        ]) * congestion_multiplier

        self.payoff_B = np.array([
            [5, 8],
            [3, 2]
        ]) * congestion_multiplier

        self.game = nash.Game(
            self.payoff_A,
            self.payoff_B
        )

    # NASH EQUILIBRIA
   

    def find_equilibria(self):

        return list(
            self.game.support_enumeration()
        )

  
    # WEAK DOMINANCE


    def check_weak_dominance(self):

        weakly_dominated = []

    
        # CHECK AGENT A (ROWS)
        
        for i in range(2):

            for j in range(2):

                if i != j:

                    if (
                        np.all(
                            self.payoff_A[j]
                            >= self.payoff_A[i]
                        )
                        and
                        np.any(
                            self.payoff_A[j]
                            > self.payoff_A[i]
                        )
                    ):

                        weakly_dominated.append(
                            f"Agent A Strategy {i} "
                            f"is weakly dominated "
                            f"by Strategy {j}"
                        )

        
        # CHECK AGENT B (COLUMNS)
        

        for i in range(2):

            for j in range(2):

                if i != j:

                    col_i = self.payoff_B[:, i]
                    col_j = self.payoff_B[:, j]

                    if (
                        np.all(col_j >= col_i)
                        and
                        np.any(col_j > col_i)
                    ):

                        weakly_dominated.append(
                            f"Agent B Strategy {i} "
                            f"is weakly dominated "
                            f"by Strategy {j}"
                        )

        return (
            weakly_dominated
            if weakly_dominated
            else ["No weak dominance found."]
        )

    
    # IESDS
   

    def run_iesds(self):

        A = self.payoff_A.copy()
        B = self.payoff_B.copy()

        rows = list(range(A.shape[0]))
        cols = list(range(B.shape[1]))

        eliminated_log = []

        while True:

            start_count = (
                len(rows) + len(cols)
            )

       
            # CHECK ROWS
         

            for i in rows:

                for j in rows:

                    if i != j:

                        if np.all(
                            A[j, cols]
                            > A[i, cols]
                        ):

                            if i in rows:

                                rows.remove(i)

                                eliminated_log.append(
                                    f"Eliminated "
                                    f"Agent A Strategy {i}"
                                )

        
            # CHECK COLUMNS
          

            for i in cols:

                for j in cols:

                    if i != j:

                        if np.all(
                            B[rows, j]
                            > B[rows, i]
                        ):

                            if i in cols:

                                cols.remove(i)

                                eliminated_log.append(
                                    f"Eliminated "
                                    f"Agent B Strategy {i}"
                                )

            if (
                len(rows) + len(cols)
                == start_count
            ):

                break

        return (
            eliminated_log
            if eliminated_log
            else ["No strict dominance found."]
        )

    
    # EPSILON GREEDY
 

    def get_q_best_response(
            self,
            q_values,
            epsilon=0.1):

        if np.random.random() < epsilon:

            return np.random.choice([0, 1])

        return np.argmax(q_values)