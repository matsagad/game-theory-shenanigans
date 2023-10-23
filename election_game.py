import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class Agent:
    pass


class DiscreteAgent(Agent):
    def get_payoffs(self, board_state: np.ndarray) -> np.ndarray:
        if not np.any(board_state):
            return []

        occupied = np.argwhere(board_state).flatten()
        occupied_counts = board_state[occupied].astype(int)
        indiv_weights = self.ref[occupied]

        weights = np.min(indiv_weights, axis=0)
        weight_counts = np.sum(
            (weights == indiv_weights).astype(int) * occupied_counts.reshape(-1, 1),
            axis=0,
        )

        shares = weight_counts * (indiv_weights == weights).astype(int)
        division = np.zeros(shares.shape)
        division[shares > 0] = 1 / shares[shares > 0]

        payoffs = np.sum(self.board_weights * division, axis=1)

        return np.array(
            [
                payoff
                for payoff, count in zip(payoffs, occupied_counts)
                for _ in range(count)
            ]
        )


class ContinuousAgent(Agent):
    pass


class DiscreteBestResponseAgent(DiscreteAgent):
    def __init__(self, board_weights: np.ndarray):
        self.board_weights = board_weights
        self.curr_pos = -1

        n_places = len(board_weights)
        self.ref = np.sum(
            np.fromiter(
                (i * np.eye(n_places, k=i) for i in range(n_places)),
                dtype=np.ndarray,
            )
        )
        self.ref += self.ref.T

    # Get pos in the board to move to
    def move(self, board_state: np.ndarray) -> int:
        # If no one else in board then randomly choose spot
        if not np.any(board_state):
            return np.random.choice(len(board_state))

        occupied = np.argwhere(board_state).flatten()
        occupied_counts = board_state[occupied].astype(int)

        indiv_weights = self.ref[occupied]
        weights = np.min(indiv_weights, axis=0)
        weight_counts = np.sum(
            (weights == indiv_weights).astype(int) * occupied_counts.reshape(-1, 1),
            axis=0,
        )

        shares = (self.ref < weights).astype(int) + (weight_counts + 1) * (
            self.ref == weights
        ).astype(int)

        division = np.zeros(shares.shape)
        division[shares > 0] = 1 / shares[shares > 0]

        payoffs = np.sum(self.board_weights * division, axis=1)

        # Choose random best response move
        best_responses = np.argwhere(payoffs == np.max(payoffs)).flatten()

        if self.curr_pos in best_responses:
            return self.curr_pos
        return np.random.choice(best_responses)


def plot_state(state: np.ndarray, target_ax: any = None):
    cmap = get_cmap("coolwarm")

    n_places = len(state)
    x_labels = np.arange(1, n_places + 1)
    ax = plt if target_ax is None else target_ax

    ax.bar(
        x_labels,
        state,
        color=[cmap(i / n_places) for i in range(n_places)],
        edgecolor="black",
    )

    if not target_ax:
        plt.show()


def plot_states(states: List[np.ndarray], board_weights: np.ndarray = None):
    if not states:
        return

    plot_board_weights = 1 if board_weights is not None else 0
    fig, axes = plt.subplots(
        nrows=len(states) + plot_board_weights, ncols=1, sharex=True
    )

    for ax, state in zip(axes, states):
        plot_state(state, target_ax=ax)
    if plot_board_weights:
        axes[-1].plot(np.arange(1, len(states[0]) + 1), board_weights, color="black")

    plt.show()


def simulate_discrete(
    n_places: int,
    n_players: int,
    n_turns_per_player: int,
    board_weights: np.ndarray = None,
):
    if board_weights is None:
        board_weights = np.repeat(1 / n_places, n_places)
    players = [DiscreteBestResponseAgent(board_weights) for _ in range(n_players)]

    board_state = np.zeros(n_places, dtype=int)
    state_trail = np.zeros(n_places)

    n_turns = 0
    is_equilibrium = False

    for _ in range(n_turns_per_player):
        prev_board_state = board_state.copy()

        for player in players:
            curr_pos = player.curr_pos
            if curr_pos != -1:
                board_state[curr_pos] -= 1

            new_pos = player.move(board_state)
            player.curr_pos = new_pos

            board_state[new_pos] += 1

        state_trail += board_state
        n_turns += 1

        if np.array_equal(prev_board_state, board_state):
            is_equilibrium = True
            break

    print(f"Equilibrium: {is_equilibrium}.")
    print(f"No. of turns: {n_turns}")
    print(f"Board state:\n{board_state}")
    print(f"Payoffs: {players[0].get_payoffs(board_state)}")

    avg_trail = state_trail / n_turns_per_player

    """
    Plot legend:
    - first row is final board state (could be equilibrium)
    - second row is the average activity per location across the entire simulation
    - third row (if board_weights is passed) is the payoff distribution
    """

    # plot_states([board_state, avg_trail])
    plot_states([board_state, avg_trail], board_weights)


def main():
    # Discrete case
    n_places = 50
    n_players = 4
    n_turns_per_player = 1000

    ## Uniformly distributed voters
    print("\n[Uniform case]")
    simulate_discrete(n_places, n_players, n_turns_per_player)

    ## Normally distributed voters
    print("\n[Normally distributed case]")
    from scipy.stats import norm

    k = 1 / 50
    board_weights = norm.pdf(
        np.linspace(norm.ppf(k / n_places), norm.ppf(1 - k / n_places), n_places)
    )
    simulate_discrete(n_places, n_players, n_turns_per_player, board_weights)


if __name__ == "__main__":
    main()
