from typing import List, Tuple
import random


class Agent:
    def __init__(self, n: int):
        self.n = n


class RandomAgent(Agent):
    def __init__(self, n: int):
        self.n = n

    def move(
        self, observation: Tuple[List[int], List[Tuple[int, int]]], choices: List[int]
    ) -> int:
        return random.choice(choices)


class OneUpAgent(Agent):
    def __init__(self, n: int):
        self.n = n

    def move(
        self, observation: Tuple[List[int], List[Tuple[int, int]]], choices: List[int]
    ) -> int:
        opp_moves = {move for _, move in observation[1]}
        rem_opp_choices = [i for i in range(1, self.n + 1) if i not in opp_moves]
        opp_avg_choice = sum(rem_opp_choices) / len(rem_opp_choices)

        if max(choices) < opp_avg_choice:
            return min(choices)
        return min(i for i in choices if i >= opp_avg_choice)


class MatchAgent(Agent):
    def __init__(self, n: int):
        self.n = n

    def move(
        self, observation: Tuple[List[int], List[Tuple[int, int]]], choices: List[int]
    ) -> int:
        most_recent_card = observation[0][-1]
        return most_recent_card


class OneUpMatchAgent(Agent):
    def __init__(self, n: int):
        self.n = n

    def move(
        self, observation: Tuple[List[int], List[Tuple[int, int]]], choices: List[int]
    ) -> int:
        most_recent_card = observation[0][-1]
        if most_recent_card == self.n:
            return 1
        return most_recent_card + 1


class GameInstance:
    def __init__(self, n: int, p1: Agent, p2: Agent):
        self.n = n
        self.p2 = p2
        self.p1 = p1

    def simulate(self, verbose: bool = False):
        black_cards = [i for i in range(1, self.n + 1)]
        random.shuffle(black_cards)

        choices1 = [i for i in range(1, self.n + 1)]
        choices2 = [i for i in range(1, self.n + 1)]

        cards_dealt = []
        moves = []

        for _ in range(self.n):
            black_card = black_cards.pop()
            cards_dealt.append(black_card)

            choice1 = self.p1.move((cards_dealt, moves), choices1)
            choice2 = self.p2.move((cards_dealt, moves), choices2)

            moves.append((choice1, choice2))
            choices1.remove(choice1)
            choices2.remove(choice2)

        score1 = 0
        score2 = 0

        for stake, (move1, move2) in zip(cards_dealt, moves):
            if move1 > move2:
                score1 += stake
            elif move2 > move1:
                score2 += stake

        if verbose:
            if score1 == score2:
                print(f"It's a tie! Both scored {score1}")
            else:
                winner = 1 if score1 > score2 else 2
                print(f"P{winner} won! P1: {score1}, P2: {score2}")

        return score1, score2


def main():
    """
    See link for game details: https://en.wikipedia.org/wiki/Goofspiel
    """
    n = 10
    p1 = RandomAgent(n)
    p2 = MatchAgent(n)

    game = GameInstance(n, p1, p2)

    n_games = 100000
    wins1 = 0
    wins2 = 0
    draws = 0

    for _ in range(n_games):
        score1, score2 = game.simulate()
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1
        else:
            draws += 1

    print(f"P1 won {wins1} times. ({100 * wins1 / n_games:.3f}%)")
    print(f"P2 won {wins2} times. ({100 * wins2 / n_games:.3f}%)")
    print(f"There were {draws} draws. ({100 * draws / n_games:.3f}%)")


if __name__ == "__main__":
    main()
