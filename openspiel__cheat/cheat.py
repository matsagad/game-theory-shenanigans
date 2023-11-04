import enum
import numpy as np
import pyspiel
import math

"""
Game details:
https://en.wikipedia.org/wiki/Cheat_(game)
"""

class Action(enum.IntEnum):
    ACCUSE = 0
    PASS = -1


_NUM_SUITES = 4
_NUM_NUMBERS = 6
_DECK_SIZE = _NUM_SUITES * _NUM_NUMBERS
_DECK = [i for i in range(_NUM_SUITES * _NUM_NUMBERS)]

_MAX_NUM_PLAYERS = 10
_MIN_NUM_PLAYERS = 2
_MAX_GAME_LENGTH = 1000
_MIN_FACTOR_NON_AMBIGUOUS = 3

ENCODING_OFFSET = 6
CLAIM_BITMASK = (1 << ENCODING_OFFSET) - 1

_COMB_MAP = {i: math.comb(_DECK_SIZE, i) for i in range(1, 1 + _NUM_SUITES)}

_ENCODE_ACTION = {Action.ACCUSE: 0, Action.PASS: 1}
_DECODE_ACTION = {0: Action.ACCUSE, 1: Action.PASS}

curr = 2
for n_cards_down in range(1, _NUM_SUITES + 1):
    v = (1 << n_cards_down) - 1
    for _ in range(_COMB_MAP[n_cards_down]):
        for card_no in range(1, 1 + _NUM_NUMBERS):
            action = (
                (v << ENCODING_OFFSET) + _NUM_NUMBERS * (n_cards_down - 1) + card_no
            )
            _ENCODE_ACTION[action] = curr
            _DECODE_ACTION[curr] = action
            curr += 1

        t = (v | (v - 1)) + 1
        v = t | ((((t & -t) // (v & -v)) >> 1) - 1)


_GAME_TYPE = pyspiel.GameType(
    short_name="python_cheat",
    long_name="Python Cheat",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    # reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_NUM_PLAYERS,
    min_num_players=_MIN_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(_ENCODE_ACTION),
    max_chance_outcomes=_MIN_NUM_PLAYERS * _MIN_FACTOR_NON_AMBIGUOUS,
    num_players=_MIN_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)


class CheatGame(pyspiel.Game):
    """A Python version of Cheat."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        game_parameters = self.get_parameters()
        self._num_players = game_parameters.get("num_players", _MIN_NUM_PLAYERS)

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return CheatState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return CheatObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            self._num_players,
            params,
        )


class CheatState(pyspiel.State):
    """A Python version of the Cheat state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self.pile = 0

        shuffled_deck = _DECK.copy()
        np.random.shuffle(shuffled_deck)
        self.hands = [
            sum(1 << i for i in arr)
            for arr in np.array_split(shuffled_deck, game._num_players)
        ]

        self._fixed_chance_outcomes = [
            (i, 1 / (_MIN_FACTOR_NON_AMBIGUOUS * game._num_players))
            for i in range(_MIN_FACTOR_NON_AMBIGUOUS * game._num_players)
        ]

        self._is_chance = False
        self._game_over = False
        self.winner = -1

        self._prev_player = -1
        self._next_player = 0
        self.num_players = game._num_players

        self.last_action = -1
        self.last_claim_action = -1
        self.num_turns = 0
        self.chance_target = 0

        self._ENCODE_ACTION = _ENCODE_ACTION
        self._DECODE_ACTION = _DECODE_ACTION

        self._NUM_NUMBERS = _NUM_NUMBERS
        self._NUM_SUITES = _NUM_SUITES

        self._rewards = np.zeros(self.num_players)
        self._returns = np.zeros(self.num_players)

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._is_chance:
            return pyspiel.PlayerId.CHANCE
        return pyspiel.PlayerId.SIMULTANEOUS

    def _decode_play(self, action):
        truth = action >> ENCODING_OFFSET
        claim = action & CLAIM_BITMASK

        count = ((claim - 1) // _NUM_NUMBERS) + 1
        num = _NUM_NUMBERS - (-claim % _NUM_NUMBERS)

        return truth, count, num

    def _encode_play(self, truth, count, num):
        return (truth << ENCODING_OFFSET) + _NUM_NUMBERS * (count - 1) + num

    def _legal_next_number(self):
        if self.last_claim_action == -1 or self.last_action == Action.ACCUSE:
            return [i for i in range(1, _NUM_NUMBERS + 1)]
        last_num = (
            _NUM_NUMBERS - (-(CLAIM_BITMASK & self.last_claim_action)) % _NUM_NUMBERS
        )
        return [last_num, _NUM_NUMBERS - (-(last_num + 1) % _NUM_NUMBERS)]

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player != pyspiel.PlayerId.TERMINAL

        play_options = []
        # Can only accuse previous player if pile is non-empty.
        if self.pile != 0 and player != self._prev_player and self._prev_player != -1:
            play_options.append(_ENCODE_ACTION[Action.ACCUSE])

        # If not supposed to be next player, then can only accuse (as above)
        # or pass (i.e. not accuse).
        if player != self._next_player:
            return [_ENCODE_ACTION[Action.PASS]] + play_options

        curr_hand = int(self.hands[player])
        smallest_playable_hand = min(_NUM_SUITES, int.bit_count(curr_hand))

        if self.pile != 0:
            legal_nums = self._legal_next_number()
        else:
            legal_nums = [i for i in range(1, _NUM_NUMBERS + 1)]

        for n_cards_down in range(1, smallest_playable_hand + 1):
            v = (1 << n_cards_down) - 1
            for _ in range(_COMB_MAP[n_cards_down]):
                if curr_hand & v == v:
                    play_options.extend(
                        _ENCODE_ACTION[
                            (v << ENCODING_OFFSET)
                            + _NUM_NUMBERS * (n_cards_down - 1)
                            + card_no
                        ]
                        for card_no in legal_nums
                    )
                t = (v | (v - 1)) + 1
                v = t | ((((t & -t) // (v & -v)) >> 1) - 1)

        play_options.sort()

        assert play_options

        return play_options

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance
        return self._fixed_chance_outcomes

    def _action_was_truth(self, action):
        assert action != Action.PASS and action != Action.ACCUSE
        truth, _, num = self._decode_play(action)
        check_mask = sum(1 << (i * _NUM_NUMBERS + num - 1) for i in range(_NUM_SUITES))
        return truth | check_mask == check_mask

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        assert self._is_chance and not self._game_over
        self.chance_target = action

        self.num_turns += 1
        self._is_chance = False

        if self.num_turns >= self.get_game().max_game_length():
            self._game_over = True
            self._rewards.fill(0)

            hand_sizes = np.array([-int.bit_count(int(hand)) for hand in self.hands])

            self._rewards += (
                _DECK_SIZE
                * (hand_sizes - hand_sizes.min())
                / (hand_sizes.max() - hand_sizes.min())
                - _DECK_SIZE / 2
            )
            self._returns += self.rewards

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        self._rewards.fill(0)

        actions = [_DECODE_ACTION[action] for action in actions]
        accusers = [i for i, action in enumerate(actions) if action == Action.ACCUSE]

        if accusers:
            _, accuser = min(
                (
                    min(
                        sgn
                        * abs(self.chance_target - _MIN_FACTOR_NON_AMBIGUOUS * i)
                        % (_MIN_FACTOR_NON_AMBIGUOUS * self.num_players)
                        for sgn in (1, -1)
                    ),
                    i,
                )
                for i in accusers
            )

            assert self.last_action != Action.ACCUSE and self.last_action != Action.PASS

            cards_in_pile = int.bit_count(int(self.pile))

            accuser_mask = np.ones(self._rewards.shape, dtype=bool)
            accuser_mask[accusers] = True

            if self._action_was_truth(self.last_claim_action):
                if np.any(accuser_mask):
                    self._rewards[accuser_mask] -= cards_in_pile / accuser_mask.sum()
                if np.any(~accuser_mask):
                    self._rewards[~accuser_mask] += (
                        cards_in_pile / (~accuser_mask).sum()
                    )

                self.hands[accuser] += self.pile

                if not self.hands[self._prev_player]:
                    self.game_over = True
                    self.winner = self._prev_player

                    self.pile = 0
                    self.last_action = Action.ACCUSE

                    self._rewards -= _DECK_SIZE / (self.num_players - 1)
                    self._rewards[self.winner] += _DECK_SIZE

                    self._returns += self._rewards
                    return

                self._next_player = self._prev_player
            else:
                if np.any(accuser_mask):
                    self._rewards[accuser_mask] += cards_in_pile / accuser_mask.sum()
                if np.any(~accuser_mask):
                    self._rewards[~accuser_mask] -= (
                        cards_in_pile / (~accuser_mask).sum()
                    )

                self.hands[self._prev_player] += self.pile
                self._next_player = accuser

            self.pile = 0
            self.last_action = Action.ACCUSE
            self._returns += self._rewards
            return

        if (
            self.last_claim_action == self.last_action
            and self.last_claim_action != -1
            and not self._action_was_truth(self.last_action)
        ):
            cards_in_pile = int.bit_count(int(self.pile))
            self._rewards.fill(-cards_in_pile / (self.num_players - 1))
            self._rewards[self._prev_player] = cards_in_pile

        if not self.hands[self._prev_player]:
            self._game_over = True
            self.winner = self._prev_player

            self._rewards -= _DECK_SIZE / (self.num_players - 1)
            self._rewards[self.winner] += _DECK_SIZE

            self._returns += self._rewards
            return

        moves = [
            (i, action)
            for i, action in enumerate(actions)
            if action != Action.ACCUSE and action != Action.PASS
        ]
        assert len(moves) == 1

        curr_player, action = moves[0]

        assert curr_player == self._next_player, f"{curr_player} != {self._next_player}"

        truth, *_ = self._decode_play(action)

        self.hands[curr_player] -= truth
        self.pile += truth
        self.last_action = action
        self.last_claim_action = action

        self._prev_player = curr_player
        self._next_player = (curr_player + 1) % self.num_players
        self._is_chance = True
        self._returns += self._rewards

    def _action_to_string(self, player, action):
        action = _DECODE_ACTION[action]
        if action == Action.ACCUSE:
            return f"P{player} accuses P{self._prev_player}."
        if action == Action.PASS:
            return f"P{player} passes."

        played, claim_count, claim_num = self._decode_play(action)
        nums_played = self.hand_bits_to_nums(played)

        return f"P{player} claims {claim_count} x {claim_num}s but plays {','.join(map(str, nums_played))}."

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        output = []
        for player in range(self.num_players):
            output.append(self.hand_bits_to_nums(self.hands[player]))

        return "\n".join(
            f"P{i}: [{','.join(map(str, hand))}]" for i, hand in enumerate(output)
        )

    def lead_player(self):
        if self.winner != -1:
            return self.winner
        return min((reward, i) for i, reward in enumerate(self.returns()))[1]

    def hand_bits_to_nums(self, hand):
        nums = []
        curr = 1
        while hand:
            if hand % 2 == 1:
                nums.append(curr)
            hand >>= 1
            curr = _NUM_NUMBERS - (-(curr + 1) % _NUM_NUMBERS)
        return sorted(nums)


class CheatObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, num_players, params):
        """Initialises an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")

        self.num_players = num_players
        pieces = [("player", num_players, (num_players,))]

        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("private_hand", _DECK_SIZE, (_DECK_SIZE,)))
        if iig_obs_type.public_info:
            pieces.append(
                (
                    "pile_claims",
                    num_players * _NUM_NUMBERS * (_NUM_SUITES + 1),
                    (num_players, _NUM_NUMBERS, _NUM_SUITES + 1),
                )
            )
            pieces.append(
                (
                    "last_claim",
                    _DECK_SIZE,
                    (_NUM_NUMBERS, _NUM_SUITES),
                )
            )
            pieces.append(("revealed_cards", _DECK_SIZE, (_DECK_SIZE,)))

        total_size = sum(size for _, size, _ in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index : index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

        if "player" in self.dict:
            self.dict["player"].fill(0)
            self.dict["player"][state._next_player] += 1

        if "private_hand" in self.dict:
            self.dict["private_hand"].fill(0)

            self.dict["private_hand"] += np.array(
                list(np.binary_repr(state.hands[player]).zfill(_DECK_SIZE))
            ).astype(np.int8)

        if "pile_claims" in self.dict:
            if state.last_action == Action.ACCUSE or state.pile == 0:
                self.dict["pile_claims"].fill(0)
            else:
                _, count, num = state._decode_play(state.last_action)
                num_count = self.dict["pile_claims"][state._prev_player][num - 1]
                if not np.any(num_count):
                    num_count[count - 1] = 1
                else:
                    curr_count = np.argmax(num_count)
                    if curr_count < _NUM_SUITES:
                        num_count[curr_count] = 0
                        num_count[curr_count + 1] = 1

        if "last_claim" in self.dict:
            self.dict["last_claim"].fill(0)
            if state.last_action != Action.ACCUSE and state.last_action != Action.PASS:
                _, count, num = state._decode_play(state.last_action)
                self.dict["last_claim"][num - 1][count - 1] = 1

        if "revealed_cards" in self.dict:
            self.dict["revealed_cards"].fill(0)
            if state.last_action == Action.ACCUSE:
                truth, *_ = state._decode_play(state.last_claim_action)

                self.dict["revealed_cards"] += np.array(
                    list(np.binary_repr(truth).zfill(_DECK_SIZE))
                ).astype(np.int8)

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        return ""


pyspiel.register_game(_GAME_TYPE, CheatGame)
