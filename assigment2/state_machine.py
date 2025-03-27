"""
Finite Automation
A finite automaton (FA) is a 5-tuple (Q, Σ, q0, F, δ), where
Q is a finite set of states;
Σ is a finite input alphabet;
q0 ∈ Q is the initial state;
F ⊆ Q is the set of accepting/final states; and
δ: Q×Σ→Q is the transition function.

For any element q of Q and any symbol σ ∈ Σ, we interpret δ (q, σ) as the state to which the FA
moves, if it is in state q and receives the input σ.

Mod-Three FA
Based on the notation from the definition, the modulo three FSM would be configured as
follows:
Q = (S0, S1, S2)
Σ = (0, 1)
q0 = S0
F = (S0, S1, S2)
δ(S0,0) = S0; δ(S0,1) = S1; δ(S1,0) = S2; δ(S1,1) = S0; δ(S2,0) = S1; δ(S2,1) = S2

Remainder Table
Final State Remainder
S0          0
S1          1
S2          2

"""
from typing import TypeVar, Generic, List
from abc import ABC, abstractmethod
from functools import reduce

import unittest


# L - Alphabet type 
L = TypeVar('L')
# S - State type
S = TypeVar('S')

class InvalidAlphabetContent(Exception):
    pass

class InvalidState(Exception):
    pass

#
# Finite State Machine interface to compute final state. It requires the
# state transition function
#
class FiniteStateMachine(ABC, Generic[L, S]):
    # current state
    current_state: S

    # map of state transitions, indexed by state 
    # where each value is the array state transitions by letter index
    # ex: {'State1': ['State3', 'State1', 'State2'], ...}
    state_transitions: dict[S, List[S]] = {}

    #
    # returns the alphabet letter transition index in a state
    # transitions dictionary value. By default throws invalid content exception
    #
    @abstractmethod
    def letter_transition_index(self, letter: L) -> int:
        raise InvalidAlphabetContent()
    
    #
    # checks the validity of a state name, used to access the
    # transitions dictionary
    #
    @abstractmethod
    def state_check(self, state: S) -> S:
        raise InvalidState()
    
    #
    # returns the transition state from a given state and alphabet letter
    #
    def transition(self, state: S, letter: L) -> S:
        return self.transition_matrix[self.state_check(state)][self.letter_transition_index(letter)]

    #
    # Returns the machine initial state
    #
    @abstractmethod
    def initial_state(self) -> S:
        pass

    #
    # returns the final state given an initial state and alphabet letter sequence
    #
    def sequence_transition(self, alphabet_sequence: list[L], initial_state: S | None = None) -> S:
        if not initial_state:
            initial_state = self.current_state
        final_state = reduce(self.transition, alphabet_sequence, initial_state)
        self.current_state = final_state
        return final_state

#
# Finite State Machine implementation for division by three remainder
#
# States are strings and alphabet is composed by integers
#
class ThreeFiniteStateMachine(FiniteStateMachine[int, str]):
    # setup the state transitions
    transition_matrix = {
        'S0': ['S0', 'S1'],
        'S1': ['S2', 'S0'],
        'S2': ['S1', 'S2']
    }
    # setup the current state as the inital state
    current_state = 'S0'

    def letter_transition_index(self, letter):
        if letter in [0, 1]:
            return letter
        # invalid letter, cannot compute index
        return super().letter_transition_index(letter)
    
    def state_check(self, state):
        if state in ['S0', 'S1', 'S2']:
            return state
        # invalid state
        return super().state_check(state)

    def initial_state(self):
        return 'S0'

#
# Calculates the division by 3 remainder of a binary representation
# of an unsigned int
#       
def binary_divide_by_three(sequence: list[int]) -> int:
    # empty list is considered invalid
    if len(sequence) == 0:
        raise InvalidAlphabetContent()
    
    # init the state machine
    state_machine = ThreeFiniteStateMachine()

    # get the final state
    final_state = state_machine.sequence_transition(sequence)

    # Translate the final state into the remainder
    match final_state:
        case 'S0':
            return 0
        case 'S1':
            return 1
        case 'S2':
            return 2
        case _:
            raise InvalidState()
#
# Transforms a digit string into a list of numbers
#
def string_digit_list(digit_string: str) -> list[int]:
    return [int(digit) for digit in digit_string]

class TestBinarySequenceDividedByThreeRemainder(unittest.TestCase):
    def test_expect_1(self):
        self.assertEqual(binary_divide_by_three([1,1,0,1]), 1)

    def test_expect_2(self):
        self.assertEqual(binary_divide_by_three([1,1,1,0]), 2)

    def test_expect_0(self):
        self.assertEqual(binary_divide_by_three([1,1,1,1]), 0)

    def test_prefix_zero(self):
        self.assertEqual(binary_divide_by_three([0,1,1,1,1]), 0)

    def test_invalid_non_binary_input(self):
        with self.assertRaises(InvalidAlphabetContent):
            binary_divide_by_three([1,2])

    def test_invalid__input(self):
        with self.assertRaises(InvalidAlphabetContent):
            binary_divide_by_three([])

    def test_string_input_1(self):
        self.assertEqual(binary_divide_by_three(string_digit_list("1101")), 1)

    def test_string_input_2(self):
        self.assertEqual(binary_divide_by_three(string_digit_list("1110")), 2)

    def test_string_input_0(self):
        self.assertEqual(binary_divide_by_three(string_digit_list("1111")), 0)

if __name__ == "__main__":
    unittest.main()
