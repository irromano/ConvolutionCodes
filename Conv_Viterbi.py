import numpy as np
import enum


class EncodingType(enum.Enum):
    BSC = 1
    AWGN = 2

# The unit level used to create an individual state of the trelis diagram


class State:
    def __init__(self, D, encoding: EncodingType, Eb):
        self.backbranch0 = -1
        self.backbranch1 = -1
        if isinstance(D, list):
            self.state_value = D.copy()
            self.state_digit = value_to_digit(D)
        else:
            self.state_digit = D
            self.state_value = digit_to_Value(D)

        self.branch0_state = self.state_value.copy()
        self.branch1_state = self.state_value.copy()
        self.branch0_output = conv(0, self.branch0_state, encoding, Eb)
        self.branch1_output = conv(1, self.branch1_state, encoding, Eb)
        self.branch0_digit = value_to_digit(self.branch0_state)
        self.branch1_digit = value_to_digit(self.branch1_state)
        return


class Trellis:
    def __init__(self, encoding: EncodingType, Eb=1.0):
        self.encoding = encoding
        self.states = []
        self.memory = 3
        self.availableNodes = np.zeros(2 ** self.memory, dtype=int)
        self.availableNodes[0] = 1
        self.availableNodes_ClosingSet = []
        self.reverseNode = 0
        for num in range(2 ** self.memory):
            self.states.append(State(num, encoding, Eb))

        for state in self.states:
            for otherState in self.states:
                if otherState.branch0_state == state.state_value:
                    state.backbranch0 = otherState.state_digit
                    state.backbranch0_output = otherState.branch0_output
                if otherState.branch1_state == state.state_value:
                    state.backbranch1 = otherState.state_digit
                    state.backbranch1_output = otherState.branch1_output

    # Helper function of updateBranchCost()
    def nextAvailableNodes(self, time) -> list[int]:
        newAvailable = self.availableNodes.copy()
        self.availableNodes_ClosingSet.append(self.availableNodes)
        for i in range(len(newAvailable)):
            if self.availableNodes[i]:
                newAvailable[self.states[i].branch0_digit] = 1
                newAvailable[self.states[i].branch1_digit] = 1
        return newAvailable

    def finalAvailableNodes(self, time, blockLength) -> list[int]:
        return self.availableNodes_ClosingSet[blockLength - time - 1]  # This is blocklength minus current time minus 1

    def distance(self, A: list[int], B: list[int], encoding: EncodingType, hard):

        if encoding == EncodingType.BSC:
            # Calculate Hamming Distance
            d = 0
            if A[0] != B[0]:
                d += 1
            if A[1] != B[1]:
                d += 1
            return d
        else:
            # Return Euclidean Distance
            a = A.copy()
            b = B.copy()
            if hard:  # Force values into -1.0 or 1.0
                for i in range(len(a)):
                    a[i] = -1.0 if a[i] < 0 else 1.0
                for i in range(len(b)):
                    b[i] = -1.0 if b[i] < 0 else 1.0
            return np.linalg.norm(a - b)

    def updateBranchCost(self, cost: list[int], input: list[int], time: int, hard=False):
        if time <= self.memory:
            nextAvailableNodes = self.nextAvailableNodes(time)
        else:
            nextAvailableNodes = self.availableNodes
        newcost = cost.copy()
        for i in range(len(self.states)):
            if nextAvailableNodes[i]:
                distance0 = self.distance(
                    input, self.states[i].backbranch0_output, self.encoding, hard) + cost[self.states[i].backbranch0]
                distance1 = self.distance(
                    input, self.states[i].backbranch1_output, self.encoding, hard) + cost[self.states[i].backbranch1]

                if not self.availableNodes[self.states[i].backbranch1]:
                    newcost[i] = distance0
                elif not self.availableNodes[self.states[i].backbranch0]:
                    newcost[i] = distance1

                elif distance0 <= distance1:
                    newcost[i] = distance0
                else:
                    newcost[i] = distance1

        self.availableNodes = nextAvailableNodes
        return newcost

    def viterbi_Decoder(self, c: list[int], cost: list[int], time: int) -> int:
        current_state = self.states[self.reverseNode]
        cost_0 = cost[current_state.backbranch0]
        cost_1 = cost[current_state.backbranch1]

        if time < self.memory:
            if not self.availableNodes_ClosingSet[time][self.states[self.reverseNode].backbranch0]:
                cost_0 = 10000
            if not self.availableNodes_ClosingSet[time][self.states[self.reverseNode].backbranch1]:
                cost_1 = 10000

        if cost_0 > cost_1:
            self.reverseNode = self.states[self.reverseNode].backbranch1
            return 1

        else:
            self.reverseNode = self.states[self.reverseNode].backbranch0
            return 0


def digit_to_Value(dig: int) -> list[int]:
    value = [0, 0, 0]
    if dig % 2 > 0:
        value[0] = 1
        dig -= 1
    if dig % 4 > 0:
        value[1] = 1
        dig -= 2
    if dig % 8 > 0:
        value[2] = 1

    return value


def value_to_digit(value: list) -> int:
    return value[0] + value[1] * 2 + value[2] * 4


# Memory State Elements
D = np.zeros(3, dtype=np.int16)


def conv(u, D: list, encoding, Eb=1.0):
    R = (D[1] + D[2]) % 2

    c1 = u
    c2 = (R + u + D[0] + D[1] + D[2]) % 2

    D[2] = D[1]
    D[1] = D[0]
    D[0] = (u + R) % 2

    if encoding == EncodingType.AWGN:
        c1 = Eb if c1 else -Eb
        c2 = Eb if c2 else -Eb

    return [c1, c2]
