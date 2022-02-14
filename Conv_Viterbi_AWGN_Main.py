import numpy as np
import matplotlib.pyplot as plt
import random

import Conv_Viterbi as Conv

# Constants
BLOCK_LENGTH = 16000
MEMORY = 3
STATE_COUNT = (2 ** MEMORY)
ITERATIONS = 10
TRIALS = 10

# Plot Data
Eb_data = np.zeros(ITERATIONS, dtype=np.double)
BER_data_soft = np.zeros(ITERATIONS, dtype=np.double)
BER_data_hard = np.zeros(ITERATIONS, dtype=np.double)


# initialize randomizer
random.seed()

for iter in range(ITERATIONS):
    # Declaring uncoded and coded matrixes
    uncoded = np.zeros(BLOCK_LENGTH, dtype=int)
    #uncoded = [1, 1, 0, 0, 1, 0, 1, 0]

    coded = np.zeros((BLOCK_LENGTH, 2), dtype=np.double)
    coded_guess = coded.copy()
    uncoded_guess_soft = uncoded.copy()
    uncoded_guess_hard = uncoded.copy()

    # Randomize uncoded message
    v = 0.5
    for i in range(BLOCK_LENGTH - MEMORY):
        if random.random() < v:
            uncoded[i] = 1

    # encoding uncoded
    Eb = 2.5 + iter * 0.4
    d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
    D = np.zeros((BLOCK_LENGTH, MEMORY), dtype=np.int16)
    for i in range(BLOCK_LENGTH - MEMORY):
        D[i] = d.copy()
        coded[i] = Conv.conv(uncoded[i], d, Conv.EncodingType.AWGN, Eb)

    # Using the last 3 bits to reset state to [ 0, 0, 0 ]
    for i in range(BLOCK_LENGTH - MEMORY, BLOCK_LENGTH):
        D[i] = d.copy()
        resetbit = (d[1] + d[2]) % 2
        uncoded[i] = resetbit
        coded[i] = Conv.conv(resetbit, d, Conv.EncodingType.AWGN, Eb)

    # introducing noise
    BER_soft = np.zeros(TRIALS, dtype=np.double)
    BER_hard = np.zeros(TRIALS, dtype=np.double)
    #p = iter * 0.001 + 0.001
    p = 0.002
    coded_observation = coded.copy()
    for trial in range(TRIALS):

        for i in range(BLOCK_LENGTH):
            for j in range(2):
                coded_observation[i][j] = random.gauss(coded_observation[i][j], 1.0)
        trel_soft = Conv.Trellis(Conv.EncodingType.AWGN, Eb)
        trel_hard = Conv.Trellis(Conv.EncodingType.AWGN, Eb)
        costMatrix_soft = np.zeros((BLOCK_LENGTH, STATE_COUNT), dtype=np.double)
        cost_soft = np.zeros(STATE_COUNT, dtype=np.double)
        costMatrix_hard = np.zeros((BLOCK_LENGTH, STATE_COUNT), dtype=np.double)
        cost_hard = np.zeros(STATE_COUNT, dtype=np.double)
        for i in range(1, BLOCK_LENGTH):
            costMatrix_soft[i] = trel_soft.updateBranchCost(costMatrix_soft[i-1], coded_observation[i-1], i)
            costMatrix_hard[i] = trel_hard.updateBranchCost(costMatrix_hard[i-1], coded_observation[i-1], i, True)

        for i in range(BLOCK_LENGTH - 1, -1, -1):
            uncoded_guess_soft[i] = trel_soft.viterbi_Decoder(coded_observation[i], costMatrix_soft[i], i)
            uncoded_guess_hard[i] = trel_hard.viterbi_Decoder(coded_observation[i], costMatrix_hard[i], i)

        # BER
        misses_soft = 0
        misses_hard = 0
        for i in range(BLOCK_LENGTH):
            if uncoded_guess_soft[i] != uncoded[i]:
                misses_soft += 1
            if uncoded_guess_hard[i] != uncoded[i]:
                misses_hard += 1

        BER_soft[trial] = misses_soft / BLOCK_LENGTH
        BER_hard[trial] = misses_hard / BLOCK_LENGTH

    Eb_data[iter] = Eb
    BER_data_soft[iter] = np.mean(BER_soft)
    BER_data_hard[iter] = np.mean(BER_hard)

# Ploting Chart
plt.title("Viterbi BER for Eb/No")
plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Eb_data, BER_data_soft, color="red", label="Soft Desicion")
plt.plot(Eb_data, BER_data_hard, color="blue", label="Hard Desicion")
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()
