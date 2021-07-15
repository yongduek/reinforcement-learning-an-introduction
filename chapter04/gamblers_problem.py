#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use('Agg')

# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4


def figure_4_3():
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        ar = np.column_stack((actions[1:], np.round(action_returns[1:], 7)))  # round to find the same values
        ar = ar[ar[:,1].argsort()]
        # print(ar)
        br = ar[ar[:,1] == ar[-1,1]]  # the same maximal [(action, action_return)]
        br = br[br[:,0].argsort()]    # sort along the action (stake)
        # print('br: ', br)
        print(f'AT [{state}] {br} ')

        policy[state] = br[-1][0]    # one among equivalent actions, just triangle
        policy[state] = br[(br.shape[0]-1)//2][0]  # the figure is interesting!!
        # policy[state] = br[0][0]    # one among equivalent actions, the same as Figure 4.3

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        # policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
    #
    print('optimal policy: ', policy)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    # plt.scatter(STATES, policy)
    plt.stem(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('../images/figure_4_3.png')
    plt.pause(10)
    plt.close()


if __name__ == '__main__':
    figure_4_3()
