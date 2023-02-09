import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint

from hidden_rust.models_fp_v0 import HiddenDDC


def diff(parameter, true_dynamics, init_believes):
    # parameter is a 1-d array
    # true dynamics is dictionary
    # init_believes: (-1, 2) array
    dynamics = {
        0: parameter[0: 8].reshape(2, 2, 2),
        1: parameter[8: 16].reshape(2, 2, 2),
        2: parameter[16:].reshape(2, 2, 2)
    }
    agent = HiddenDDC(dynamics, reward=np.random.rand(2, 3))
    agent_true = HiddenDDC(true_dynamics, reward=np.random.rand(2, 3))

    def prob(path, init_belief):  # path: array, N * 2 shape
        x = init_belief
        x_true = init_belief
        for cell in path[:-1]:
            x = agent.x_next(x, cell[0], cell[1])
            x_true = agent_true.x_next(x_true, cell[0], cell[1])
        return agent.sigma(x, path[-1, 0]), agent_true.sigma(x_true, path[-1, 0])

    paths = {}
    paths[0] = [[a, z] for a in range(3) for z in range(2)]
    paths[1] = [cell + [a, z] for a in range(3) for z in range(2) for cell in paths[0]]

    paths[0] = np.array(paths[0])
    paths[1] = np.array(paths[1])

    diff = 0
    for belief in init_believes:
        for t in range(2):
            for path in paths[t]:
                path = path.reshape(-1, 2)
                est_sigma, true_sigma = prob(path, belief)
                diff += np.square(est_sigma - true_sigma).sum()

    return diff


def estimation(dynamics, init_believes):

    ###################################################################################
    ###################################################################################

    dyn_res = {}

    """1. estimation for dynamics"""
    count = 0
    while True:
        x0 = np.array([0.25] * 24)
        bounds = Bounds([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # sum to 1 constraint: Ax = b
        A = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ])
        linear_constraint = LinearConstraint(A, lb=[1] * 6, ub=[1] * 6)

        res = minimize(diff,
                       x0=x0,
                       args=(dynamics, init_believes),
                       bounds=bounds,
                       constraints=[linear_constraint],
                       method='SLSQP',
                       options={'disp': 2, 'iprint': 2})

        # # solver COBYLA
        # cons1 = [{'type': 'ineq', 'fun': lambda x: x[i]} for i in range(24)]
        # cons2 = [{'type': 'ineq', 'fun': lambda x: -x[i] + 1} for i in range(24)]
        #
        # cons3 = [
        #     {'type': 'ineq', 'fun': lambda x: sum(x[0:4]) - 1},
        #     {'type': 'ineq', 'fun': lambda x: sum(x[4:8]) - 1},
        #     {'type': 'ineq', 'fun': lambda x: sum(x[8:12]) - 1},
        #     {'type': 'ineq', 'fun': lambda x: sum(x[12:16]) - 1},
        #     {'type': 'ineq', 'fun': lambda x: sum(x[16:20]) - 1},
        #     {'type': 'ineq', 'fun': lambda x: sum(x[20:24]) - 1}
        # ]
        #
        # cons4 = [
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[0:4]) + 1},
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[4:8]) + 1},
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[8:12]) + 1},
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[12:16]) + 1},
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[16:20]) + 1},
        #     {'type': 'ineq', 'fun': lambda x: -sum(x[20:24]) + 1}
        # ]
        #
        # res = minimize(dynamic_initializer,
        #                x0=x0,
        #                args=(data_clip, init_believes),
        #                constraints=[*cons1, *cons2, *cons3, *cons4],
        #                method='COBYLA',
        #                options={'disp': 3})

        if res.success or count >= 3:
            if count >= 3:
                dyn_res['msg_dynamics'] = 'failed'  # todo: should report res.message if unsuccessful
            else:
                dyn_res['msg_dynamics'] = 'succeeded'
            break

        count += 1

    dyn_res['dynamics'] = res.x
    dyn_res['dynamic_ll'] = res.fun

    return dyn_res


if __name__ == '__main__':
    """
    this file tests the identifiability fit the model with 2 period under true parameters
    to see if estimation converges with more and more data
    
    maximum difference can converge to less than 1e-6
    
    """

    reward = np.array([
        [10, 6, 3],
        [3, 5, 7]
    ])
    transition = np.array([
        [
            [0.8, 0.2],
            [0, 1]
        ],
        [
            [0.9, 0.1],
            [0, 1]
        ],
        [
            [1, 0],
            [0.4, 0.6]
        ]
    ])
    observation = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    dynamics = {}
    for a in range(3):
        l = [transition[a, s, ss] * observation[ss, z] for s in range(2) for ss in range(2) for z in range(2)]
        arr = np.array(l).reshape((2, 2, 2))
        dynamics[a] = arr

    init_believes = []
    for i in range(10):
        x0 = np.random.rand()
        init_believes.append([x0, 1 - x0])
    init_believes = np.array(init_believes)

    t0 = time.time()
    res = estimation(dynamics, init_believes)
    elapsed_time = time.time() - t0

    true_dynamics = np.concatenate([dynamics[0].flatten(), dynamics[1].flatten(),
                                    dynamics[2].flatten()])
    max_diff = abs(res['dynamics'] - true_dynamics).max()
    mean_diff = abs(res['dynamics'] - true_dynamics).mean()

    print(f'max_diff: {max_diff}, mean_diff: {mean_diff}')
