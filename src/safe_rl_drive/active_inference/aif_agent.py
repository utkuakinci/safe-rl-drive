"""Active-inference agent (pymdp, numpy API) for :mod:`lane_pomdp`.

The agent is given the generative model (A, B, C, D) - no reward, no training samples. It selects
policies by minimising expected free energy G = -(pragmatic value) - (epistemic value), so it
trades off "avoid predicted crash" against "wait for the detector to become informative".
"""

from __future__ import annotations

import numpy as np
from pymdp import utils
from pymdp.agent import Agent

from safe_rl_drive.active_inference.lane_pomdp import D_MAX, DEFAULT_ACCURACY, N_LANES, detector_likelihood


def generative_model(accuracy=DEFAULT_ACCURACY, crash_pref: float = -8.0, pass_pref: float = 2.0):
    ns = [N_LANES, N_LANES, D_MAX + 1]  # ego lane, obstacle lane, distance
    no = [N_LANES, N_LANES, 3]  # ego lane obs, detector, outcome

    A = utils.obj_array(3)
    A[0] = np.zeros((no[0], *ns))
    for e in range(N_LANES):
        A[0][e, e, :, :] = 1.0
    lik = detector_likelihood(accuracy)  # [o, s, d]
    A[1] = np.broadcast_to(lik[:, None, :, :], (no[1], *ns)).copy()
    A[2] = np.zeros((no[2], *ns))
    for e in range(N_LANES):
        for o in range(N_LANES):
            A[2][0, e, o, 1:] = 1.0  # neutral while approaching
            A[2][2 if e == o else 1, e, o, 0] = 1.0  # crash / passed at distance 0

    B = utils.obj_array(3)
    B[0] = np.zeros((N_LANES, N_LANES, 3))
    for e in range(N_LANES):
        for a in range(3):
            B[0][int(np.clip(e + a - 1, 0, N_LANES - 1)), e, a] = 1.0
    B[1] = np.eye(N_LANES)[:, :, None]
    B[2] = np.zeros((D_MAX + 1, D_MAX + 1, 1))
    for d in range(D_MAX + 1):
        B[2][max(d - 1, 0), d, 0] = 1.0

    C = utils.obj_array_zeros(no)
    C[2][:] = [0.0, pass_pref, crash_pref]

    D = utils.obj_array(3)
    D[0] = np.eye(N_LANES)[1]
    D[1] = np.ones(N_LANES) / N_LANES
    D[2] = np.eye(D_MAX + 1)[D_MAX]
    return A, B, C, D


class ActiveInferenceDriver:
    def __init__(self, accuracy=DEFAULT_ACCURACY, policy_len: int = 2, gamma: float = 16.0, seed: int = 0):
        A, B, C, D = generative_model(accuracy)
        self.agent = Agent(
            A=A,
            B=B,
            C=C,
            D=D,
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_states_info_gain=True,
            action_selection="deterministic",
        )
        self.D = D
        np.random.seed(seed)  # noqa: NPY002 - pymdp samples from the legacy global RNG

    def reset(self):
        self.agent.reset()

    def act(self, raw_obs: list[int]) -> int:
        self.agent.infer_states(raw_obs)
        self.agent.infer_policies()
        return int(self.agent.sample_action()[0])

    def belief_obstacle(self) -> np.ndarray:
        return np.asarray(self.agent.qs[1])
