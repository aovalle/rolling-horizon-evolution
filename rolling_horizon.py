'''
29 april 2020
Vanilla RHE
Assumes access to a simulator and thus the existence of a perfect forward model
'''

import numpy as np
import copy

class RollingHorizon():

    def __init__(self, args, num_actions):
        self.horizon = args.plan_horizon
        self.optim_ters = args.optim_iters
        self.candidates = args.n_candidates
        self.mutation_rate = args.mutation_rate
        self.shift_buffer_on = args.shift_buffer_on
        self.num_actions = num_actions
        self.curr_rollout = None

    def select_action(self, simulator, cloned_state=None):

        if self.curr_rollout is None or not self.shift_buffer_on:
            # Generate random rollout
            rollout = np.random.randint(self.num_actions, size=self.horizon)
        else: # if we want to use shift buffer
            rollout = self.shift_buffer(np.copy(self.curr_rollout))

        # Mutate rollout
        rollouts = self.mutate(rollout)

        # Evaluate rollouts and get the best one
        self.curr_rollout = self.eval_seq(cloned_state, rollouts, simulator)
        action = self.curr_rollout[0]
        return action, self.curr_rollout

    def eval_seq(self, init_state, rollouts, simulator):
        cloned_env = copy.deepcopy(simulator)
        highest_reward = float("-inf")
        best_rollout = None
        for rollout in rollouts:

            # (Re)init simulated environment
            env = copy.deepcopy(cloned_env)
            env.world_state = copy.deepcopy(init_state)

            rollout_reward = 0
            for step in range(self.horizon):
                action = rollout[step] #1
                state, reward, done, _ = env.step(action)
                rollout_reward += reward
                if done:
                    break
            if rollout_reward > highest_reward:
                best_rollout = np.copy(rollout)
                highest_reward = rollout_reward
        return best_rollout

    def mutate(self, rollout):
        # Clone sequence
        rollouts = np.tile(rollout, (self.candidates, 1))
        # Generate indices to mutate
        idx = np.random.rand(*rollouts.shape) < self.mutation_rate
        # Generate new actions and place them accordingly
        rollouts[idx] = np.random.randint(self.num_actions, size=len(idx[idx is True]))
        return rollouts

    def shift_buffer(self, rollout):
        # append new random action at the end
        sb_rollout = np.append(rollout, np.random.randint(self.num_actions))
        # remove first action
        sb_rollout = np.delete(sb_rollout, 0)
        return sb_rollout
