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

        # Evaluate rollouts and get the best one
        self.curr_rollout = self.eval_seq(cloned_state, rollout, simulator)
        action = self.curr_rollout[0]
        return action, self.curr_rollout

    def eval_seq(self, init_state, rollout, simulator):
        cloned_env = copy.deepcopy(simulator)
        highest_reward = float("-inf")
        best_rollout = None
        for candidate in range(self.candidates):
            if candidate == 0:
                mutated_rollout = rollout
            else:
                mutated_rollout = self.mutate(np.copy(rollout))

            # (Re)init simulated environment
            env = copy.deepcopy(cloned_env)
            env.world_state = copy.deepcopy(init_state)

            rollout_reward = 0
            for step in range(self.horizon):
                action = mutated_rollout[step] #1
                state, reward, done, _ = env.step(action)
                rollout_reward += reward
                if done:
                    break
            if rollout_reward > highest_reward:
                best_rollout = np.copy(mutated_rollout)
                highest_reward = rollout_reward
        return best_rollout

    def mutate(self, rollout):
        # Determine how many elements to sample from the rollout
        n = np.random.binomial(len(rollout), self.mutation_rate)
        # Generate indices of the sequence to be mutated
        idx = d = np.unique(np.random.randint(0, len(rollout), size=n))
        # Sample new actions
        new_actions = np.random.randint(self.num_actions, size=len(d))
        # Substitute values and obtain mutated sequence
        rollout[idx] = new_actions
        return rollout

    def shift_buffer(self, rollout):
        # append new random action at the end
        sb_rollout = np.append(rollout, np.random.randint(self.num_actions))
        # remove first action
        sb_rollout = np.delete(sb_rollout, 0)
        return sb_rollout
