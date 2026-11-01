"""Spark balance curriculum: randomize four-step direction per episode.

All physical controls, rewards, and failure predicates remain inherited from V7.
"""
import torch

from .balance_env import BalanceLegEnv


class SparkBalanceEnv(BalanceLegEnv):
    """Train both LRLR and RLRL without mirroring physical transitions."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not hasattr(self, "spark_start_right"):
            self.spark_start_right = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self.spark_start_right[env_ids] = torch.randint(
            2, (len(env_ids),), device=self.device, dtype=torch.long
        ).bool()

    def _training_schedule(self):
        super()._training_schedule()
        swap = (self.balance_task < 5) & self.spark_start_right
        left = swap & (self.mode == 2)
        right = swap & (self.mode == 3)
        self.mode[left] = 3
        self.mode[right] = 2
