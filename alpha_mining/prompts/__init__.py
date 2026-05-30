"""Prompts for agents."""

from alpha_mining.prompts.leader_prompts import LEADER_SYSTEM_PROMPT, LEADER_ITERATION_PROMPT
from alpha_mining.prompts.proposer_prompts import PROPOSER_SYSTEM_PROMPT, PROPOSER_USER_PROMPT
from alpha_mining.prompts.critic_prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT

__all__ = [
    "LEADER_SYSTEM_PROMPT",
    "LEADER_ITERATION_PROMPT",
    "PROPOSER_SYSTEM_PROMPT",
    "PROPOSER_USER_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
    "CRITIC_USER_PROMPT",
]
