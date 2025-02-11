"""Stepping controller for the simulation."""

from enum import Enum, auto
from typing import Protocol


class StepMode(Enum):
    """Defines how the simulation stepping should behave."""

    CONTINUOUS = auto()  # Run continuously in real-time
    MANUAL = auto()  # Only step when explicitly called


class Steppable(Protocol):
    """Protocol for objects that can be stepped."""

    async def step(self) -> None: ...


class StepController:
    """Controls the stepping behavior of a simulation."""

    def __init__(self, steppable: Steppable, mode: StepMode = StepMode.CONTINUOUS) -> None:
        self.steppable = steppable
        self.mode = mode
        self._paused = False
        self._step_request = False
        self._num_steps = 0

    @property
    def paused(self) -> bool:
        return self._paused

    async def set_paused(self, paused: bool) -> None:
        """Pause or unpause the simulation."""
        self._paused = paused

    async def request_steps(self, num_steps: int = 1) -> None:
        """Request a number of steps to be taken."""
        self._step_request = True
        self._num_steps = num_steps

    async def should_step(self) -> bool:
        """Check if a step should be taken."""
        if self.mode == StepMode.CONTINUOUS:
            return not self._paused

        if self._step_request and self._num_steps > 0:
            self._num_steps -= 1
            if self._num_steps == 0:
                self._step_request = False
            return True

        return False
