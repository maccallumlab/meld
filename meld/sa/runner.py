import numpy as np
from meld import interfaces
from meld.system import meld_system
from meld.system import options

from typing import List, Sequence, Optional


class SimulatedAnnealingRunner:
    """
    class to coordinate running simulated annealing in meld
    """

    @property
    def alphas(self) -> List[float]:
        """
        values of alpha
        """
        return self._alphas

    @property
    def n_alpha_steps(self) -> int:
        """
        number of alpha steps
        """
        return self._n_alpha_steps

    @property
    def n_steps_total(self) -> int:
        """
        total number of steps n_steps_per_alpha * n_alphas
        """
        return self._timesteps_per_alpha * len(self._alphas)

    _alphas: List[float]

    def __init__(
        self,
        n_alpha_steps,
        system: meld_system.System,
        runner: interfaces.IRunner,
        platform: Optional[str] = None,
    ) -> None:
        """
        Initialize a  SimulatedAnnealingRunner

        ARgs:
            system: MELD system created in the normal way
            options: meld options
            platform: CUDA or CPU
            n_alpha_steps: number of steps between alpha = 0 and alpha = 1.0
        """
        self.id = np.random.randint(0x7FFFFFFF)
        self._n_alpha_steps = n_alpha_steps
        self.system = system
        self.platform = platform
        self.runner = runner
        self._setup_alphas()

    def run(
        self,
    ):
        state = self.system.get_state_template()
        self._step = 0

        for alpha1, alpha2 in zip(self._alphas[0:-1], self._alphas[1:]):
            # Prepare system for first alpha and run - keep track of energy
            self.runner.prepare_for_timestep(state, alpha1, self._step)
            state.alpha = alpha1
            state = self.runner.run(state)
            energy1 = self._compute_energy(state)

            # This is a bit of unnecessary overhead because we will redo this at the start of next looop
            # but makes calculating work easier
            # update state with new alpha - keep same mappings/coordinates
            state.alpha = alpha2
            self.runner.prepare_for_timestep(state, alpha2, self._step)

            # Compute energy after alpha value is changed and state is updated.
            energy2 = self._compute_energy(state)
            work = energy2 - energy1

            self._output_energy(alpha1, energy1)
            self._output_work(alpha2, work)

            # If we are on the last loop, we need to run the last alpha value
            # This will not have a work value associated because we there is no next alpha value
            # So we run and compute/save energy
            if alpha2 == self.alphas[-1]:
                state = self.runner.run(state)
                last_energy = self._compute_energy(state)
                self._output_energy(alpha2, last_energy)

        np.save(f"log_mappings_{self.id}.npy", state.mappings)

    def _setup_alphas(self) -> None:
        delta = 1.0 / (self._n_alpha_steps - 1.0)
        self._alphas = [i * delta for i in range(self._n_alpha_steps)]
        self.alphas.reverse()

    def _compute_energy(self, state) -> float:
        return self.runner.get_energy(state)

    def _output_energy(self, alpha, energy) -> None:
        with open(f"log_energy_{self.id}.txt", "a+") as energyfile:
            energyfile.write(f"{alpha} {energy} \n")

    def _output_work(self, alpha, work) -> None:
        with open(f"log_work_{self.id}.txt", "a+") as wfile:
            wfile.write(f"{alpha} {work} \n")
