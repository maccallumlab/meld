import numpy as np
from meld.runner.openmm_runner import OpenMMRunner
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

    @property
    def timesteps_per_alpha(self) -> int:
        """
        number of alpha steps
        """
        return self._timesteps_per_alpha

    _alphas: List[float]

    def __init__(
        self,
        n_alpha_steps,
        system: meld_system.System,
        options: options.RunOptions,
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
        self.options = options
        self._timesteps_per_alpha = self.options.timesteps
        self.system = system
        self.platform = platform
        self._initialize_runner()
        self._setup_alphas()

    def run(
        self,
    ):
        state = self.system.get_state_template()

        energy = 0
        for idx, alpha in enumerate(self._alphas):
            state.alpha = alpha
            self._step = 1

            while self._step <= self.timesteps_per_alpha:
                self.runner.prepare_for_timestep(state, alpha, self._step)

                if self._step == 1:
                    energy2 = self._compute_and_output_energy(state)
                    work = self._compute_and_output_work(state, energy, energy2)

                state = self.runner.run(state)
                self._step = self._step + 1

            energy = self._compute_and_output_energy(state)
            if idx < len(self._alphas) - 1:
                work = self._compute_and_output_work(
                    state, self.alphas[idx], self.alphas[idx + 1]
                )

        np.save(f"mappings_{self.id}.npy", state.mappings)

    def _setup_alphas(self) -> None:
        delta = 1.0 / (self._n_alpha_steps - 1.0)
        self._alphas = [i * delta for i in range(self._n_alpha_steps)]
        self.alphas.reverse()

    def _initialize_runner(self) -> None:
        self.runner = OpenMMRunner(
            self.system, options=self.options, communicator=None, platform=self.platform
        )

    def _compute_and_output_energy(self, state) -> float:
        energy = self.runner.get_energy(state)
        with open(f"energy_{self.id}.txt", "a+") as energyfile:
            energyfile.write(f"{state.alpha} {energy} \n")
        return energy

    def _compute_and_output_work(self, state, energy1, energy2) -> float:
        work = energy2 - energy1

        with open(f"work_{self.id}.txt", "a+") as wfile:
            wfile.write(f"{state.alpha} {work} \n")

        return work
