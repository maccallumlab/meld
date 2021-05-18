from meld import interfaces


class FakeSystemRunner(interfaces.IRunner):
    """
    Fake runner for test purposes.
    """

    def __init__(
        self,
        system,
        options,
        communicator,
    ) -> None:
        self.temperature_scaler = system.temperature_scaler

    def set_alpha_and_timestep(self, alpha: float, timestep: int) -> None:
        pass

    def minimize_then_run(self, state: interfaces.IState) -> interfaces.IState:
        return state

    def run(self, state: interfaces.IState) -> interfaces.IState:
        return state

    def get_energy(self, state: interfaces.IState) -> float:
        return 0.0

    def prepare_for_timestep(self, alpha: float, timestep: int) -> None:
        pass
