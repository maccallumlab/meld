"""
Add RDC alignments to the system
"""

import math

import openmm as mm  # type: ignore
from openmm import unit as u

GAS_CONSTANT = 8.314e-3


class CustomRDCIntegrator:
    _temperature: float
    _stepsize: int
    _num_alignments: int
    _alignment_mass: float
    _wrapped_integrator: mm.CustomIntegrator

    def __init__(
        self,
        temperature,
        friction_coefficient,
        step_size,
        num_alignments,
        alignment_mass=1e4,
    ):
        self._temperature = self._handle_temp(temperature)

        if isinstance(friction_coefficient, u.Quantity):
            friction_coefficient = friction_coefficient.value_in_unit(
                (1 / u.picoseconds).unit
            )

        assert friction_coefficient > 0
        self._friction_coefficient = friction_coefficient

        if isinstance(step_size, u.Quantity):
            step_size = step_size.value_in_unit(u.picosecond)
        assert step_size > 0

        assert num_alignments > 0
        assert alignment_mass > 0
        if isinstance(step_size, u.Quantity):
            step_size = step_size.value_in_unit(u.picosecond)
        self._step_size = step_size
        self._num_alignments = num_alignments
        self._alignment_mass = alignment_mass
        self._wrapped_integrator = self._create_custom_integrator()

    def getTemperature(self):
        return self._temperature * u.kelvin

    def setTemperature(self, temp):
        self._temperature = self._handle_temp(temp)
        self._wrapped_integrator.setGlobalVariableByName(
            "kT", self._temperature * GAS_CONSTANT
        )

    @property
    def num_alignments(self):
        return self._num_alignments

    def step(self, steps):
        self._wrapped_integrator.step(steps)

    def _create_custom_integrator(self):
        #
        # Setup variables
        #
        integrator = mm.CustomIntegrator(self._step_size)

        integrator.addGlobalVariable(
            "a", math.exp(-self._friction_coefficient * self._step_size)
        )
        integrator.addGlobalVariable(
            "b",
            math.sqrt(1 - math.exp(-2 * self._friction_coefficient * self._step_size)),
        )
        integrator.addGlobalVariable("kT", GAS_CONSTANT * self._temperature)
        # Add in the alignment tensor components
        for i in range(self._num_alignments):
            for j in range(5):
                integrator.addGlobalVariable(f"rdc_{i}_s{j + 1}", 0.0)
                integrator.addGlobalVariable(f"rdc_{i}_s{j + 1}_vel", 0.0)
        integrator.addPerDofVariable("x1", 0)

        #
        # Steps during integration
        #
        integrator.addUpdateContextState()
        integrator.addComputePerDof("v", "v + dt*f/m")
        for i in range(self._num_alignments):
            for j in range(5):
                integrator.addComputeGlobal(
                    f"rdc_{i}_s{j + 1}_vel",
                    f"rdc_{i}_s{j + 1}_vel - dt * deriv(energy, rdc_{i}_s{j + 1} / {self._alignment_mass}",
                )

        integrator.addConstrainVelocities()

        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        for i in range(self._num_alignments):
            for j in range(5):
                integrator.addComputeGlobal(
                    f"rdc_{i}_s{j + 1}",
                    f"rdc_{i}_s{j + 1} + 0.5*dt*rdc_{i}_s{j + 1}_vel",
                )

        integrator.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        for i in range(self._num_alignments):
            for j in range(5):
                integrator.addComputeGlobal(
                    f"rdc_{i}_s{j + 1}_vel",
                    f"a*rdc_{i}_s{j + 1}_vel + b*sqrt(kT/{self._alignment_mass})*gaussian",
                )

        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        for i in range(self._num_alignments):
            for j in range(5):
                integrator.addComputeGlobal(
                    f"rdc_{i}_s{j + 1}",
                    f"rdc_{i}_s{j + 1} + 0.5*dt*rdc_{i}_s{j + 1}_vel",
                )

        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v + (x-x1)/dt")

        return integrator

    def _handle_temp(self, temp):
        if isinstance(temp, u.Quantity):
            temp = temp.value_in_unit(u.kelvin)
        assert temp > 0
        return temp
