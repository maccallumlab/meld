#!/bin/sh


mpirun -np 4 python test_broadcast_lambda_to_slaves.py
mpirun -np 4 python test_broadcast_states_to_slaves.py
mpirun -np 4 python test_gather_states_from_slaves.py
mpirun -np 4 python test_broadcast_states_for_energy_calc_to_slaves.py
mpirun -np 4 python test_gather_energies_from_slaves.py
