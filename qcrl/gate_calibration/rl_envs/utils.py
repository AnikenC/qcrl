from qiskit.opflow import H, I, X, S
from qiskit.extensions import CXGate

from qiskit.circuit import ParameterVector, QuantumCircuit

import numpy as np

from typing import Tuple, List, Union, Dict


def circuit_func():
    # Target gate: CNOT
    circuit_Plus_i = S @ H
    circuit_Minus_i = S @ H @ X
    cnot_target = {
        "target_type": "gate",
        "gate": CXGate("CNOT"),
        "register": [0, 1],
        "input_states": [
            {
                "name": "|00>",  # Drawn from Ref [21] of PhysRevLett.93.080502
                "circuit": I ^ 2,
            },
            {
                "name": "|01>",
                "circuit": X ^ I,
            },
            {
                "name": "|10>",
                "circuit": I ^ X,
            },
            {
                "name": "|11>",
                "circuit": X ^ X,
            },
            {
                "name": "|+_1>",
                "circuit": X ^ H,
            },
            {
                "name": "|0_->",
                "circuit": (H @ X) ^ I,
            },
            {
                "name": "|+_->",
                "circuit": (H @ X) ^ H,
            },
            {
                "name": "|1_->",
                "circuit": (H @ X) ^ X,
            },
            {
                "name": "|+_0>",
                "circuit": I ^ H,
            },
            {
                "name": "|0_->",
                "circuit": (H @ X) ^ I,
            },
            {
                "name": "|i_0>",
                "circuit": I ^ circuit_Plus_i,
            },
            {
                "name": "|i_1>",
                "circuit": X ^ circuit_Plus_i,
            },
            {
                "name": "|0_i>",
                "circuit": circuit_Plus_i ^ I,
            },
            {
                "name": "|i_i>",
                "circuit": circuit_Plus_i ^ circuit_Plus_i,
            },
            {
                "name": "|i_->",
                "circuit": (H @ X) ^ circuit_Plus_i,
            },
            {
                "name": "|+_i->",
                "circuit": circuit_Minus_i ^ H,
            },
        ],
    }
    return cnot_target


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """

    n_actions = 7
    params = ParameterVector("theta", n_actions)
    qc.u(np.pi * params[0], np.pi * params[1], np.pi * params[2], 0)
    qc.u(np.pi * params[3], np.pi * params[4], np.pi * params[5], 1)
    qc.rzx(np.pi * params[6], 0, 1)
