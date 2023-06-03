from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    Pauli,
    SparsePauliOp,
    state_fidelity,
    Operator,
    average_gate_fidelity,
)

from qiskit.primitives import Estimator
from qiskit.opflow import Zero

import numpy as np
from itertools import product
from typing import Dict, Union, Optional, List, Tuple
from copy import deepcopy

import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from .utils import apply_parametrized_circuit, circuit_func


class QuantumEnvironment:
    def __init__(self):
        self.estimator = Estimator()
        self._n_qubits = 2
        self.parametrized_circuit_func = apply_parametrized_circuit

        self.Pauli_ops = [
            {"name": "".join(s), "matrix": Pauli("".join(s)).to_matrix()}
            for s in product(["I", "X", "Y", "Z"], repeat=self._n_qubits)
        ]
        self.c_factor = 0.25
        self.d = 2**self._n_qubits  # Dimension of Hilbert space
        self.sampling_Pauli_space = 100
        self.n_shots = 50

        self.target, self.target_type, self.tgt_register = self._define_target(
            circuit_func()
        )
        self.q_register = QuantumRegister(self._n_qubits)

        self.time_step = 0
        self.episode_ended = False

    def _define_target(self, target: Dict):
        for i, input_state in enumerate(target["input_states"]):
            assert ("circuit" or "dm") in input_state, (
                f"input_state {i} does not have a "
                f"DensityMatrix or circuit description"
            )

            gate_op = Operator(target["gate"])
            target["input_states"][i]["target_state"] = {"target_type": "state"}
            target["input_states"][i]["dm"] = DensityMatrix(
                input_state["circuit"] @ (Zero ^ self._n_qubits)
            )
            target["input_states"][i]["target_state"]["dm"] = DensityMatrix(
                gate_op @ input_state["circuit"] @ (Zero ^ self._n_qubits)
            )

            target["input_states"][i]["target_state"] = self.calculate_chi_target_state(
                target["input_states"][i]["target_state"]
            )
        return target, "gate", target["register"]

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: Target state supplemented with appropriate "Chi" key
        """
        target_state["Chi"] = np.array(
            [
                np.trace(
                    np.array(target_state["dm"].to_operator())
                    @ self.Pauli_ops[k]["matrix"]
                ).real
                for k in range(self.d**2)
            ]
        )
        # Real part is taken to convert it in good format,
        # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
        return target_state

    def perform_action(self, actions):
        qc = QuantumCircuit(
            self.q_register
        )  # Reset the QuantumCircuit instance for next iteration
        angles, batch_size = np.array(actions), len(np.array(actions))

        # Pick random input state from the list of possible input states (forming a tomographically complete set)
        index = np.random.randint(len(self.target["input_states"]))
        input_state = self.target["input_states"][index]
        target_state = self.target["input_states"][index][
            "target_state"
        ]  # Deduce target state associated to input
        # Append input state circuit to full quantum circuit for gate calibration
        qc.append(input_state["circuit"].to_instruction(), self.tgt_register)

        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round(
            [
                self.c_factor * target_state["Chi"][p] / (self.d * distribution.prob(p))
                for p in pauli_index
            ],
            5,
        )

        # Figure out which observables to sample
        observables = SparsePauliOp.from_list(
            [
                (self.Pauli_ops[p]["name"], reward_factor[i])
                for i, p in enumerate(pauli_index)
            ]
        )

        # Apply parametrized quantum circuit (action), for benchmarking only
        parametrized_circ = QuantumCircuit(self._n_qubits)
        self.parametrized_circuit_func(parametrized_circ)

        qc_list = [parametrized_circ.bind_parameters(angle_set) for angle_set in angles]
        q_process_list = [Operator(qc) for qc in qc_list]
        avg_fidelity = np.mean(
            [
                average_gate_fidelity(q_process, Operator(self.target["gate"]))
                for q_process in q_process_list
            ]
        )

        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.parametrized_circuit_func(qc)

        # print("\n Sending job to Estimator...")
        job = self.estimator.run(
            circuits=[qc] * batch_size,
            observables=[observables] * batch_size,
            parameter_values=angles,
            shots=self.sampling_Pauli_space * self.n_shots,
        )

        # print("\n Job done")
        reward_table = job.result().values
        return reward_table, avg_fidelity, index
