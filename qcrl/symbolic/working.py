import numpy as np

from sympy import *
from sympy.physics.quantum import Commutator, Operator, Dagger

# Custom Functions
import general_canonical  # Method, evaluate_with_ccr
import utils  # Method, quantum_transformations, neglect_small_terms
from utils import custom_printer

# Args, all values in MHz
W_C = 7062.0 + 102.9**2 / (7062 - 5092)  # Dressed Resonator Freq
W_Q = 5092.0 - 102.9**2 / (7062 - 5092)  # Dressed Transmon Freq
DELTA = 7062.0 - 5092.0  # Bare Delta
RWA_MAX_FREQ = DELTA  # The maximum frequency for a term we'd keep under RWA
MAX_PHOTONS_RES = 7.0  # Maximum Res Photon Population we'd reasonably have
MAX_PHOTONS_TRANS = 2.0  # Maximum Transmon Photon Population we'd reasonably have

ANHARM = 314.0  # First Anharmonicity Approximately
RATIO = 1 / 62.5  # E_C/E_J
KERR = 0.00467
CHI = 1.712
G = 102.9

# Operators and Sympy Variables
c = Operator("c")
cd = Dagger(c)

q = Operator("q")
qd = Dagger(q)

g, delta, w_c, w_q, t, anharm, chi, kerr, ratio = symbols(
    "g Delta omega_c omega_q t alpha chi K E_C/E_J",
    positive=True,
    commutative=True,
)

simplify_subs_list = [
    (anharm * g**2 / delta**2, chi / 2),
    (chi * g**2 / delta**2, 2 * kerr),
]

small_subs_list = [
    (anharm, ANHARM),
    (kerr, KERR),
    (chi, CHI),
    (g, G),
    (delta, DELTA),
    (c, np.sqrt(MAX_PHOTONS_RES)),
    (q, np.sqrt(MAX_PHOTONS_TRANS)),
    (ratio, RATIO),
]

non_linear_ham = (
    -anharm
    / 12
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        # + g / delta * c * exp(-I * w_c * t)
        # + g / delta * cd * exp(I * w_c * t)
    )
    ** 4
    + anharm
    * sqrt(2 * ratio)
    / 360
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        # + g / delta * c * exp(-I * w_c * t)
        # + g / delta * cd * exp(I * w_c * t)
    )
    ** 6
)

ccr_c_cd = Eq(Commutator(c, cd), 1)
ccr_c_q = Eq(Commutator(c, q), 0)
ccr_c_qd = Eq(Commutator(c, qd), 0)
ccr_cd_q = Eq(Commutator(cd, q), 0)
ccr_cd_qd = Eq(Commutator(cd, qd), 0)
ccr_q_qd = Eq(Commutator(q, qd), 1)

ccr_list = (ccr_c_cd, ccr_c_q, ccr_c_qd, ccr_cd_q, ccr_cd_qd, ccr_q_qd)

res = powsimp(non_linear_ham.expand().evaluate_with_ccr(ccr_list))


transformed = powsimp(
    res.quantum_transformations(
        only_op_terms=True,
        rwa_freq=RWA_MAX_FREQ,
        rwa_args={w_c: W_C, w_q: W_Q},
        time_symbol=t,
    )
)

com_term_c = powsimp(
    -I * Commutator(c, transformed).doit().expand().evaluate_with_ccr(ccr_list)
)

com_term_q = powsimp(
    -I * Commutator(q, transformed).doit().expand().evaluate_with_ccr(ccr_list)
)

custom_printer(
    transformed,
    name="Hamiltonian",
    subs_list=simplify_subs_list,
)

"""
custom_printer(
    com_term_c,
    name="Commutator with respect to c",
    subs_list=simplify_subs_list,
    full_subs_list=small_subs_list,
    cutoff_val=1e-3,
)
"""

custom_printer(
    com_term_q,
    name="Commutator with respect to q",
    subs_list=simplify_subs_list,
    full_subs_list=small_subs_list,
    cutoff_val=1e-4,
)
