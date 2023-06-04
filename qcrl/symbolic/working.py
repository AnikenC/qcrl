import numpy as np

from sympy import *
from sympy.physics.quantum import Commutator, Operator, Dagger

# Custom Functions
import general_canonical  # Method, evaluate_with_ccr
import utils  # Method, quantum_transformations, neglect_small_terms
from utils import custom_printer

# Args
W_C = 7062.0 + 102.9**2 / (7062 - 5092)
W_Q = 5092.0 - 102.9**2 / (7062 - 5092)
W_2 = W_Q
DELTA = W_C - W_Q
RWA_MAX_FREQ = DELTA
MAX_PHOTONS_RES = 7.0
MAX_PHOTONS_TRANS = 1.0

WA = 7062.0
WB = 5092.0
ALT_MAX_FREQ = WA - WB

ANHARM = 1971.3118648588484 / 2 / np.pi
RATIO = 1 / 62.5
KERR = 0.02934823080752646 / 2 / np.pi
CHI = 10.75681324589145 / 2 / np.pi
G = 646.5397681087794 / 2 / np.pi
DELTA = 12377.875055143788 / 2 / np.pi

# Operators and Sympy Variables
c = Operator("c")
cd = Dagger(c)

q = Operator("q")
qd = Dagger(q)

g, delta, w_c, w_q, w_2, t, anharm, chi, kerr, delta_2, ratio = symbols(
    "g Delta omega_c omega_q omega_2 t alpha chi K Delta_2 E_C/E_J",
    positive=True,
    commutative=True,
)
D_b, xi_2, conj_xi_2, c_dot, q_dot = symbols(
    "D_b xi_2 xi_2c c_dot q_dot", complex=True, commutative=True
)

alt_subs_list = [
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

b = Operator("b")
bd = Dagger(b)

wb = symbols("omega_b", positive=True, commutative=True)

alt_fourth_order = (
    -anharm
    / 12
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        # + c * g / delta * exp(-I * w_c * t)
        # + cd * g / delta * exp(I * w_c * t)
    )
    ** 4
    + anharm
    / 360
    * sqrt(2 * ratio)
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        # + c * g / delta * exp(-I * w_c * t)
        # + cd * g / delta * exp(I * w_c * t)
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

ccr_b_bd = Eq(Commutator(b, bd), 1)

alt_ccr_list = [ccr_b_bd]

res_alt = powsimp(alt_fourth_order.expand().evaluate_with_ccr(ccr_list))
transformed_alt = powsimp(
    res_alt.quantum_transformations(
        only_op_terms=True,
        rwa_freq=RWA_MAX_FREQ,
        rwa_args={w_c: W_C, w_q: W_Q},
        time_symbol=t,
    )
)

com_term_c = powsimp(
    -I * Commutator(c, transformed_alt).doit().expand().evaluate_with_ccr(ccr_list)
)

com_term_q = powsimp(
    -I * Commutator(q, transformed_alt).doit().expand().evaluate_with_ccr(ccr_list)
)

custom_printer(
    transformed_alt,
    name="Alt Hamiltonian",
    subs_list=alt_subs_list,
)
# custom_printer(
#    com_term_c,
#    name="Alt Commutator c",
#    subs_list=alt_subs_list,
#    full_subs_list=small_subs_list,
#    cutoff_val=1e-3,
# )
custom_printer(
    com_term_q,
    name="Alt Commutator q",
    subs_list=alt_subs_list,
    full_subs_list=small_subs_list,
    cutoff_val=1e-4,
)
