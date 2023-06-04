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

# We also define a time symbol, t, but its not necessary
# Its good to pass assumptions of symbols like positive, or commutative,
# helps with the sympy methods
g, delta, w_c, w_q, t, anharm, chi, kerr, ratio = symbols(
    "g Delta omega_c omega_q t alpha chi K E_C/E_J",
    positive=True,
    commutative=True,
)

# This is a list of substitutions we make for the final form we want the Hamiltonian in
# Notice that the second substitution is defined partially in terms of the first one
# This is since all substitutions are made sequentially in sympy
simplify_subs_list = [
    (anharm * g**2 / delta**2, chi / 2),
    (chi * g**2 / delta**2, kerr),
]

# List of substitutions with rough values of all parameters (including the approximate maximum values
# of operators) so that we can neglect very small terms
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

# The Hamiltonian Terms we evaluate
# Notice that all operators are in their respective rotating frames
# This is necessary if you want to use the RWA feature
non_linear_ham = (
    -anharm
    / 12
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        + g / delta * c * exp(-I * w_c * t)
        + g / delta * cd * exp(I * w_c * t)
    )
    ** 4
    + anharm
    * sqrt(ratio)
    / 360
    * (
        q * exp(-I * w_q * t)
        + qd * exp(I * w_q * t)
        + g / delta * c * exp(-I * w_c * t)
        + g / delta * cd * exp(I * w_c * t)
    )
    ** 6
)

# Defining the canonical commutation relations of all operators
# For now you have to list all possible combinations
# I might use a Ladder Operator class in the future to simplify this
ccr_c_cd = Eq(Commutator(c, cd), 1)
ccr_c_q = Eq(Commutator(c, q), 0)
ccr_c_qd = Eq(Commutator(c, qd), 0)
ccr_cd_q = Eq(Commutator(cd, q), 0)
ccr_cd_qd = Eq(Commutator(cd, qd), 0)
ccr_q_qd = Eq(Commutator(q, qd), 1)

ccr_list = (ccr_c_cd, ccr_c_q, ccr_c_qd, ccr_cd_q, ccr_cd_qd, ccr_q_qd)

# Expands all terms, this way we can rearrange terms and normally order when possible
res_expanded = non_linear_ham.expand()

# A custom method I made, applies the canonical commutation relations we specified whenever possible
# And it implicitly normally orders all terms
res_with_ccr = res_expanded.evaluate_with_ccr(ccr_list)

# A sympy function, combines exponential powers when possible to simplify terms
res_combined = powsimp(res_with_ccr)

# This is what the full line would look like
# res = powsimp(non_linear_ham.expand().evaluate_with_ccr(ccr_list))

# Applies a quantum_transformations method I made
# passing only_op_terms=True means only terms with operators are kept
# passing the rwa_freq means the RWA will be applied
# rwa_args are necessary if rwa_freq is passed,
#   provide all the rotating frequencies of terms involved
# time_symbol needs to be passed if you're using one in the expression
transformed = powsimp(
    res_combined.quantum_transformations(
        only_op_terms=True,
        rwa_freq=RWA_MAX_FREQ,
        rwa_args={w_c: W_C, w_q: W_Q},
        time_symbol=t,
    )
)

# Evaluating the commutators, and then simplifying the expressions again
# Useful for the Master Equations
# Notice that commutators have an additional doit() method which explictly expands the commutator
com_term_c = powsimp(
    -I * Commutator(c, transformed).doit().expand().evaluate_with_ccr(ccr_list)
)

com_term_q = powsimp(
    -I * Commutator(q, transformed).doit().expand().evaluate_with_ccr(ccr_list)
)

# A custom printing method I made
# You can pass the name of the expression you're printing
# The optional subs_list will replace symbols whenever possible according to the subs_list
# The optional full_subs_list is required to evaluate and neglect any small terms
# The cutoff_val specifies the smallest magnitude of a term before its removed
# An extra latex version of each final expression is provided for any writeups!
custom_printer(
    transformed,
    name="Hamiltonian",
    subs_list=simplify_subs_list,
)
custom_printer(
    com_term_c,
    name="Commutator with respect to c",
    subs_list=simplify_subs_list,
    full_subs_list=small_subs_list,
    cutoff_val=1e-3,
)
custom_printer(
    com_term_q,
    name="Commutator with respect to q",
    subs_list=simplify_subs_list,
    full_subs_list=small_subs_list,
    cutoff_val=1e-4,
)
