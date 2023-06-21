import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from jax.experimental import sparse


def complex_plotter(ts, complex_1, complex_2, rot_1, rot_2, name_1, name_2, fig_name):
    fig, ax = plt.subplots(4, 2, num=fig_name)
    ax[0, 0].plot(
        ts, jnp.absolute(complex_1) ** 2, label=f"{name_1} photons", color="red"
    )
    ax[0, 0].legend()

    ax[1, 0].plot(
        ts, jnp.absolute(complex_2) ** 2, label=f"{name_2} photons", color="blue"
    )
    ax[1, 0].legend()

    ax[2, 0].plot(
        complex_1.real, complex_1.imag, label=f"{name_1} phase", color="orange"
    )
    ax[2, 0].legend()

    ax[3, 0].plot(
        complex_2.real, complex_2.imag, label=f"{name_2} phase", color="green"
    )
    ax[3, 0].legend()

    ax[0, 1].plot(
        ts, jnp.absolute(rot_1) ** 2, label=f"rot {name_1} photons", color="red"
    )
    ax[0, 1].legend()

    ax[1, 1].plot(
        ts, jnp.absolute(rot_2) ** 2, label=f"rot {name_2} photons", color="blue"
    )
    ax[1, 1].legend()

    ax[2, 1].plot(rot_1.real, rot_1.imag, label=f"rot {name_1} phase", color="orange")
    ax[2, 1].legend()

    ax[3, 1].plot(rot_2.real, rot_2.imag, label=f"rot {name_2} phase", color="green")
    ax[3, 1].legend()

    return fig, ax


def simple_plotter(ts, complex_1, complex_2, name_1, name_2, fig_name):
    fig, ax = plt.subplots(2, 2, num=fig_name)
    ax[0, 0].plot(ts, jnp.absolute(complex_1), label=f"{name_1} photons")
    ax[0, 0].legend()

    ax[1, 0].plot(complex_1.real, complex_1.imag, label=f"{name_1} phase")
    ax[1, 0].legend()

    ax[0, 1].plot(ts, jnp.absolute(complex_2), label=f"{name_2} photons")
    ax[0, 1].legend()

    ax[1, 1].plot(complex_2.real, complex_2.imag, label=f"{name_2} phase")
    ax[1, 1].legend()

    return fig, ax


def simple_plotter_debug(ts, complex_1, complex_2, name_1, name_2, fig_name):
    fig, ax = plt.subplots(4, 2, num=fig_name)
    ax[0, 0].plot(
        ts, jnp.absolute(complex_1) ** 2, label=f"{name_1} photons", color="red"
    )
    ax[0, 0].legend()

    ax[1, 0].plot(
        complex_1.real, complex_1.imag, label=f"{name_1} phase", color="orange"
    )
    ax[1, 0].legend()

    ax[2, 0].plot(ts, complex_1.real, label=f"{name_1} real", color="blue")
    ax[2, 0].legend()

    ax[3, 0].plot(ts, complex_1.imag, label=f"{name_1} imag", color="green")
    ax[3, 0].legend()

    ax[0, 1].plot(
        ts, jnp.absolute(complex_2) ** 2, label=f"{name_2} photons", color="red"
    )
    ax[0, 1].legend()

    ax[1, 1].plot(
        complex_2.real, complex_2.imag, label=f"{name_2} phase", color="orange"
    )
    ax[1, 1].legend()

    ax[2, 1].plot(ts, complex_2.real, label=f"{name_2} real", color="blue")
    ax[2, 1].legend()

    ax[3, 1].plot(ts, complex_2.imag, label=f"{name_2} imag", color="green")
    ax[3, 1].legend()

    return fig, ax


def drive_plotter(ts, complex_drive, fig_name="Complex Drive"):
    fig, ax = plt.subplots(2, num=f"{fig_name}")
    ax[0].plot(ts, complex_drive.real, label="Real", color="red")
    ax[0].legend()

    ax[1].plot(ts, complex_drive.imag, label="Imaginary", color="red")
    ax[1].legend()
    return fig, ax


t_pi = 1.0  # 2.0 * jnp.pi

CHI = 0.16 * 5.35 * t_pi
KAPPA = CHI
G = 102.9 * t_pi
WA = 7062.0 * t_pi
WB = 5092.0 * t_pi
DELTA = WA - WB
GAMMA = 1 / 39.2 * t_pi
KERR = CHI * 0.5 * (G / DELTA) ** 2
ANHARM = 0.5 * CHI / (G / DELTA) ** 2
SQRT_RATIO = jnp.sqrt(2.0 / 62.5)
SIN_ANGLE = G / DELTA
WC = 0.5 * (
    WA
    + WB
    - 1j * KAPPA / 2
    - 1j * GAMMA / 2
    + jnp.sqrt(
        DELTA**2
        + 1j * DELTA * (GAMMA - KAPPA)
        - 0.25 * (KAPPA + GAMMA) ** 2
        - 4 * G**2
        - KAPPA * GAMMA
    )
)
WQ = 0.5 * (
    WA
    + WB
    - 1j * KAPPA / 2
    - 1j * GAMMA / 2
    - jnp.sqrt(
        DELTA**2
        + 1j * DELTA * (GAMMA - KAPPA)
        - 0.25 * (KAPPA + GAMMA) ** 2
        - 4 * G**2
        - KAPPA * GAMMA
    )
)
WC_EFF = WC + 0.5 * SQRT_RATIO * KERR + 0.125 * SQRT_RATIO * CHI - KERR - 0.5 * CHI
WQ_EFF = (
    WQ
    + 0.5 * SQRT_RATIO * KERR
    - 0.25 * SQRT_RATIO * ANHARM
    + 0.25 * SQRT_RATIO * CHI
    - ANHARM
    - CHI / 2
)
KERR_EFF = KERR - 0.5 * SQRT_RATIO * KERR
CHI_EFF = CHI - 0.5 * CHI * SQRT_RATIO - SQRT_RATIO * KERR
DELTA_EFF = WC_EFF.real - WQ_EFF.real + 0.5 * CHI_EFF
KAPPA_EFF = 2 * jnp.abs(WC_EFF.imag)


def get_params(print_params=True):
    physics_params = {
        "kappa": KAPPA,
        "G": G,
        "WA": WA,
        "WB": WB,
        "delta": DELTA,
        "gamma": GAMMA,
        "anharm": ANHARM,
        "sqrt_ratio": SQRT_RATIO,
        "sin_angle": SIN_ANGLE,
        "wc": WC,
        "wq": WQ,
        "wc_eff": WC_EFF,
        "wq_eff": WQ_EFF,
        "kerr": KERR,
        "kerr_eff": KERR_EFF,
        "chi": CHI,
        "chi_eff": CHI_EFF,
        "delta_eff": DELTA_EFF,
        "kappa_eff": KAPPA_EFF,
    }

    if print_params:
        print("Physics Params")
        for key, value in physics_params.items():
            print(f"{key}: {value}")

    return physics_params


def dagger(input):
    return jnp.conjugate(input).T


def state2dm(state: Array):
    """
    Takes an input state (bra or ket) and outputs a Density Matrix formed from the Outer Product.

    Input:
    State -> Jax 2D Array

    Let N be the dimensions of the state, a ket should have shape (N, 1) and a bra should have shape (1, N), where N >= 2
    """
    state_type = "ket"
    if len(state.shape) != 2:
        raise ValueError(
            "Ensure the state is a 2D array with shape (N, 1) for a ket, or (1, N) for a bra"
        )

    if state.shape[0] == 1:
        state_type = "bra"
    elif state.shape[1] == 1:
        state_type = "ket"
    else:
        raise ValueError(
            "At least one of the dimensions of the state must be of length greater than 1"
        )

    if state_type == "ket":
        dm = state * dagger(state)
    if state_type == "bra":
        dm = dagger(state) * state
    return dm


def number_op(N, dtype=jnp.complex64, use_sparse=True):
    """
    Makes a diagonal matrix representation of the number operator
    By default uses jnp.complex64 as the dtype (a jax compatible dtype must be passed)
    By default use_sparse = True, so a sparse Jax array is made. If sparse = False, the dense representation is made instead.
    """
    if not isinstance(N, int):
        raise ValueError("N must be of int type and non-zero")

    out_matrix = jnp.diag(jnp.arange(0, N, dtype=dtype), k=0)
    if use_sparse:
        out_matrix = sparse.BCOO.fromdense(out_matrix)
    return out_matrix


def annihilation_op(N, dtype=jnp.complex64, use_sparse=True):
    """
    Outputs matrix representation of Ladder Operator in Fock Basis
    """
    if not isinstance(N, int):
        raise ValueError("N must be of int type and non-zero")

    out_matrix = jnp.diag(jnp.sqrt(jnp.arange(1, N, dtype=dtype)), k=1)
    if use_sparse:
        out_matrix = sparse.BCOO.fromdense(out_matrix)
    return out_matrix


def creation_op(N, dtype=jnp.complex64, use_sparse=True):
    """
    Outputs matrix representation of Ladder Operator in Fock Basis
    """
    if not isinstance(N, int):
        raise ValueError("N must be of int type and non-zero")

    out_matrix = jnp.diag(jnp.sqrt(jnp.arange(1, N, dtype=dtype)), k=-1)
    if use_sparse:
        out_matrix = sparse.BCOO.fromdense(out_matrix)
    return out_matrix


def q_eye(N, dtype=jnp.complex64, use_sparse=True):
    """
    Outputs matrix representation of Ladder Operator in Fock Basis
    """
    if not isinstance(N, int):
        raise ValueError("N must be of int type and non-zero")

    out_matrix = jnp.identity(N, dtype=dtype)
    if use_sparse:
        out_matrix = sparse.BCOO.fromdense(out_matrix)
    return out_matrix


def handle_diffrax_stats(stats, name="Solution"):
    max_steps = stats["max_steps"][0]
    num_accepted_steps = stats["num_accepted_steps"][0]
    num_rejected_steps = stats["num_rejected_steps"][0]
    num_steps = stats["num_steps"][0]
    print(f"{name} Statistics")
    print(f"max_steps: {max_steps}")
    print(f"num_accepted_steps: {num_accepted_steps}")
    print(f"num_rejected_steps: {num_rejected_steps}")
    print(f"num_steps: {num_steps}")
