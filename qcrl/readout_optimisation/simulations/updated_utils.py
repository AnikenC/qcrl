import time
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


t_pi = 2.0 * jnp.pi

CHI = 0.16 * 5.35 * t_pi
KAPPA = CHI
G = 102.9 * t_pi
WA = 7062.0 * t_pi
WB = 5092.0 * t_pi
DELTA = WA - WB
GAMMA = 1 / 39.2
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
        'kappa_eff': KAPPA_EFF,
    }

    if print_params:
        print("Physics Params")
        for key, value in physics_params.items():
            print(f"{key}: {value}")

    return physics_params
