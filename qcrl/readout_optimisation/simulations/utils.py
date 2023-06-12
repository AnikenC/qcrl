import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

KAPPA = 5.35 * 2 * jnp.pi
CHI = 0.16 * KAPPA * 2.0  # extra 2. factor in case of alternate cross-Kerr definition
WR = 7062.0 * 2 * jnp.pi
WQ = 5092.0 * 2 * jnp.pi
DELTA = WR - WQ
G = 102.9 * 2 * jnp.pi
GAMMA = 1 / 39.2 * 2 * jnp.pi
ANHARM_1 = CHI * DELTA**2 / (2 * G**2)
ANHARM_2 = CHI * DELTA**2 / (2 * G**2 + CHI * DELTA)
KERR = CHI * G**2 / DELTA**2 / 2  # 0.00467 * 2 * pi MHz
RATIO = 1 / 62.5  # E_C/E_J


def get_params():
    return {
        "kappa": KAPPA,
        "chi": CHI,
        "wr": WR,
        "wq": WQ,
        "delta": DELTA,
        "g": G,
        "gamma": GAMMA,
        "anharm": ANHARM_1,
        "kerr": KERR,
        "ratio": RATIO,
    }


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
    ax[0, 0].plot(ts, jnp.absolute(complex_1), label=f"{name_1} photons", color="red")
    ax[0, 0].legend()

    ax[1, 0].plot(
        complex_1.real, complex_1.imag, label=f"{name_1} phase", color="orange"
    )
    ax[1, 0].legend()

    ax[2, 0].plot(ts, complex_1.real, label=f"{name_1} real", color="blue")
    ax[2, 0].legend()

    ax[3, 0].plot(ts, complex_1.imag, label=f"{name_1} imag", color="green")
    ax[3, 0].legend()

    ax[0, 1].plot(ts, jnp.absolute(complex_2), label=f"{name_2} photons", color="red")
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


def timer_func(
    func, num_reps, name, func_args, block=True, trials_in_each=1, units=1.0
):
    """
    Uses time for timing, and jax
    """
    if units not in {1.0, 1e-3, 1e-6}:
        raise ValueError("only units of 1., 1e-3, and 1e-6 are supported for now")
    time_arr = jnp.zeros(num_reps, dtype=jnp.float32)
    if func_args is None:
        _ = func()
        if block:
            for i in range(num_reps):
                start = time.time()
                res = jax.block_until_ready(func())
                end = time.time()
                time_arr = time_arr.at[i].set(end - start)
        else:
            for i in range(num_reps):
                start = time.time()
                res = func()
                end = time.time()
                time_arr = time_arr.at[i].set(end - start)
    else:
        _ = func(*func_args)
        if block:
            for i in range(num_reps):
                start = time.time()
                res = jax.block_until_ready(func(*func_args))
                end = time.time()
                time_arr = time_arr.at[i].set(end - start)
        else:
            for i in range(num_reps):
                start = time.time()
                res = func(*func_args)
                end = time.time()
                time_arr = time_arr.at[i].set(end - start)
    time_mean = jnp.mean(time_arr)
    time_std = jnp.std(time_arr)
    if units == 1.0:
        symbol = "s"
    if units == 1e-6:
        symbol = "us"
    if units == 1e-3:
        symbol = "ms"
    print(f"time taken for {name} averaged over {num_reps} reps")
    print(f"mean: {time_mean/units}{symbol}")
    print(f"std: {time_std/units}{symbol}")
    print(f"mean per trial: {time_mean/trials_in_each/units}{symbol}")
    return res


"""
print(f"anharm_1: {ANHARM_1 / 2 / jnp.pi}")
print(f"anharm_2: {ANHARM_2 / 2 / jnp.pi}")


chi = 2 * alpha * g^2/delta^2
alpha = chi * delta^2/(2 * g^2)
alpha

chi = 2 * alpha * g^2/delta / (delta - alpha)
chi * delta - chi * alpha = 2 * alpha * g^2/delta
chi * delta = alpha * (2 * g^2/delta + chi)
alpha = chi * delta / (2 * g^2/delta + chi)
alpha = chi * delta^2 / (2 * g^2 + chi * delta)

"""
