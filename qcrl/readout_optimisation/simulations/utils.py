import jax.numpy as jnp

KAPPA = 5.35 * 2 * jnp.pi
CHI = 0.16 * KAPPA * 2.0  # extra 2. factor in case of alternate cross-Kerr definition
WR = 7062.0 * 2 * jnp.pi
WQ = 5092.0 * 2 * jnp.pi
DELTA = WR - WQ
G = 102.9 * 2 * jnp.pi
GAMMA = 1 / 39.2 * 2 * jnp.pi
ANHARM_1 = CHI * DELTA**2 / (2 * G**2)
ANHARM_2 = CHI * DELTA**2 / (2 * G**2 + CHI * DELTA)
KERR = CHI * G**2 / DELTA**2  # 0.00467 * 2 * pi MHz


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
    }


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
