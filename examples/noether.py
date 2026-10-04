"""Periodic flux examples from pmocz/noether; see docs/noether.md for provenance."""
import jax.numpy as jnp


def advect(u, c):
    flux = c * u
    return u - (flux - jnp.roll(flux, 1))


def burgers(u, lam):
    flux = 0.5 * u**2
    return u - lam * (flux - jnp.roll(flux, 1))
