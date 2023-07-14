import pytest
import jax.numpy as jnp
from diffmpm.material import Bingham
from diffmpm.particle import Particles
from diffmpm.element import Quadrilateral4Node
from diffmpm.constraint import Constraint
from diffmpm.node import Nodes

particles_element_targets = [
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 771.8,
                        "mu": 0.0451,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1)),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 771.8,
                        "mu": 0.0451,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 0.02)), (0, Constraint(1, 0.03))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array([-52083.3333333333, -52083.3333333333, 0.0, 0.0, 0.0, 0.0]).reshape(
            (6, 1)
        ),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 200.0,
                        "mu": 200.0,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 2.0)), (0, Constraint(1, 3.0))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [-5208520.35574006, -5208613.86694342, 0.0, -233.778008402801, 0.0, 0.0]
        ).reshape((6, 1)),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 200.0,
                        "mu": 200.0,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                        "incompressible": True,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 2.0)), (0, Constraint(1, 3.0))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [-187.0224067222, -280.5336100834, 0.0, -233.778008402801, 0.0, 0.0]
        ).reshape((6, 1)),
    ),
]


@pytest.fixture
def state_vars():
    return {"pressure": jnp.zeros(1)}


@pytest.fixture
def dt():
    return 1.0


@pytest.mark.parametrize(
    "particles, element, target",
    particles_element_targets,
)
def test_compute_stress(particles, state_vars, element, target, dt):
    particles.update_natural_coords(element)
    if element.constraints:
        element.apply_boundary_constraints()
    particles.compute_strain(element, dt)
    stress = particles.material.compute_stress(None, particles, state_vars)
    assert jnp.allclose(stress, target)
