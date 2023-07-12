import pytest
import jax.numpy as jnp
from diffmpm.material import Bingham
from diffmpm.particle import Particles
from diffmpm.element import Quadrilateral4Node
from diffmpm.constraint import Constraint
from diffmpm.node import Nodes

material_dstrain_particles_state_vars_element_targets_dt = [
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
        ),
        None,
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
        {"pressure": jnp.zeros(1)},
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1)),
        1.0,
    ),
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
        ),
        None,
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
        {"pressure": jnp.zeros(1)},
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
        1.0,
    ),
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
        ),
        None,
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
        {"pressure": jnp.zeros(1)},
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
        1.0,
    ),
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
        ),
        None,
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
        {"pressure": jnp.zeros(1)},
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
        1.0,
    ),
]


@pytest.mark.parametrize(
    "material, dstrain, particles, state_vars, element, target, dt",
    material_dstrain_particles_state_vars_element_targets_dt,
)
def test_compute_stress(material, dstrain, particles, state_vars, element, target, dt):
    if element.constraints != []:
        element.apply_boundary_constraints()
    particles.compute_strain(element, dt)
    stress = material.compute_stress(dstrain, particles, state_vars)
    assert jnp.allclose(stress, target)
