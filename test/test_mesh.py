import jax.numpy as jnp
from diffmpm.material import Material
from diffmpm.mesh import Mesh1D


def test_particles_uniform_initialization():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, ppe=3, particle_type="uniform")
    assert jnp.allclose(
        mesh.particles.x, jnp.array([1 / 6, 1 / 2, 5 / 6, 7 / 6, 3 / 2, 11 / 6])
    )


def test_particle_initial_element_mapping():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, ppe=3)
    assert (mesh.particles.element_ids == jnp.array([0, 0, 0, 1, 1, 1])).all()


def test_particle_element_mapping_update():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, ppe=3)
    mesh.particles.x += 1
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([1, 1, 1, -1, -1, -1])
    ).all()
    mesh.particles.x -= 2
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, 0, 0, 0])
    ).all()
    mesh.particles.x -= 1
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, -1, -1, -1])
    ).all()
    mesh.particles.x += 5
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, -1, -1, -1])
    ).all()
