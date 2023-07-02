from jax.tree_util import register_pytree_node_class
import abc
import jax.numpy as jnp


class Material(abc.ABC):
    """Base material class."""

    _props = ()

    def __init__(self, material_properties):
        """
        Initialize material properties.

        Arguments
        ---------
        material_properties: dict
            A key-value map for various material properties.
        """
        self.properties = material_properties

    # @abc.abstractmethod
    def tree_flatten(self):
        """Flatten this class as PyTree Node."""
        return (tuple(), self.properties)

    # @abc.abstractmethod
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten this class as PyTree Node."""
        del children
        return cls(aux_data)

    @abc.abstractmethod
    def __repr__(self):
        """Repr for Material class."""
        ...

    @abc.abstractmethod
    def compute_stress(self):
        """Compute stress for the material."""
        ...

    def validate_props(self, material_properties):
        for key in self._props:
            if key not in material_properties:
                raise KeyError(
                    f"'{key}' should be present in `material_properties` "
                    f"for {self.__class__.__name__} materials."
                )


@register_pytree_node_class
class LinearElastic(Material):
    """Linear Elastic Material."""

    _props = ("density", "youngs_modulus", "poisson_ratio")

    def __init__(self, material_properties):
        """
        Create a Linear Elastic material.

        Arguments
        ---------
        material_properties: dict
            Dictionary with material properties. For linear elastic
        materials, 'density' and 'youngs_modulus' are required keys.
        """
        self.validate_props(material_properties)
        youngs_modulus = material_properties["youngs_modulus"]
        poisson_ratio = material_properties["poisson_ratio"]
        density = material_properties["density"]
        bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
        constrained_modulus = (
            youngs_modulus
            * (1 - poisson_ratio)
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
        # Wave velocities
        vp = jnp.sqrt(constrained_modulus / density)
        vs = jnp.sqrt(shear_modulus / density)
        self.properties = {
            **material_properties,
            "bulk_modulus": bulk_modulus,
            "pwave_velocity": vp,
            "swave_velocity": vs,
        }
        self._compute_elastic_tensor()

    def __repr__(self):
        return f"LinearElastic(props={self.properties})"

    def _compute_elastic_tensor(self):
        G = self.properties["youngs_modulus"] / (
            2 * (1 + self.properties["poisson_ratio"])
        )

        a1 = self.properties["bulk_modulus"] + (4 * G / 3)
        a2 = self.properties["bulk_modulus"] - (2 * G / 3)

        self.de = jnp.array(
            [
                [a1, a2, a2, 0, 0, 0],
                [a2, a1, a2, 0, 0, 0],
                [a2, a2, a1, 0, 0, 0],
                [0, 0, 0, G, 0, 0],
                [0, 0, 0, 0, G, 0],
                [0, 0, 0, 0, 0, G],
            ]
        )

    def compute_stress(self, dstrain):
        """
        Compute material stress.
        """
        dstress = self.de @ dstrain
        return dstress


@register_pytree_node_class
class SimpleMaterial(Material):
    _props = ("E", "density")

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        self.properties = material_properties

    def __repr__(self):
        return f"SimpleMaterial(props={self.properties})"

    def compute_stress(self, dstrain):
        return dstrain * self.properties["E"]


# commenting to create bingham model
@register_pytree_node_class
class Bingham(Material):
    _props = (
        "density",
        "youngs_modulus",
        "poisson_ratio",
        "tau0",
        "mu_",
        "critical_shear_rate",
    )

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        density = material_properties["density"]
        youngs_modulus = material_properties["youngs_modulus"]
        poisson_ratio = material_properties["poisson_ratio"]
        tau0 = material_properties["tau0"]
        mu_ = material_properties["mu_"]
        critical_shear_rate = material_properties["critical_shear_rate"]
        bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
        compressibility_multiplier_ = 1.0
        if "incompressible" in material_properties.keys():
            incompressible = material_properties["incompressible"]
            if incompressible:
                compressibility_multiplier_ = 0.0
        self.properties = {
            **material_properties,
            "bulk_modulus": bulk_modulus,
            "compressibility_multiplier": compressibility_multiplier_,
        }

    def __repr__(self):
        return f"Bingham(props={self.properties})"

    def thermodynamic_pressure(self, volumetric_strain):
        return -self.properties["bulk_modulus"] * volumetric_strain

    def compute_stress(self, dstrain, particle):
        def compute_stress_per_particle(i):
            strain_rate = particle.strain_rate[i]
            strain_rate = strain_rate.at[-3:].set(strain_rate[-3:] * 0.5)
            strain_rate_threshold = 1e-15
            if self.critical_shear_rate < shear_rate_threshold:
                self.critical_shear_rate = shear_rate_threshold
            shear_rate = jnp.sqrt(
                2.0
                * (
                    strain_rate.T @ (strain_rate)
                    + strain_rate[-3].T @ strain_rate[-3, jnp.newaxis]
                )
            )
            apparent_viscosity = 0.0
            if shear_rate[0, 0] * shear_rate[0, 0] > self.critical_shear_rate:
                apparent_viscosity = 2.0 * ((self.tau0 / shear_rate[0, 0]))
            tau = apparent_viscosity * strain_rate
            trace_invariant2 = 0.5 * jnp.dot(tau[:, 0], tau[:, 0])
            if trace_invariant2 < (self.tau0 * self.tau0):
                tau = tau.at[:].set(0)

        strain_rate = particle.strain_rate
        # Convert strain rate to rate of deformation tensor
        strain_rate[:, -3:, :] *= 0.5
        shear_rate_threshold = 1e-15
        if self.critical_shear_rate < shear_rate_threshold:
            self.critical_shear_rate = shear_rate_threshold
        shear_rate = jnp.sqrt(
            strain_rate @ (strain_rate.transpose(0, 2, 1))
            + (strain_rate[:, -3, :] @ (strain_rate[:, -3, :].transpose(0, 2, 1)))
        )


if __name__ == "__main__":
    from diffmpm.utils import _show_example

    _show_example(SimpleMaterial({"E": 2, "density": 1}))
