from jax.tree_util import register_pytree_node_class
import abc
import jax.numpy as jnp
from jax import vmap, lax, jit


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


@register_pytree_node_class
class Bingham(Material):
    _props = (
        "density",
        "youngs_modulus",
        "poisson_ratio",
        "tau0",
        "mu",
        "critical_shear_rate",
    )

    # Passing ndim as an extra parameter for the material to work for both 1D and 2D case
    def __init__(self, material_properties, ndim):
        """
        Create a Bingham material model.

        Arguments
        ---------
        material_properties: dict
            Dictionary with material properties. For Bingham
        material, 'density','youngs_modulus','poisson_ratio','tau0','mu'
        and 'critical_shear_rate' are required keys.

        ndim: int
            Dimension of the problem supports 1D and 2D
        """
        self.validate_props(material_properties)
        self.ndim = ndim
        density = material_properties["density"]
        youngs_modulus = material_properties["youngs_modulus"]
        poisson_ratio = material_properties["poisson_ratio"]
        tau0 = material_properties["tau0"]
        mu_ = material_properties["mu"]
        critical_shear_rate = material_properties["critical_shear_rate"]
        # Calculate the bulk modulus
        bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
        compressibility_multiplier_ = 1.0
        # Special Material Properties
        if "incompressible" in material_properties:
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

    # Initialise State Variables
    def initialise_state_variables(particles):
        state_vars = {}
        state_vars["pressure"] = jnp.zeros((particles.loc.shape[0]))
        return state_vars

    # State Variables
    def state_variables():
        return ["pressure"]

    # Compute the pressure
    def thermodynamic_pressure(self, volumetric_strain):
        return -self.properties["bulk_modulus"] * volumetric_strain

    # Compute the stress
    def compute_stress(self, dstrain, particles, state_vars):
        shear_rate_threshold = 1e-15
        dirac_delta = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))
        if self.ndim == 1:
            dirac_delta = dirac_delta.at[0, 0].set(1.0)
        elif self.ndim == 2:
            dirac_delta = dirac_delta.at[0:2, 0].set(1.0)
        if self.properties["critical_shear_rate"] < shear_rate_threshold:
            self.properties["critical_shear_rate"] = shear_rate_threshold
        
        def compute_stress_per_particle(
            particle_strain_rate,
            self,
            state_vars_pressure,
            dvolumetric_strain_per_particle,
            dirac_delta,
        ):
            strain_r = particle_strain_rate
            strain_r = strain_r.at[-3:].multiply(0.5)
            shear_rate = jnp.sqrt(
                2.0 * (strain_r.T @ (strain_r) + strain_r[-3:].T @ strain_r[-3:])
            )
            apparent_velocity_true = 2.0 * (
                (self.properties["tau0"] / shear_rate[0, 0]) + self.properties["mu"]
            )
            condition = (shear_rate[0, 0] * shear_rate[0, 0]) > (
                self.properties["critical_shear_rate"]
                * self.properties["critical_shear_rate"]
            )
            apparent_viscosity = lax.select(condition, apparent_velocity_true, 0.0)
            tau = apparent_viscosity * strain_r
            trace_invariant2 = 0.5 * jnp.dot(tau[:3, 0], tau[:3, 0])
            lax.cond(
                trace_invariant2 < (self.properties["tau0"] * self.properties["tau0"]),
                lambda x: x.at[:].set(0),
                lambda x: x,
                tau,
            )
            state_vars_pressure += self.properties[
                "compressibility_multiplier"
            ] * self.thermodynamic_pressure(dvolumetric_strain_per_particle)
            updated_stress_per_particle = (
                -(state_vars_pressure)
                * dirac_delta
                * self.properties["compressibility_multiplier"]
                + tau
            )
            return updated_stress_per_particle,state_vars_pressure

        updated_stress,state_vars["pressure"] = jit(vmap(compute_stress_per_particle, in_axes=(0, None, 0, 0,None)))(
            particles.strain_rate,
            self,
            state_vars["pressure"],
            particles.dvolumetric_strain,
            dirac_delta,
        )

        return updated_stress


if __name__ == "__main__":
    from diffmpm.utils import _show_example

    _show_example(SimpleMaterial({"E": 2, "density": 1}))
