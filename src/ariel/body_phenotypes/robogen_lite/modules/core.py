"""TODO(jmdm): description of script."""
from functools import lru_cache

# Third-party libraries
import mujoco
import numpy as np
import quaternion as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleType,
    printable,
)
from ariel.body_phenotypes.robogen_lite.modules.module import Module

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- Robogen Configuration --- #
# Module weights (kg)
CORE_MASS: WeightType = 1.363

# Module dimensions (length, width, height) in meters
CORE_DIMENSIONS: DimensionType = (0.075, 0.075, 0.075)
# ------------------------------ #


class CoreModule(Module):
    """Core module specifications."""

    module_type: ModuleType = ModuleType.CORE

    def __init__(self) -> None:
        super().__init__()

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Core =========
        core_name = self.module_type.name.lower()
        core = spec.worldbody.add_body(
            name=core_name,
        )
        core.add_geom(
            name=core_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=CORE_MASS,
            size=CORE_DIMENSIONS,
            # pos=[0, CORE_DIMENSIONS[0], 0],
            pos=[0, 0, 0],
            rgba=(253 / 255, 202 / 255, 64 / 255, 1),
        )

        core.add_camera(
            name=f"{core_name}_mycamera",
            pos=[0, 0, CORE_DIMENSIONS[0]-0.02],
            euler=[-90, 0, 180],
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = core.add_site(
            name=f"{core_name}-front",
            pos=[CORE_DIMENSIONS[0], 0, -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.BACK] = self.add_site(
            core,
            name=f"{core_name}-back",
            pos=[-CORE_DIMENSIONS[0], 0, -CORE_DIMENSIONS[1] / 2],

            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            -np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = self.add_site(
            core,
            name=f"{core_name}-left",
            pos=[0, CORE_DIMENSIONS[1], -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(180),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.RIGHT] = self.add_site(
            core,
            name=f"{core_name}-right",
            pos=[0, -CORE_DIMENSIONS[1], -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            np.deg2rad(0),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),

        )

        if not printable:
            self.sites[ModuleFaces.TOP] = self.add_site(
                core,
                name=f"{core_name}-top",
                pos=[0, 0, CORE_DIMENSIONS[2]],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(180),
                                np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )
            self.sites[ModuleFaces.BOTTOM] = self.add_site(
                core,
                name=f"{core_name}-bottom",
                pos=[0, 0, -CORE_DIMENSIONS[2]],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(0),
                                -np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )

        # Save model specifications
        self.spec = spec

    def rotate(self, angle: float) -> None:
        """
        Rotate the core module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the core.

        Raises
        ------
        AttributeError
            Core module does not support rotation.
        """
        if angle != 0:
            msg = f"Attempted to rotate the core module by: {angle}."
            msg += f"Core ({self.index}) module does not support rotation."
            raise AttributeError(msg)

    @property
    @lru_cache
    def hinges(self):
        return self.spec.actuators
