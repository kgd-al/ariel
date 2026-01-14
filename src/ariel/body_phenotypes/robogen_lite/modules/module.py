"""TODO(jmdm): description of script.

Todo:
----
    [ ] Autoincrement of index in subclasses.
"""

# Standard library
import typing
from abc import ABC, abstractmethod

import quaternion as qnp
import numpy as np
from mujoco import MjsBody, MjsSite

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces


class Module(ABC):
    """Base class for all modules."""

    def __init__(self):
        self.body: typing.Optional[MjsBody] = None
        self.sites: typing.Mapping[ModuleFaces, MjsSite] = dict()

    @staticmethod
    def add_site(body: MjsBody, *args, **kwargs):
        return body.add_site(*args, **kwargs, group=5)

    @property
    def children(self):
        for site in self.sites.values():
            print(site)
        return {}

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the brick module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the brick.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        quat = np.roll(qnp.as_float_array(quat), shift=-1)

        # Set the quaternion for the brick body
        self.body.quat = np.round(quat, decimals=3)
