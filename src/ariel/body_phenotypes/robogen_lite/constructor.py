"""TODO(jmdm): description of script."""

# Handle forward references in type hints
from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING, Any

# Third-party libraries
# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

import networkx as nx

# Type checking
if TYPE_CHECKING:
    from networkx import DiGraph

    from ariel.body_phenotypes.robogen_lite.modules.module import Module


def construct_mjspec_from_graph(graph: DiGraph[Any]) -> CoreModule:
    """
    Construct a MuJoCo specification from a graph representation.

    Can be used for constructing the body of a robot after crossover using the
    graph representation.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules.

    Raises
    ------
    ValueError
        If the graph contains unknown module types.
    """
    assert nx.is_tree(graph)

    modules: dict[int, Module] = {}
    for node in graph.nodes:
        # Extract module type and rotation from the graph node
        module_type = graph.nodes[node]["type"]
        module_rotation = graph.nodes[node]["rotation"]

        # Create the module based on its type
        match module_type:
            case ModuleType.CORE.name:
                module = CoreModule()
            case ModuleType.HINGE.name:
                module = HingeModule()
            case ModuleType.BRICK.name:
                module = BrickModule()
            case ModuleType.NONE.name:
                module = None
            case _:
                msg = f"Unknown module type: {module_type}"
                raise ValueError(msg)

        # Check that the module is not None
        if module:
            rotation_angle = ModuleRotationsTheta[module_rotation].value
            module.rotate(rotation_angle)
            modules[node] = module
        else:
            modules[node] = None

    core_module = modules[IDX_OF_CORE]
    names = {IDX_OF_CORE: "C"}

    for parent, children in nx.dfs_successors(graph).items():
        parent_module = modules[parent]
        for child in children:
            child_module = modules[child]
            face = ModuleFaces[graph[parent][child]["face"]]
            assert parent_module is not None and child_module is not None
            names[child] = name = f"{names[parent]}-{face.name[0].lower()}{child_module.__class__.__name__[0]}"
            parent_module.sites[face].attach_body(
                body=child_module.body,
                prefix=f"{name}-",
            )

    if isinstance(core_module, CoreModule):
        for site in core_module.spec.sites:
            core_module.spec.delete(site)

        return core_module

    msg = "The core module is not of type CoreModule."
    raise ValueError(msg)
