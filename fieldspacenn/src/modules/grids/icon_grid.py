"""ICON's triangular grid with the two properties FieldSpaceNN takes from
HEALPix: the same cell order inside every coarse cell, and a neighbour list
whose slots mean the same relative place for every cell.

An ICON grid file is nested (the parent of cell ``c`` is ``c // 4``), and
child 0 is the centre child, but the three corner children come in any order
and so do a cell's vertices and neighbours. Everything here is derived from the
finest grid file alone (`vertex_of_cell` and the vertex coordinates).

Frames
------
A cell's *frame* is a labelling A, B, C of its corners, counter-clockwise.

- Coarsest level (zoom 0): A is the corner that lies most nearly due north or
  due south of the cell centre. Triangles point up or down, so the two frames
  are turned by 180 degrees against each other.
- Finer levels: a corner child keeps the frame of its parent (its corner at
  the parent's A is its A), the centre child gets the turned one (its A is the
  midpoint of the parent's edge BC). This uses vertex ids only.

Cell order (`canonical_order`)
    Child 0 is the centre child, child 1/2/3 the one at the parent's A/B/C.
    The order stays nested, so `to_zoom`, `encode_zooms` and the tokenizer work
    as on HEALPix, and position ``k`` inside a zoom-0 cell is the same place
    for all of them.

Neighbours (`ring12`)
    The 12 cells that touch a cell (3 by an edge, 9 by a corner),
    counter-clockwise, starting with the neighbour across edge AB::

              .-------.-------.
             / \     / \     / \
            / 12\ 11/ 10\   /
          .-------A-------.
         / \  1  / \  9  / \
        / 2 \   / x \   / 8 \
      .-------B-------C-------.
       \ 3   / \  5  / \  7  /
        \   / 4 \   / 6 \   /
          .-------.-------.

    At the 12 corners of the icosahedron only 5 cells meet; slot 4, 8 or 12 is
    then missing.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr

RING = 12  # cells that touch a triangle


def parent_corners(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Corners of the parent cells, from their children.

    :param corners: ``(4 * P, 3)`` vertex ids of the cells, counter-clockwise,
        the 4 children of a parent in consecutive rows.
    :return: ``(P, 3)`` vertex ids of the parents, counter-clockwise, and
        ``(P, 4)`` the parent corner (0, 1, 2) each child holds, -1 for the
        centre child.
    """
    children = corners.reshape(-1, 4, 3)
    # The centre child shares an edge (2 vertices) with each of the 3 others.
    shared = (children[:, :, None, :, None] == children[:, None, :, None, :]).any(-1).sum(-1)
    is_centre = (shared == 2).sum(-1) == 3
    assert (is_centre.sum(1) == 1).all(), "cells are not nested: 4 consecutive cells must form a parent"
    midpoints = children[is_centre]  # (P, 3): the midpoints of the parent's edges
    is_midpoint = children[:, :, :, None] == midpoints[:, None, None, :]  # (P, child, child slot, midpoint)
    holds = is_midpoint.any(2)  # (P, child, midpoint)
    # A corner child holds two midpoints; the parent corner i is the one opposite midpoint i,
    # so the parent's corners are counter-clockwise like the centre child's.
    corner = np.where(holds.sum(-1) == 2, holds.argmin(-1), -1)
    own_vertex = np.take_along_axis(children, (~is_midpoint.any(3)).argmax(-1)[..., None], 2)[..., 0]
    parents = np.empty_like(midpoints)
    rows, kids = np.nonzero(corner >= 0)
    parents[rows, corner[rows, kids]] = own_vertex[rows, kids]
    return parents, corner


def _north_east(origin: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """North and east component of the unit direction from ``origin`` to
    ``target`` (unit vectors ``(..., 3)``) in the tangent plane at ``origin``."""
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, origin)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(origin, east)
    step = target - origin
    n, e = (step * north).sum(-1), (step * east).sum(-1)
    length = np.hypot(n, e)
    return n / length, e / length


def _centres(xyz: np.ndarray) -> np.ndarray:
    """Cell centres ``(n, 3)`` on the unit sphere from the corners ``(n, 3, 3)``."""
    centre = xyz.sum(1)
    return centre / np.linalg.norm(centre, axis=-1, keepdims=True)


def canonical_order(vertex_of_cell: np.ndarray, vertex_xyz: np.ndarray, n_zooms: int) -> List[Dict[str, np.ndarray]]:
    """The cells of the finest grid and of its ``n_zooms - 1`` coarser levels
    in canonical order, coarsest first.

    :param vertex_of_cell: ``(n, 3)`` vertex ids of the finest grid's cells
        (0-based, counter-clockwise, in the file's nested order).
    :param vertex_xyz: ``(n_vertices, 3)`` vertices on the unit sphere.
    :return: Per zoom ``{"corners": (n_zoom, 3) vertex ids A, B, C of every
        cell, "file_index": (n_zoom,) the file-order index of every cell}``.
        File order at a coarser level is ``fine index // 4**levels``.
    """
    triples, child_corner = [vertex_of_cell.astype(np.int64)], []
    for _ in range(n_zooms - 1):
        parents, corner = parent_corners(triples[0])
        triples.insert(0, parents)
        child_corner.insert(0, corner)

    # Zoom 0 keeps the file's cell order; A is the corner most nearly due north or south.
    xyz = vertex_xyz[triples[0]]
    turn = np.abs(_north_east(_centres(xyz)[:, None], xyz)[1]).argmin(1)  # slot of A in the cell's triple
    place = np.zeros(len(turn), dtype=np.int64)  # canonical place inside the zoom-0 cell

    def level(triple, turn, place):
        per_cell = len(place) // len(triples[0])  # cells of this level in a zoom-0 cell
        file_index = np.argsort(np.arange(len(place)) // per_cell * per_cell + place)
        corners = np.take_along_axis(triple, (np.arange(3) + turn[:, None]) % 3, 1)
        return {"corners": corners[file_index], "file_index": file_index}

    levels = [level(triples[0], turn, place)]
    for parents, triple, corner in zip(triples, triples[1:], child_corner):
        children = triple.reshape(-1, 4, 3)
        is_corner = corner >= 0
        # The parent corner a child holds, as a slot of the parent's frame (A, B, C = 0, 1, 2) ...
        frame_slot = np.where(is_corner, (corner - turn[:, None]) % 3, 0)
        # ... and where the child has that vertex in its own triple.
        held = np.take_along_axis(parents, np.maximum(corner, 0), 1)
        at = (children == held[:, :, None]).argmax(-1)
        # Corner child: that vertex gets the same frame slot. Centre child: its triple is the
        # midpoints opposite the parent's corners, in the same order, so the turn carries over.
        turn = np.where(is_corner, (at - frame_slot) % 3, turn[:, None]).ravel()
        place = (4 * place[:, None] + np.where(is_corner, 1 + frame_slot, 0)).ravel()
        levels.append(level(triple, turn, place))
    return levels


def ring12(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The 12 cells around every cell of a closed triangular grid, in the
    order of the module docstring.

    :param corners: ``(n, 3)`` vertex ids A, B, C of every cell (its frame).
    :return: ``(n, 13)`` cell indices, the cell itself first, and ``(n, 13)``
        True where a neighbour is missing (its index is then the cell itself).
    """
    n = len(corners)
    n_vertices = corners.max() + 1
    # Edge j runs from corner j to corner j + 1. The cell across it has the reversed edge.
    start, end = corners, np.roll(corners, -1, 1)
    key = (start * n_vertices + end).ravel()
    order = np.argsort(key)
    twin = order[np.searchsorted(key[order], (end * n_vertices + start).ravel())].reshape(n, 3)
    across, enters = twin // 3, twin % 3  # the cell across edge j, and the slot of that edge there

    ring = np.empty((n, RING), dtype=np.int64)
    missing = np.zeros((n, RING), dtype=bool)
    for j in range(3):  # walk around B, then C, then A
        # In the neighbour, the reversed edge starts at the corner we walk around.
        cell, slot = across[:, j], enters[:, j]
        ring[:, 4 * j] = cell
        for step in range(1, 4):  # counter-clockwise around a corner: leave by the edge that ends at it
            leave = (slot - 1) % 3
            cell, slot = across[cell, leave], enters[cell, leave]
            ring[:, 4 * j + step] = cell
        missing[:, 4 * j + 3] = cell == across[:, (j + 1) % 3]  # only 5 cells around this corner
    own = np.arange(n)[:, None]
    adjc = np.concatenate([own, np.where(missing, own, ring)], 1)
    return adjc, np.concatenate([np.zeros((n, 1), dtype=bool), missing], 1)


def canonical_mgrid(vertex_of_cell: np.ndarray, vertex_xyz: np.ndarray, n_zooms: int) -> List[Dict]:
    """FieldSpaceNN's ``mgrids`` of a triangular grid and its ``n_zooms - 1``
    coarser levels (coarsest first), in canonical order with 12 neighbours.

    Each grid dict has, besides "coords" (lon, lat of the cell centres, the
    mean of the corners), "adjc", "adjc_mask" and "zoom":

    - "grid_type": "icon", which `GridLayer` needs for this neighbour list;
    - "file_index" ``(n,)``: the file-order index of every cell, so a field
      read in the file's order is ``field[file_index]`` here;
    - "frame_direction" ``(n, 2)``: north and east component of the unit
      direction from the cell centre to its corner A.

    :param vertex_of_cell: ``(n, 3)`` vertex ids of the finest grid's cells
        (0-based, counter-clockwise, in the file's nested order).
    :param vertex_xyz: ``(n_vertices, 3)`` vertices on the unit sphere.
    """
    grids = []
    for zoom, level in enumerate(canonical_order(vertex_of_cell, vertex_xyz, n_zooms)):
        adjc, missing = ring12(level["corners"])
        xyz = vertex_xyz[level["corners"]]
        centre = _centres(xyz)
        lonlat = np.stack([np.arctan2(centre[:, 1], centre[:, 0]), np.arcsin(centre[:, 2])], -1)
        grids.append(
            {
                "coords": torch.from_numpy(lonlat).float(),
                "adjc": torch.from_numpy(adjc),
                "adjc_mask": torch.from_numpy(missing),
                "zoom": zoom,
                "grid_type": "icon",
                "file_index": torch.from_numpy(level["file_index"]),
                "frame_direction": torch.from_numpy(np.stack(_north_east(centre, xyz[:, 0]), -1)).float(),
            }
        )
    return grids


def icon_canonical_mgrid(fine_grid_file: str, n_zooms: int) -> List[Dict]:
    """`canonical_mgrid` of an ICON grid file (the finest level) and its
    ``n_zooms - 1`` coarser levels. The finest level's "coords" are the file's
    own cell centres."""
    with xr.open_dataset(fine_grid_file) as ds:
        vertex_of_cell = ds["vertex_of_cell"].values.T - 1
        vertex_xyz = np.stack([ds[f"cartesian_{axis}_vertices"].values for axis in "xyz"], -1)
        lon_name, lat_name = ("clon", "clat") if "clon" in ds else ("lon_cell_centre", "lat_cell_centre")
        centres = torch.from_numpy(np.stack([ds[lon_name].values, ds[lat_name].values], -1)).float()
    grids = canonical_mgrid(vertex_of_cell, vertex_xyz, n_zooms)
    grids[-1]["coords"] = centres[grids[-1]["file_index"]]
    return grids
