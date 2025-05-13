"""Digital Holography module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from muscopy.backend_manager import ArrayProtocol, BackendManager


def demultiplex_cp_arrays(
    bmg: BackendManager,
    cp_arrays: Sequence[ArrayProtocol],
    demultiplexing_matrix: ArrayProtocol,
) -> list[ArrayProtocol]:
    r"""Demultiplex a set of CP arrays using a demultiplexing matrix.

    Parameters
    ----------
    bmg : `BackendManager`
        Backend manager to use for the operation.
    cp_arrays : `collections.abc.Sequence`\[`ArrayProtocol`\]
        List of CP arrays to be demultiplexed.
    demultiplexing_matrix : `ArrayProtocol`
        Demultiplexing coefficient matrix.

    Returns
    -------
    `list`\[`ArrayProtocol`\]
        List of demultiplexed CP arrays.

    Raises
    ------
    ValueError
        1. If the demultiplexing matrix is not square
        2. If the number of CP arrays does not match the size of the demultiplexing matrix.
    """
    if demultiplexing_matrix.shape[0] != demultiplexing_matrix.shape[1]:
        msg = "Demultiplexing matrix must be square."
        raise ValueError(msg)
    if demultiplexing_matrix.shape[0] != len(cp_arrays):
        msg = "number of CP arrays must match the size of the demultiplexing matrix."
        raise ValueError(msg)

    xp = bmg.get_backend()
    cp_arrays = xp.asarray(cp_arrays)
    demultiplexing_matrix = xp.asarray(demultiplexing_matrix)

    demultiplexed_arrays = xp.tensordot(
        cp_arrays,
        demultiplexing_matrix,
        axes=(0, 0),
    )

    return [demultiplexed_arrays[i] for i in range(demultiplexed_arrays.shape[0])]
