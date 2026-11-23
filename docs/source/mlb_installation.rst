.. _mlb-installation:

Installing the optional MLB simulator
=====================================

The examples ``odt_with_mlb_simulation.py`` and
``idt_with_mlb_simulation.py`` generate synthetic measurements using
``muscopy-mlbsim`` (Python import name: ``muscopy_mlbsim``).
This simulator is currently private and is not distributed on PyPI or included in
muscopy's extras. The reconstruction APIs and the other examples can be
used without it. Without the simulator, these two examples print an
installation hint and skip execution.

Access requirements
-------------------

You need read access to ``Ideguchi-Lab/muscopy-mlbsim`` on GitHub and a
GitHub-authenticated SSH key. Contact the Ideguchi-Lab repository maintainers
if you need access. Installing public muscopy does not grant access to the
simulator. If Git reports "Repository not found" or "Permission denied",
check your repository access and SSH authentication first.

Install and run from a muscopy checkout
---------------------------------------

Run these commands from the muscopy repository root. The ``doc`` extra
supplies the plotting dependencies used by the examples.

.. code-block:: sh

   uv sync --extra doc
   uv pip install --python .venv/bin/python \
     "muscopy-mlbsim @ git+ssh://git@github.com/Ideguchi-Lab/muscopy-mlbsim.git"
   uv run --no-sync python examples/odt_with_mlb_simulation.py
   uv run --no-sync python examples/idt_with_mlb_simulation.py

On Windows, use ``.venv\Scripts\python.exe`` in place of
``.venv/bin/python``. Add ``--extra cuda`` to the initial ``uv sync``
command if using a compatible CUDA 12 environment.

Use ``uv run --no-sync`` after the separate installation: a subsequent
``uv sync`` can remove the simulator because it is not a project dependency.
If that happens, repeat the ``uv pip install`` command. This procedure
replaces the removed ``uv sync --extra extra`` option.

Install in an existing Python environment
-----------------------------------------

Alternatively, activate your environment and install both packages with pip:

.. code-block:: sh

   python -m pip install "muscopy[doc]==1.0.2"
   python -m pip install \
     "muscopy-mlbsim @ git+ssh://git@github.com/Ideguchi-Lab/muscopy-mlbsim.git"

The example scripts are in the muscopy source repository. Run them from
a matching checkout using that environment's ``python``.

For reproducible simulations, append ``@<commit-sha>`` to the simulator's
Git URL, replacing the placeholder with the full commit hash used in your
experiment, and record the remaining environment versions as well.
The simulator has its own license; muscopy's MIT license does not apply to it.
