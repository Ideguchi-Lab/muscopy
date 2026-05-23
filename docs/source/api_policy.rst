Public API and Compatibility
============================

Stable imports
--------------

For v1.0, muscopy's stable import contract is submodule-based. Import stable
functions and classes from their documented modules:

.. code-block:: python

    from muscopy.qpi import qpi
    from muscopy.odt import ODTConfig, odt
    from muscopy.idt import IDTConfig, compute_idt

The package does not provide top-level re-exports from ``muscopy`` as part of
the v1.0 API contract.

Stable modules
--------------

The stable v1.0 modules are:

- ``muscopy.cfg``
- ``muscopy.dh``
- ``muscopy.qpi``
- ``muscopy.qpi_utils``
- ``muscopy.odt``
- ``muscopy.idt``
- ``muscopy.phasor``
- ``muscopy.phase_noise``
- ``muscopy.dir_parser``

Within these modules, public names listed in ``__all__`` are covered by the
compatibility guarantee. Private names beginning with ``_``, implementation
constants, examples, and generated documentation files are not stable API.

Experimental modules
--------------------

``muscopy.image_checker`` is experimental and is not covered by the v1.0
compatibility guarantee. It may change or be removed with changelog notice.

Deprecation policy
------------------

After v1.0, stable public APIs should not be removed or changed incompatibly
without a deprecation notice, migration guidance, and a changelog entry.
Removal should happen only in a future major release. Experimental APIs may
change outside that process, but changes should still be documented in the
changelog.
