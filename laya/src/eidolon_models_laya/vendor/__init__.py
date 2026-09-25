"""Third-party code carried in-tree.

``laya`` (Copyright Convai Innovations, Apache License 2.0): the upstream package at the git
tag named in ``laya/VENDOR_VERSION``, synced by ``scripts/sync-laya-vendor.py``. Carried
in-tree because upstream deletes old releases from PyPI (0.3.7 vanished on 2026-09-25) while
iterating daily; following a tag keeps installs reproducible and upgrades deliberate. Our
torch-free port in ``eidolon_models_laya.sequence`` is verified against this copy by
``tests/test_parity.py``; that test and the smart-home eval are the upgrade gate.
Import it as ``eidolon_models_laya.vendor.laya``.
"""
