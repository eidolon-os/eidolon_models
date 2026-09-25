"""Third-party code carried in-tree.

``laya``: laya 0.3.7 (Copyright Convai Innovations, Apache License 2.0), byte-for-byte the
PyPI wheel's package. Vendored on 2026-09-25 because upstream removed the 0.3.7 release from
PyPI (only 0.3.15+ remain), which would have broken every fresh install of the service and
the ECS deploy. Our torch-free port in ``eidolon_models_laya.sequence`` is verified against
this copy by ``tests/test_parity.py``. Import it as ``eidolon_models_laya.vendor.laya``.
"""
