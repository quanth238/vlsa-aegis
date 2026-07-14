"""CRFS oracle experiment harness.

The package deliberately keeps its orchestration layer dependency-free. Heavy
OpenPI, MuJoCo, SciPy, and Torch imports live behind experiment integrations so
that manifests, schemas, aggregation, and restart semantics remain testable on
a CPU-only workstation.
"""

__version__ = "0.1.0"
