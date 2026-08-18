# Licenses and provenance

The `optimizers/` directory is derived from the Meta Distributed Shampoo code
included in the uploaded artifact. It retains the original copyright notices
and is distributed under the **Apache License 2.0** contained in
`optimizers/LICENSE.md`.

The files changed after the upload are enumerated in `MODIFICATIONS.md`.
The FOAM-specific experiment harness, tests, configurations, reports, and
analysis scripts were produced to repair and make executable the uploaded
research artifact. They do not alter the license obligations of the
upstream-derived optimizer files.

The optional SOAP source is not bundled. `foam_experiments/soap_adapter.py`
only loads a separately supplied `soap.py`, so the license of that external
implementation remains independent of this package.
