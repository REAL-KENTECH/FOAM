# Modification notice

The uploaded artifact included a modified copy of Meta's Distributed Shampoo
implementation. The reconstructed package further changes the following
upstream-derived files:

- `optimizers/matrix_functions.py`
- `optimizers/distributed_shampoo/distributed_shampoo.py`
- `optimizers/distributed_shampoo/shampoo_types.py`
- `optimizers/distributed_shampoo/utils/shampoo_preconditioner_list.py`
- `optimizers/distributed_shampoo/utils/shampoo_ddp_distributor.py`
- `submission.py`

The principal modifications are:

1. one canonical inverse-root refresh policy interface;
2. mathematically corrected FOAM proxy and damping floor;
3. stale, FOAM, ablation, and diagonal-residual modes;
4. persistent eigenspace/controller tensor state;
5. diagnostics, profiling, and factor snapshot export;
6. CPU/single-process safety repairs;
7. scalar matrix inverse-root return-contract repair;
8. AlgoPerf parameter plumbing for FOAM modes.

The original copyright headers and the Apache License 2.0 text in
`optimizers/LICENSE.md` are retained. This notice documents the changed files
as required for redistribution of modified upstream-derived source.
