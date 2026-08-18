# Optional third-party baselines

The uploaded FOAM artifact did not contain the SOAP implementation. The
reconstructed suite therefore does not copy or silently replace it.

To enable `optimizer: soap`, place the external reference implementation at:

```text
third_party/SOAP/soap.py
```

or set `soap_module_path` in the YAML configuration to a directory containing
`soap.py`. The adapter expects that module to expose a class named `SOAP`.
