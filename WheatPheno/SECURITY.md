# Security and responsible reporting

## Fixed model files

Python pickle and joblib files can execute code when loaded. Bundled models have pinned hashes. Local refits can be loaded through the CLI only with the explicit `--trust-model` flag and matching exported JSON metadata. This flag is appropriate for models you trained yourself or received from a trusted source; metadata hashes do not make an untrusted joblib safe. The web interface does not accept model uploads. Do not bypass that check or load a model attached to an unsolicited message. A checksum is not a signature: the source distribution and the code containing the expected hash must also come from a trusted source.

The original serialized selector is handled by a small inference compatibility module. Loading the historical training script is unnecessary and could trigger unrelated processing. Model inference must not execute training, overwrite source observations or download replacement weights.

## Local files and interfaces

Mapping configurations refer to files on the machine running the program. Review the paths and output destination before executing a configuration from another person, and avoid placing outputs over original research data. Large feature rasters and spatial alignment can consume substantial memory, disk space and time.

If using the optional web interface, treat it as a local research tool. Do not expose it to an untrusted network without a separate deployment review, authentication and resource limits. The project does not claim a hardened multiuser service or a guarantee of data anonymity.

## Reporting an issue

Report security issues privately to the maintainers through an established contact channel, or the repository's private vulnerability reporting facility when enabled. Avoid including exploit details or private research data in public issue reports.

Provide the software version, platform, minimal reproduction and impact, but remove private observations, original field coordinates, credentials and executable model files. Do not publish sensitive exploit details or private research data in a public issue while a private report is being considered.
