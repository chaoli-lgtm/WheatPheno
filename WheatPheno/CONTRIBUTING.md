# Contributing to WheatPheno

Contributions should preserve reproducibility, explicit data provenance and the separation between deployment inference and model development.

Use Python 3.12 and the dependency versions in the package metadata. Install with `python -m pip install -e ".[test]"`, run `python -m pytest`, and exercise `python -m wheatpheno verify`, `demo` and `review` before proposing a change. Optional spatial or interface changes require the appropriate extras and a relevant execution check.

Keep contributions focused and describe the observable change, its scientific implications and how it was checked. Include a small nonprivate reproducer for a bug. Tests should verify meaningful behavior such as input order, units, coordinate handling, prediction stability or the distinction between raw and filled cells.

The bundled estimators are frozen. Do not replace them, alter observations, change feature definitions or select favorable evaluation groups to obtain a better score. A new model or corrected scientific feature definition belongs in a separately identified development workflow with its own provenance and validation. Preserve legacy NDRE and task specific texture behavior when using the existing artifacts.

Do not add full private datasets, original coordinate records, credentials, personal identifiers or hardcoded author machine paths. State the provenance and license of any new example data or third party material, and keep data/model licensing separate from source code licensing.

Original code contributions are made under the MIT license in `LICENSE`. Only contribute code you have the right to distribute on those terms; do not copy code from a project with incompatible or unspecified permissions. Demonstration and aggregate data contributions require a documented license consistent with `DATA_LICENSE.md`.

Before preparing a release, inspect its file manifest, verify model hashes and review all included data licenses. Keep private research material outside the distribution and record any unresolved methodological limitations in the release notes.
