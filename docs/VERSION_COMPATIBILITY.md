# Reference version and compatibility

## What is actually identified

This independent add-on release is version **0.1.0**. Its default-path behavior was compared to the project's historical added-module source, not to an unmodified detector freshly downloaded from GitHub.

The historical training-start record dated 2026-09-05 records checkout HEAD:

```text
6c7b0290bbe43c56d9c289392d086d07b06bba65
branch: main
working tree: has uncommitted changes
```

This is a **recorded historical checkout identifier**, not a verified official upstream pin. The working tree was dirty, so the commit alone cannot identify the code actually used. Its presence in a historical log does not establish a publicly fetchable commit in the current official DQ-DETR repository, nor identify the source of the final reported paper numbers.

## Exact file snapshot

The historical local integration/reference files inspected during release preparation have these SHA256 hashes. Their hashes also match the files recorded in the previously saved historical forward-capture manifest.

| Reference file within the detector checkout | SHA256 |
| --- | --- |
| `models/dqdetr/slb_dq_modules.py` | `f5856cf5649b01182aef5238dd59e42db3ab976e436013c4f4170aadfa6683e7` |
| `models/dqdetr/dqdetr.py` | `9494838784fa104aca3f288f05888b65ca7300df0ba4ee3f7a18e1df4079e6c6` |
| `models/dqdetr/deformable_transformer.py` | `f80af434855ad2764eaf9d156ffd1a99505c04dbeb1b844670dc0aa9223327f8` |
| `main_aitod.py` | `75e10c5a783576a1dfda21009d8e6ec86314cfc69dce342f60dc8632b7d99850` |

These hashes identify reference content without redistributing it. They are not a complete detector source-tree lock, and a hash does not make an unavailable file publicly obtainable. Machine-readable metadata are provided in [REFERENCE_SNAPSHOT.json](../REFERENCE_SNAPSHOT.json).

## Compatibility claims

| Scope | Status |
| --- | --- |
| Default spatial-head output and parameter-key parity against historical added module | checked with matching parameters on synthetic inputs |
| Default ranking, SARI, target generation, and loss parity | checked as recorded in `VALIDATION.md` |
| Entire historical detector checkpoint compatibility | not certified |
| End-to-end integration with a clean official upstream commit | not certified |
| Publicly fetchable official upstream commit matching all historical files | not established |
| Full source revision associated with final manuscript result assets | not independently verified |

Obtain the detector from its official repository, record the actual commit you use, implement the documented hooks, and validate them. Do not label an untested current upstream HEAD as a supported or verified version. No automatic upstream patch is included in this distribution.
