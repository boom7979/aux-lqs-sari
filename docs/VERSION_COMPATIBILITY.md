# DQ-DETR interface reference

The [integration guide](INTEGRATION.md) documents hook locations in [official DQ-DETR commit c5246797b111d7e5da9e6c20667fe0901aae35ba](https://github.com/hoiliu-0801/DQ-DETR/tree/c5246797b111d7e5da9e6c20667fe0901aae35ba).

Check the source identity and function locations of an existing checkout with:

```bash
python examples/check_dqdetr_interfaces.py /path/to/DQ-DETR
```

This is a read-only source check. After implementing the documented hooks, run the integration checks in the guide to verify tensor alignment, reference geometry, loss weighting, and gradients in your detector environment.
