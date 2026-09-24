# Bundled SimPEG synthetic FTG (standalone copy)

| File | Role |
| --- | --- |
| `grids/truth_*.asc` | 65×65 high-resolution truth (ESRI ASCII, 25 m cells) |
| `synthetic_highres.npy` | Full grid `(4225, 12)` — built by `build_local_data.py` |
| `synthetic_lowres_200.npy` | 200 m flight lines `(N, 12)` — built by `build_local_data.py` |

If the `.npy` files are missing, run from the workshop folder:

```bash
python build_local_data.py
```

The notebook loader calls `ensure_bundled_npy()` automatically when possible.

`parallelFoldLayers.svg` is the digitised single-layer parallel fold used by `02_ldi_restore.ipynb`.

`model3_surface_points.csv` and `model3_orientations.csv` are the GemPy tutorial model-3 recumbent fold. `03_ldi_restore_recumbent.ipynb` uses the `Y = 500` m section.
