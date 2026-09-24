# Workshop notebooks

Run the notebooks from this folder.

```bash
pip install -r requirements.txt
jupyter notebook 01_neural_fields_to_tensorweave.ipynb
jupyter notebook 02_ldi_restore.ipynb
jupyter notebook 03_ldi_restore_recumbent.ipynb
```

**FTG.** `01_neural_fields_to_tensorweave.ipynb` fits a scalar potential to SimPEG flight-line gravity gradients. The model lives in `tensorweave.py`. Set `num_fourier_features=0` for a coordinate MLP and `>0` for the random Fourier bank. Grids are in `data/` (see `data/README.md`).

**Restoration.** `02_ldi_restore.ipynb` fits the digitised single-layer fold in `data/parallelFoldLayers.svg`. `03_ldi_restore_recumbent.ipynb` runs the same two maps on the GemPy model-3 recumbent fold (`data/model3_surface_points.csv`, the `Y = 500` m section). `ldi.py` holds the loaders, the direct displacement field, a two-dimensional Euler velocity, its RK4 flow, and the figures.
