"""
Drop-in Neural Fourier Field for FTG / Hessian interpolation.

Put this file next to the notebook and import it::

    from tensorweave import NeuralFourierField

Two heads and two Fourier geometries:

* **``harmonic=True``** (default): horizontal wavenumbers ``k_xy`` only;
  height enters via ``exp(-z ‖k_xy‖)`` (Curlew / Laplace-style).
* **``harmonic=False``**: full ``input_dim`` FSF or stacked wavenumbers
  ``k ∈ ℝ^D`` and features ``cos(k·x), sin(k·x)`` on all coordinates.

* **Basis only** (no hidden layers): a linear combination of the sinusoids.
  Gradient and Hessian are analytic — one matrix multiply, no autodiff.
* **MLP**: the original Tensorweave head. Derivatives come from autodiff.
  ``fit`` then mixes in a Poisson-disk Laplacian penalty.
* **``num_fourier_features=0``**: no Fourier bank — coordinates feed the MLP
  directly (coordinate network on ``input_dim``).

Two frequency banks:

* **``length_scales=[ℓ₁, …]``** (original): i.i.d. Gaussian (or Cauchy)
  directions, stacked as ``k_eff = k_xy / ℓ``. Feature width is
  ``2 M n_scales``. If ``learnable=True``, the scales are trained in log10
  space along with ``potential_scale``.
* **``length_scale_range=(λ_min, λ_max)``** (default): Curlew FSF. Scrambled
  Sobol in ``(wavelength, direction)``, ``k = 2π k̂ / λ``, one wavelength per
  mode. Amplitudes damped by ``1/|k|^{freq_damp}`` (default ``2``).
  Frequencies stay frozen.

FTG / Hessian component order: ``[Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]``.

``RasterGrid`` builds a regular evaluation mesh from a bounding box or
point cloud and writes masked fields as ESRI ASCII rasters (``.asc``).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, grad, jacrev, vmap
from tqdm import tqdm

TensorLike = Union[np.ndarray, torch.Tensor]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_tensor(x: TensorLike, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.array(t.tolist(), dtype=np.float32)


def _chunk_slices(n: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def tensor6_to_matrix(data: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` → symmetric ``(N, 3, 3)``."""
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected data with shape (N, 6), got {tuple(data.shape)}")
    H = data.new_zeros(data.shape[0], 3, 3)
    H[:, 0, 0] = data[:, 0]
    H[:, 0, 1] = H[:, 1, 0] = data[:, 1]
    H[:, 0, 2] = H[:, 2, 0] = data[:, 2]
    H[:, 1, 1] = data[:, 3]
    H[:, 1, 2] = H[:, 2, 1] = data[:, 4]
    H[:, 2, 2] = data[:, 5]
    return H


def matrix_to_tensor6(H: torch.Tensor) -> torch.Tensor:
    """Symmetric ``(N, 3, 3)`` → ``(N, 6)``."""
    if H.ndim != 3 or H.shape[-2:] != (3, 3):
        raise ValueError(f"Expected H with shape (N, 3, 3), got {tuple(H.shape)}")
    return torch.stack(
        [H[:, 0, 0], H[:, 0, 1], H[:, 0, 2], H[:, 1, 1], H[:, 1, 2], H[:, 2, 2]],
        dim=-1,
    )


def prepare_ftg_data(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accept ``(N, 6)`` or ``(N, 3, 3)`` and return both."""
    if data.ndim == 2 and data.shape[1] == 6:
        return data, tensor6_to_matrix(data)
    if data.ndim == 3 and data.shape[-2:] == (3, 3):
        return matrix_to_tensor6(data), data
    raise ValueError("Data must have shape (N, 6) or (N, 3, 3).")


def poisson_disk_indices(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    max_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fast Poisson-disk subset of an unstructured (x, y) cloud."""
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if max_points <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    coords = np.c_[x, y]
    n = coords.shape[0]
    cell = radius / np.sqrt(2.0)
    xmin, ymin = coords.min(axis=0)
    ix = np.floor((coords[:, 0] - xmin) / cell).astype(np.int32)
    iy = np.floor((coords[:, 1] - ymin) / cell).astype(np.int32)

    order = rng.permutation(n)
    grid: Dict[int, Dict[int, int]] = {}
    chosen: List[int] = []
    r2 = radius * radius
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    for p in order:
        if len(chosen) >= max_points:
            break
        cx, cy = int(ix[p]), int(iy[p])
        ok = True
        for dx, dy in neighbours:
            col = grid.get(cx + dx)
            if col is None:
                continue
            q = col.get(cy + dy)
            if q is None:
                continue
            ddx = coords[p, 0] - coords[q, 0]
            ddy = coords[p, 1] - coords[q, 1]
            if ddx * ddx + ddy * ddy < r2:
                ok = False
                break
        if ok:
            grid.setdefault(cx, {})[cy] = p
            chosen.append(p)
    return np.asarray(chosen, dtype=np.int64)


def add_ftg_noise_by_snr(
    ftg: np.ndarray,
    snr_db: Union[float, Sequence[float], np.ndarray],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add Gaussian noise so each FTG component matches a target SNR (dB).

    ``ftg`` is ``(N, 6)``. ``snr_db`` is a scalar or length-6 sequence.
    """
    ftg = np.asarray(ftg, dtype=np.float64)
    if ftg.ndim != 2 or ftg.shape[1] != 6:
        raise ValueError(f"ftg must have shape (N, 6), got {ftg.shape}.")
    snr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if snr.size == 1:
        snr = np.full(6, float(snr[0]))
    if snr.size != 6:
        raise ValueError("snr_db must be a scalar or a length-6 sequence.")
    rng = np.random.default_rng(seed)
    power = np.mean(ftg * ftg, axis=0)
    std = np.sqrt(power / (10.0 ** (snr / 10.0)))
    return (ftg + rng.normal(size=ftg.shape) * std).astype(ftg.dtype, copy=False)


def _sample_wavelengths(
    u0: torch.Tensor,
    length_scale_range: Tuple[float, float],
    sampling: str = "log",
) -> torch.Tensor:
    """Map Sobol coordinate ``u0 ∈ (0, 1)`` onto ``length_scale_range`` (FSF)."""
    lo, hi = float(length_scale_range[0]), float(length_scale_range[1])
    if lo <= 0.0 or hi <= 0.0 or hi < lo:
        raise ValueError(
            f"length_scale_range must satisfy 0 < min <= max, got {length_scale_range}"
        )
    lo_t = torch.tensor(lo, dtype=u0.dtype, device=u0.device)
    hi_t = torch.tensor(hi, dtype=u0.dtype, device=u0.device)
    if sampling == "log":
        log_lo, log_hi = torch.log10(lo_t), torch.log10(hi_t)
        return torch.pow(torch.tensor(10.0, dtype=u0.dtype, device=u0.device), log_lo + u0 * (log_hi - log_lo))
    if sampling == "uniform":
        return lo_t + u0 * (hi_t - lo_t)
    raise ValueError(f"wavelength_sampling must be 'log' or 'uniform', got {sampling!r}")


def _rff_spatial_dim(input_dim: int, harmonic: bool) -> int:
    """Fourier wavenumber dimension: ``D`` if non-harmonic, else ``D-1`` (no ``k_z`` in bank)."""
    if harmonic:
        return max(1, int(input_dim) - 1)
    return int(input_dim)


def _draw_fsf_k(
    n_features: int,
    length_scale_range: Tuple[float, float],
    n_dim: int,
    seed: int,
    wavelength_sampling: str = "log",
) -> torch.Tensor:
    """
    FSF-style Sobol bank: one wavelength per mode, joint with direction.

    Returns
    -------
    k : (n_dim, M)
        ``k = 2π k̂ / λ`` with ``λ`` log-uniform in ``length_scale_range``.
    """
    engine = torch.quasirandom.SobolEngine(dimension=n_dim + 1, scramble=True, seed=int(seed))
    u = engine.draw(int(n_features)).clamp(1e-6, 1.0 - 1e-6)
    lengths = _sample_wavelengths(u[:, 0], length_scale_range, wavelength_sampling)
    k_raw = torch.special.ndtri(u[:, 1:].clamp(1e-6, 1.0 - 1e-6))
    k_hat = torch.nn.functional.normalize(k_raw, dim=1, eps=1e-8)
    return (2.0 * math.pi * k_hat / lengths.unsqueeze(1)).T.to(dtype=torch.float32)


def _draw_raw_k(
    n_features: int,
    n_dim: int,
    seed: int,
    distribution: str = "Normal",
    dist_variance: float = 1.0,
) -> torch.Tensor:
    """Original Tensorweave draw: i.i.d. Gaussian or Cauchy, shape ``(n_dim, M)``."""
    _set_all_seeds(seed)
    if distribution == "Normal":
        return torch.randn(n_dim, n_features, dtype=torch.float32)
    if distribution == "Cauchy":
        return torch.distributions.Cauchy(0.0, float(dist_variance)).sample(
            (n_dim, n_features)
        ).to(dtype=torch.float32)
    raise ValueError(f"distribution must be 'Normal' or 'Cauchy', got {distribution!r}")


def _sinusoid_derivative_weights(
    wc: torch.Tensor,
    ws: torch.Tensor,
    k: torch.Tensor,
    harmonic: bool,
) -> torch.Tensor:
    """
    Linear cos/sin coefficients → columns
    ``phi, gx, gy, gz, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz``. Shape ``(2 M, 10)``.
    """
    kx, ky = k[0], k[1]
    if harmonic:
        kz = torch.linalg.norm(k, dim=0)
    elif k.shape[0] >= 3:
        kz = k[2]
    else:
        kz = torch.zeros_like(kx)

    def pack(w_c: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
        return torch.cat((w_c, w_s), dim=0)

    kx2, ky2, kz2 = kx * kx, ky * ky, kz * kz
    kxy, kxz, kyz = kx * ky, kx * kz, ky * kz
    return torch.stack(
        (
            pack(wc, ws),
            pack(ws * kx, -wc * kx),
            pack(ws * ky, -wc * ky),
            pack(-wc * kz, -ws * kz),
            pack(-wc * kx2, -ws * kx2),
            pack(-wc * ky2, -ws * ky2),
            pack(wc * kz2, ws * kz2),
            pack(-wc * kxy, -ws * kxy),
            pack(-ws * kxz, wc * kxz),
            pack(-ws * kyz, wc * kyz),
        ),
        dim=1,
    )


# --------------------------------------------------------------------------- #
# Gridding / ASCII raster
# --------------------------------------------------------------------------- #

FTG_COMPONENT_NAMES: Tuple[str, ...] = ("xx", "xy", "xz", "yy", "yz", "zz")
_ASCII_SUFFIXES = {".asc", ".grd", ".txt", ".ascii"}


def _as_float2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("coordinate vector must be 1-D with length >= 2.")
    if np.any(~np.isfinite(x)):
        raise ValueError("coordinate vector must be finite.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("coordinate vector must be strictly increasing.")
    return x


def _regular_spacing(x: np.ndarray, name: str) -> float:
    step = np.diff(x)
    if not np.allclose(step, step[0], rtol=1e-6, atol=1e-9):
        raise ValueError(f"{name} must be regularly spaced.")
    return float(step[0])


def _bool_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    ny, nx = shape
    if m.shape == shape:
        return m
    if m.size == ny * nx:
        return m.reshape(shape)
    raise ValueError(
        f"mask must have shape {shape} or length {ny * nx}, got {m.shape}."
    )


def _component_paths(path: Union[str, Path], names: Sequence[str]) -> List[Path]:
    path = Path(path)
    as_dir = str(path).endswith(("/", "\\")) or path.is_dir()
    if as_dir:
        parent, stem, ext = path, "", ".asc"
    elif path.suffix.lower() in _ASCII_SUFFIXES:
        parent, stem, ext = path.parent, path.stem, path.suffix
    else:
        parent, stem, ext = path.parent, path.name, ".asc"
    parent.mkdir(parents=True, exist_ok=True)
    if len(names) == 1:
        fname = f"{stem}{ext}" if stem else f"{names[0]}{ext}"
        if Path(fname).suffix.lower() not in _ASCII_SUFFIXES:
            fname += ".asc"
        return [parent / fname]
    prefix = f"{stem}_" if stem else ""
    return [parent / f"{prefix}{name}{ext}" for name in names]


def write_ascii_raster(
    path: Union[str, Path],
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    nodata: float = -9999.0,
    fmt: str = "%.8g",
) -> Path:
    """
    Write a 2-D array as an ESRI ASCII raster (``.asc``).

    ``x`` and ``y`` are cell centres, strictly increasing. The file uses
    ``xllcorner`` / ``yllcorner`` (half a cell south-west of the first
    centre). Rows are written north to south. Cells must be square.
    """
    x = _as_float2d(np.asarray(x, dtype=np.float64).reshape(-1))
    y = _as_float2d(np.asarray(y, dtype=np.float64).reshape(-1))
    dx = _regular_spacing(x, "x")
    dy = _regular_spacing(y, "y")
    if not np.isclose(dx, dy, rtol=1e-6, atol=1e-9):
        raise ValueError(f"ASCII rasters need square cells, got dx={dx}, dy={dy}.")
    img = np.asarray(values, dtype=np.float64)
    if img.shape != (y.size, x.size):
        raise ValueError(f"values must have shape {(y.size, x.size)}, got {img.shape}.")

    out = np.where(np.isfinite(img), img, nodata)
    header = (
        f"ncols         {x.size}\n"
        f"nrows         {y.size}\n"
        f"xllcorner     {x[0] - 0.5 * dx:.10g}\n"
        f"yllcorner     {y[0] - 0.5 * dy:.10g}\n"
        f"cellsize      {dx:.10g}\n"
        f"NODATA_value  {nodata:.10g}"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.flipud(out), fmt=fmt, header=header, comments="")
    return path


@dataclass
class RasterGrid:
    """
    Regular 2-D evaluation grid with an optional validity mask.

    ``x`` and ``y`` are cell-centre vectors (west→east, south→north).
    ``mask`` is True on cells that should be evaluated / written; False
    cells become ``NODATA`` on export. Ravel order matches
    ``np.meshgrid(x, y, indexing="xy")`` (row-major, x fastest).

    Examples
    --------
    Build from a bounding box or a point cloud, then export FTG::

        grid = RasterGrid.from_domain(xyz, spacing=20.0, padding=300.0, z=0.0)
        grid = RasterGrid(grid.x, grid.y, z=0.0, mask=actv)
        grid.to_ascii(tensor6, "grids/ftg")   # ftg_xx.asc … ftg_zz.asc
    """

    x: np.ndarray
    y: np.ndarray
    z: float = 0.0
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.x = _as_float2d(np.asarray(self.x, dtype=np.float64).reshape(-1))
        self.y = _as_float2d(np.asarray(self.y, dtype=np.float64).reshape(-1))
        self.z = float(self.z)
        self._dx = _regular_spacing(self.x, "x")
        self._dy = _regular_spacing(self.y, "y")
        self.mask = _bool_mask(self.mask, self.shape)

    @classmethod
    def from_domain(
        cls,
        domain: Union[Sequence[float], np.ndarray],
        spacing: float,
        z: float = 0.0,
        padding: float = 0.0,
        mask: Optional[np.ndarray] = None,
    ) -> "RasterGrid":
        """
        Build cell centres with ``np.arange`` over a rectangle.

        ``domain`` is ``(xmin, xmax, ymin, ymax)`` or an ``(N, ≥2)`` point
        cloud (bounds from the first two columns). Centres run from
        ``xmin - padding`` up to but not including ``xmax + padding``.
        """
        dx = float(spacing)
        if dx <= 0.0:
            raise ValueError(f"spacing must be > 0, got {spacing}.")
        pad = float(padding)
        arr = np.asarray(domain, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 4:
            xmin, xmax, ymin, ymax = (float(v) for v in arr)
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
            ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())
        else:
            raise ValueError(
                "domain must be (xmin, xmax, ymin, ymax) or an (N, ≥2) array."
            )
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"empty domain: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}].")
        x = np.arange(xmin - pad, xmax + pad, dx, dtype=np.float64)
        y = np.arange(ymin - pad, ymax + pad, dx, dtype=np.float64)
        if x.size < 2 or y.size < 2:
            raise ValueError("domain is smaller than two cells at this spacing.")
        return cls(x=x, y=y, z=z, mask=mask)

    @property
    def nx(self) -> int:
        return int(self.x.size)

    @property
    def ny(self) -> int:
        return int(self.y.size)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.ny, self.nx

    @property
    def spacing(self) -> float:
        return self._dx

    @property
    def xx(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y, indexing="xy")[0]

    @property
    def yy(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y, indexing="xy")[1]

    @property
    def coords(self) -> np.ndarray:
        """All cell centres as ``(ny*nx, 3)`` in ravel order."""
        xx, yy = np.meshgrid(self.x, self.y, indexing="xy")
        z = np.full(xx.shape, self.z, dtype=np.float64)
        return np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))

    @property
    def active(self) -> np.ndarray:
        """Boolean mask, ravelled to ``(ny*nx,)``."""
        if self.mask is None:
            return np.ones(self.ny * self.nx, dtype=bool)
        return self.mask.reshape(-1)

    @property
    def active_coords(self) -> np.ndarray:
        """Cell centres where ``mask`` is True."""
        return self.coords[self.active]

    def embed(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """
        Place values onto ``(ny, nx)`` or ``(ny, nx, C)``.

        Accepts a full raster, a ravelled full grid, or active-cell values
        only (length ``mask.sum()``).
        """
        values = np.asarray(values)
        ny, nx = self.shape
        n = ny * nx
        n_act = int(self.active.sum())
        fill = float(fill)

        def _one(vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(vec, dtype=np.float64).reshape(-1)
            if vec.size == n:
                img = vec.reshape(ny, nx)
            elif vec.size == n_act:
                img = np.full((ny, nx), fill, dtype=np.float64)
                img.reshape(-1)[self.active] = vec
            else:
                raise ValueError(
                    f"expected {n} grid values or {n_act} active values, got {vec.size}."
                )
            if self.mask is not None:
                img = np.where(self.mask, img, fill)
            return img

        if values.ndim == 2 and values.shape == (ny, nx):
            return _one(values.ravel())
        if values.ndim == 1:
            return _one(values)
        if values.ndim == 3 and values.shape[:2] == (ny, nx):
            return np.stack([_one(values[:, :, k]) for k in range(values.shape[2])], axis=-1)
        if values.ndim == 2:
            return np.stack([_one(values[:, k]) for k in range(values.shape[1])], axis=-1)
        raise ValueError(
            f"values must be (ny, nx), (N,), (N, C) or (ny, nx, C); got {values.shape}."
        )

    def to_ascii(
        self,
        values: np.ndarray,
        path: Union[str, Path],
        nodata: float = -9999.0,
        names: Optional[Sequence[str]] = None,
        fmt: str = "%.8g",
    ) -> List[Path]:
        """
        Write ``values`` as one or more ESRI ASCII rasters.

        Multi-channel arrays (``(N, C)``, ``(ny, nx, C)``) write ``C`` files.
        Default names for ``C=6`` are ``xx … zz``. Inactive / non-finite
        cells are stored as ``nodata``.
        """
        img = self.embed(values, fill=nodata)
        if img.ndim == 2:
            channels = [img]
            labels: Sequence[str] = names if names is not None else ("z",)
        else:
            channels = [img[:, :, k] for k in range(img.shape[2])]
            if names is None:
                labels = FTG_COMPONENT_NAMES if len(channels) == 6 else tuple(
                    f"b{k}" for k in range(len(channels))
                )
            else:
                labels = names
        if len(labels) != len(channels):
            raise ValueError(
                f"names has length {len(labels)} but values have {len(channels)} channels."
            )
        paths = _component_paths(path, labels)
        return [
            write_ascii_raster(p, ch, self.x, self.y, nodata=nodata, fmt=fmt)
            for p, ch in zip(paths, channels)
        ]


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


@dataclass
class EarlyStopper:
    mode: str = "min"
    patience: int = 50
    min_delta: float = 0.002
    percent: bool = True
    ema_alpha: Optional[float] = 0.3
    cooldown: int = 5
    verbose: bool = True

    best: Optional[float] = None
    smooth: Optional[float] = None
    bad_epochs: int = 0
    cd_left: int = 0
    best_state: Optional[dict] = None
    best_epoch: Optional[int] = None

    def _better(self, cur: float, ref: float) -> bool:
        if self.percent:
            rel = (ref - cur) / (abs(ref) + 1e-12)
            return rel > self.min_delta if self.mode == "min" else -rel > self.min_delta
        delta = (ref - cur) if self.mode == "min" else (cur - ref)
        return delta > self.min_delta

    def step(self, metric: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None) -> bool:
        if self.ema_alpha is not None:
            self.smooth = metric if self.smooth is None else (
                self.ema_alpha * metric + (1.0 - self.ema_alpha) * self.smooth
            )
            val = self.smooth
        else:
            val = metric

        if self.best is None or self._better(val, self.best):
            if self.best is not None:
                self.bad_epochs = 0
                self.cd_left = self.cooldown
            self.best = val
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch
            return False

        if self.cd_left > 0:
            self.cd_left -= 1
            return False

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            if self.verbose:
                print(
                    f"[EarlyStopper] stop after {self.bad_epochs} bad epochs "
                    f"(best at {self.best_epoch}, value={self.best:.6g})."
                )
            return True
        return False


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class NeuralFourierField(nn.Module):
    """
    Scalar potential on a Fourier basis, with an MLP or linear head.

    Parameters
    ----------
    input_dim : int
        Coordinate dimension. Analytic Hessian requires 3.
    num_fourier_features : int
        Number of Fourier modes ``M`` per scale (stacked) or in total (FSF).
        ``0`` skips encoding; the MLP reads raw ``input_dim`` coordinates.
    length_scales : list of float or None
        Original Tensorweave: stacked scales. Each scale reuses the same
        Gaussian (or Cauchy) ``k_xy`` as ``k_eff = k_xy / ℓ``. Feature width
        is ``2 M n_scales``. If set, ``length_scale_range`` is ignored.
    length_scale_range : (float, float)
        FSF mode when ``length_scales`` is None. ``(λ_min, λ_max)``; each mode
        gets one wavelength, log-uniform in this interval by default.
    wavelength_sampling : ``"log"`` | ``"uniform"``
        FSF only. How the Sobol radial coordinate is mapped onto the range.
    distribution : ``"Normal"`` | ``"Cauchy"``
        Stacked-scale draw only.
    dist_variance : float
        Cauchy scale when ``distribution="Cauchy"``.
    freq_damp : float
        FSF only. Exponent ν of ``1/|k|^ν``. ``2`` (default) flattens Hessian
        contributions; ``0`` turns damping off. Ignored for stacked scales.
    harmonic : bool
        If True, use horizontal ``k_xy`` and ``exp(-z ‖k_xy‖)``. If False,
        draw ``k`` in all ``input_dim`` coordinates and use ``k·x``.
    potential_scale : float
        Output multiplier, stored (and optionally learned) in log10 space.
    learnable : bool
        If True, ``potential_scale`` is a parameter. In stacked-scale mode the
        length scales are parameters as well. FSF frequencies stay fixed.
    hidden_layers : list of int or None
        MLP widths. Empty / None → linear head with analytic derivatives.
        A non-empty list is the original Tensorweave MLP; ``fit`` then
        includes a Laplacian penalty because the field is no longer harmonic.
    activation : nn.Module or None
        Used between hidden layers only. Ignored when there are no hidden
        layers.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_fourier_features: int = 32,
        length_scales: Optional[Sequence[float]] = None,
        length_scale_range: Tuple[float, float] = (1e2, 1e3),
        wavelength_sampling: str = "log",
        distribution: str = "Normal",
        dist_variance: float = 1.0,
        freq_damp: float = 2.0,
        harmonic: bool = True,
        potential_scale: float = 1e2,
        learnable: bool = True,
        hidden_layers: Optional[Sequence[int]] = None,
        output_dim: int = 1,
        activation: Optional[nn.Module] = None,
        seed: int = 404,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if input_dim < 2:
            raise ValueError("input_dim must be >= 2.")
        if output_dim != 1:
            raise ValueError("Only scalar output_dim=1 is supported.")
        if wavelength_sampling not in ("log", "uniform"):
            raise ValueError(
                "wavelength_sampling must be 'log' or 'uniform', "
                f"got {wavelength_sampling!r}"
            )
        if distribution not in ("Normal", "Cauchy"):
            raise ValueError(
                "distribution must be 'Normal' or 'Cauchy', "
                f"got {distribution!r}"
            )
        if freq_damp < 0:
            raise ValueError(f"freq_damp must be non-negative, got {freq_damp}")

        self.device = _as_device(device)
        self.input_dim = int(input_dim)
        self.num_features = int(num_fourier_features)
        if self.num_features < 0:
            raise ValueError("num_fourier_features must be >= 0.")
        self.harmonic = bool(harmonic)
        self.wavelength_sampling = str(wavelength_sampling)
        self.distribution = str(distribution)
        self.use_stacked = length_scales is not None
        if self.num_features == 0 and self.use_stacked:
            raise ValueError(
                "num_fourier_features=0 (coordinate MLP) cannot use stacked length_scales."
            )
        hidden_layers = list(hidden_layers) if hidden_layers else []

        _set_all_seeds(seed)
        rff_dim = _rff_spatial_dim(self.input_dim, self.harmonic)
        if self.use_stacked:
            scales = [float(s) for s in length_scales]
            if not scales or min(scales) <= 0.0:
                raise ValueError("length_scales must be a non-empty sequence of positive values.")
            self.length_scale_range = (min(scales), max(scales))
            self.freq_damp = 0.0
            k_xy = _draw_raw_k(
                n_features=self.num_features,
                n_dim=rff_dim,
                seed=seed,
                distribution=self.distribution,
                dist_variance=float(dist_variance),
            )
            log_ls = torch.log10(torch.tensor(scales, dtype=torch.float32))
            if learnable:
                self.log_length_scales = nn.Parameter(log_ls)
            else:
                self.register_buffer("log_length_scales", log_ls)
        else:
            self.length_scale_range = (float(length_scale_range[0]), float(length_scale_range[1]))
            self.freq_damp = float(freq_damp)
            if self.num_features == 0:
                k_xy = torch.zeros(rff_dim, 0, dtype=torch.float32)
            else:
                k_xy = _draw_fsf_k(
                    n_features=self.num_features,
                    length_scale_range=self.length_scale_range,
                    n_dim=rff_dim,
                    seed=seed,
                    wavelength_sampling=self.wavelength_sampling,
                )
            self.register_buffer(
                "log_length_scales",
                torch.log10(torch.tensor(self.length_scale_range, dtype=torch.float32)),
            )

        self.register_buffer("k_xy", k_xy)
        self.register_buffer("coord_offset", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("coord_scale", torch.ones(self.input_dim, dtype=torch.float32))
        if self.num_features == 0:
            self.register_buffer("amplitude_denom", torch.zeros(0, dtype=torch.float32))
        else:
            kn = torch.linalg.norm(k_xy, dim=0).clamp_min(1e-8)
            if self.freq_damp == 0.0:
                denom = torch.ones_like(kn)
            else:
                denom = kn.pow(self.freq_damp)
            self.register_buffer("amplitude_denom", denom)

        log_scale = torch.log10(torch.tensor(float(potential_scale), dtype=torch.float32))
        if learnable:
            self.log_potential = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_potential", log_scale)

        if self.num_features == 0:
            n_in = self.input_dim
        else:
            n_in = 2 * self.num_features * (
                int(self.log_length_scales.numel()) if self.use_stacked else 1
            )
        dims = [n_in] + hidden_layers + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 and activation is not None:
                layers.append(activation)
        self.mlp = nn.Sequential(*layers)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        self.loss_history: List[List[float]] = []
        self.laplacian_sample_counts: List[int] = []
        self.to(self.device)

    # ----- properties ------------------------------------------------------ #

    @property
    def is_linear(self) -> bool:
        return len(self.mlp) == 1 and isinstance(self.mlp[0], nn.Linear)

    @property
    def uses_fourier(self) -> bool:
        """``False`` when ``num_fourier_features=0`` (direct coordinate MLP)."""
        return self.num_features > 0

    @property
    def scale(self) -> torch.Tensor:
        return (10.0 ** self.log_potential).to(dtype=torch.float32)

    @property
    def n_scales(self) -> int:
        return int(self.log_length_scales.numel()) if self.use_stacked else 1

    @property
    def scales_learnable(self) -> bool:
        p = getattr(self, "log_length_scales", None)
        return self.use_stacked and isinstance(p, nn.Parameter)

    def _scale_bank(self) -> List[torch.Tensor]:
        """Effective ``k`` per stacked scale: ``k_xy / ℓ``."""
        scales = 10.0 ** self.log_length_scales
        return [self.k_xy / s for s in scales]

    def _set_coord_affine_from_data(self, coords: torch.Tensor) -> None:
        """
        Map physical coordinates to ~O(1) per axis before the coordinate MLP.

        Constant axes (e.g. a single survey height) use the largest horizontal
        span so ``z`` is not drowned out by ``x,y`` in raw metres.
        """
        if self.num_features != 0:
            return
        off = coords.mean(dim=0)
        span = coords.max(dim=0).values - coords.min(dim=0).values
        ref = span[span > 1e-6].max() if bool((span > 1e-6).any()) else coords.new_tensor(1.0)
        scale = torch.where(span > 1e-6, span, ref)
        self.coord_offset.copy_(off.to(self.coord_offset.dtype))
        self.coord_scale.copy_(scale.to(self.coord_scale.dtype))

    def mode_wavelengths(self) -> torch.Tensor:
        """``λ = 2π / |k|``. FSF: ``(M,)``. Stacked: ``(M n_scales,)``."""
        if not self.use_stacked:
            return 2.0 * math.pi / torch.linalg.norm(self.k_xy, dim=0).clamp_min(1e-30)
        parts = [
            2.0 * math.pi / torch.linalg.norm(k, dim=0).clamp_min(1e-30)
            for k in self._scale_bank()
        ]
        return torch.cat(parts, dim=0)

    # ----- forward --------------------------------------------------------- #

    def _mode_damp(self) -> torch.Tensor:
        """Per-mode ``1/|k|^ν`` (M,), FSF amplitude damping."""
        return 1.0 / self.amplitude_denom

    def _feature_damp(self) -> torch.Tensor:
        """``(2 M,)`` copy of ``1/|k|^ν`` for the cos then sin block."""
        d = self._mode_damp()
        return torch.cat((d, d))

    def _encode_rff(
        self,
        coords: torch.Tensor,
        k_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cos/sin features. FSF: ``(N, 2 M)``. Stacked: ``(N, 2 M n_scales)``."""
        if self.num_features == 0:
            x = coords.to(dtype=torch.float32)
            return (x - self.coord_offset) / self.coord_scale.clamp_min(1e-8)
        if self.use_stacked:
            blocks: List[torch.Tensor] = []
            for k in self._scale_bank():
                if self.harmonic:
                    xy, z = coords[:, :-1], coords[:, -1:]
                    proj = xy @ k
                    cs, ss = torch.cos(proj), torch.sin(proj)
                    kn = torch.linalg.norm(k, dim=0, keepdim=True)
                    decay = torch.exp(-z @ kn)
                    cs, ss = cs * decay, ss * decay
                else:
                    proj = coords @ k
                    cs, ss = torch.cos(proj), torch.sin(proj)
                blocks.append(torch.cat((cs, ss), dim=-1))
            return torch.cat(blocks, dim=-1)

        k = self.k_xy if k_xy is None else k_xy
        if self.harmonic:
            proj = coords[:, :-1] @ k
            cs, ss = torch.cos(proj), torch.sin(proj)
            kn = torch.linalg.norm(k, dim=0, keepdim=True)
            decay = torch.exp(-coords[:, -1:] @ kn)
            cs, ss = cs * decay, ss * decay
        else:
            proj = coords @ k
            cs, ss = torch.cos(proj), torch.sin(proj)
        return torch.cat((cs, ss), dim=-1)

    def _first_linear(self, feats: torch.Tensor) -> torch.Tensor:
        """First linear layer with FSF ``A_k / |k|^ν`` column scaling."""
        layer = self.mlp[0]
        weight = layer.weight
        if self.freq_damp != 0.0:
            weight = weight * self._feature_damp()
        return torch.nn.functional.linear(feats, weight, layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.k_xy.device, dtype=torch.float32)
        feats = self._encode_rff(x)
        if not self.uses_fourier or self.use_stacked or self.freq_damp == 0.0:
            y = self.mlp(feats)
        else:
            h = self._first_linear(feats)
            rest = self.mlp[1:]
            y = rest(h) if len(rest) else h
        return y * self.scale

    # ----- analytic derivatives (linear head, no autodiff) ----------------- #

    def _derivative_weights(self, k_xy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map linear coefficients onto {phi, grad, Hessian} of the sinusoids.

        Returns
        -------
        W_all : (2 M n_scales, 10)  or  (2 M, 10) in FSF mode
            Columns: phi, gx, gy, gz, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz.
        bias : (1,)
            Potential bias; Hessian/gradient of a constant is zero.
        """
        pot = self.scale
        bias = self.mlp[0].bias * pot
        if self.use_stacked:
            width = 2 * self.num_features
            w = (self.mlp[0].weight * pot).view(-1)
            parts = []
            for i, k in enumerate(self._scale_bank()):
                block = w[i * width : (i + 1) * width].view(2, self.num_features)
                parts.append(_sinusoid_derivative_weights(block[0], block[1], k, self.harmonic))
            return torch.cat(parts, dim=0), bias

        k = self.k_xy if k_xy is None else k_xy
        w = (self.mlp[0].weight * pot).view(2, self.num_features) * self._mode_damp()
        return _sinusoid_derivative_weights(w[0], w[1], k, self.harmonic), bias

    def _analytic_out(
        self,
        x: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        k_xy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(N, 10)`` analytic components and the potential bias."""
        if feats is None:
            feats = self._encode_rff(x, k_xy=k_xy)
        W_all, bias = self._derivative_weights(k_xy)
        return feats @ W_all, bias

    @staticmethod
    def _unpack_ftg(out: torch.Tensor) -> torch.Tensor:
        """Columns 4,7,8,5,9,6 → ``[Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]``."""
        return torch.stack(
            (out[:, 4], out[:, 7], out[:, 8], out[:, 5], out[:, 9], out[:, 6]),
            dim=-1,
        )

    @staticmethod
    def _unpack_grad_hess(out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Hxx, Hyy, Hzz = out[:, 4], out[:, 5], out[:, 6]
        Hxy, Hxz, Hyz = out[:, 7], out[:, 8], out[:, 9]
        grad_field = out[:, 1:4]
        H = torch.stack(
            (
                torch.stack((Hxx, Hxy, Hxz), dim=-1),
                torch.stack((Hxy, Hyy, Hyz), dim=-1),
                torch.stack((Hxz, Hyz, Hzz), dim=-1),
            ),
            dim=1,
        )
        return grad_field, H

    def potential_grad_hess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scalar, gradient and Hessian in one pass."""
        x = x.to(self.k_xy.device, dtype=torch.float32)
        if self.is_linear and self.input_dim == 3 and self.uses_fourier:
            out, bias = self._analytic_out(x)
            grad_field, H = self._unpack_grad_hess(out)
            return out[:, :1] + bias, grad_field, H
        return _autodiff_hessian(self, x, detach_params=not self.training)

    def gradient_and_hessian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``grad (N, D)`` and ``H (N, D, D)``.

        No hidden layers: analytic derivatives of the sinusoids (no autodiff).
        """
        _, grad_field, H = self.potential_grad_hess(x)
        return grad_field, H

    def _analytic_tensor6(
        self,
        x: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        k_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self._analytic_out(x, feats=feats, k_xy=k_xy)
        return self._unpack_ftg(out)

    def _hessian_batched(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        if self.is_linear and self.input_dim == 3 and self.uses_fourier:
            return self.potential_grad_hess(x)[2]
        parts = []
        for sl in _chunk_slices(x.shape[0], max(1, chunk_size)):
            _, _, H = self.potential_grad_hess(x[sl])
            parts.append(H)
        return torch.cat(parts, dim=0)

    # ----- training -------------------------------------------------------- #

    def fit(
        self,
        coords: TensorLike,
        data: TensorLike,
        grid: Optional[TensorLike] = None,
        epochs: int = 250,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 1e-4,
        wd: float = 1e-4,
        lap: Optional[bool] = None,
        lap_hyperparam: float = 0.01, # hyperparameter for the Laplacian penalty
        lap_spacing: Union[float, Tuple[float, float, float, int, int]] = 100.0,
        lap_samples: int = 2000,
        chunk_size: int = 512,
        patience: int = 50,
        min_delta: float = 0.002,
        percent: bool = True,
        ema_alpha: float = 0.3,
        cooldown: int = 5,
        verbose: bool = True,
        plot_every: int = 0,
        plotter: Optional[Callable[[torch.Tensor], None]] = None,
        eval_grid: Optional[TensorLike] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Fit Hessian / FTG components.

        ``lap`` defaults to True whenever the field is not an analytic
        harmonic linear head (MLP, or linear with ``harmonic=False``).
        Collocation points are a Poisson-disk subset of ``grid``, or of
        ``coords`` if ``grid`` is omitted.
        """
        self.train()
        loss_fn = loss_fn or nn.L1Loss()
        use_analytic = bool(self.is_linear and self.input_dim == 3 and self.uses_fourier)
        if lap is None:
            lap = not (use_analytic and self.harmonic)

        coords_t = _to_tensor(coords, self.k_xy.device)
        data6, _ = prepare_ftg_data(_to_tensor(data, self.k_xy.device))
        if coords_t.shape != (data6.shape[0], self.input_dim):
            raise ValueError(
                f"coords must have shape (N, {self.input_dim}) matching data, "
                f"got {tuple(coords_t.shape)} vs N={data6.shape[0]}."
            )

        grid_t: Optional[torch.Tensor] = None
        x_vec = y_vec = None
        if lap:
            grid_t = _to_tensor(coords if grid is None else grid, self.k_xy.device)
            if grid_t.ndim != 2 or grid_t.shape[1] != self.input_dim:
                raise ValueError(
                    f"grid must have shape (N, {self.input_dim}), got {tuple(grid_t.shape)}."
                )
            grid_np = _tensor_to_numpy(grid_t)
            x_vec, y_vec = grid_np[:, 0], grid_np[:, 1]

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=max(5, patience // 3)
        )
        stopper = EarlyStopper(
            patience=patience,
            min_delta=min_delta,
            percent=percent,
            ema_alpha=ema_alpha,
            cooldown=cooldown,
            verbose=verbose,
        )

        # Frozen frequencies → encode training coordinates once.
        freeze_k = use_analytic and not self.scales_learnable
        cached_k = self.k_xy if (freeze_k and not self.use_stacked) else None
        cached_feats = self._encode_rff(coords_t, k_xy=cached_k) if freeze_k else None

        self.loss_history = []
        self.laplacian_sample_counts = []
        r = float(lap_spacing) if not isinstance(lap_spacing, tuple) else float(lap_spacing[1])

        with tqdm(range(epochs), desc="Training", disable=not verbose) as bar:
            for epoch in bar:
                optimizer.zero_grad(set_to_none=True)

                if lap and isinstance(lap_spacing, tuple):
                    r_max, r_min, decay, cycles, update_schedule = lap_spacing
                    cycle_len = max(1, int(epochs / max(1, int(cycles))))
                    if epoch % int(update_schedule) == 0:
                        phase = (epoch % cycle_len) / max(1, cycle_len)
                        r = r_min + (r_max - r_min) * np.exp(-decay * phase)

                lap_loss = coords_t.new_zeros(())
                lap_pts = None
                if lap:
                    idx = poisson_disk_indices(x_vec, y_vec, r, lap_samples)
                    if idx.size == 0:
                        idx = np.arange(int(grid_t.shape[0]), dtype=np.int64)
                    lap_pts = grid_t[torch.as_tensor(idx, device=grid_t.device, dtype=torch.long)]
                    self.laplacian_sample_counts.append(int(lap_pts.shape[0]))

                if use_analytic and not lap:
                    pred6 = self._analytic_tensor6(coords_t, feats=cached_feats, k_xy=cached_k)
                    self.laplacian_sample_counts.append(0)
                elif use_analytic and lap:
                    all_xyz = torch.cat((coords_t, lap_pts), dim=0)
                    out, _ = self._analytic_out(all_xyz)
                    pred6 = self._unpack_ftg(out[: coords_t.shape[0]])
                    lap_h = out[coords_t.shape[0] :, 4] + out[coords_t.shape[0] :, 5] + out[coords_t.shape[0] :, 6]
                    lap_loss = lap_h.abs().mean()
                else:
                    if lap:
                        all_xyz = torch.cat((coords_t, lap_pts), dim=0)
                        H_all = self._hessian_batched(all_xyz, chunk_size)
                        pred6 = matrix_to_tensor6(H_all[: coords_t.shape[0]])
                        lap_loss = H_all[coords_t.shape[0] :].diagonal(dim1=-2, dim2=-1).sum(-1).abs().mean()
                    else:
                        pred6 = matrix_to_tensor6(self._hessian_batched(coords_t, chunk_size))
                        self.laplacian_sample_counts.append(0)

                if isinstance(loss_fn, nn.L1Loss):
                    comp = (pred6 - data6).abs().mean(dim=0)
                elif isinstance(loss_fn, nn.MSELoss):
                    comp = (pred6 - data6).square().mean(dim=0)
                else:
                    comp = torch.stack([loss_fn(pred6[:, i], data6[:, i]) for i in range(6)])

                w = 1.0 / (comp.detach() + 1e-12)
                total = (w * comp).sum()
                if lap:
                    # adaptive fitting
                    #total = total + lap_loss * (1.0 / (lap_loss.detach() + 1e-12))
                    
                    # fixed hyperparameter
                    total = total + lap_loss * lap_hyperparam

                total.backward()
                optimizer.step()

                unweighted = comp.sum() + (lap_loss if lap else 0.0)
                unweighted_f = float(unweighted.detach())
                scheduler.step(unweighted_f)
                row = [float(v) for v in comp.detach()]
                if lap:
                    row.append(float(lap_loss.detach()))
                self.loss_history.append(row)

                postfix = {
                    "loss": f"{unweighted_f:.4f}",
                    "ftg": "[" + ", ".join(f"{v:.3f}" for v in row[:6]) + "]",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "stall": stopper.bad_epochs,
                }
                if lap:
                    postfix["lap"] = f"{row[6]:.3e}"
                bar.set_postfix(postfix)

                if stopper.step(unweighted_f, model=self, epoch=epoch):
                    if verbose:
                        print("Early stopping triggered.")
                    break

                if plotter is not None and plot_every > 0 and epoch % plot_every == 0:
                    g = _to_tensor(eval_grid if eval_grid is not None else grid_t, self.k_xy.device)
                    _, _, H_plot = compute_hessian_eval(self, g, chunk_size=max(1024, chunk_size))
                    plotter(H_plot)

        if stopper.best_state is not None:
            self.load_state_dict(stopper.best_state)

        losses = np.asarray(self.loss_history, dtype=float)
        return losses, self.laplacian_sample_counts

    # ----- prediction ------------------------------------------------------ #

    def predict(
        self,
        coords: np.ndarray,
        output: str = "hessian",
        chunk_size: int = 2048,
        normalize_grad: bool = False,
    ) -> np.ndarray:
        if not isinstance(coords, np.ndarray):
            coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != self.input_dim:
            raise ValueError(f"coords must have shape (N, {self.input_dim}), got {coords.shape}.")

        key = {"phi": "potential", "grad": "gradient", "hess": "hessian", "ftg": "tensor6"}.get(
            output.strip().lower(), output.strip().lower()
        )
        allowed = {"potential", "gradient", "hessian", "tensor6", "laplacian"}
        if key not in allowed:
            raise ValueError(f"output must be one of {sorted(allowed)}.")

        prior = self.training
        self.eval()
        analytic = self.is_linear and self.input_dim == 3
        if analytic:
            chunk_size = max(int(chunk_size), 65536)
        chunks: List[torch.Tensor] = []
        try:
            n = coords.shape[0]
            ctx = torch.inference_mode() if analytic else torch.enable_grad()
            with ctx:
                for sl in _chunk_slices(n, chunk_size):
                    x = torch.as_tensor(coords[sl], device=self.k_xy.device, dtype=torch.float32)
                    if key == "potential":
                        chunks.append(self.forward(x).detach().cpu())
                        continue
                    if analytic:
                        out, _ = self._analytic_out(x)
                        if key == "gradient":
                            grad_field = out[:, 1:4]
                            if normalize_grad:
                                grad_field = grad_field / (torch.linalg.norm(grad_field, dim=-1, keepdim=True) + 1e-8)
                            chunks.append(grad_field.detach().cpu())
                        elif key == "hessian":
                            chunks.append(self._unpack_grad_hess(out)[1].detach().cpu())
                        elif key == "tensor6":
                            chunks.append(self._unpack_ftg(out).detach().cpu())
                        else:
                            chunks.append((out[:, 4] + out[:, 5] + out[:, 6]).detach().cpu())
                    else:
                        grad_field, H = self.gradient_and_hessian(x)
                        if key == "gradient":
                            if normalize_grad:
                                grad_field = grad_field / (torch.linalg.norm(grad_field, dim=-1, keepdim=True) + 1e-8)
                            chunks.append(grad_field.detach().cpu())
                        elif key == "hessian":
                            chunks.append(H.detach().cpu())
                        elif key == "tensor6":
                            chunks.append(matrix_to_tensor6(H).detach().cpu())
                        else:
                            chunks.append(H.diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu())
            return _tensor_to_numpy(torch.cat(chunks, dim=0))
        finally:
            self.train(prior)

    def export_ascii(
        self,
        grid: RasterGrid,
        path: Union[str, Path],
        output: str = "tensor6",
        nodata: float = -9999.0,
        chunk_size: int = 2048,
        names: Optional[Sequence[str]] = None,
        fmt: str = "%.8g",
    ) -> List[Path]:
        """
        Evaluate on ``grid`` (active cells only) and write ESRI ASCII rasters.

        ``output`` is forwarded to :meth:`predict`. ``tensor6`` / ``hessian``
        write six component files; ``gradient`` writes ``gx, gy, gz``;
        scalar fields write a single ``.asc``.
        """
        key = {"phi": "potential", "grad": "gradient", "hess": "hessian", "ftg": "tensor6"}.get(
            output.strip().lower(), output.strip().lower()
        )
        if grid.active_coords.shape[0] == 0:
            raise ValueError("grid mask is empty; nothing to evaluate.")
        pred = self.predict(grid.active_coords.astype(np.float32), output=key, chunk_size=chunk_size)
        if key == "hessian":
            if pred.ndim == 3:
                pred = _tensor_to_numpy(matrix_to_tensor6(torch.as_tensor(pred)))
            key = "tensor6"
        if names is None:
            if key == "tensor6":
                names = FTG_COMPONENT_NAMES
            elif key == "gradient":
                names = ("gx", "gy", "gz")
            elif key == "potential":
                names = ("phi",)
            elif key == "laplacian":
                names = ("lap",)
        return grid.to_ascii(pred, path, nodata=nodata, names=names, fmt=fmt)


# --------------------------------------------------------------------------- #
# Autodiff (MLP head)
# --------------------------------------------------------------------------- #


def _autodiff_hessian(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 1024,
    detach_params: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if detach_params:
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
    else:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

    def f_single(x, params, buffers):
        return functional_call(model, (params, buffers), x.unsqueeze(0)).squeeze()

    g_fn = grad(f_single, argnums=0)
    h_fn = jacrev(g_fn, argnums=0)
    calc = vmap(
        lambda x, p, b: (f_single(x, p, b), g_fn(x, p, b), h_fn(x, p, b)),
        in_dims=(0, None, None),
        randomness="same",
    )

    scalars, grads, hessians = [], [], []
    for sl in _chunk_slices(coords.shape[0], chunk_size):
        s, g, h = calc(coords[sl], params, buffers)
        scalars.append(s)
        grads.append(g)
        hessians.append(h)
    return torch.cat(scalars, 0), torch.cat(grads, 0), torch.cat(hessians, 0)


def compute_gradient(
    model: nn.Module,
    coords: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "k_xy"):
        coords = coords.to(model.k_xy.device)
    if hasattr(model, "potential_grad_hess"):
        phi, grad_field, _ = model.potential_grad_hess(coords)
    elif hasattr(model, "gradient_and_hessian"):
        phi = model(coords)
        grad_field, _ = model.gradient_and_hessian(coords)
    else:
        coords = coords.requires_grad_(True)
        phi = model(coords)
        grad_field = torch.autograd.grad(
            phi, coords, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
    if normalize:
        grad_field = grad_field / (torch.linalg.norm(grad_field, dim=-1, keepdim=True) + 1e-8)
    return phi, grad_field


def compute_hessian(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is not None:
        coords = coords.to(device)
    if hasattr(model, "potential_grad_hess"):
        phi, grad_field, H = model.potential_grad_hess(coords)
        if phi.ndim == 1:
            phi = phi.unsqueeze(-1)
        return phi, grad_field, H
    return _autodiff_hessian(model, coords, chunk_size=chunk_size, detach_params=False)


def compute_hessian_eval(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior = model.training
    model.eval()
    try:
        with torch.inference_mode() if getattr(model, "is_linear", False) else torch.no_grad():
            phi, g, H = compute_hessian(model, coords, chunk_size=chunk_size, device=device)
        return phi.detach().cpu(), g.detach().cpu(), H.detach().cpu()
    finally:
        model.train(prior)


def compute_ftg_trainable(model: nn.Module, coords: torch.Tensor):
    if hasattr(model, "potential_grad_hess"):
        return model.potential_grad_hess(coords)
    phi = model(coords)
    grad_field, H = model.gradient_and_hessian(coords)
    return phi, grad_field, H


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #


@dataclass
class RFFEnsemble:
    n_members: int
    model_kwargs: Dict
    base_seed: int = 404
    bootstrap: bool = True
    bagging_frac: float = 1.0
    device: Optional[str] = None
    keep_members: bool = True

    members: List[NeuralFourierField] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    histories: List[Tuple[np.ndarray, List[int]]] = field(default_factory=list)

    def _make_member(self, seed: int) -> NeuralFourierField:
        kwargs = dict(self.model_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        kwargs["seed"] = seed
        return NeuralFourierField(**kwargs)

    def fit(self, coords, data, grid=None, **fit_kwargs) -> "RFFEnsemble":
        coords_t = coords if isinstance(coords, torch.Tensor) else torch.as_tensor(coords)
        data_t = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
        n = coords_t.shape[0]
        rng = np.random.default_rng(self.base_seed + 12345)

        self.members.clear()
        self.histories.clear()
        self.seeds = [self.base_seed + i for i in range(self.n_members)]

        for i, seed in enumerate(self.seeds):
            print(f"Model #{i}")
            _set_all_seeds(seed)
            member = self._make_member(seed)
            if self.bootstrap:
                boot = rng.choice(n, size=max(1, int(self.bagging_frac * n)), replace=True)
                c_i, d_i = coords_t[boot], data_t[boot]
            else:
                c_i, d_i = coords_t, data_t
            history = member.fit(c_i, d_i, grid=coords if grid is None else grid, **fit_kwargs)
            self.histories.append(history)
            if self.keep_members:
                self.members.append(member)
            else:
                del member
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self

    def predict(
        self,
        coords: np.ndarray,
        output: str = "hessian",
        chunk_size: int = 2048,
        normalize_grad: bool = False,
        return_std: bool = True,
        return_quantiles: Optional[Tuple[float, float]] = None,
    ):
        if not self.members:
            raise RuntimeError("Ensemble has no trained members. Call fit() first.")
        stack = np.stack(
            [
                m.predict(coords, output=output, chunk_size=chunk_size, normalize_grad=normalize_grad)
                for m in tqdm(self.members, desc="Evaluating")
            ],
            axis=0,
        )
        mean = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=1) if return_std else None
        qpair = None
        if return_quantiles is not None:
            q_lo, q_hi = return_quantiles
            qpair = (np.quantile(stack, q_lo, axis=0), np.quantile(stack, q_hi, axis=0))
        return mean, std, qpair


__all__ = [
    "NeuralFourierField",
    "RFFEnsemble",
    "EarlyStopper",
    "RasterGrid",
    "write_ascii_raster",
    "FTG_COMPONENT_NAMES",
    "compute_gradient",
    "compute_hessian",
    "compute_hessian_eval",
    "compute_ftg_trainable",
    "poisson_disk_indices",
    "add_ftg_noise_by_snr",
    "tensor6_to_matrix",
    "matrix_to_tensor6",
    "prepare_ftg_data",
]
