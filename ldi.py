"""
Workshop helpers for a single-layer restoration.

Two representations of the same map, scored on the same observations:

* :class:`DisplacementField` — a random-Fourier neural field that predicts a
  displacement ``u(x)``. The trial restoration is ``Φ⁻¹(x) = x + u(x)``.
* :class:`EulerVelocity` and :class:`RestoringFlow` — one Fourier potential
  ``α``, the divergence-free velocity ``v = ∇α × e_z``, and an RK4 integral
  of that velocity. This is the two-dimensional restore event, kept in this file.

``load_analogue`` reads the digitised parallel fold (``parallelFoldLayers.svg``).
``load_recumbent_fold`` reads the GemPy model-3 section, the Y = 500 m slice.
"""

from __future__ import annotations

import csv
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from svgpathtools import parse_path
from tqdm import tqdm

Array = np.ndarray
MapFn = Callable[[Array], Array]
JacFn = Callable[[Array], Array]

SVG_NS = "http://www.w3.org/2000/svg"


# --------------------------------------------------------------------------- #
# Analogue section
# --------------------------------------------------------------------------- #


def _tag(element) -> str:
    name = element.tag
    return name.split("}", 1)[-1] if name.startswith("{") else name


def _parse_svg_length(value: str) -> float:
    """SVG length to pixels at 96 dpi. Bare numbers are returned as-is."""
    conversions = {
        "mm": 3.779527559,
        "cm": 37.79527559,
        "in": 96.0,
        "pt": 96.0 / 72.0,
        "pc": 16.0,
        "px": 1.0,
    }
    text = value.strip()
    for unit, factor in conversions.items():
        if text.endswith(unit):
            return float(text[: -len(unit)].strip()) * factor
    return float(text)


def _viewbox_height(root) -> float:
    """User-unit height. Path data live in the viewBox, not in millimetres."""
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[3])
    return _parse_svg_length(root.get("height", "0") or "0")


def _group_transform(element) -> Tuple[float, float, float, float]:
    """Read ``translate(tx ty)`` and ``scale(sx sy)`` from a group."""
    tx, ty, sx, sy = 0.0, 0.0, 1.0, 1.0
    text = element.get("transform", "") or ""
    translated = re.search(r"translate\(\s*([-\d.eE]+)(?:[\s,]+([-\d.eE]+))?\s*\)", text)
    if translated:
        tx = float(translated.group(1))
        ty = float(translated.group(2) or 0.0)
    scaled = re.search(r"scale\(\s*([-\d.eE]+)(?:[\s,]+([-\d.eE]+))?\s*\)", text)
    if scaled:
        sx = float(scaled.group(1))
        sy = float(scaled.group(2) or sx)
    return tx, ty, sx, sy


def _points_from_polyline(element) -> Optional[Array]:
    raw = element.get("points", "")
    nums = np.fromstring(re.sub(r"[,\s]+", " ", raw.strip()), sep=" ")
    if nums.size < 4:
        return None
    return nums.reshape(-1, 2)


def _points_from_path(element, n_samples: int) -> Optional[Array]:
    spec = element.get("d", "")
    if not spec:
        return None
    try:
        path = parse_path(spec)
    except Exception:
        return None
    if path.length() == 0:
        return None
    ts = np.linspace(0.0, 1.0, n_samples)
    return np.array([[path.point(t).real, path.point(t).imag] for t in ts], dtype=np.float64)


def load_svg_layers(
    path: str,
    *,
    flip_y: bool = True,
    n_samples: int = 80,
    min_points: int = 3,
) -> Dict[str, Array]:
    """
    Digitised layer polylines from an SVG.

    Each ``path``, ``polyline`` or ``polygon`` becomes one ``(N, 2)`` array,
    in document order. ``flip_y`` turns SVG's downward axis upward.
    """
    root = ET.parse(path).getroot()
    height = _viewbox_height(root)
    layers: Dict[str, Array] = {}

    def _store(element, group: str, tx: float, ty: float, sx: float, sy: float) -> None:
        kind = _tag(element)
        if kind in ("polyline", "polygon"):
            pts = _points_from_polyline(element)
        elif kind == "path":
            pts = _points_from_path(element, n_samples)
        else:
            return
        if pts is None or len(pts) < min_points:
            return
        pts = pts.copy()
        pts[:, 0] = pts[:, 0] * sx + tx
        pts[:, 1] = pts[:, 1] * sy + ty
        if flip_y:
            origin = height if height > 0 else float(pts[:, 1].max())
            pts[:, 1] = origin - pts[:, 1]
        name = element.get("id") or group
        if name in layers:
            name = f"{name}_{sum(1 for key in layers if key.startswith(name))}"
        layers[name] = pts

    for child in root:
        kind = _tag(child)
        if kind == "g":
            tx, ty, sx, sy = _group_transform(child)
            group = child.get("id") or "layer"
            for element in child:
                _store(element, group, tx, ty, sx, sy)
        elif kind in ("polyline", "polygon", "path"):
            _store(child, "layer", 0.0, 0.0, 1.0, 1.0)
    if not layers:
        raise ValueError(f"no layer geometry found in {path}")
    return layers


def layer_normals(layers: Dict[str, Array]) -> Tuple[Array, Array, Array]:
    """
    Unit bedding normals from each polyline.

    The tangent is a central difference along the trace. The normal is that
    tangent rotated 90° counter-clockwise, so a left-to-right horizontal
    contact youngs upward.
    """
    points: List[Array] = []
    normals: List[Array] = []
    layer_ids: List[Array] = []
    for lid, pts in enumerate(layers.values()):
        tangent = np.zeros_like(pts)
        tangent[1:-1] = pts[2:] - pts[:-2]
        tangent[0] = pts[1] - pts[0]
        tangent[-1] = pts[-1] - pts[-2]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-8)
        points.append(pts)
        normals.append(normal)
        layer_ids.append(np.full(len(pts), lid, dtype=np.int64))
    return np.vstack(points), np.vstack(normals), np.concatenate(layer_ids)


@dataclass
class AnalogueFold:
    """Sparse fit points, full traces, and bedding normals, recentered."""

    names: List[str]
    horizons: List[Array]
    horizons_full: List[Array]
    gp: Array
    gv: Array
    center: Array

    @property
    def extent(self) -> float:
        pts = np.vstack(self.horizons_full)
        return float(np.ptp(pts, axis=0).max())


def length_scales_for(data: AnalogueFold) -> Tuple[float, float]:
    """Wavelength band from the section width: short enough for the hinge, long enough for a tilt."""
    extent = data.extent
    short = max(0.5 * extent, 1e-3)
    long = max(3.0 * extent, short)
    return (float(short), float(long))


def load_analogue(
    path: str,
    *,
    n_orientation: int = 10,
    n_horizon: int = 5,
    n_path_samples: int = 80,
    seed: int = 42,
    point_normals_up: bool = True,
) -> AnalogueFold:
    """
    Digitised contacts, subsampled for the fit.

    ``n_orientation`` normals and ``n_horizon`` points per trace are the fit.
    ``horizons_full`` keeps every sample of each contact for the figures.
    Coordinates are shifted so the mean of the bedding points sits at the origin.
    With ``point_normals_up``, a downward normal is reversed so younging points
    up, the upright-fold convention. Leave it off on a recumbent fold so an
    overturned limb keeps the younging the tangent already carries.
    """
    layers = load_svg_layers(path, flip_y=True, n_samples=n_path_samples)
    gp, gv, _ids = layer_normals(layers)
    center = gp.mean(axis=0)

    rng = random.Random(seed)
    take = rng.sample(range(len(gp)), k=min(n_orientation, len(gp)))
    gp = gp[take] - center
    gv = gv[take].copy()
    if point_normals_up:
        gv[gv[:, 1] < 0.0] *= -1.0

    names = list(layers)
    horizons_full = [(pts - center) for pts in layers.values()]
    horizons = []
    for trace in horizons_full:
        k = min(n_horizon, len(trace))
        horizons.append(trace[rng.sample(range(len(trace)), k=k)])

    return AnalogueFold(
        names=names,
        horizons=horizons,
        horizons_full=horizons_full,
        gp=np.asarray(gp, dtype=np.float64),
        gv=np.asarray(gv, dtype=np.float64),
        center=np.asarray(center, dtype=np.float64),
    )


def _section_normal(azimuth_deg: float, dip_deg: float) -> Array:
    """GemPy azimuth/dip as a unit normal in the vertical section, components (x, z)."""
    az = math.radians(float(azimuth_deg))
    dp = math.radians(float(dip_deg))
    normal = np.array(
        [math.sin(az) * math.sin(dp), math.cos(dp)],
        dtype=np.float64,
    )
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    return normal


def _fold_trace(points: Array) -> Array:
    """Upper limb left to right, the hinge, then the lower limb right to left."""
    by_x: Dict[float, List[float]] = {}
    for x, z in np.asarray(points, dtype=np.float64):
        by_x.setdefault(float(x), []).append(float(z))
    upper: List[Tuple[float, float]] = []
    lower: List[Tuple[float, float]] = []
    hinge: List[Tuple[float, float]] = []
    for x in sorted(by_x):
        heights = sorted(by_x[x])
        if len(heights) == 1:
            hinge.append((x, heights[0]))
        else:
            lower.append((x, heights[0]))
            upper.append((x, heights[-1]))
    return np.asarray(upper + hinge + lower[::-1], dtype=np.float64)


def load_recumbent_fold(
    surface_points: str,
    orientations: str,
    *,
    section_y: float = 500.0,
) -> AnalogueFold:
    """
    GemPy tutorial model 3 on the slice that holds the bedding measurements.

    ``rock2`` is the younger contact and ``rock1`` the older one, so the order
    term in the fit asks the younger horizon to restore above the older.
    Every contact point on ``Y = section_y`` enters the fit. Normals are the
    section components of the GemPy azimuth/dip, including the downward
    younging on the overturned limb. Coordinates are shifted to the mean of
    the contacts and the normals, and the vertical axis is stored in column 1.
    """
    contacts: Dict[str, List[Tuple[float, float]]] = {}
    with open(surface_points, newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Y"]) - float(section_y)) > 1e-6:
                continue
            contacts.setdefault(row["formation"], []).append(
                (float(row["X"]), float(row["Z"]))
            )
    missing = [name for name in ("rock2", "rock1") if name not in contacts]
    if missing:
        raise ValueError(f"section Y={section_y} is missing {missing}")

    names = ["Younger horizon", "Older horizon"]
    traces = [_fold_trace(contacts["rock2"]), _fold_trace(contacts["rock1"])]

    gp_rows: List[Array] = []
    gv_rows: List[Array] = []
    with open(orientations, newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Y"]) - float(section_y)) > 1e-6:
                continue
            gp_rows.append(np.array([float(row["X"]), float(row["Z"])], dtype=np.float64))
            gv_rows.append(_section_normal(float(row["azimuth"]), float(row["dip"])))
    if not gp_rows:
        raise ValueError(f"no orientations on Y={section_y}")

    gp = np.vstack(gp_rows)
    gv = np.vstack(gv_rows)
    center = np.vstack(traces + [gp]).mean(axis=0)
    horizons = [trace - center for trace in traces]
    return AnalogueFold(
        names=names,
        horizons=[h.copy() for h in horizons],
        horizons_full=horizons,
        gp=gp - center,
        gv=gv,
        center=np.asarray(center, dtype=np.float64),
    )


def grid_axes(
    data: AnalogueFold,
    *,
    nx: int = 90,
    ny: int = 60,
    margin: float = 0.12,
) -> Tuple[Array, Array]:
    """Axis vectors for a regular section covering the full traces."""
    pts = np.vstack(data.horizons_full)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - margin * span
    hi = hi + margin * span
    xs = np.linspace(lo[0], hi[0], int(nx))
    ys = np.linspace(lo[1], hi[1], int(ny))
    return xs, ys


# --------------------------------------------------------------------------- #
# Direct displacement field
# --------------------------------------------------------------------------- #


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw_wavenumbers(n: int, length_scales: Sequence[float], seed: int) -> torch.Tensor:
    """Log-uniform wavelengths, direction uniform on a half-turn. ``k = 2π / λ``."""
    lo, hi = float(length_scales[0]), float(length_scales[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"length scales must satisfy 0 < min <= max, got {length_scales}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    log_lambda = torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=generator)
    theta = torch.empty(n).uniform_(0.0, math.pi, generator=generator)
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    return (2.0 * math.pi / torch.exp(log_lambda)).unsqueeze(-1) * direction


def _curlew_weight(spec: float, value: torch.Tensor) -> float:
    """Weight for one loss term, frozen from its value on the first call.

    A Curlew ``HSet`` entry written as a string (``"1.0"``, ``"0.1"``) is
    replaced, on the first loss evaluation, by ``spec / initial_loss`` and
    then held fixed. A non-positive initial term keeps weight 0. The floor
    ``1e-8`` is the one in ``RestorationField._push_term``.
    """
    level = float(value.detach())
    if level <= 0.0:
        return 0.0
    return float(spec) / max(level, 1e-8)


def _fit_loop(
    module: nn.Module,
    loss_fn,
    *,
    epochs: int,
    lr: float,
    patience: Optional[int],
    tol: float,
    verbose: bool,
    desc: str,
    targets: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """Adam on flatness, bedding and order. Weights freeze after the first epoch."""
    weights = {"grad": 10.0, "eq": 1.0, "iq": 1.0}
    if targets:
        weights.update(targets)
    module.scales = {}
    module.history = []
    optimiser = torch.optim.Adam(module.parameters(), lr=lr)
    best = math.inf
    best_state = None
    stale = 0
    bar = tqdm(range(epochs), desc=desc, disable=not verbose)
    for _epoch in bar:
        raw = loss_fn()
        if not module.scales:
            module.scales = {
                key: _curlew_weight(target, raw[key]) for key, target in weights.items()
            }
        weighted = {key: module.scales[key] * raw[key] for key in weights}
        total = sum(weighted.values())
        value = float(total.detach())
        row = {key: float(weighted[key].detach()) for key in ("grad", "eq", "iq")}
        row["total"] = value
        module.history.append(row)
        if verbose:
            bar.set_postfix(
                grad=f"{row['grad']:.3g}",
                eq=f"{row['eq']:.3g}",
                iq=f"{row['iq']:.3g}",
                total=f"{value:.3g}",
            )
        if not math.isfinite(value):
            break
        if value < best - tol:
            best = value
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if patience is not None and stale > patience:
            break
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        optimiser.step()
    if best_state is not None:
        module.load_state_dict(best_state)
    return module.history


def _normal_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Squared angle between unit normals. Anti-parallel counts as π, not 0."""
    pred_n = F.normalize(predicted, dim=-1)
    obs_n = F.normalize(observed, dim=-1)
    dot = (pred_n * obs_n).sum(dim=-1).clamp(-1.0, 1.0)
    cross = pred_n[..., 0] * obs_n[..., 1] - pred_n[..., 1] * obs_n[..., 0]
    theta = torch.atan2(cross.abs().clamp_min(1e-12), dot)
    return theta.square().mean()


class DisplacementField(nn.Module):
    """
    Neural field ``u(x)`` and the map ``Φ⁻¹(x) = x + u(x)``.

    Restored depth is the vertical component of that map. Flatness, bedding
    and stratigraphic order are scored on restored depth exactly as in a
    Curlew restore event with a flat reference. The Jacobian ``I + Du`` is
    free: nothing in :meth:`fit` asks ``det J`` to stay positive, or near 1.

    A constant shift of the restored frame is removed after fitting
    (``pin``), so the anchor stays put. The shift does not enter the loss
    and does not change ``det J``.
    """

    def __init__(
        self,
        *,
        num_fourier_features: int = 128,
        length_scales: Sequence[float] = (20.0, 200.0),
        hidden_layers: Sequence[int] = (128, 128),
        depth_axis: int = 1,
        seed: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if depth_axis not in (0, 1):
            raise ValueError("depth_axis must be 0 or 1")
        if num_fourier_features < 0:
            raise ValueError("num_fourier_features must be >= 0")
        _set_seed(seed)
        self.num_fourier_features = int(num_fourier_features)
        self.depth_axis = int(depth_axis)
        self.seed = int(seed)
        self.scales: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

        if self.num_fourier_features > 0:
            wavenumbers = _draw_wavenumbers(self.num_fourier_features, length_scales, seed)
        else:
            wavenumbers = torch.zeros(0, 2)
        self.register_buffer("k", wavenumbers)
        self.register_buffer("shift", torch.zeros(2))

        width = 2 * self.num_fourier_features if self.num_fourier_features else 2
        blocks: List[nn.Module] = []
        for hidden in hidden_layers:
            blocks.append(nn.Linear(width, int(hidden)))
            blocks.append(nn.SiLU())
            width = int(hidden)
        blocks.append(nn.Linear(width, 2))
        self.mlp = nn.Sequential(*blocks)
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        chosen = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(chosen)

    def _as_tensor(self, xy: Union[Array, torch.Tensor]) -> torch.Tensor:
        if isinstance(xy, torch.Tensor):
            return xy.to(device=self.shift.device, dtype=self.shift.dtype)
        return torch.as_tensor(np.asarray(xy), device=self.shift.device, dtype=self.shift.dtype)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_fourier_features == 0:
            return x
        projected = x @ self.k.T
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)

    def _displacement(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._features(x))

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._displacement(x) + self.shift

    def inverse_map(self, xy: Union[Array, torch.Tensor]) -> Array:
        """Restored coordinates, ``(N, 2)`` numpy."""
        with torch.no_grad():
            return self._inverse(self._as_tensor(xy)).detach().cpu().numpy()

    def jacobian(self, xy: Union[Array, torch.Tensor], chunk_size: int = 2048) -> Array:
        """Autodiff Jacobian of ``Φ⁻¹``, shape ``(N, 2, 2)``."""
        pts = self._as_tensor(xy).detach()
        enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            differentiate = torch.func.jacrev(lambda p: self._inverse(p.unsqueeze(0)).squeeze(0))
            blocks = [
                torch.func.vmap(differentiate)(pts[start : start + chunk_size])
                for start in range(0, pts.shape[0], chunk_size)
            ]
            stacked = torch.cat(blocks, dim=0)
        finally:
            torch.set_grad_enabled(enabled)
        return stacked.detach().cpu().numpy()

    def pin(self, point: Union[Array, Sequence[float]]) -> "DisplacementField":
        """Hold one present-day point fixed. Applied after fitting."""
        anchor = self._as_tensor(np.asarray(point, dtype=np.float32).reshape(1, 2))
        with torch.no_grad():
            self.shift.copy_(-self._displacement(anchor).squeeze(0))
        return self

    def _batch(
        self,
        gp: torch.Tensor,
        gv: torch.Tensor,
        horizons: Sequence[torch.Tensor],
        n_iq: int,
    ) -> Dict[str, torch.Tensor]:
        x = gp.detach().requires_grad_(True)
        depth = self._inverse(x)[:, self.depth_axis]
        gradient = torch.autograd.grad(depth.sum(), x, create_graph=True)[0]
        grad_loss = _normal_loss(gradient, gv)

        eq_terms = []
        for trace in horizons:
            values = self._inverse(trace)[:, self.depth_axis]
            centre = values.mean().detach()
            eq_terms.append((values - centre).square().mean())
        eq_loss = torch.stack(eq_terms).mean()

        iq_terms = []
        for younger, older in zip(horizons, horizons[1:]):
            iy = torch.randint(0, younger.shape[0], (n_iq,), device=younger.device)
            io = torch.randint(0, older.shape[0], (n_iq,), device=older.device)
            y_depth = self._inverse(younger[iy])[:, self.depth_axis]
            o_depth = self._inverse(older[io])[:, self.depth_axis]
            # ">" in a Curlew CSet: younger restored depth sits at or above older.
            iq_terms.append(torch.relu(o_depth - y_depth).square().mean())
        iq_loss = torch.stack(iq_terms).mean() if iq_terms else depth.new_zeros(())
        return {"grad": grad_loss, "eq": eq_loss, "iq": iq_loss}

    def fit(
        self,
        data: AnalogueFold,
        *,
        epochs: int = 250,
        lr: float = 1e-2,
        n_iq: int = 8,
        targets: Optional[Dict[str, float]] = None,
        patience: Optional[int] = 40,
        tol: float = 1e-4,
        pin: bool = True,
        anchor: Optional[Sequence[float]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Fit flatness, bedding and order.

        ``targets`` are the Curlew string weights (bedding 10, flatness 1,
        order 1, the analogue ``HSet`` ratio). On the first epoch each weight
        becomes ``target / loss`` and then stays fixed, so the three
        contributions that are logged and differentiated start at those
        targets.
        """
        if data.gp.shape[0] == 0 or len(data.horizons) < 2:
            raise ValueError("analogue fold needs bedding normals and at least two horizons")
        gp = self._as_tensor(data.gp)
        gv = self._as_tensor(data.gv)
        horizons = [self._as_tensor(trace) for trace in data.horizons]
        self.shift.zero_()
        _fit_loop(
            self,
            lambda: self._batch(gp, gv, horizons, n_iq),
            epochs=epochs,
            lr=lr,
            patience=patience,
            tol=tol,
            verbose=verbose,
            desc="displacement",
            targets=targets,
        )
        if pin:
            self.pin(np.zeros(2) if anchor is None else anchor)
        return self.history


# --------------------------------------------------------------------------- #
# Euler velocity and its RK4 flow
# --------------------------------------------------------------------------- #


class EulerVelocity(nn.Module):
    """
    Two-dimensional Euler velocity from one Fourier potential.

    ``α`` is a linear Fourier series with the same normalisation as Curlew's
    ``FSF``. Each mode is divided by ``|k|^{freq_damp}`` (default 2), and the
    sum is scaled by ``s/√F``:

        α(x) = (s/√F) Σ_k (A_k cos(k·x) + B_k sin(k·x)) / |k|^{freq_damp}

    The second potential is the fixed section normal ``e_z``, so

        v = ∇α × e_z = (−∂α/∂y, ∂α/∂x).

    The Jacobian of ``v`` has trace zero for any amplitudes: the velocity is
    divergence-free before it is fitted. Amplitudes start at zero, which is
    a zero velocity and an identity map once the flow is integrated.
    """

    def __init__(
        self,
        *,
        num_fourier_features: int = 128,
        length_scales: Sequence[float] = (20.0, 200.0),
        scale: float = 1.0,
        freq_damp: float = 2.0,
        seed: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if num_fourier_features < 1:
            raise ValueError("num_fourier_features must be >= 1")
        if freq_damp < 0:
            raise ValueError(f"freq_damp must be non-negative, got {freq_damp}")
        _set_seed(seed)
        self.num_fourier_features = int(num_fourier_features)
        self.freq_damp = float(freq_damp)
        self.seed = int(seed)
        wavenumbers = _draw_wavenumbers(self.num_fourier_features, length_scales, seed)
        self.register_buffer("k", wavenumbers)
        wave = wavenumbers.norm(dim=1).clamp_min(1e-8)
        if self.freq_damp == 0.0:
            denom = torch.ones(self.num_fourier_features)
        else:
            denom = wave.pow(self.freq_damp)
        self.register_buffer("amplitude_denom", denom)
        self.register_buffer(
            "series_scale",
            torch.tensor(float(scale) / math.sqrt(self.num_fourier_features)),
        )
        self.A = nn.Parameter(torch.zeros(self.num_fourier_features))
        self.B = nn.Parameter(torch.zeros(self.num_fourier_features))
        chosen = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(chosen)

    def randomise(self, std: float = 1.0) -> "EulerVelocity":
        """Draw amplitudes. Used to inspect a nonzero velocity before fitting."""
        with torch.no_grad():
            self.A.normal_(0.0, float(std))
            self.B.normal_(0.0, float(std))
        return self

    def _as_tensor(self, xy: Union[Array, torch.Tensor]) -> torch.Tensor:
        if isinstance(xy, torch.Tensor):
            return xy.to(device=self.A.device, dtype=self.A.dtype)
        return torch.as_tensor(np.asarray(xy), device=self.A.device, dtype=self.A.dtype)

    def _grad_hess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """∇α and the Hessian, both closed form. ``x`` is ``(N, 2)``."""
        theta = x @ self.k.T
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        weight = self.series_scale / self.amplitude_denom
        d_dtheta = weight * (-self.A * sine + self.B * cosine)
        d2_dtheta = weight * (-(self.A * cosine + self.B * sine))
        grad = d_dtheta @ self.k
        outer = self.k[:, :, None] * self.k[:, None, :]
        hess = torch.einsum("nm,mij->nij", d2_dtheta, outer)
        return grad, hess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Velocity ``(N, 2)``."""
        grad, _hess = self._grad_hess(x)
        return torch.stack((-grad[:, 1], grad[:, 0]), dim=-1)

    def forward_and_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Velocity and ``∂v/∂x``, shapes ``(N, 2)`` and ``(N, 2, 2)``."""
        grad, hess = self._grad_hess(x)
        velocity = torch.stack((-grad[:, 1], grad[:, 0]), dim=-1)
        jac = torch.stack(
            (
                torch.stack((-hess[:, 1, 0], -hess[:, 1, 1]), dim=-1),
                torch.stack((hess[:, 0, 0], hess[:, 0, 1]), dim=-1),
            ),
            dim=-2,
        )
        return velocity, jac


class RestoringFlow(nn.Module):
    """
    Inverse map obtained by integrating an :class:`EulerVelocity` with RK4.

    ``n_steps`` equal substeps run from τ = 1 to τ = 0. The Jacobian is the
    derivative of that discrete step, so ``det J = 1 + O(Δτ⁴)`` with
    ``Δτ = 1/n_steps``. Restored depth is the vertical component of the
    image point. Flatness, bedding and order match :class:`DisplacementField`.

    A constant shift of the restored frame is removed after fitting. It does
    not enter the loss and does not change ``det J``.
    """

    def __init__(
        self,
        *,
        num_fourier_features: int = 128,
        length_scales: Sequence[float] = (20.0, 200.0),
        scale: float = 1.0,
        freq_damp: float = 2.0,
        n_steps: int = 8,
        depth_axis: int = 1,
        seed: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if int(n_steps) < 1:
            raise ValueError("n_steps must be >= 1")
        if depth_axis not in (0, 1):
            raise ValueError("depth_axis must be 0 or 1")
        self.velocity = EulerVelocity(
            num_fourier_features=num_fourier_features,
            length_scales=length_scales,
            scale=scale,
            freq_damp=freq_damp,
            seed=seed,
            device=device,
        )
        self.n_steps = int(n_steps)
        self.depth_axis = int(depth_axis)
        self.scales: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []
        self.register_buffer("shift", torch.zeros(2))
        chosen = torch.device(device) if device is not None else self.velocity.A.device
        self.to(chosen)

    def _as_tensor(self, xy: Union[Array, torch.Tensor]) -> torch.Tensor:
        return self.velocity._as_tensor(xy)

    def _integrate(self, x: torch.Tensor, *, jac: bool):
        """RK4 for ``dx/dτ = v(x)`` run backwards. Optional chain-rule Jacobian."""
        h = -1.0 / self.n_steps
        eye = torch.eye(2, dtype=x.dtype, device=x.device)
        J = eye.expand(x.shape[0], -1, -1).clone() if jac else None
        for _step in range(self.n_steps):
            if jac:
                k1, L1 = self.velocity.forward_and_jacobian(x)
                k2, L2 = self.velocity.forward_and_jacobian(x + (h / 2.0) * k1)
                k3, L3 = self.velocity.forward_and_jacobian(x + (h / 2.0) * k2)
                k4, L4 = self.velocity.forward_and_jacobian(x + h * k3)
                d1 = L1
                d2 = torch.bmm(L2, eye + (h / 2.0) * d1)
                d3 = torch.bmm(L3, eye + (h / 2.0) * d2)
                d4 = torch.bmm(L4, eye + h * d3)
                J = torch.bmm(eye + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4), J)
            else:
                k1 = self.velocity.forward(x)
                k2 = self.velocity.forward(x + (h / 2.0) * k1)
                k3 = self.velocity.forward(x + (h / 2.0) * k2)
                k4 = self.velocity.forward(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (x, J) if jac else x

    def _image(self, x: torch.Tensor) -> torch.Tensor:
        return self._integrate(x, jac=False) + self.shift

    def inverse_map(self, xy: Union[Array, torch.Tensor]) -> Array:
        """Restored coordinates, ``(N, 2)`` numpy."""
        with torch.no_grad():
            return self._image(self._as_tensor(xy)).detach().cpu().numpy()

    def jacobian(self, xy: Union[Array, torch.Tensor], chunk_size: int = 1500) -> Array:
        """Jacobian of the discrete RK4 map, shape ``(N, 2, 2)``."""
        pts = self._as_tensor(xy).detach()
        enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            blocks = []
            for start, stop in _chunk_ranges(pts.shape[0], chunk_size):
                _x0, jac = self._integrate(pts[start:stop], jac=True)
                blocks.append(jac)
            stacked = torch.cat(blocks, dim=0)
        finally:
            torch.set_grad_enabled(enabled)
        return stacked.detach().cpu().numpy()

    def pin(self, point: Union[Array, Sequence[float]]) -> "RestoringFlow":
        """Hold one present-day point fixed. Applied after fitting."""
        anchor = self._as_tensor(np.asarray(point, dtype=np.float32).reshape(1, 2))
        with torch.no_grad():
            image = self._integrate(anchor, jac=False)
            self.shift.copy_(-(image - anchor).squeeze(0))
        return self

    def _batch(
        self,
        gp: torch.Tensor,
        gv: torch.Tensor,
        horizons: Sequence[torch.Tensor],
        n_iq: int,
    ) -> Dict[str, torch.Tensor]:
        x0, jac = self._integrate(gp, jac=True)
        depth = x0[:, self.depth_axis] + self.shift[self.depth_axis]
        gradient = jac[:, self.depth_axis, :]
        grad_loss = _normal_loss(gradient, gv)

        eq_terms = []
        for trace in horizons:
            values = self._image(trace)[:, self.depth_axis]
            centre = values.mean().detach()
            eq_terms.append((values - centre).square().mean())
        eq_loss = torch.stack(eq_terms).mean()

        iq_terms = []
        for younger, older in zip(horizons, horizons[1:]):
            iy = torch.randint(0, younger.shape[0], (n_iq,), device=younger.device)
            io = torch.randint(0, older.shape[0], (n_iq,), device=older.device)
            y_depth = self._image(younger[iy])[:, self.depth_axis]
            o_depth = self._image(older[io])[:, self.depth_axis]
            iq_terms.append(torch.relu(o_depth - y_depth).square().mean())
        iq_loss = torch.stack(iq_terms).mean() if iq_terms else depth.new_zeros(())
        return {"grad": grad_loss, "eq": eq_loss, "iq": iq_loss}

    def fit(
        self,
        data: AnalogueFold,
        *,
        epochs: int = 250,
        lr: float = 5e-2,
        n_iq: int = 8,
        targets: Optional[Dict[str, float]] = None,
        patience: Optional[int] = 40,
        tol: float = 1e-4,
        pin: bool = True,
        anchor: Optional[Sequence[float]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Fit flatness, bedding and order by integrating the Euler velocity."""
        if data.gp.shape[0] == 0 or len(data.horizons) < 2:
            raise ValueError("analogue fold needs bedding normals and at least two horizons")
        gp = self._as_tensor(data.gp)
        gv = self._as_tensor(data.gv)
        horizons = [self._as_tensor(trace) for trace in data.horizons]
        self.shift.zero_()
        _fit_loop(
            self,
            lambda: self._batch(gp, gv, horizons, n_iq),
            epochs=epochs,
            lr=lr,
            patience=patience,
            tol=tol,
            verbose=verbose,
            desc="flow",
            targets=targets,
        )
        if pin:
            self.pin(np.zeros(2) if anchor is None else anchor)
        return self.history


# --------------------------------------------------------------------------- #
# Diagnostics shared by both maps
# --------------------------------------------------------------------------- #


def _chunk_ranges(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def depth_std(inverse_map: MapFn, traces: Sequence[Array], depth_axis: int = 1) -> Array:
    """Standard deviation of restored depth along each trace."""
    spreads = []
    for trace in traces:
        restored = np.asarray(inverse_map(trace))
        spreads.append(float(restored[:, depth_axis].std()))
    return np.asarray(spreads, dtype=np.float64)


def determinant_grid(jacobian: JacFn, xs: Array, ys: Array) -> Array:
    """``det J`` on the tensor-product grid, shape ``(len(ys), len(xs))``."""
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    jac = np.asarray(jacobian(pts))
    return np.linalg.det(jac).reshape(xx.shape)


def summarise_restoration(
    name: str,
    inverse_map: MapFn,
    jacobian: JacFn,
    data: AnalogueFold,
    xs: Array,
    ys: Array,
    *,
    depth_axis: int = 1,
) -> Dict[str, Union[float, Array]]:
    """Depth scatter on the fit points and on the full traces, plus ``det J`` on the grid."""
    sparse = depth_std(inverse_map, data.horizons, depth_axis)
    full = depth_std(inverse_map, data.horizons_full, depth_axis)
    det = determinant_grid(jacobian, xs, ys)
    finite = det[np.isfinite(det)]
    report: Dict[str, Union[float, Array]] = {
        "name": name,
        "sparse_std": sparse,
        "full_std": full,
        "mean_sparse_std": float(sparse.mean()),
        "mean_full_std": float(full.mean()),
        "det_min": float(finite.min()),
        "det_max": float(finite.max()),
        "det_mean_abs_dev": float(np.mean(np.abs(finite - 1.0))),
        "det_max_abs_dev": float(np.max(np.abs(finite - 1.0))),
        "fraction_negative": float(np.mean(finite < 0.0)),
        "det": det,
    }
    print(f"{name}")
    print(
        f"  restored-depth std   sparse {report['mean_sparse_std']:.3f}"
        f"    full traces {report['mean_full_std']:.3f}"
    )
    for i, spread in enumerate(full):
        print(f"    horizon {i} ({data.names[i]})  std φ = {spread:.3f}")
    print(
        f"  det J   min {report['det_min']:.4f}   max {report['det_max']:.4f}"
        f"   mean |det J - 1| {report['det_mean_abs_dev']:.3e}"
        f"   max |det J - 1| {report['det_max_abs_dev']:.3e}"
        f"   fraction det J < 0  {report['fraction_negative']:.3f}"
    )
    return report


def velocity_snapshot(
    velocity: EulerVelocity,
    xy: Array,
    *,
    chunk_size: int = 2048,
) -> Tuple[Array, Array]:
    """
    Speed and divergence of an :class:`EulerVelocity` on ``xy``.

    Divergence is the trace of ``∂v/∂x``. It is zero up to floating-point
    roundoff for any amplitudes, before any observations are fitted.
    """
    pts = np.asarray(xy, dtype=np.float32)
    speeds = []
    divs = []
    with torch.no_grad():
        for start, stop in _chunk_ranges(len(pts), chunk_size):
            batch = velocity._as_tensor(pts[start:stop])
            v, jac = velocity.forward_and_jacobian(batch)
            speeds.append(v.norm(dim=-1).detach().cpu().numpy())
            divs.append(jac.diagonal(dim1=-2, dim2=-1).sum(dim=-1).detach().cpu().numpy())
    return np.concatenate(speeds), np.concatenate(divs)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _colors(n: int):
    return [plt.cm.tab10(i % 10) for i in range(n)]


def _equal(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _overlay_traces(ax, data: AnalogueFold, lw: float = 1.0) -> None:
    for color, trace in zip(_colors(len(data.horizons_full)), data.horizons_full):
        ax.plot(trace[:, 0], trace[:, 1], color=color, lw=lw)


def plot_constraints(
    data: AnalogueFold,
    *,
    title: str = "Analogue fold — fit points and bedding",
) -> None:
    """Full contacts, the points that enter the fit, and the bedding normals."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    span = max(data.extent, 1e-6)
    for color, full, sparse in zip(_colors(len(data.names)), data.horizons_full, data.horizons):
        ax.plot(full[:, 0], full[:, 1], color=color, lw=1.2, alpha=0.85)
        ax.scatter(sparse[:, 0], sparse[:, 1], s=28, color=color, zorder=3, edgecolors="k", linewidths=0.3)
    ax.quiver(
        data.gp[:, 0],
        data.gp[:, 1],
        data.gv[:, 0],
        data.gv[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / (0.08 * span),
        width=0.004,
        color="k",
        zorder=4,
    )
    _equal(ax, title)
    plt.show()


def plot_history(history: Sequence[Dict[str, float]], *, title: str = "Displacement field — data losses") -> None:
    """Weighted flatness, bedding and order through training.

    Each curve is ``target / loss₀ × loss``, so the three terms start at
    their Curlew targets and the curves are what Adam sees.
    """
    if not history:
        return
    epochs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
    for key, label in (("grad", "bedding"), ("eq", "flatness"), ("iq", "order")):
        ax.plot(epochs, [row[key] for row in history], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted loss")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_traces(data: AnalogueFold, inverse_map: MapFn, *, title: str) -> None:
    """Present-day contacts beside their image under ``inverse_map``."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    colors = _colors(len(data.names))
    span = max(data.extent, 1e-6)
    for color, full, sparse in zip(colors, data.horizons_full, data.horizons):
        axes[0].plot(full[:, 0], full[:, 1], color=color, lw=1.2)
        axes[0].scatter(sparse[:, 0], sparse[:, 1], s=22, color=color, zorder=3, edgecolors="k", linewidths=0.3)
        restored = inverse_map(full)
        restored_sparse = inverse_map(sparse)
        axes[1].plot(restored[:, 0], restored[:, 1], color=color, lw=1.2)
        axes[1].scatter(
            restored_sparse[:, 0],
            restored_sparse[:, 1],
            s=22,
            color=color,
            zorder=3,
            edgecolors="k",
            linewidths=0.3,
        )
    axes[0].quiver(
        data.gp[:, 0],
        data.gp[:, 1],
        data.gv[:, 0],
        data.gv[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / (0.08 * span),
        width=0.003,
        color="k",
        zorder=4,
    )
    _equal(axes[0], "Present-day contacts")
    _equal(axes[1], r"Restored $\Phi^{-1}(x)$")
    fig.suptitle(title, y=1.02)
    plt.show()


def plot_determinants(
    xs: Array,
    ys: Array,
    panels: Sequence[Tuple[str, Array]],
    data: Optional[AnalogueFold] = None,
) -> None:
    """``det J`` on the section. The colour scale is shared and centred on 1."""
    stack = np.concatenate([np.asarray(det).ravel() for _title, det in panels])
    finite = stack[np.isfinite(stack)]
    limit = max(float(np.max(np.abs(finite - 1.0))), 1e-3)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), squeeze=False, constrained_layout=True)
    image = None
    for ax, (title, det) in zip(axes[0], panels):
        image = ax.imshow(
            det,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=1.0 - limit,
            vmax=1.0 + limit,
            aspect="equal",
        )
        ax.contour(xs, ys, det, levels=[0.0], colors="k", linewidths=0.7)
        if data is not None:
            _overlay_traces(ax, data, lw=0.9)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, label=r"$\det J$")
    plt.show()


def plot_warps(
    xs: Array,
    ys: Array,
    panels: Sequence[Tuple[str, MapFn]],
    *,
    n_lines: int = 14,
    samples: int = 60,
) -> None:
    """Image of a regular mesh under each map. Crossed lines are a folded section."""
    x_lines = np.linspace(xs[0], xs[-1], n_lines)
    y_lines = np.linspace(ys[0], ys[-1], n_lines)
    vertical = np.linspace(ys[0], ys[-1], samples)
    horizontal = np.linspace(xs[0], xs[-1], samples)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), squeeze=False, constrained_layout=True)
    for ax, (title, inverse_map) in zip(axes[0], panels):
        for x in x_lines:
            line = np.column_stack((np.full(samples, x), vertical))
            warped = inverse_map(line)
            ax.plot(warped[:, 0], warped[:, 1], color="0.65", lw=0.6)
        for y in y_lines:
            line = np.column_stack((horizontal, np.full(samples, y)))
            warped = inverse_map(line)
            ax.plot(warped[:, 0], warped[:, 1], color="0.65", lw=0.6)
        _equal(ax, title)
    plt.show()


def plot_scalar(
    xs: Array,
    ys: Array,
    panels: Sequence[Tuple[str, MapFn]],
    data: Optional[AnalogueFold] = None,
    *,
    depth_axis: int = 1,
) -> None:
    """Restored depth ``φ`` on the section, with its level sets.

    ``φ(x) = (Φ⁻¹ x)[depth_axis]``. Panels share one colour scale. The
    present-day contacts are drawn on top when ``data`` is given.
    """
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    fields = [
        np.asarray(inverse_map(pts))[:, depth_axis].reshape(len(ys), len(xs))
        for _title, inverse_map in panels
    ]
    finite = np.concatenate([phi.ravel() for phi in fields])
    finite = finite[np.isfinite(finite)]
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    levels = np.linspace(lo, hi, 12)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.2 * len(panels), 4.4), squeeze=False, constrained_layout=True
    )
    image = None
    for ax, (title, _), phi in zip(axes[0], panels, fields):
        image = ax.imshow(
            phi,
            origin="lower",
            extent=extent,
            cmap="cividis",
            vmin=lo,
            vmax=hi,
            aspect="equal",
        )
        ax.contour(xs, ys, phi, levels=levels, colors="k", linewidths=0.35, alpha=0.55)
        if data is not None:
            _overlay_traces(ax, data, lw=1.15)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, label=r"$\varphi$")
    plt.show()


def plot_velocity(xs: Array, ys: Array, speed: Array, divergence: Array) -> None:
    """Speed of an Euler velocity beside its divergence."""
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    speed2 = speed.reshape(len(ys), len(xs))
    div2 = divergence.reshape(len(ys), len(xs))
    lim = max(float(np.max(np.abs(div2))), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(speed2, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[0].set_title(r"$|v|$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(div2, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    axes[1].set_title(r"$\nabla \cdot v$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    fig.suptitle("Euler velocity before fitting", y=1.02)
    plt.show()
