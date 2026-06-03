#!/usr/bin/env python3
"""Reduced Grad-Shafranov-style equilibrium generator.

This module provides a fixed-boundary axisymmetric equilibrium state for the
desktop workflow.  It is intentionally lightweight: it does not replace EFIT,
CHEASE, or HELENA, but it produces the shared IMAS ``.nc`` state required by
the RF, transport, PIC, and low-rank prototypes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.core.imas_compat import load_imas_state, write_module_imas_state
from modules.core.plotting import save_2d_projection_figure, save_profile_figure

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional preview dependency
    Image = None
    ImageDraw = None
    ImageFont = None


Array = np.ndarray
MU0 = 4.0 * math.pi * 1.0e-7


@dataclass(frozen=True)
class EquilibriumParams:
    input_mode: str = "manual"
    interface_state: str = ""
    device_machine_state: str = ""
    machine_config: str = ""
    geqdsk_input: str = ""
    geqdsk_output: str = "equilibrium.geqdsk"
    shape_constraint: str = ""
    diagnostics_constraint: str = ""
    reconstruction_mode: str = "off"
    reconstruction_gain: float = 0.5
    reconstruction_fit_params: str = "b0_t,plasma_current_ma,beta_percent,q_axis,q_edge,poloidal_field_fraction"
    reconstruction_regularization: float = 1.0e-2
    profile_model: str = "power-law"
    pressure_profile: str = ""
    current_profile: str = ""
    benchmark_geqdsk: str = ""
    cocos_index: int = 11
    psi_sign: float = 1.0
    ip_sign: float = 1.0
    btor_sign: float = 1.0
    export_formats: str = "profiles,chease,helena,efit"
    shape_control: str = "off"
    shape_control_gain: float = 0.65
    shape_control_current_limit_ma: float = 2.0
    shape_control_damping: float = 1.0e-5
    equilibrium_model: str = "free-boundary-gs"
    n_r: int = 72
    n_z: int = 96
    major_radius_m: float = 1.85
    minor_radius_m: float = 0.55
    elongation: float = 1.65
    triangularity: float = 0.18
    b0_t: float = 3.0
    plasma_current_ma: float = 0.75
    beta_percent: float = 2.2
    q_axis: float = 1.05
    q_edge: float = 4.0
    pressure_alpha: float = 1.6
    current_alpha: float = 1.2
    pressure_current_fraction: float = 0.55
    density_axis_1e19_m3: float = 1.0
    density_edge_fraction: float = 0.22
    temperature_axis_kev: float = 8.0
    poloidal_field_fraction: float = 0.08
    gs_iterations: int = 1500
    gs_relaxation: float = 1.15
    gs_tolerance: float = 1.0e-3
    free_boundary_extent: float = 1.45
    boundary_update_every: int = 50
    boundary_relaxation: float = 0.03
    pf_coil_current_ma: float = 0.0
    pf_coil_r_offset_m: float = 0.72
    pf_coil_z_m: float = 1.08
    pf_coil_turns: float = 1.0
    limiter_points: int = 192
    wall_clearance_m: float = 0.08


@dataclass(frozen=True)
class PFCoil:
    name: str
    r_m: float
    z_m: float
    current_ma: float
    turns: float = 1.0
    control: bool = True
    width_m: float = 0.0
    height_m: float = 0.0
    resistance_ohm: float = 0.0
    voltage_v: float = 0.0


@dataclass(frozen=True)
class PassiveStructure:
    name: str
    r_m: float
    z_m: float
    width_m: float
    height_m: float
    resistance_ohm: float = 0.0
    current_ma: float = 0.0
    turns: float = 1.0


@dataclass(frozen=True)
class MachineGeometry:
    device: str
    limiter_r: Array
    limiter_z: Array
    wall_r: Array
    wall_z: Array
    coils: tuple[PFCoil, ...]
    passive_structures: tuple[PassiveStructure, ...] = ()


@dataclass(frozen=True)
class EquilibriumState:
    r: Array
    z: Array
    major_r: Array
    psi: Array
    psi_norm: Array
    rho_pol: Array
    br: Array
    bz: Array
    b_phi: Array
    b_total: Array
    density: Array
    pressure_pa: Array
    temperature_kev: Array
    j_phi: Array
    q_profile_rho: Array
    q_profile: Array
    p_profile_pa: Array
    density_profile: Array
    inside: Array
    solver_iterations: int = 0
    solver_residual: float = 0.0
    operator_residual: float = 0.0
    pprime: Array | None = None
    ffprime: Array | None = None
    coil_flux: Array | None = None
    magnetic_axis_r_m: float = 0.0
    magnetic_axis_z_m: float = 0.0
    psi_lcfs: float = 0.0
    x_point_count: int = 0
    primary_x_point_r_m: float = 0.0
    primary_x_point_z_m: float = 0.0
    primary_x_point_psi: float = 0.0
    strike_point_count: int = 0
    primary_strike_point_r_m: float = 0.0
    primary_strike_point_z_m: float = 0.0
    primary_strike_point_psi_norm: float = 0.0
    separatrix_topology: str = "limited"
    divertor_balance: float = 0.0
    lower_strike_point_count: int = 0
    upper_strike_point_count: int = 0
    shape_control_rms_error: float = 0.0
    shape_control_max_error: float = 0.0
    shape_control_rank: int = 0
    reconstruction_rms_error: float = 0.0
    reconstruction_max_error: float = 0.0
    reconstruction_chi2_reduced: float = 0.0
    reconstruction_constraint_count: int = 0
    benchmark_lcfs_rms_m: float = 0.0
    benchmark_q_rms: float = 0.0
    machine: MachineGeometry | None = None


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def validate_params(params: EquilibriumParams) -> None:
    if params.input_mode not in {"manual", "interface"}:
        raise ValueError("input_mode must be manual or interface.")
    if params.input_mode == "interface" and not params.interface_state:
        raise ValueError("interface mode requires --interface-state.")
    if params.equilibrium_model not in {"free-boundary-gs", "iterative-gs", "circular-tokamak", "elongated-tokamak", "solovev"}:
        raise ValueError("equilibrium_model must be free-boundary-gs, iterative-gs, circular-tokamak, elongated-tokamak or solovev.")
    if params.n_r < 8 or params.n_z < 8:
        raise ValueError("n_r and n_z must both be at least 8.")
    if params.major_radius_m <= 0.0 or params.minor_radius_m <= 0.0:
        raise ValueError("major_radius_m and minor_radius_m must be positive.")
    if params.elongation <= 0.2:
        raise ValueError("elongation must be > 0.2.")
    if params.b0_t <= 0.0:
        raise ValueError("b0_t must be positive.")
    if params.q_edge <= params.q_axis:
        raise ValueError("q_edge must be larger than q_axis.")
    if params.gs_iterations < 1:
        raise ValueError("gs_iterations must be at least 1.")
    if not 0.1 <= params.gs_relaxation < 1.95:
        raise ValueError("gs_relaxation must be in [0.1, 1.95).")
    if params.gs_tolerance <= 0.0:
        raise ValueError("gs_tolerance must be positive.")
    if not 0.0 <= params.pressure_current_fraction <= 1.0:
        raise ValueError("pressure_current_fraction must be in [0, 1].")
    if params.free_boundary_extent < 1.05:
        raise ValueError("free_boundary_extent must be at least 1.05.")
    if params.boundary_update_every < 1:
        raise ValueError("boundary_update_every must be at least 1.")
    if not 0.0 <= params.boundary_relaxation <= 1.0:
        raise ValueError("boundary_relaxation must be in [0, 1].")
    if params.pf_coil_r_offset_m < 0.0:
        raise ValueError("pf_coil_r_offset_m must be non-negative.")
    if params.pf_coil_z_m < 0.0:
        raise ValueError("pf_coil_z_m must be non-negative.")
    if params.pf_coil_turns <= 0.0:
        raise ValueError("pf_coil_turns must be positive.")
    if params.limiter_points < 24:
        raise ValueError("limiter_points must be at least 24.")
    if params.wall_clearance_m < 0.0:
        raise ValueError("wall_clearance_m must be non-negative.")
    if params.machine_config and not Path(params.machine_config).expanduser().exists():
        raise FileNotFoundError(f"Missing machine_config: {params.machine_config}")
    if params.geqdsk_input and not Path(params.geqdsk_input).expanduser().exists():
        raise FileNotFoundError(f"Missing geqdsk_input: {params.geqdsk_input}")
    if params.shape_constraint and not Path(params.shape_constraint).expanduser().exists():
        raise FileNotFoundError(f"Missing shape_constraint: {params.shape_constraint}")
    for label, path_text in (
        ("diagnostics_constraint", params.diagnostics_constraint),
        ("pressure_profile", params.pressure_profile),
        ("current_profile", params.current_profile),
        ("benchmark_geqdsk", params.benchmark_geqdsk),
    ):
        if path_text and not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Missing {label}: {path_text}")
    if params.reconstruction_mode not in {"off", "weighted", "least-squares"}:
        raise ValueError("reconstruction_mode must be off, weighted or least-squares.")
    if not 0.0 <= params.reconstruction_gain <= 1.0:
        raise ValueError("reconstruction_gain must be in [0, 1].")
    if params.reconstruction_regularization < 0.0:
        raise ValueError("reconstruction_regularization must be non-negative.")
    if params.profile_model not in {"power-law", "spline"}:
        raise ValueError("profile_model must be power-law or spline.")
    if params.cocos_index < 1:
        raise ValueError("cocos_index must be positive.")
    for sign_name, sign_value in (("psi_sign", params.psi_sign), ("ip_sign", params.ip_sign), ("btor_sign", params.btor_sign)):
        if sign_value not in {-1.0, 1.0}:
            raise ValueError(f"{sign_name} must be -1 or 1.")
    if params.shape_control not in {"off", "forward", "isoflux", "inverse", "forward-inverse"}:
        raise ValueError("shape_control must be off, forward, isoflux, inverse or forward-inverse.")
    if not 0.0 <= params.shape_control_gain <= 1.5:
        raise ValueError("shape_control_gain must be in [0, 1.5].")
    if params.shape_control_current_limit_ma <= 0.0:
        raise ValueError("shape_control_current_limit_ma must be positive.")
    if params.shape_control_damping < 0.0:
        raise ValueError("shape_control_damping must be non-negative.")


def _as_text(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _array_from_data(data: Mapping[str, Any], names: Sequence[str], dtype: Any = float) -> Array | None:
    for name in names:
        if name not in data:
            continue
        values = np.asarray(data[name], dtype=dtype).reshape(-1)
        if values.size:
            return values
    return None


def _numeric_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?", text)


def _float_token(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "E"))


_JSON_CACHE: dict[str, dict[str, Any]] = {}


def load_json_object(path: str | Path) -> dict[str, Any]:
    key = str(Path(path).expanduser())
    if not key:
        return {}
    if key not in _JSON_CACHE:
        data = json.loads(Path(key).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} must contain a JSON object.")
        _JSON_CACHE[key] = dict(data)
    return dict(_JSON_CACHE[key])


def _weighted_mean(values: list[float], weights: list[float], fallback: float) -> float:
    if not values:
        return fallback
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        return fallback
    return float(np.sum(v[good] * w[good]) / max(float(np.sum(w[good])), 1.0e-30))


def _profile_data(path: str) -> dict[str, Any]:
    return load_json_object(path) if path else {}


def _profile_interpolate(path: str, names: Sequence[str], rho: Array, fallback: Array) -> Array:
    if not path:
        return np.asarray(fallback, dtype=float)
    raw = _profile_data(path)
    rho_raw = raw.get("rho", raw.get("rho_pol", raw.get("psi_norm")))
    value_raw: Any = None
    for name in names:
        if name in raw:
            value_raw = raw[name]
            break
    if rho_raw is None or value_raw is None:
        return np.asarray(fallback, dtype=float)
    xp = np.asarray(rho_raw, dtype=float).reshape(-1)
    yp = np.asarray(value_raw, dtype=float).reshape(-1)
    count = min(xp.size, yp.size)
    if count < 2:
        return np.asarray(fallback, dtype=float)
    xp = np.clip(xp[:count], 0.0, 1.0)
    yp = yp[:count]
    order = np.argsort(xp)
    xp = xp[order]
    yp = yp[order]
    unique = np.concatenate(([True], np.diff(xp) > 1.0e-9))
    xp = xp[unique]
    yp = yp[unique]
    if xp.size < 2:
        return np.asarray(fallback, dtype=float)
    return np.interp(np.clip(np.asarray(rho, dtype=float), 0.0, 1.0), xp, yp)


def pressure_shape_from_flux(params: EquilibriumParams, psi_norm: Array) -> Array:
    rho = np.sqrt(np.clip(np.asarray(psi_norm, dtype=float), 0.0, 1.0))
    fallback = np.maximum(1.0 - np.clip(psi_norm, 0.0, 1.0), 0.0) ** max(params.pressure_alpha, 0.05)
    if params.profile_model != "spline" or not params.pressure_profile:
        return fallback
    values = _profile_interpolate(params.pressure_profile, ("pressure_norm", "pressure", "pressure_pa", "p"), rho, fallback)
    if "pressure_pa" in _profile_data(params.pressure_profile):
        values = values / max(float(np.nanmax(values)), 1.0e-30)
    return np.clip(values, 0.0, None)


def current_shape_from_flux(params: EquilibriumParams, psi_norm: Array) -> Array:
    clipped = np.clip(np.asarray(psi_norm, dtype=float), 0.0, 1.0)
    fallback = (1.0 - clipped ** max(params.pressure_alpha, 0.05)) ** max(params.current_alpha, 0.05)
    if params.profile_model != "spline" or not params.current_profile:
        return fallback
    rho = np.sqrt(clipped)
    values = _profile_interpolate(
        params.current_profile,
        ("current_norm", "j_norm", "jphi_norm", "pprime_norm", "ffprime_norm", "current"),
        rho,
        fallback,
    )
    return np.clip(values, 0.0, None)


def pressure_profile_1d(params: EquilibriumParams, rho: Array) -> Array:
    p_axis = (params.beta_percent / 100.0) * params.b0_t * params.b0_t / (2.0 * MU0)
    fallback = p_axis * np.maximum(1.0 - np.asarray(rho, dtype=float) * np.asarray(rho, dtype=float), 0.0) ** max(params.pressure_alpha, 0.05)
    if params.profile_model != "spline" or not params.pressure_profile:
        return fallback
    raw = _profile_data(params.pressure_profile)
    if "pressure_norm" in raw:
        values = p_axis * _profile_interpolate(params.pressure_profile, ("pressure_norm",), np.asarray(rho, dtype=float), fallback / max(p_axis, 1.0e-30))
    else:
        values = _profile_interpolate(params.pressure_profile, ("pressure_pa", "pressure", "p"), np.asarray(rho, dtype=float), fallback)
    return np.clip(values, 0.0, None)


def read_geqdsk(path: str | Path) -> dict[str, Any]:
    """Read the core fields of an EFIT-style G-EQDSK file.

    The parser is intentionally small and local: it reads the standard scalar
    block, profile arrays, psi grid, LCFS and limiter polygons without pulling
    in an external backend.
    """

    text = Path(path).expanduser().read_text(errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty GEQDSK file.")
    header = lines[0]
    ints = [int(token) for token in _numeric_tokens(header)[-2:]]
    if len(ints) < 2:
        raise ValueError("GEQDSK header must contain nw and nh.")
    nw, nh = ints[-2], ints[-1]
    body_tokens = _numeric_tokens("\n".join(lines[1:]))
    scalars = [_float_token(token) for token in body_tokens]
    minimum = 20 + 5 * nw + nw * nh
    if len(scalars) < minimum:
        raise ValueError("GEQDSK numeric block is shorter than expected.")
    idx = 0

    def take(count: int) -> list[float]:
        nonlocal idx
        values = scalars[idx : idx + count]
        idx += count
        return values

    rdim, zdim, rcentr, rleft, zmid = take(5)
    rmagx, zmagx, simagx, sibdry, bcentr = take(5)
    cpasma, *_ = take(5)
    take(5)
    fpol = np.asarray(take(nw), dtype=float)
    pres = np.asarray(take(nw), dtype=float)
    ffprime = np.asarray(take(nw), dtype=float)
    pprime = np.asarray(take(nw), dtype=float)
    psi = np.asarray(take(nw * nh), dtype=float).reshape((nw, nh))
    qpsi = np.asarray(take(nw), dtype=float)
    nbbbs = int(round(scalars[idx])) if idx < len(scalars) else 0
    idx += 1
    limitr = int(round(scalars[idx])) if idx < len(scalars) else 0
    idx += 1
    boundary = np.asarray(take(max(0, 2 * nbbbs)), dtype=float).reshape((-1, 2)) if nbbbs > 0 else np.zeros((0, 2))
    limiter = np.asarray(take(max(0, 2 * limitr)), dtype=float).reshape((-1, 2)) if limitr > 0 else np.zeros((0, 2))
    return {
        "header": header,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmagx": rmagx,
        "zmagx": zmagx,
        "simagx": simagx,
        "sibdry": sibdry,
        "bcentr": bcentr,
        "cpasma": cpasma,
        "fpol": fpol,
        "pres": pres,
        "ffprime": ffprime,
        "pprime": pprime,
        "psi": psi,
        "qpsi": qpsi,
        "rbdry": boundary[:, 0] if boundary.size else np.asarray([], dtype=float),
        "zbdry": boundary[:, 1] if boundary.size else np.asarray([], dtype=float),
        "rlim": limiter[:, 0] if limiter.size else np.asarray([], dtype=float),
        "zlim": limiter[:, 1] if limiter.size else np.asarray([], dtype=float),
    }


def apply_geqdsk_input(params: EquilibriumParams) -> EquilibriumParams:
    if not params.geqdsk_input:
        return params
    data = read_geqdsk(params.geqdsk_input)
    rbdry = np.asarray(data["rbdry"], dtype=float)
    zbdry = np.asarray(data["zbdry"], dtype=float)
    r0 = float(data["rcentr"]) if float(data["rcentr"]) > 0.0 else params.major_radius_m
    if rbdry.size:
        a = max(float(np.max(np.abs(rbdry - r0))), 1.0e-6)
    else:
        a = max(float(data["rdim"]) / 4.0, params.minor_radius_m)
    if zbdry.size:
        kappa = max(float(np.max(np.abs(zbdry - float(data["zmagx"])))) / a, 0.3)
    else:
        kappa = max(float(data["zdim"]) / max(2.0 * a, 1.0e-12), 0.3)
    qpsi = np.asarray(data["qpsi"], dtype=float)
    q_axis = float(qpsi[0]) if qpsi.size and np.isfinite(qpsi[0]) else params.q_axis
    q_edge = float(qpsi[-1]) if qpsi.size and np.isfinite(qpsi[-1]) else params.q_edge
    return replace(
        params,
        major_radius_m=r0,
        minor_radius_m=a,
        elongation=kappa,
        b0_t=float(data["bcentr"]) if float(data["bcentr"]) > 0.0 else params.b0_t,
        plasma_current_ma=abs(float(data["cpasma"])) / 1.0e6 if float(data["cpasma"]) != 0.0 else params.plasma_current_ma,
        q_axis=max(q_axis, 0.2),
        q_edge=max(q_edge, max(q_axis + 0.05, params.q_edge)),
    )


def load_diagnostics_constraints(path: str | Path) -> dict[str, Any]:
    return load_json_object(path) if path else {}


RECONSTRUCTION_PARAM_SCALES = {
    "b0_t": 1.0,
    "plasma_current_ma": 0.25,
    "beta_percent": 1.0,
    "q_axis": 0.25,
    "q_edge": 0.75,
    "poloidal_field_fraction": 0.02,
    "pressure_current_fraction": 0.1,
}


def _reconstruction_fit_param_names(params: EquilibriumParams) -> list[str]:
    requested = [item.strip() for item in params.reconstruction_fit_params.split(",") if item.strip()]
    return [name for name in requested if name in RECONSTRUCTION_PARAM_SCALES]


def _q_proxy_from_params(params: EquilibriumParams, rho: float) -> tuple[float, dict[str, float]]:
    shape = float(np.clip(rho, 0.0, 1.0) ** 1.55)
    predicted = params.q_axis + (params.q_edge - params.q_axis) * shape
    return predicted, {"q_axis": 1.0 - shape, "q_edge": shape}


def _pressure_proxy_from_params(params: EquilibriumParams, rho: float) -> tuple[float, dict[str, float]]:
    p_axis = (params.beta_percent / 100.0) * params.b0_t * params.b0_t / (2.0 * MU0)
    shape = float(np.maximum(1.0 - np.clip(rho, 0.0, 1.0) ** 2, 0.0) ** max(params.pressure_alpha, 0.05))
    predicted = p_axis * shape
    return predicted, {
        "beta_percent": predicted / max(params.beta_percent, 1.0e-30),
        "b0_t": 2.0 * predicted / max(params.b0_t, 1.0e-30),
    }


def _mse_pitch_proxy_from_params(params: EquilibriumParams, rho: float) -> tuple[float, dict[str, float]]:
    radial_weight = 0.25 + 0.75 * float(np.clip(rho, 0.0, 1.0))
    ratio = params.poloidal_field_fraction * radial_weight
    predicted = math.degrees(math.atan(ratio))
    sensitivity = math.degrees(radial_weight / (1.0 + ratio * ratio))
    return predicted, {"poloidal_field_fraction": sensitivity}


def _magnetic_probe_proxy_from_params(params: EquilibriumParams, item: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    rv_major = float(item.get("r_m", item.get("R", params.major_radius_m)))
    rv_minor = rv_major - params.major_radius_m
    zv = float(item.get("z_m", item.get("Z", 0.0)))
    component = str(item.get("component", "b_total")).lower()
    r_scale = params.major_radius_m / max(rv_major, 1.0e-12)
    poloidal = params.poloidal_field_fraction * params.b0_t
    radius = max(math.hypot(rv_minor, zv), 1.0e-9)
    if component in {"bphi", "b_phi", "btor", "bt"}:
        return params.b0_t * r_scale, {"b0_t": r_scale}
    if component == "br":
        direction = -zv / radius
        return poloidal * direction, {"b0_t": params.poloidal_field_fraction * direction, "poloidal_field_fraction": params.b0_t * direction}
    if component == "bz":
        direction = rv_minor / radius
        return poloidal * direction, {"b0_t": params.poloidal_field_fraction * direction, "poloidal_field_fraction": params.b0_t * direction}
    predicted = math.hypot(params.b0_t * r_scale, poloidal)
    return predicted, {
        "b0_t": (params.b0_t * r_scale * r_scale + params.poloidal_field_fraction * poloidal) / max(predicted, 1.0e-30),
        "poloidal_field_fraction": params.b0_t * poloidal / max(predicted, 1.0e-30),
    }


def _flux_loop_proxy_from_params(params: EquilibriumParams, item: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    rv_major = float(item.get("r_m", item.get("R", params.major_radius_m)))
    zv = float(item.get("z_m", item.get("Z", 0.0)))
    rv_minor = rv_major - params.major_radius_m
    rho = min(math.hypot(rv_minor / max(params.minor_radius_m, 1.0e-12), zv / max(params.elongation * params.minor_radius_m, 1.0e-12)), 1.5)
    shape = min(rho * rho, 1.5)
    base = 0.5 * params.b0_t * params.poloidal_field_fraction * params.major_radius_m * params.minor_radius_m * shape
    return base, {
        "b0_t": base / max(params.b0_t, 1.0e-30),
        "poloidal_field_fraction": base / max(params.poloidal_field_fraction, 1.0e-30),
    }


def reconstruction_linear_equations(params: EquilibriumParams) -> list[dict[str, Any]]:
    if not params.diagnostics_constraint:
        return []
    raw = load_diagnostics_constraints(params.diagnostics_constraint)
    equations: list[dict[str, Any]] = []

    def add(kind: str, name: str, predicted: float, target: float, sigma: float, unit: str, sensitivities: Mapping[str, float]) -> None:
        safe_sigma = max(abs(float(sigma)), 1.0e-12)
        equations.append(
            {
                "kind": kind,
                "name": name,
                "predicted": float(predicted),
                "target": float(target),
                "sigma": safe_sigma,
                "unit": unit,
                "sensitivities": {key: float(value) for key, value in sensitivities.items() if key in RECONSTRUCTION_PARAM_SCALES and np.isfinite(value)},
            }
        )

    global_section = raw.get("global", {})
    if isinstance(global_section, Mapping):
        direct_keys = {
            "b0_t": ("b0_t", "T"),
            "plasma_current_ma": ("plasma_current_ma", "MA"),
            "q_axis": ("q_axis", "1"),
            "q_edge": ("q_edge", "1"),
            "beta_percent": ("beta_percent", "%"),
            "poloidal_field_fraction": ("poloidal_field_fraction", "1"),
        }
        for source_key, (target_key, unit) in direct_keys.items():
            if source_key in global_section:
                sigma = float(global_section.get(f"{source_key}_sigma", global_section.get("sigma", 1.0)))
                add("global", source_key, float(getattr(params, target_key)), float(global_section[source_key]), sigma, unit, {target_key: 1.0})

    for item in raw.get("q_points", []):
        if isinstance(item, Mapping):
            rho = float(item.get("rho", item.get("rho_pol", 1.0)))
            predicted, sensitivities = _q_proxy_from_params(params, rho)
            add("q_point", str(item.get("name", f"q_{rho:.2f}")), predicted, float(item.get("q", item.get("target", predicted))), float(item.get("sigma", item.get("sigma_q", 0.05))), "1", sensitivities)

    for item in raw.get("pressure_points", []):
        if isinstance(item, Mapping):
            rho = float(item.get("rho", item.get("rho_pol", 0.0)))
            predicted, sensitivities = _pressure_proxy_from_params(params, rho)
            add("pressure_point", str(item.get("name", f"pressure_{rho:.2f}")), predicted, float(item.get("pressure_pa", item.get("target", predicted))), float(item.get("sigma", item.get("sigma_pa", max(0.05 * abs(predicted), 1.0)))), "Pa", sensitivities)

    for item in raw.get("mse_points", []):
        if isinstance(item, Mapping):
            rho = float(item.get("rho", item.get("rho_pol", 0.5)))
            predicted, sensitivities = _mse_pitch_proxy_from_params(params, rho)
            add("mse_pitch", str(item.get("name", f"mse_{rho:.2f}")), predicted, float(item.get("pitch_angle_deg", item.get("gamma_deg", item.get("target", predicted)))), float(item.get("sigma", item.get("sigma_deg", 0.5))), "deg", sensitivities)

    for item in raw.get("magnetic_probes", []):
        if isinstance(item, Mapping):
            predicted, sensitivities = _magnetic_probe_proxy_from_params(params, item)
            add("magnetic_probe_proxy", str(item.get("name", f"probe_{len(equations)+1}")), predicted, float(item.get("value_t", item.get("target", predicted))), float(item.get("sigma", item.get("sigma_t", 1.0e-3))), "T", sensitivities)

    for item in raw.get("flux_loops", []):
        if isinstance(item, Mapping):
            predicted, sensitivities = _flux_loop_proxy_from_params(params, item)
            add("flux_loop_proxy", str(item.get("name", f"flux_loop_{len(equations)+1}")), predicted, float(item.get("psi_wb", item.get("target", predicted))), float(item.get("sigma", item.get("sigma_wb", 1.0e-3))), "Wb", sensitivities)

    for section in ("rogowski_current", "ip_constraints"):
        for item in raw.get(section, []):
            if isinstance(item, Mapping):
                target = float(item.get("plasma_current_ma", item.get("ip_ma", item.get("target", params.plasma_current_ma))))
                add("rogowski_current", str(item.get("name", f"ip_{len(equations)+1}")), params.plasma_current_ma, target, float(item.get("sigma", item.get("sigma_ma", 0.05))), "MA", {"plasma_current_ma": 1.0})

    for section in ("diamagnetic_loop", "beta_constraints"):
        for item in raw.get(section, []):
            if isinstance(item, Mapping):
                target = float(item.get("beta_percent", item.get("target", params.beta_percent)))
                add("diamagnetic_beta", str(item.get("name", f"beta_{len(equations)+1}")), params.beta_percent, target, float(item.get("sigma", item.get("sigma_percent", 0.2))), "%", {"beta_percent": 1.0})

    return equations


def solve_least_squares_reconstruction(params: EquilibriumParams) -> tuple[EquilibriumParams, list[list[object]], list[list[object]]]:
    equations = reconstruction_linear_equations(params)
    param_names = _reconstruction_fit_param_names(params)
    active_equations = [eq for eq in equations if any(name in eq["sensitivities"] for name in param_names)]
    if not param_names or not active_equations:
        return params, [], []

    a_rows: list[list[float]] = []
    b_rows: list[float] = []
    for eq in active_equations:
        sigma = float(eq["sigma"])
        a_rows.append([float(eq["sensitivities"].get(name, 0.0)) * RECONSTRUCTION_PARAM_SCALES[name] / sigma for name in param_names])
        b_rows.append((float(eq["target"]) - float(eq["predicted"])) / sigma)
    a_matrix = np.asarray(a_rows, dtype=float)
    b_vec = np.asarray(b_rows, dtype=float)
    if params.reconstruction_regularization > 0.0:
        damp = math.sqrt(params.reconstruction_regularization)
        a_matrix = np.vstack([a_matrix, damp * np.eye(len(param_names))])
        b_vec = np.concatenate([b_vec, np.zeros(len(param_names))])
    solution, *_ = np.linalg.lstsq(a_matrix, b_vec, rcond=None)
    updates: dict[str, float] = {}
    parameter_rows: list[list[object]] = []
    for idx, name in enumerate(param_names):
        current = float(getattr(params, name))
        delta = params.reconstruction_gain * RECONSTRUCTION_PARAM_SCALES[name] * float(solution[idx])
        max_delta = max(abs(current) * 0.45, RECONSTRUCTION_PARAM_SCALES[name] * 2.0)
        delta = float(np.clip(delta, -max_delta, max_delta))
        updated = current + delta
        if name in {"b0_t", "plasma_current_ma", "beta_percent"}:
            updated = max(updated, 1.0e-9)
        elif name == "poloidal_field_fraction":
            updated = float(np.clip(updated, 1.0e-4, 0.65))
        elif name == "pressure_current_fraction":
            updated = float(np.clip(updated, 0.0, 1.0))
        updates[name] = updated
        parameter_rows.append([name, f"{current:.10e}", f"{delta:.10e}", f"{updated:.10e}", RECONSTRUCTION_PARAM_SCALES[name], params.reconstruction_gain])

    candidate = replace(params, **updates)
    if candidate.q_edge <= candidate.q_axis:
        candidate = replace(candidate, q_edge=candidate.q_axis + 0.05)

    cost_rows: list[list[object]] = []
    for idx, eq in enumerate(active_equations, 1):
        predicted_after = float(eq["predicted"])
        for name in param_names:
            predicted_after += float(eq["sensitivities"].get(name, 0.0)) * (float(getattr(candidate, name)) - float(getattr(params, name)))
        residual_before = float(eq["predicted"]) - float(eq["target"])
        residual_after = predicted_after - float(eq["target"])
        sigma = float(eq["sigma"])
        cost_rows.append(
            [
                idx,
                eq["kind"],
                eq["name"],
                f"{float(eq['predicted']):.10e}",
                f"{predicted_after:.10e}",
                f"{float(eq['target']):.10e}",
                f"{residual_before:.10e}",
                f"{residual_after:.10e}",
                f"{sigma:.10e}",
                f"{residual_before / sigma:.10e}",
                f"{residual_after / sigma:.10e}",
                eq["unit"],
                ";".join(f"{name}:{float(eq['sensitivities'].get(name, 0.0)):.6e}" for name in param_names if name in eq["sensitivities"]),
            ]
        )
    return candidate, parameter_rows, cost_rows


def _apply_weighted_reconstruction_constraints(params: EquilibriumParams) -> EquilibriumParams:
    if params.reconstruction_mode == "off" or not params.diagnostics_constraint:
        return params
    raw = load_diagnostics_constraints(params.diagnostics_constraint)
    gain = float(params.reconstruction_gain)
    updates: dict[str, object] = {}
    global_section = raw.get("global", raw)
    if isinstance(global_section, Mapping):
        direct_keys = {
            "b0_t": "b0_t",
            "plasma_current_ma": "plasma_current_ma",
            "q_axis": "q_axis",
            "q_edge": "q_edge",
            "beta_percent": "beta_percent",
            "major_radius_m": "major_radius_m",
            "minor_radius_m": "minor_radius_m",
            "elongation": "elongation",
            "triangularity": "triangularity",
        }
        for source_key, target_key in direct_keys.items():
            if source_key not in global_section:
                continue
            target = float(global_section[source_key])
            current = float(getattr(params, target_key))
            updates[target_key] = (1.0 - gain) * current + gain * target

    q_values: list[float] = []
    q_weights: list[float] = []
    for item in raw.get("q_points", []):
        if isinstance(item, Mapping) and float(item.get("rho", 0.0)) >= 0.95 and item.get("q") is not None:
            sigma = max(float(item.get("sigma", item.get("sigma_q", 1.0))), 1.0e-9)
            q_values.append(float(item["q"]))
            q_weights.append(1.0 / (sigma * sigma))
    if q_values:
        target_q_edge = _weighted_mean(q_values, q_weights, params.q_edge)
        updates["q_edge"] = (1.0 - gain) * params.q_edge + gain * target_q_edge

    pressure_values: list[float] = []
    pressure_weights: list[float] = []
    for item in raw.get("pressure_points", []):
        if isinstance(item, Mapping) and float(item.get("rho", 1.0)) <= 0.15 and item.get("pressure_pa") is not None:
            sigma = max(float(item.get("sigma", item.get("sigma_pa", 1.0))), 1.0e-9)
            pressure_values.append(float(item["pressure_pa"]))
            pressure_weights.append(1.0 / (sigma * sigma))
    if pressure_values:
        p_axis = _weighted_mean(pressure_values, pressure_weights, 0.0)
        beta = 100.0 * p_axis * 2.0 * MU0 / max(params.b0_t * params.b0_t, 1.0e-30)
        updates["beta_percent"] = (1.0 - gain) * params.beta_percent + gain * beta

    ip_values: list[float] = []
    ip_weights: list[float] = []
    for section in ("rogowski_current", "ip_constraints"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            target = item.get("plasma_current_ma", item.get("ip_ma", item.get("target")))
            if target is None:
                continue
            sigma = max(float(item.get("sigma", item.get("sigma_ma", 0.05))), 1.0e-9)
            ip_values.append(abs(float(target)))
            ip_weights.append(1.0 / (sigma * sigma))
    if ip_values:
        target_ip = _weighted_mean(ip_values, ip_weights, params.plasma_current_ma)
        updates["plasma_current_ma"] = (1.0 - gain) * params.plasma_current_ma + gain * target_ip

    beta_values: list[float] = []
    beta_weights: list[float] = []
    for section in ("diamagnetic_loop", "beta_constraints"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            target = item.get("beta_percent", item.get("target"))
            if target is None:
                continue
            sigma = max(float(item.get("sigma", item.get("sigma_percent", 0.2))), 1.0e-9)
            beta_values.append(float(target))
            beta_weights.append(1.0 / (sigma * sigma))
    if beta_values:
        target_beta = _weighted_mean(beta_values, beta_weights, params.beta_percent)
        updates["beta_percent"] = (1.0 - gain) * params.beta_percent + gain * target_beta

    if not updates:
        return params
    candidate = replace(params, **updates)
    if candidate.q_edge <= candidate.q_axis:
        candidate = replace(candidate, q_edge=candidate.q_axis + 0.05)
    return candidate


def apply_reconstruction_constraints_with_report(params: EquilibriumParams) -> tuple[EquilibriumParams, list[list[object]], list[list[object]]]:
    if params.reconstruction_mode == "off" or not params.diagnostics_constraint:
        return params, [], []
    if params.reconstruction_mode == "least-squares":
        return solve_least_squares_reconstruction(params)

    before = params
    after = _apply_weighted_reconstruction_constraints(params)
    parameter_rows: list[list[object]] = []
    for name in _reconstruction_fit_param_names(params):
        old = float(getattr(before, name))
        new = float(getattr(after, name))
        if abs(new - old) > 0.0:
            parameter_rows.append([name, f"{old:.10e}", f"{new - old:.10e}", f"{new:.10e}", RECONSTRUCTION_PARAM_SCALES[name], params.reconstruction_gain])
    _, _, cost_rows = solve_least_squares_reconstruction(before)
    return after, parameter_rows, cost_rows


def apply_reconstruction_constraints(params: EquilibriumParams) -> EquilibriumParams:
    return apply_reconstruction_constraints_with_report(params)[0]


def _d_shape_points(params: EquilibriumParams, count: int, scale: float = 1.0) -> tuple[Array, Array]:
    theta = np.linspace(0.0, 2.0 * math.pi, max(count, 24), endpoint=False)
    a = params.minor_radius_m * scale
    shaped_angle = theta + params.triangularity * np.sin(theta)
    r = a * np.cos(shaped_angle)
    z = params.elongation * a * np.sin(theta)
    return r.astype(float), z.astype(float)


def default_machine_geometry(params: EquilibriumParams) -> MachineGeometry:
    limiter_r, limiter_z = _d_shape_points(params, params.limiter_points, 1.0)
    wall_scale = 1.0 + params.wall_clearance_m / max(params.minor_radius_m, 1.0e-12)
    wall_r, wall_z = _d_shape_points(params, params.limiter_points, wall_scale)
    coils: tuple[PFCoil, ...] = (
        PFCoil(
            "upper_pf",
            params.pf_coil_r_offset_m,
            params.pf_coil_z_m,
            params.pf_coil_current_ma,
            params.pf_coil_turns,
        ),
        PFCoil(
            "lower_pf",
            params.pf_coil_r_offset_m,
            -params.pf_coil_z_m,
            params.pf_coil_current_ma,
            params.pf_coil_turns,
        ),
    )
    return MachineGeometry("parametric_d_shape", limiter_r, limiter_z, wall_r, wall_z, coils)


def _coils_from_device_state(data: Mapping[str, Any], params: EquilibriumParams) -> tuple[PFCoil, ...]:
    r_values = _array_from_data(data, ("pf_coil_r_m", "coil_r_m"))
    z_values = _array_from_data(data, ("pf_coil_z_m", "coil_z_m"))
    current_values = _array_from_data(data, ("pf_coil_current_ma", "coil_current_ma"))
    turns_values = _array_from_data(data, ("pf_coil_turns", "coil_turns"))
    name_values = _array_from_data(data, ("pf_coil_name", "coil_name"), dtype=object)
    coils: list[PFCoil] = []
    if r_values is not None and z_values is not None and current_values is not None:
        count = min(r_values.size, z_values.size, current_values.size)
        for idx in range(count):
            name = str(name_values[idx]) if name_values is not None and idx < name_values.size else f"pf_{idx + 1}"
            turns = float(turns_values[idx]) if turns_values is not None and idx < turns_values.size else params.pf_coil_turns
            coils.append(
                PFCoil(
                    name=name,
                    r_m=float(r_values[idx]) - params.major_radius_m,
                    z_m=float(z_values[idx]),
                    current_ma=float(current_values[idx]),
                    turns=turns,
                )
            )
    if coils:
        return tuple(coils)

    upper_r = _scalar_or_none(data, "upper_pf_coil_r_m")
    upper_z = _scalar_or_none(data, "upper_pf_coil_z_m")
    upper_i = _scalar_or_none(data, "upper_pf_coil_current_ma")
    lower_r = _scalar_or_none(data, "lower_pf_coil_r_m")
    lower_z = _scalar_or_none(data, "lower_pf_coil_z_m")
    lower_i = _scalar_or_none(data, "lower_pf_coil_current_ma")
    if upper_r is not None and upper_z is not None and upper_i is not None:
        coils.append(PFCoil("upper_pf", upper_r - params.major_radius_m, upper_z, upper_i, params.pf_coil_turns))
    if lower_r is not None and lower_z is not None and lower_i is not None:
        coils.append(PFCoil("lower_pf", lower_r - params.major_radius_m, lower_z, lower_i, params.pf_coil_turns))
    return tuple(coils)


def _machine_from_mapping(config: Mapping[str, Any], params: EquilibriumParams) -> MachineGeometry:
    device = str(config.get("device", config.get("name", "machine_config")))
    limiter = config.get("limiter", {})
    wall = config.get("wall", {})

    def polygon_points(section: Any, fallback_scale: float) -> tuple[Array, Array]:
        if isinstance(section, Mapping):
            r_raw = section.get("r_m", section.get("R", section.get("r")))
            z_raw = section.get("z_m", section.get("Z", section.get("z")))
            if r_raw is not None and z_raw is not None:
                r_array = np.asarray(r_raw, dtype=float).reshape(-1)
                z_array = np.asarray(z_raw, dtype=float).reshape(-1)
                count = min(r_array.size, z_array.size)
                if count >= 3:
                    r_minor = r_array[:count] - params.major_radius_m
                    return r_minor, z_array[:count]
        return _d_shape_points(params, params.limiter_points, fallback_scale)

    limiter_r, limiter_z = polygon_points(limiter, 1.0)
    wall_scale = 1.0 + params.wall_clearance_m / max(params.minor_radius_m, 1.0e-12)
    wall_r, wall_z = polygon_points(wall, wall_scale)
    coils = []
    for idx, raw in enumerate(config.get("pf_coils", config.get("coils", []))):
        if not isinstance(raw, Mapping):
            continue
        r_actual = raw.get("r_m", raw.get("R"))
        r_offset = raw.get("r_offset_m", raw.get("minor_r_m"))
        if r_actual is None and r_offset is None:
            continue
        r_minor = float(r_offset) if r_offset is not None else float(r_actual) - params.major_radius_m
        coils.append(
            PFCoil(
                name=str(raw.get("name", raw.get("label", f"pf_{idx + 1}"))),
                r_m=r_minor,
                z_m=float(raw.get("z_m", raw.get("Z", 0.0))),
                current_ma=float(raw.get("current_ma", raw.get("current", 0.0))),
                turns=float(raw.get("turns", 1.0)),
                control=bool(raw.get("control", True)),
                width_m=float(raw.get("width_m", raw.get("width", 0.0))),
                height_m=float(raw.get("height_m", raw.get("height", 0.0))),
                resistance_ohm=float(raw.get("resistance_ohm", raw.get("resistance", 0.0))),
                voltage_v=float(raw.get("voltage_v", raw.get("voltage", 0.0))),
            )
        )
    if not coils:
        coils = list(default_machine_geometry(params).coils)
    passive: list[PassiveStructure] = []
    for idx, raw in enumerate(config.get("passive_structures", config.get("passive", []))):
        if not isinstance(raw, Mapping):
            continue
        r_actual = raw.get("r_m", raw.get("R"))
        r_offset = raw.get("r_offset_m", raw.get("minor_r_m"))
        if r_actual is None and r_offset is None:
            continue
        r_minor = float(r_offset) if r_offset is not None else float(r_actual) - params.major_radius_m
        passive.append(
            PassiveStructure(
                name=str(raw.get("name", raw.get("label", f"passive_{idx + 1}"))),
                r_m=r_minor,
                z_m=float(raw.get("z_m", raw.get("Z", 0.0))),
                width_m=float(raw.get("width_m", raw.get("width", 0.02))),
                height_m=float(raw.get("height_m", raw.get("height", 0.02))),
                resistance_ohm=float(raw.get("resistance_ohm", raw.get("resistance", 0.0))),
                current_ma=float(raw.get("current_ma", raw.get("current", 0.0))),
                turns=float(raw.get("turns", 1.0)),
            )
        )
    return MachineGeometry(device, limiter_r, limiter_z, wall_r, wall_z, tuple(coils), tuple(passive))


def load_machine_geometry(params: EquilibriumParams) -> MachineGeometry:
    if params.machine_config:
        path = Path(params.machine_config).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("machine_config must contain a JSON object.")
        return _machine_from_mapping(data, params)

    if params.device_machine_state:
        path = Path(params.device_machine_state).expanduser()
        if path.exists():
            data = load_imas_state(path, "device_machine_state")
            limiter_r_actual = _array_from_data(data, ("limiter_r_m", "wall_limiter_r_m"))
            limiter_z = _array_from_data(data, ("limiter_z_m", "wall_limiter_z_m"))
            if limiter_r_actual is not None and limiter_z is not None and min(limiter_r_actual.size, limiter_z.size) >= 3:
                count = min(limiter_r_actual.size, limiter_z.size)
                limiter_r = limiter_r_actual[:count] - params.major_radius_m
                limiter_z = limiter_z[:count]
            else:
                limiter_r, limiter_z = _d_shape_points(params, params.limiter_points, 1.0)
            wall_r_actual = _array_from_data(data, ("wall_r_m", "first_wall_r_m"))
            wall_z = _array_from_data(data, ("wall_z_m", "first_wall_z_m"))
            if wall_r_actual is not None and wall_z is not None and min(wall_r_actual.size, wall_z.size) >= 3:
                count = min(wall_r_actual.size, wall_z.size)
                wall_r = wall_r_actual[:count] - params.major_radius_m
                wall_z = wall_z[:count]
            else:
                wall_scale = 1.0 + params.wall_clearance_m / max(params.minor_radius_m, 1.0e-12)
                wall_r, wall_z = _d_shape_points(params, params.limiter_points, wall_scale)
            coils = _coils_from_device_state(data, params) or default_machine_geometry(params).coils
            return MachineGeometry(_as_text(data.get("device", "device_machine_state")), limiter_r, limiter_z, wall_r, wall_z, coils)
    if params.geqdsk_input:
        data = read_geqdsk(params.geqdsk_input)
        rlim = np.asarray(data["rlim"], dtype=float)
        zlim = np.asarray(data["zlim"], dtype=float)
        rbdry = np.asarray(data["rbdry"], dtype=float)
        zbdry = np.asarray(data["zbdry"], dtype=float)
        if rlim.size >= 3 and zlim.size >= 3:
            limiter_r = rlim - params.major_radius_m
            limiter_z = zlim
        elif rbdry.size >= 3 and zbdry.size >= 3:
            limiter_r = rbdry - params.major_radius_m
            limiter_z = zbdry
        else:
            limiter_r, limiter_z = _d_shape_points(params, params.limiter_points, 1.0)
        wall_scale = 1.0 + params.wall_clearance_m / max(params.minor_radius_m, 1.0e-12)
        if rlim.size >= 3 and zlim.size >= 3:
            wall_r = (rlim - params.major_radius_m) * wall_scale
            wall_z = zlim * wall_scale
        else:
            wall_r, wall_z = _d_shape_points(params, params.limiter_points, wall_scale)
        return MachineGeometry("geqdsk_import", limiter_r, limiter_z, wall_r, wall_z, default_machine_geometry(params).coils)
    return default_machine_geometry(params)


def point_in_polygon(x: Array, y: Array, poly_x: Array, poly_y: Array) -> Array:
    px = np.asarray(poly_x, dtype=float)
    py = np.asarray(poly_y, dtype=float)
    if px.size < 3 or py.size < 3:
        return np.ones_like(np.asarray(x, dtype=float), dtype=bool)
    inside = np.zeros_like(np.asarray(x, dtype=float), dtype=bool)
    j = px.size - 1
    for i in range(px.size):
        yi = py[i]
        yj = py[j]
        crosses = ((yi > y) != (yj > y)) & (x < (px[j] - px[i]) * (y - yi) / (yj - yi + 1.0e-30) + px[i])
        inside ^= crosses
        j = i
    return inside


def machine_limiter_mask(machine: MachineGeometry, r: Array, z: Array) -> Array:
    rr, zz = np.meshgrid(r, z, indexing="ij")
    return point_in_polygon(rr, zz, machine.limiter_r, machine.limiter_z)


def sample_grid(values: Array, r: Array, z: Array, sample_r: Array, sample_z: Array) -> Array:
    arr = np.asarray(values, dtype=float)
    sr = np.asarray(sample_r, dtype=float)
    sz = np.asarray(sample_z, dtype=float)
    flat_r = sr.reshape(-1)
    flat_z = sz.reshape(-1)
    out = np.full(flat_r.shape, np.nan, dtype=float)
    if r.size < 2 or z.size < 2:
        return out.reshape(sr.shape)
    for index, (rv, zv) in enumerate(zip(flat_r, flat_z)):
        if rv < r[0] or rv > r[-1] or zv < z[0] or zv > z[-1]:
            continue
        i = int(np.searchsorted(r, rv) - 1)
        j = int(np.searchsorted(z, zv) - 1)
        i = min(max(i, 0), r.size - 2)
        j = min(max(j, 0), z.size - 2)
        tr = (rv - r[i]) / max(r[i + 1] - r[i], 1.0e-30)
        tz = (zv - z[j]) / max(z[j + 1] - z[j], 1.0e-30)
        out[index] = (
            (1.0 - tr) * (1.0 - tz) * arr[i, j]
            + tr * (1.0 - tz) * arr[i + 1, j]
            + (1.0 - tr) * tz * arr[i, j + 1]
            + tr * tz * arr[i + 1, j + 1]
        )
    return out.reshape(sr.shape)


def resample_polygon(poly_r: Array, poly_z: Array, points_per_segment: int = 8) -> tuple[Array, Array]:
    r_values = np.asarray(poly_r, dtype=float).reshape(-1)
    z_values = np.asarray(poly_z, dtype=float).reshape(-1)
    if r_values.size < 2:
        return r_values, z_values
    rr: list[float] = []
    zz: list[float] = []
    count = r_values.size
    for idx in range(count):
        nxt = (idx + 1) % count
        for step in range(max(points_per_segment, 1)):
            frac = step / max(points_per_segment, 1)
            rr.append(float((1.0 - frac) * r_values[idx] + frac * r_values[nxt]))
            zz.append(float((1.0 - frac) * z_values[idx] + frac * z_values[nxt]))
    return np.asarray(rr, dtype=float), np.asarray(zz, dtype=float)


def contour_points(values: Array, r: Array, z: Array, level: float) -> tuple[Array, Array]:
    arr = np.asarray(values, dtype=float)
    points: list[tuple[float, float]] = []
    for i in range(arr.shape[0] - 1):
        for j in range(arr.shape[1] - 1):
            corners = (
                (float(arr[i, j]), float(r[i]), float(z[j])),
                (float(arr[i + 1, j]), float(r[i + 1]), float(z[j])),
                (float(arr[i + 1, j + 1]), float(r[i + 1]), float(z[j + 1])),
                (float(arr[i, j + 1]), float(r[i]), float(z[j + 1])),
            )
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                v0, r0, z0 = corners[a]
                v1, r1, z1 = corners[b]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                if (v0 - level) == 0.0:
                    points.append((r0, z0))
                if (v0 - level) * (v1 - level) < 0.0:
                    denom = v1 - v0
                    if abs(denom) <= 1.0e-30:
                        continue
                    frac = (level - v0) / denom
                    if not 0.0 <= frac <= 1.0:
                        continue
                    points.append((r0 + frac * (r1 - r0), z0 + frac * (z1 - z0)))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    # De-duplicate points at shared cell edges.
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[float, float]] = []
    scale = max(float(np.ptp(r)), float(np.ptp(z)), 1.0)
    for rv, zv in points:
        key = (int(round(rv / (scale * 1.0e-5))), int(round(zv / (scale * 1.0e-5))))
        if key in seen:
            continue
        seen.add(key)
        unique.append((rv, zv))
    return np.asarray([p[0] for p in unique], dtype=float), np.asarray([p[1] for p in unique], dtype=float)


def sorted_lcfs_points(state: "EquilibriumState", max_points: int = 300) -> tuple[Array, Array]:
    cr, cz = contour_points(state.psi_norm, state.r, state.z, 1.0)
    if cr.size == 0:
        return cr, cz
    angles = np.arctan2(cz - state.magnetic_axis_z_m, cr - state.magnetic_axis_r_m)
    order = np.argsort(angles)
    cr = cr[order]
    cz = cz[order]
    if cr.size > max_points:
        indices = np.linspace(0, cr.size - 1, max_points, dtype=int)
        cr = cr[indices]
        cz = cz[indices]
    return cr, cz


def _span_from_axis(data: dict[str, Array], names: Sequence[str]) -> float | None:
    for name in names:
        if name not in data:
            continue
        values = np.asarray(data[name], dtype=float).reshape(-1)
        if values.size >= 2:
            return float(max(abs(np.nanmin(values)), abs(np.nanmax(values))))
    return None


def _scalar_from_data(data: dict[str, Array], names: Sequence[str], mode: str = "mean") -> float | None:
    for name in names:
        if name not in data:
            continue
        values = np.asarray(data[name], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if mode == "max":
            return float(np.max(finite))
        if mode == "min":
            return float(np.min(finite))
        return float(np.mean(finite))
    return None


def apply_interface_state(params: EquilibriumParams) -> EquilibriumParams:
    if params.input_mode == "manual":
        return params
    path = Path(params.interface_state).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing interface_state: {path}")
    data = load_imas_state(path, "equilibrium_state")

    updates: dict[str, object] = {"interface_state": str(path)}
    r_span = _span_from_axis(data, ("r", "rho_m", "radius_m", "r_m"))
    z_span = _span_from_axis(data, ("z", "z_m"))
    b_axis = _scalar_from_data(data, ("b0_t", "b_axis_t", "B0"), "mean")
    b_total = _scalar_from_data(data, ("b_total", "bmag", "B_total", "b_total_t"), "max")
    density = _scalar_from_data(data, ("density", "ne_1e19_m3", "density_1e19_m3"), "max")
    if r_span is not None and r_span > 0.0:
        updates["minor_radius_m"] = r_span
    if z_span is not None and z_span > 0.0:
        updates["elongation"] = max(z_span / max(float(updates.get("minor_radius_m", params.minor_radius_m)), 1.0e-12), 0.4)
    if b_axis is not None and b_axis > 0.0:
        updates["b0_t"] = b_axis
    elif b_total is not None and b_total > 0.0:
        updates["b0_t"] = b_total
    if density is not None and density > 0.0:
        updates["density_axis_1e19_m3"] = density
    return replace(params, **updates)


def _scalar_or_none(data: dict[str, Array], name: str) -> float | None:
    if name not in data:
        return None
    values = np.asarray(data[name], dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite[0])


def apply_device_machine_state(params: EquilibriumParams) -> EquilibriumParams:
    if not params.device_machine_state:
        return params
    path = Path(params.device_machine_state).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing device_machine_state: {path}")
    data = load_imas_state(path, "device_machine_state")
    updates: dict[str, object] = {"device_machine_state": str(path)}
    state_key_map = {
        "major_radius_m": ("major_radius_m", "R0_m"),
        "minor_radius_m": ("minor_radius_m", "a_m"),
        "elongation": ("elongation", "kappa"),
        "triangularity": ("triangularity", "delta"),
        "b0_t": ("b0_t",),
        "density_axis_1e19_m3": ("density_axis_1e19_m3", "density_1e19_m3"),
        "free_boundary_extent": ("free_boundary_extent",),
        "pf_coil_current_ma": ("pf_coil_current_ma", "upper_pf_coil_current_ma"),
        "pf_coil_r_offset_m": ("pf_coil_r_offset_m",),
        "pf_coil_z_m": ("pf_coil_z_m", "upper_pf_coil_z_m"),
        "pf_coil_turns": ("pf_coil_turns",),
    }
    for key, names in state_key_map.items():
        for name in names:
            value = _scalar_or_none(data, name)
            if value is not None:
                updates[key] = value
                break
    return replace(params, **updates)


def model_adjusted_params(params: EquilibriumParams) -> EquilibriumParams:
    if params.equilibrium_model == "circular-tokamak":
        return replace(params, elongation=1.0, triangularity=0.0)
    if params.equilibrium_model == "solovev":
        return replace(params, pressure_alpha=1.0, current_alpha=0.4)
    return params


def build_analytic_equilibrium(params: EquilibriumParams) -> EquilibriumState:
    params = model_adjusted_params(params)
    machine = load_machine_geometry(params)
    a = params.minor_radius_m
    kappa = params.elongation
    r_extent = max(a, float(np.max(np.abs(machine.wall_r))) if machine.wall_r.size else a)
    z_extent = max(kappa * a, float(np.max(np.abs(machine.wall_z))) if machine.wall_z.size else kappa * a)
    r = np.linspace(-r_extent, r_extent, params.n_r)
    z = np.linspace(-z_extent, z_extent, params.n_z)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    major_r = params.major_radius_m + rr

    z_norm = zz / max(kappa * a, 1.0e-12)
    triangular_shift = params.triangularity * a * z_norm * z_norm
    r_eff = np.maximum(rr + triangular_shift, 0.0)
    psi_norm = (r_eff / max(a, 1.0e-12)) ** 2 + z_norm * z_norm
    if params.equilibrium_model == "solovev":
        psi_norm += 0.08 * (rr / max(a, 1.0e-12)) ** 4 - 0.04 * z_norm * z_norm * (rr / max(a, 1.0e-12)) ** 2
    psi_norm = np.maximum(psi_norm, 0.0)
    rho_pol = np.sqrt(np.maximum(psi_norm, 0.0))
    limiter_mask = machine_limiter_mask(machine, r, z)
    inside = (psi_norm <= 1.0) & limiter_mask
    core_shape = np.maximum(1.0 - psi_norm, 0.0)

    psi_scale = 0.5 * params.poloidal_field_fraction * params.b0_t * params.major_radius_m * a
    psi = psi_scale * psi_norm
    dpsi_dr = np.gradient(psi, r, axis=0, edge_order=2)
    dpsi_dz = np.gradient(psi, z, axis=1, edge_order=2)
    br = -dpsi_dz / np.maximum(major_r, 1.0e-12)
    bz = dpsi_dr / np.maximum(major_r, 1.0e-12)

    ff_slope = 0.04 if params.equilibrium_model != "solovev" else 0.02
    f_flux = params.major_radius_m * params.b0_t * (1.0 + ff_slope * core_shape)
    b_phi = f_flux / np.maximum(major_r, 1.0e-12)
    b_total = np.sqrt(br * br + bz * bz + b_phi * b_phi)

    beta = params.beta_percent / 100.0
    p_axis = beta * params.b0_t * params.b0_t / (2.0 * MU0)
    pressure_pa = p_axis * pressure_shape_from_flux(params, psi_norm)
    pressure_pa = np.where(inside, pressure_pa, 0.0)
    temperature_kev = params.temperature_axis_kev * (0.25 + 0.75 * core_shape ** 0.55)
    temperature_kev = np.where(inside, temperature_kev, 0.25 * params.temperature_axis_kev)
    density = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * core_shape ** 0.7
    )
    density = np.where(inside, density, params.density_axis_1e19_m3 * params.density_edge_fraction * np.exp(-(rho_pol - 1.0) ** 2 / 0.18))

    current_shape = current_shape_from_flux(params, psi_norm)
    current_shape = np.where(inside, current_shape, 0.0)
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    norm = float(np.sum(current_shape) * dr * dz)
    if norm <= 0.0:
        j_phi = np.zeros_like(current_shape)
    else:
        j_phi = params.plasma_current_ma * 1.0e6 * current_shape / norm

    q_profile_rho = np.linspace(0.0, 1.0, 80)
    q_profile = params.q_axis + (params.q_edge - params.q_axis) * q_profile_rho ** 1.65
    p_profile_pa = pressure_profile_1d(params, q_profile_rho)
    density_profile = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * np.maximum(1.0 - q_profile_rho * q_profile_rho, 0.0) ** 0.7
    )
    axis_r, axis_z, lcfs_value = magnetic_axis_and_lcfs(psi, psi_norm, r, z)
    x_points = estimate_x_points(psi, r, z)
    primary_x = x_points[0] if x_points else (0.0, 0.0, 0.0)
    strike_points = estimate_strike_points(machine, psi_norm, psi, r, z)
    primary_strike = strike_points[0] if strike_points else (0.0, 0.0, 0.0)
    topology = separatrix_topology_from_points(x_points, strike_points)
    strike_psi_norm = (
        float(sample_grid(psi_norm, r, z, np.asarray([primary_strike[0]]), np.asarray([primary_strike[1]]))[0])
        if strike_points
        else 0.0
    )

    return EquilibriumState(
        r=r,
        z=z,
        major_r=major_r,
        psi=psi,
        psi_norm=psi_norm,
        rho_pol=rho_pol,
        br=br,
        bz=bz,
        b_phi=b_phi,
        b_total=b_total,
        density=density,
        pressure_pa=pressure_pa,
        temperature_kev=temperature_kev,
        j_phi=j_phi,
        q_profile_rho=q_profile_rho,
        q_profile=q_profile,
        p_profile_pa=p_profile_pa,
        density_profile=density_profile,
        inside=inside,
        magnetic_axis_r_m=axis_r,
        magnetic_axis_z_m=axis_z,
        psi_lcfs=lcfs_value,
        x_point_count=len(x_points),
        primary_x_point_r_m=primary_x[0],
        primary_x_point_z_m=primary_x[1],
        primary_x_point_psi=primary_x[2],
        strike_point_count=len(strike_points),
        primary_strike_point_r_m=primary_strike[0],
        primary_strike_point_z_m=primary_strike[1],
        primary_strike_point_psi_norm=strike_psi_norm,
        separatrix_topology=str(topology["topology"]),
        divertor_balance=float(topology["divertor_balance"]),
        lower_strike_point_count=int(topology["lower_strike_count"]),
        upper_strike_point_count=int(topology["upper_strike_count"]),
        machine=machine,
    )


def grad_shafranov_residual(psi: Array, r: Array, z: Array, major_r: Array, j_phi: Array, inside: Array) -> float:
    if r.size < 3 or z.size < 3:
        return 0.0
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    interior = inside[1:-1, 1:-1]
    if not np.any(interior):
        interior = np.ones_like(interior, dtype=bool)
    r_here = np.maximum(major_r[1:-1, 1:-1], 1.0e-12)
    d2_dr2 = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (dr * dr)
    d_dr = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dr)
    d2_dz2 = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dz * dz)
    lhs = d2_dr2 - d_dr / r_here + d2_dz2
    rhs = MU0 * r_here * j_phi[1:-1, 1:-1]
    raw = lhs - rhs
    scale = max(float(np.sqrt(np.mean(rhs[interior] * rhs[interior]))), 1.0e-30)
    return float(np.sqrt(np.mean(raw[interior] * raw[interior])) / scale)


def current_components_from_flux(
    params: EquilibriumParams,
    psi_norm: Array,
    inside: Array,
    major_r: Array,
    dr: float,
    dz: float,
    psi_span: float | None = None,
) -> tuple[Array, Array, Array, Array]:
    pressure_shape = np.where(inside, pressure_shape_from_flux(params, psi_norm), 0.0)
    current_shape = np.where(inside, current_shape_from_flux(params, psi_norm), 0.0)
    if not np.any(pressure_shape > 0.0) and not np.any(current_shape > 0.0):
        zeros = np.zeros_like(current_shape)
        return zeros, zeros, zeros, current_shape

    ip_total = params.plasma_current_ma * 1.0e6
    p_axis = (params.beta_percent / 100.0) * params.b0_t * params.b0_t / (2.0 * MU0)
    normalized_grid = np.linspace(0.0, 1.0, 512)
    shape_1d = pressure_shape_from_flux(params, normalized_grid)
    shape_integral = float(np.trapezoid(shape_1d, normalized_grid) * max(float(psi_span or 1.0), 1.0e-12))
    pprime_axis = p_axis / max(shape_integral, 1.0e-30)
    pprime = pprime_axis * pressure_shape

    pressure_current = float(np.sum(major_r * pprime) * dr * dz)
    pressure_limit = max(float(params.pressure_current_fraction), 0.0) * abs(ip_total)
    if pressure_current > pressure_limit > 0.0:
        pprime *= pressure_limit / pressure_current
        pressure_current = pressure_limit
    elif params.pressure_current_fraction <= 0.0:
        pprime *= 0.0
        pressure_current = 0.0

    remaining_current = ip_total - pressure_current
    ff_integral = float(np.sum(current_shape / np.maximum(MU0 * major_r, 1.0e-30)) * dr * dz)
    ff_axis = remaining_current / max(ff_integral, 1.0e-30)
    ffprime = ff_axis * current_shape
    j_phi = major_r * pprime + ffprime / np.maximum(MU0 * major_r, 1.0e-30)
    return j_phi, pprime, ffprime, current_shape


def normalize_q_profile(q_values: Array, params: EquilibriumParams) -> Array:
    q = np.asarray(q_values, dtype=float).copy()
    if q.size == 0:
        return q
    q = np.where(np.isfinite(q), q, params.q_axis)
    q = np.maximum.accumulate(np.clip(q, 0.2, 25.0))
    source_span = float(q[-1] - q[0])
    target_span = float(params.q_edge - params.q_axis)
    if abs(source_span) > 1.0e-12 and target_span > 0.0:
        q = params.q_axis + (q - q[0]) * target_span / source_span
    else:
        rho = np.linspace(0.0, 1.0, q.size)
        q = params.q_axis + target_span * rho ** 1.65
    q = np.maximum.accumulate(np.clip(q, 0.2, 25.0))
    q[0] = params.q_axis
    q[-1] = params.q_edge
    return q


def field_estimated_q_profile(
    params: EquilibriumParams,
    r: Array,
    z: Array,
    psi_norm: Array,
    br: Array,
    bz: Array,
    b_phi: Array,
    inside: Array,
) -> Array:
    rho = np.linspace(0.0, 1.0, 80)
    fallback = normalize_q_profile(params.q_axis + (params.q_edge - params.q_axis) * rho ** 1.65, params)
    mid = int(np.argmin(np.abs(z)))
    mask = inside[:, mid] & np.isfinite(psi_norm[:, mid])
    if np.count_nonzero(mask) < 5:
        return fallback
    psi_line = np.clip(psi_norm[:, mid][mask], 0.0, 1.0)
    r_line = np.abs(r[mask])
    order = np.argsort(psi_line)
    psi_sorted = psi_line[order]
    r_sorted = r_line[order]
    unique = np.concatenate(([True], np.diff(psi_sorted) > 1.0e-6))
    psi_sorted = psi_sorted[unique]
    r_sorted = r_sorted[unique]
    if psi_sorted.size < 5 or psi_sorted[-1] < 0.35:
        return fallback

    bpol_line = np.sqrt(br[:, mid] * br[:, mid] + bz[:, mid] * bz[:, mid])
    bphi_line = np.abs(b_phi[:, mid])
    bpol_sorted = bpol_line[mask][order][unique]
    bphi_sorted = bphi_line[mask][order][unique]
    target = np.clip(rho * rho, psi_sorted[0], psi_sorted[-1])
    r_at = np.interp(target, psi_sorted, r_sorted)
    bpol_at = np.interp(target, psi_sorted, bpol_sorted)
    bphi_at = np.interp(target, psi_sorted, bphi_sorted)
    q_raw = np.abs(r_at) * bphi_at / np.maximum(params.major_radius_m * bpol_at, 1.0e-8)
    finite = np.isfinite(q_raw) & (q_raw > 0.0)
    if np.count_nonzero(finite) < 8:
        return fallback
    q = np.where(finite, q_raw, np.nan)
    first_valid = float(q[finite][0])
    q = np.where(np.isfinite(q), q, first_valid)
    edge_index = int(np.max(np.nonzero(finite)))
    edge_value = max(float(q[edge_index]), 1.0e-8)
    q *= params.q_edge / edge_value
    q[0] = params.q_axis
    q = np.maximum.accumulate(np.clip(q, 0.2, 25.0))
    q += (fallback - q) * np.linspace(1.0, 0.0, q.size) ** 3
    return normalize_q_profile(q, params)


def magnetic_axis_and_lcfs(psi: Array, psi_norm: Array, r: Array, z: Array) -> tuple[float, float, float]:
    axis_index = np.unravel_index(int(np.argmin(psi)), psi.shape)
    axis_r = float(r[axis_index[0]])
    axis_z = float(z[axis_index[1]])
    lcfs_values = psi[np.abs(psi_norm - 1.0) <= 0.03]
    psi_lcfs = float(np.median(lcfs_values)) if lcfs_values.size else float(np.percentile(psi, 75.0))
    return axis_r, axis_z, psi_lcfs


def core_mask_from_axis(psi_norm: Array, axis_index: tuple[int, int], geometry_mask: Array | None = None) -> Array:
    allowed = np.asarray(psi_norm <= 1.0, dtype=bool)
    if geometry_mask is not None:
        allowed &= np.asarray(geometry_mask, dtype=bool)
    nx, nz = allowed.shape
    start_i = min(max(int(axis_index[0]), 0), nx - 1)
    start_j = min(max(int(axis_index[1]), 0), nz - 1)
    if not allowed[start_i, start_j]:
        return allowed
    mask = np.zeros_like(allowed, dtype=bool)
    stack = [(start_i, start_j)]
    while stack:
        i, j = stack.pop()
        if mask[i, j] or not allowed[i, j]:
            continue
        mask[i, j] = True
        if i > 0:
            stack.append((i - 1, j))
        if i < nx - 1:
            stack.append((i + 1, j))
        if j > 0:
            stack.append((i, j - 1))
        if j < nz - 1:
            stack.append((i, j + 1))
    return mask


def estimate_x_points(psi: Array, r: Array, z: Array, max_points: int = 4) -> list[tuple[float, float, float]]:
    if r.size < 5 or z.size < 5:
        return []
    dpsi_dr = np.gradient(psi, r, axis=0, edge_order=2)
    dpsi_dz = np.gradient(psi, z, axis=1, edge_order=2)
    bpol2 = dpsi_dr * dpsi_dr + dpsi_dz * dpsi_dz
    d2_rr = np.gradient(dpsi_dr, r, axis=0, edge_order=2)
    d2_zz = np.gradient(dpsi_dz, z, axis=1, edge_order=2)
    d2_rz = np.gradient(dpsi_dr, z, axis=1, edge_order=2)
    axis_i, axis_j = np.unravel_index(int(np.argmin(psi)), psi.shape)
    candidates: list[tuple[float, float, float, float]] = []
    for i in range(2, psi.shape[0] - 2):
        for j in range(2, psi.shape[1] - 2):
            if abs(i - axis_i) <= 2 and abs(j - axis_j) <= 2:
                continue
            local = bpol2[i - 1 : i + 2, j - 1 : j + 2]
            if bpol2[i, j] > float(np.min(local)):
                continue
            hessian_det = d2_rr[i, j] * d2_zz[i, j] - d2_rz[i, j] * d2_rz[i, j]
            if hessian_det >= 0.0:
                continue
            candidates.append((float(bpol2[i, j]), float(r[i]), float(z[j]), float(psi[i, j])))
    candidates.sort(key=lambda item: item[0])
    return [(r_value, z_value, psi_value) for _, r_value, z_value, psi_value in candidates[:max_points]]


def estimate_strike_points(
    machine: MachineGeometry,
    psi_norm: Array,
    psi: Array,
    r: Array,
    z: Array,
    max_points: int = 4,
) -> list[tuple[float, float, float]]:
    if machine.limiter_r.size < 3:
        return []
    dense_r, dense_z = resample_polygon(machine.limiter_r, machine.limiter_z, 10)
    limiter_psi_norm = sample_grid(psi_norm, r, z, dense_r, dense_z)
    limiter_psi = sample_grid(psi, r, z, dense_r, dense_z)
    error = np.abs(limiter_psi_norm - 1.0)
    finite = np.isfinite(error)
    if np.count_nonzero(finite) == 0:
        return []
    candidates: list[tuple[float, float, float, float]] = []
    count = dense_r.size
    for idx in range(count):
        if not finite[idx]:
            continue
        prev_idx = (idx - 1) % count
        next_idx = (idx + 1) % count
        local = error[idx] <= error[prev_idx] and error[idx] <= error[next_idx]
        if local or len(candidates) < max_points:
            candidates.append((float(error[idx]), float(dense_r[idx]), float(dense_z[idx]), float(limiter_psi[idx])))
    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[float, float, float]] = []
    for _, rv, zv, psiv in candidates:
        if all(math.hypot(rv - old_r, zv - old_z) > 0.08 * max(float(np.ptp(machine.limiter_r)), float(np.ptp(machine.limiter_z)), 1.0e-12) for old_r, old_z, _ in selected):
            selected.append((rv, zv, psiv))
        if len(selected) >= max_points:
            break
    return selected


def separatrix_topology_from_points(
    x_points: Sequence[tuple[float, float, float]],
    strike_points: Sequence[tuple[float, float, float]],
) -> dict[str, object]:
    lower_x = sum(1 for _, zv, _ in x_points if zv < -1.0e-3)
    upper_x = sum(1 for _, zv, _ in x_points if zv > 1.0e-3)
    lower_strike = sum(1 for _, zv, _ in strike_points if zv < -1.0e-3)
    upper_strike = sum(1 for _, zv, _ in strike_points if zv > 1.0e-3)
    if lower_x and upper_x:
        topology = "double-null"
    elif lower_x:
        topology = "lower-single-null"
    elif upper_x:
        topology = "upper-single-null"
    elif strike_points:
        topology = "limited-strike"
    else:
        topology = "limited"
    denom = max(lower_strike + upper_strike, 1)
    return {
        "topology": topology,
        "lower_strike_count": lower_strike,
        "upper_strike_count": upper_strike,
        "divertor_balance": float((upper_strike - lower_strike) / denom),
    }


def _profile_fields_from_flux(params: EquilibriumParams, psi: Array, r: Array, z: Array, major_r: Array, psi_edge: float | None = None) -> dict[str, Array]:
    psi_axis = float(np.min(psi))
    edge_value = float(max(np.max(psi) if psi_edge is None else psi_edge, psi_axis + 1.0e-30))
    psi_norm = np.clip((psi - psi_axis) / max(edge_value - psi_axis, 1.0e-30), 0.0, None)
    rho_pol = np.sqrt(np.maximum(psi_norm, 0.0))
    inside = psi_norm <= 1.0
    core_shape = np.maximum(1.0 - np.minimum(psi_norm, 1.0), 0.0)

    dpsi_dr = np.gradient(psi, r, axis=0, edge_order=2)
    dpsi_dz = np.gradient(psi, z, axis=1, edge_order=2)
    br = -dpsi_dz / np.maximum(major_r, 1.0e-12)
    bz = dpsi_dr / np.maximum(major_r, 1.0e-12)
    f_flux = params.major_radius_m * params.b0_t * (1.0 + 0.04 * core_shape)
    b_phi = f_flux / np.maximum(major_r, 1.0e-12)
    b_total = np.sqrt(br * br + bz * bz + b_phi * b_phi)

    beta = params.beta_percent / 100.0
    p_axis = beta * params.b0_t * params.b0_t / (2.0 * MU0)
    pressure_pa = p_axis * pressure_shape_from_flux(params, psi_norm)
    pressure_pa = np.where(inside, pressure_pa, 0.0)
    temperature_kev = params.temperature_axis_kev * (0.25 + 0.75 * core_shape ** 0.55)
    temperature_kev = np.where(inside, temperature_kev, 0.25 * params.temperature_axis_kev)
    density = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * core_shape ** 0.7
    )
    density = np.where(inside, density, params.density_axis_1e19_m3 * params.density_edge_fraction * np.exp(-(rho_pol - 1.0) ** 2 / 0.18))

    return {
        "psi_norm": psi_norm,
        "rho_pol": rho_pol,
        "inside": inside,
        "core_shape": core_shape,
        "br": br,
        "bz": bz,
        "b_phi": b_phi,
        "b_total": b_total,
        "pressure_pa": pressure_pa,
        "temperature_kev": temperature_kev,
        "density": density,
        "p_axis": np.asarray(p_axis),
    }


def build_iterative_gs_equilibrium(params: EquilibriumParams) -> EquilibriumState:
    machine = load_machine_geometry(params)
    base_params = replace(params, equilibrium_model="elongated-tokamak")
    base = build_analytic_equilibrium(base_params)
    r = base.r
    z = base.z
    major_r = base.major_r
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    psi = base.psi.copy()
    psi_boundary = float(np.max(base.psi[base.inside])) if np.any(base.inside) else float(np.max(base.psi))
    psi[:, 0] = psi_boundary
    psi[:, -1] = psi_boundary
    psi[-1, :] = psi_boundary
    psi[0, 1:-1] = psi[1, 1:-1]

    iterations = 0
    change = float("inf")
    j_phi = np.zeros_like(psi)
    omega = float(params.gs_relaxation)

    for iterations in range(1, params.gs_iterations + 1):
        old = psi.copy()
        psi_axis = float(np.min(psi))
        psi_span = max(psi_boundary - psi_axis, 1.0e-30)
        psi_norm = np.clip((psi - psi_axis) / psi_span, 0.0, 1.0)
        inside = psi_norm <= 1.0
        inside[:, 0] = False
        inside[:, -1] = False
        inside[-1, :] = False
        j_phi, _, _, _ = current_components_from_flux(params, psi_norm, inside, major_r, dr, dz, psi_span)

        for i in range(1, params.n_r - 1):
            r_here = max(float(major_r[i, 0]), 1.0e-12)
            c_rp = inv_dr2 - 1.0 / (2.0 * r_here * dr)
            c_rm = inv_dr2 + 1.0 / (2.0 * r_here * dr)
            c_z = inv_dz2
            c0 = -2.0 * inv_dr2 - 2.0 * inv_dz2
            for j in range(1, params.n_z - 1):
                rhs = MU0 * r_here * j_phi[i, j]
                candidate = (
                    rhs
                    - c_rp * psi[i + 1, j]
                    - c_rm * psi[i - 1, j]
                    - c_z * (psi[i, j + 1] + psi[i, j - 1])
                ) / c0
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * candidate

        psi[1:-1, 1:-1] = np.minimum(psi[1:-1, 1:-1], psi_boundary)
        psi[:, 0] = psi_boundary
        psi[:, -1] = psi_boundary
        psi[-1, :] = psi_boundary
        psi[0, 1:-1] = psi[1, 1:-1]
        change = float(np.max(np.abs(psi - old)) / max(psi_span, 1.0e-30))
        if change <= params.gs_tolerance:
            break

    fields = _profile_fields_from_flux(params, psi, r, z, major_r)
    axis_index = np.unravel_index(int(np.argmin(psi)), psi.shape)
    fields["inside"] = core_mask_from_axis(fields["psi_norm"], axis_index)
    j_phi, pprime, ffprime, _ = current_components_from_flux(
        params,
        fields["psi_norm"],
        fields["inside"],
        major_r,
        dr,
        dz,
        float(np.max(psi) - np.min(psi)),
    )
    operator_residual = grad_shafranov_residual(psi, r, z, major_r, j_phi, fields["inside"])
    axis_r, axis_z, lcfs_value = magnetic_axis_and_lcfs(psi, fields["psi_norm"], r, z)
    x_points = estimate_x_points(psi, r, z)
    primary_x = x_points[0] if x_points else (0.0, 0.0, 0.0)
    strike_points = estimate_strike_points(machine, fields["psi_norm"], psi, r, z)
    primary_strike = strike_points[0] if strike_points else (0.0, 0.0, 0.0)
    topology = separatrix_topology_from_points(x_points, strike_points)
    strike_psi_norm = (
        float(sample_grid(fields["psi_norm"], r, z, np.asarray([primary_strike[0]]), np.asarray([primary_strike[1]]))[0])
        if strike_points
        else 0.0
    )

    q_profile_rho = np.linspace(0.0, 1.0, 80)
    q_profile = field_estimated_q_profile(params, r, z, fields["psi_norm"], fields["br"], fields["bz"], fields["b_phi"], fields["inside"])
    p_axis = float(fields["p_axis"])
    p_profile_pa = pressure_profile_1d(params, q_profile_rho)
    density_profile = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * np.maximum(1.0 - q_profile_rho * q_profile_rho, 0.0) ** 0.7
    )

    return EquilibriumState(
        r=r,
        z=z,
        major_r=major_r,
        psi=psi,
        psi_norm=fields["psi_norm"],
        rho_pol=fields["rho_pol"],
        br=fields["br"],
        bz=fields["bz"],
        b_phi=fields["b_phi"],
        b_total=fields["b_total"],
        density=fields["density"],
        pressure_pa=fields["pressure_pa"],
        temperature_kev=fields["temperature_kev"],
        j_phi=j_phi,
        q_profile_rho=q_profile_rho,
        q_profile=q_profile,
        p_profile_pa=p_profile_pa,
        density_profile=density_profile,
        inside=fields["inside"],
        solver_iterations=iterations,
        solver_residual=change,
        operator_residual=operator_residual,
        pprime=pprime,
        ffprime=ffprime,
        magnetic_axis_r_m=axis_r,
        magnetic_axis_z_m=axis_z,
        psi_lcfs=lcfs_value,
        x_point_count=len(x_points),
        primary_x_point_r_m=primary_x[0],
        primary_x_point_z_m=primary_x[1],
        primary_x_point_psi=primary_x[2],
        strike_point_count=len(strike_points),
        primary_strike_point_r_m=primary_strike[0],
        primary_strike_point_z_m=primary_strike[1],
        primary_strike_point_psi_norm=strike_psi_norm,
        separatrix_topology=str(topology["topology"]),
        divertor_balance=float(topology["divertor_balance"]),
        lower_strike_point_count=int(topology["lower_strike_count"]),
        upper_strike_point_count=int(topology["upper_strike_count"]),
        machine=machine,
    )


def _limiter_flux_value(params: EquilibriumParams, psi: Array, r: Array, z: Array, machine: MachineGeometry | None = None) -> float:
    machine = machine or default_machine_geometry(params)
    samples = sample_grid(psi, r, z, machine.limiter_r, machine.limiter_z)
    finite = samples[np.isfinite(samples)]
    if finite.size:
        return float(np.median(finite))
    z_span = max(params.elongation * params.minor_radius_m, 1.0e-12)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    z_norm = zz / z_span
    triangular_shift = params.triangularity * params.minor_radius_m * z_norm * z_norm
    limiter_level = ((rr + triangular_shift) / max(params.minor_radius_m, 1.0e-12)) ** 2 + z_norm * z_norm
    band = np.abs(limiter_level - 1.0) <= max(0.03, 2.5 / max(params.n_r, params.n_z))
    if np.any(band):
        return float(np.median(psi[band]))
    return float(np.percentile(psi, 72.0))


def _pf_coil_flux(params: EquilibriumParams, r: Array, z: Array, machine: MachineGeometry | None = None) -> Array:
    machine = machine or default_machine_geometry(params)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    ref_len = max(float(np.max(r) - np.min(r)), float(np.max(z) - np.min(z)), 1.0e-12)
    flux = np.zeros_like(rr)
    smoothing = 0.35 * max(abs(r[1] - r[0]) if r.size > 1 else 1.0e-12, abs(z[1] - z[0]) if z.size > 1 else 1.0e-12)
    current_sources = [(coil.r_m, coil.z_m, coil.current_ma, coil.turns, coil.width_m, coil.height_m) for coil in machine.coils]
    current_sources.extend((item.r_m, item.z_m, item.current_ma, item.turns, item.width_m, item.height_m) for item in machine.passive_structures)
    for source_r, source_z, current_ma, turns, width_m, height_m in current_sources:
        ampere_turns = current_ma * 1.0e6 * turns
        if abs(ampere_turns) <= 0.0:
            continue
        source_smoothing = max(smoothing, 0.25 * math.hypot(width_m, height_m))
        dist = np.sqrt((rr - source_r) ** 2 + (zz - source_z) ** 2 + source_smoothing * source_smoothing)
        flux += -MU0 * ampere_turns * max(params.major_radius_m + source_r, 1.0e-12) * np.log(dist / ref_len) / (2.0 * math.pi)
    return flux


def _coil_flux_at_points(
    params: EquilibriumParams,
    coils: Sequence[PFCoil],
    sample_r: Array,
    sample_z: Array,
    ref_len: float,
    *,
    unit_coil_index: int | None = None,
) -> Array:
    sr = np.asarray(sample_r, dtype=float)
    sz = np.asarray(sample_z, dtype=float)
    flux = np.zeros_like(sr, dtype=float)
    smoothing = 1.0e-3 * max(params.minor_radius_m, 1.0e-12)
    for idx, coil in enumerate(coils):
        if unit_coil_index is not None and idx != unit_coil_index:
            continue
        ampere_turns = (1.0e6 * coil.turns) if unit_coil_index is not None else (coil.current_ma * 1.0e6 * coil.turns)
        if abs(ampere_turns) <= 0.0:
            continue
        dist = np.sqrt((sr - coil.r_m) ** 2 + (sz - coil.z_m) ** 2 + smoothing * smoothing)
        flux += -MU0 * ampere_turns * max(params.major_radius_m + coil.r_m, 1.0e-12) * np.log(dist / max(ref_len, 1.0e-12)) / (2.0 * math.pi)
    return flux


def load_shape_constraints(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("shape_constraint must contain a JSON object.")
    return dict(data)


def _constraint_points_from_json(raw: Mapping[str, Any]) -> tuple[list[tuple[float, float, float, float, float]], list[tuple[float, float, float]]]:
    isoflux_rows: list[tuple[float, float, float, float, float]] = []
    point_rows: list[tuple[float, float, float]] = []
    for item in raw.get("isoflux", []):
        if not isinstance(item, Mapping):
            continue
        r1 = item.get("r1_m", item.get("r1"))
        z1 = item.get("z1_m", item.get("z1"))
        r2 = item.get("r2_m", item.get("r2"))
        z2 = item.get("z2_m", item.get("z2"))
        if None in {r1, z1, r2, z2}:
            continue
        isoflux_rows.append((float(r1), float(z1), float(r2), float(z2), float(item.get("weight", 1.0))))
    for section in ("xpoints", "strike_points", "points"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            rv = item.get("r_m", item.get("r"))
            zv = item.get("z_m", item.get("z"))
            if rv is None or zv is None:
                continue
            point_rows.append((float(rv), float(zv), float(item.get("weight", 1.0))))
    return isoflux_rows, point_rows


def apply_shape_control_to_machine(
    params: EquilibriumParams,
    machine: MachineGeometry,
    r: Array,
    z: Array,
    base_psi: Array,
) -> tuple[MachineGeometry, dict[str, float]]:
    if params.shape_control in {"off", "forward"} or not params.shape_constraint:
        return machine, {"rms_error": 0.0, "max_error": 0.0, "rank": 0.0}
    raw = load_shape_constraints(params.shape_constraint)
    isoflux_rows, point_rows = _constraint_points_from_json(raw)
    controlled = [idx for idx, coil in enumerate(machine.coils) if coil.control]
    if not controlled or (not isoflux_rows and not point_rows):
        return machine, {"rms_error": 0.0, "max_error": 0.0, "rank": 0.0}

    ref_len = max(float(np.max(r) - np.min(r)), float(np.max(z) - np.min(z)), 1.0e-12)
    limiter_r, limiter_z = resample_polygon(machine.limiter_r, machine.limiter_z, 8)
    limiter_base = sample_grid(base_psi, r, z, limiter_r, limiter_z)
    limiter_ref = float(np.nanmedian(limiter_base)) if np.any(np.isfinite(limiter_base)) else 0.0
    rows: list[list[float]] = []
    targets: list[float] = []

    def response_at(coil_index: int, rv: float, zv: float) -> float:
        return float(_coil_flux_at_points(params, machine.coils, np.asarray([rv]), np.asarray([zv]), ref_len, unit_coil_index=coil_index)[0])

    for r1, z1, r2, z2, weight in isoflux_rows:
        r1m = r1 - params.major_radius_m
        r2m = r2 - params.major_radius_m
        base_diff = float(sample_grid(base_psi, r, z, np.asarray([r1m]), np.asarray([z1]))[0] - sample_grid(base_psi, r, z, np.asarray([r2m]), np.asarray([z2]))[0])
        rows.append([weight * (response_at(idx, r1m, z1) - response_at(idx, r2m, z2)) for idx in controlled])
        targets.append(-weight * base_diff)

    for rv, zv, weight in point_rows:
        r_minor = rv - params.major_radius_m
        base_value = float(sample_grid(base_psi, r, z, np.asarray([r_minor]), np.asarray([zv]))[0])
        limiter_response_mean = [
            float(np.nanmean(_coil_flux_at_points(params, machine.coils, limiter_r, limiter_z, ref_len, unit_coil_index=idx)))
            for idx in controlled
        ]
        rows.append([weight * (response_at(idx, r_minor, zv) - limiter_response_mean[col]) for col, idx in enumerate(controlled)])
        targets.append(-weight * (base_value - limiter_ref))

    if not rows:
        return machine, {"rms_error": 0.0, "max_error": 0.0, "rank": 0.0}
    a_matrix = np.asarray(rows, dtype=float)
    b_vec = np.asarray(targets, dtype=float)
    if params.shape_control_damping > 0.0:
        damp = math.sqrt(params.shape_control_damping)
        a_matrix = np.vstack([a_matrix, damp * np.eye(len(controlled))])
        b_vec = np.concatenate([b_vec, np.zeros(len(controlled))])
    solution, *_ = np.linalg.lstsq(a_matrix, b_vec, rcond=None)
    adjusted: list[PFCoil] = []
    limit = float(params.shape_control_current_limit_ma)
    for idx, coil in enumerate(machine.coils):
        if idx in controlled:
            pos = controlled.index(idx)
            current = float(np.clip(coil.current_ma + params.shape_control_gain * solution[pos], -limit, limit))
            adjusted.append(replace(coil, current_ma=current))
        else:
            adjusted.append(coil)
    new_machine = replace(machine, coils=tuple(adjusted))
    residual = a_matrix[: len(rows), :] @ solution - np.asarray(targets, dtype=float)
    return new_machine, {
        "rms_error": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        "max_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "rank": float(np.linalg.matrix_rank(a_matrix[: len(rows), :])),
    }


def shape_control_response_rows(params: EquilibriumParams, machine: MachineGeometry) -> list[list[object]]:
    if params.shape_control == "off" or not params.shape_constraint:
        return []
    raw = load_shape_constraints(params.shape_constraint)
    isoflux_rows, point_rows = _constraint_points_from_json(raw)
    if not isoflux_rows and not point_rows:
        return []
    base_params = replace(params, shape_control="off")
    base_machine = load_machine_geometry(base_params)
    rows: list[list[object]] = []
    for idx, coil in enumerate(machine.coils):
        base_current = base_machine.coils[idx].current_ma if idx < len(base_machine.coils) else 0.0
        delta = coil.current_ma - base_current
        limit = max(float(params.shape_control_current_limit_ma), 1.0e-30)
        margin = max(limit - abs(coil.current_ma), 0.0)
        rows.append(
            [
                idx + 1,
                coil.name,
                int(bool(coil.control)),
                f"{base_current:.10e}",
                f"{coil.current_ma:.10e}",
                f"{delta:.10e}",
                f"{limit:.10e}",
                f"{margin:.10e}",
                int(abs(coil.current_ma) >= 0.999 * limit),
            ]
        )
    return rows


def pf_control_matrix_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    if not params.shape_constraint:
        return []
    machine = state.machine or load_machine_geometry(params)
    raw = load_shape_constraints(params.shape_constraint)
    isoflux_rows, point_rows = _constraint_points_from_json(raw)
    controlled = [idx for idx, coil in enumerate(machine.coils) if coil.control]
    if not controlled or (not isoflux_rows and not point_rows):
        return []
    ref_len = max(float(np.max(state.r) - np.min(state.r)), float(np.max(state.z) - np.min(state.z)), 1.0e-12)
    limiter_r, limiter_z = resample_polygon(machine.limiter_r, machine.limiter_z, 8)
    limiter_psi = sample_grid(state.psi, state.r, state.z, limiter_r, limiter_z)
    limiter_ref = float(np.nanmedian(limiter_psi)) if np.any(np.isfinite(limiter_psi)) else 0.0

    def response_at(coil_index: int, rv_minor: float, zv: float) -> float:
        return float(_coil_flux_at_points(params, machine.coils, np.asarray([rv_minor]), np.asarray([zv]), ref_len, unit_coil_index=coil_index)[0])

    table: list[list[object]] = []
    constraint_index = 0
    for r1, z1, r2, z2, weight in isoflux_rows:
        constraint_index += 1
        r1m = r1 - params.major_radius_m
        r2m = r2 - params.major_radius_m
        residual = float(sample_grid(state.psi, state.r, state.z, np.asarray([r1m]), np.asarray([z1]))[0] - sample_grid(state.psi, state.r, state.z, np.asarray([r2m]), np.asarray([z2]))[0])
        for idx in controlled:
            response = response_at(idx, r1m, z1) - response_at(idx, r2m, z2)
            table.append([constraint_index, "isoflux_pair", f"({r1:.5g},{z1:.5g})-({r2:.5g},{z2:.5g})", machine.coils[idx].name, idx + 1, f"{weight:.10e}", f"{response:.10e}", f"{(-residual):.10e}", f"{residual:.10e}", "Wb/MA"])

    limiter_response_mean = {
        idx: float(np.nanmean(_coil_flux_at_points(params, machine.coils, limiter_r, limiter_z, ref_len, unit_coil_index=idx)))
        for idx in controlled
    }
    for rv, zv, weight in point_rows:
        constraint_index += 1
        rv_minor = rv - params.major_radius_m
        point_psi = float(sample_grid(state.psi, state.r, state.z, np.asarray([rv_minor]), np.asarray([zv]))[0])
        residual = point_psi - limiter_ref
        for idx in controlled:
            response = response_at(idx, rv_minor, zv) - limiter_response_mean[idx]
            table.append([constraint_index, "point_to_limiter", f"({rv:.5g},{zv:.5g})-limiter", machine.coils[idx].name, idx + 1, f"{weight:.10e}", f"{response:.10e}", f"{(-residual):.10e}", f"{residual:.10e}", "Wb/MA"])
    return table


def pf_control_summary_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    machine = state.machine or load_machine_geometry(params)
    matrix_rows = pf_control_matrix_rows(state, params)
    controlled = [coil for coil in machine.coils if coil.control]
    constraint_ids = sorted({int(row[0]) for row in matrix_rows}) if matrix_rows else []
    matrix = np.asarray([float(row[6]) for row in matrix_rows], dtype=float)
    response_matrix = matrix.reshape((len(constraint_ids), len(controlled))) if constraint_ids and controlled and matrix.size == len(constraint_ids) * len(controlled) else np.zeros((0, 0))
    rank = int(np.linalg.matrix_rank(response_matrix)) if response_matrix.size else 0
    condition = float(np.linalg.cond(response_matrix)) if response_matrix.size and min(response_matrix.shape) > 0 and rank == min(response_matrix.shape) else 0.0
    saturated = sum(1 for coil in machine.coils if abs(coil.current_ma) >= 0.999 * params.shape_control_current_limit_ma)
    return [
        ["shape_control_mode", params.shape_control, "off/forward/inverse/forward-inverse"],
        ["semantics", "static_forward" if params.shape_control == "forward" else "static_inverse" if params.shape_control in {"isoflux", "inverse"} else "static_forward_inverse" if params.shape_control == "forward-inverse" else "none", "FreeGS-like reduced PF control mode"],
        ["constraint_count", len(constraint_ids), "shape constraints included in response matrix"],
        ["controlled_coil_count", len(controlled), "PF coils with control=true"],
        ["response_matrix_rank", rank, "linear response rank"],
        ["response_matrix_condition", condition, "0 means unavailable or rank deficient"],
        ["shape_rms_error", state.shape_control_rms_error, "final weighted psi_norm RMS"],
        ["shape_max_error", state.shape_control_max_error, "final weighted psi_norm max error"],
        ["saturated_coil_count", saturated, "coils at current limit"],
    ]


def shape_constraint_diagnostics(
    params: EquilibriumParams,
    machine: MachineGeometry,
    r: Array,
    z: Array,
    psi: Array,
    psi_norm: Array,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    if not params.shape_constraint:
        return [], {"rms_error": 0.0, "max_error": 0.0, "rank": 0.0}
    raw = load_shape_constraints(params.shape_constraint)
    isoflux_rows, point_rows = _constraint_points_from_json(raw)
    if not isoflux_rows and not point_rows:
        return [], {"rms_error": 0.0, "max_error": 0.0, "rank": 0.0}

    limiter_r, limiter_z = resample_polygon(machine.limiter_r, machine.limiter_z, 8)
    limiter_psi = sample_grid(psi, r, z, limiter_r, limiter_z)
    limiter_psi_norm = sample_grid(psi_norm, r, z, limiter_r, limiter_z)
    limiter_ref = float(np.nanmedian(limiter_psi)) if np.any(np.isfinite(limiter_psi)) else 0.0
    limiter_norm_ref = float(np.nanmedian(limiter_psi_norm)) if np.any(np.isfinite(limiter_psi_norm)) else 1.0

    rows: list[dict[str, object]] = []
    residuals: list[float] = []

    def sample_pair(array: Array, rv_major: float, zv: float) -> float:
        rv_minor = float(rv_major) - params.major_radius_m
        return float(sample_grid(array, r, z, np.asarray([rv_minor]), np.asarray([float(zv)]))[0])

    for index, (r1, z1, r2, z2, weight) in enumerate(isoflux_rows, 1):
        psi_delta = sample_pair(psi, r1, z1) - sample_pair(psi, r2, z2)
        norm_delta = sample_pair(psi_norm, r1, z1) - sample_pair(psi_norm, r2, z2)
        residual = weight * norm_delta
        if np.isfinite(residual):
            residuals.append(float(residual))
        rows.append(
            {
                "constraint_index": index,
                "kind": "isoflux_pair",
                "r1_m": r1,
                "z1_m": z1,
                "r2_m": r2,
                "z2_m": z2,
                "weight": weight,
                "psi_delta_wb": psi_delta,
                "psi_norm_delta": norm_delta,
                "weighted_residual_norm": residual,
            }
        )

    offset = len(rows)
    for index, (rv, zv, weight) in enumerate(point_rows, 1):
        point_psi = sample_pair(psi, rv, zv)
        point_norm = sample_pair(psi_norm, rv, zv)
        psi_delta = point_psi - limiter_ref
        norm_delta = point_norm - limiter_norm_ref
        residual = weight * norm_delta
        if np.isfinite(residual):
            residuals.append(float(residual))
        rows.append(
            {
                "constraint_index": offset + index,
                "kind": "point_to_limiter",
                "r1_m": rv,
                "z1_m": zv,
                "r2_m": "",
                "z2_m": "",
                "weight": weight,
                "psi_delta_wb": psi_delta,
                "psi_norm_delta": norm_delta,
                "weighted_residual_norm": residual,
            }
        )

    residual_array = np.asarray(residuals, dtype=float)
    stats = {
        "rms_error": float(np.sqrt(np.mean(residual_array * residual_array))) if residual_array.size else 0.0,
        "max_error": float(np.max(np.abs(residual_array))) if residual_array.size else 0.0,
        "rank": float(min(len(residuals), sum(1 for coil in machine.coils if coil.control))),
    }
    return rows, stats


def _vacuum_boundary_from_current(psi: Array, r: Array, z: Array, major_r: Array, j_phi: Array) -> Array:
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    source = np.asarray(j_phi, dtype=float) * np.asarray(major_r, dtype=float) * dr * dz
    active = source > max(float(np.max(source)) * 1.0e-8, 0.0)
    boundary = psi.copy()
    if not np.any(active):
        return boundary

    rr, zz = np.meshgrid(r, z, indexing="ij")
    src_r = rr[active]
    src_z = zz[active]
    src_strength = source[active]
    ref_len = max(float(np.max(r) - np.min(r)), float(np.max(z) - np.min(z)), 1.0e-12)
    edge_points: list[tuple[int, int]] = []
    edge_points.extend((0, j) for j in range(psi.shape[1]))
    edge_points.extend((psi.shape[0] - 1, j) for j in range(psi.shape[1]))
    edge_points.extend((i, 0) for i in range(1, psi.shape[0] - 1))
    edge_points.extend((i, psi.shape[1] - 1) for i in range(1, psi.shape[0] - 1))

    for i, j in edge_points:
        dist = np.sqrt((r[i] - src_r) ** 2 + (z[j] - src_z) ** 2 + (0.35 * dr) ** 2)
        boundary[i, j] = -MU0 * float(np.sum(src_strength * np.log(dist / ref_len))) / (2.0 * math.pi)
    return boundary


def build_free_boundary_gs_equilibrium(params: EquilibriumParams) -> EquilibriumState:
    extent = float(params.free_boundary_extent)
    machine = load_machine_geometry(params)
    seed_params = replace(params, equilibrium_model="elongated-tokamak")
    a = params.minor_radius_m
    kappa = params.elongation
    coil_r_extent = max((abs(coil.r_m) for coil in machine.coils), default=0.0) + 0.15 * a
    coil_z_extent = max((abs(coil.z_m) for coil in machine.coils), default=0.0) + 0.15 * kappa * a
    r_extent = max(extent * a, float(np.max(np.abs(machine.wall_r))) if machine.wall_r.size else 0.0, coil_r_extent)
    z_extent = max(extent * kappa * a, float(np.max(np.abs(machine.wall_z))) if machine.wall_z.size else 0.0, coil_z_extent)
    r = np.linspace(-r_extent, r_extent, params.n_r)
    z = np.linspace(-z_extent, z_extent, params.n_z)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    major_r = params.major_radius_m + rr
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    z_norm = zz / max(kappa * a, 1.0e-12)
    triangular_shift = params.triangularity * a * z_norm * z_norm
    limiter_level = ((rr + triangular_shift) / max(a, 1.0e-12)) ** 2 + z_norm * z_norm
    limiter_mask = machine_limiter_mask(machine, r, z)
    psi_boundary_seed = 0.5 * params.poloidal_field_fraction * params.b0_t * params.major_radius_m * a
    base_psi = psi_boundary_seed * np.clip(limiter_level, 0.0, extent * extent)
    machine, shape_stats = apply_shape_control_to_machine(params, machine, r, z, base_psi)
    coil_flux = _pf_coil_flux(params, r, z, machine)
    psi = base_psi + coil_flux
    psi_lcfs = _limiter_flux_value(seed_params, psi, r, z, machine)
    boundary_values = psi.copy()
    boundary_values[:, 0] = psi_lcfs
    boundary_values[:, -1] = psi_lcfs
    boundary_values[-1, :] = psi_lcfs
    boundary_values[0, :] = boundary_values[1, :]

    iterations = 0
    change = float("inf")
    j_phi = np.zeros_like(psi)
    omega = float(params.gs_relaxation)
    boundary_relaxation = float(params.boundary_relaxation)
    if params.shape_control in {"isoflux", "inverse", "forward-inverse"}:
        boundary_relaxation = min(boundary_relaxation, 0.005)

    for iterations in range(1, params.gs_iterations + 1):
        old = psi.copy()
        psi_axis = float(np.min(psi))
        psi_lcfs = max(_limiter_flux_value(params, psi, r, z, machine), psi_axis + 1.0e-30)
        psi_span = max(psi_lcfs - psi_axis, 1.0e-30)
        psi_norm = np.clip((psi - psi_axis) / psi_span, 0.0, 1.0)
        inside = (psi_norm <= 1.0) & limiter_mask
        j_phi, _, _, _ = current_components_from_flux(params, psi_norm, inside, major_r, dr, dz, psi_span)

        if iterations == 1 or iterations % params.boundary_update_every == 0:
            vacuum_boundary = _vacuum_boundary_from_current(psi, r, z, major_r, j_phi)
            vacuum_boundary += coil_flux
            vacuum_boundary += psi_lcfs - float(np.median(vacuum_boundary[-1, :]))
            alpha = boundary_relaxation
            boundary_values = (1.0 - alpha) * boundary_values + alpha * vacuum_boundary

        for i in range(1, params.n_r - 1):
            r_here = max(float(major_r[i, 0]), 1.0e-12)
            c_rp = inv_dr2 - 1.0 / (2.0 * r_here * dr)
            c_rm = inv_dr2 + 1.0 / (2.0 * r_here * dr)
            c_z = inv_dz2
            c0 = -2.0 * inv_dr2 - 2.0 * inv_dz2
            for j in range(1, params.n_z - 1):
                rhs = MU0 * r_here * j_phi[i, j]
                candidate = (
                    rhs
                    - c_rp * psi[i + 1, j]
                    - c_rm * psi[i - 1, j]
                    - c_z * (psi[i, j + 1] + psi[i, j - 1])
                ) / c0
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * candidate

        psi[:, 0] = boundary_values[:, 0]
        psi[:, -1] = boundary_values[:, -1]
        psi[-1, :] = boundary_values[-1, :]
        psi[0, 1:-1] = psi[1, 1:-1]
        change = float(np.max(np.abs(psi - old)) / psi_span)
        if change <= params.gs_tolerance:
            break

    psi_min = float(np.min(psi))
    psi_lcfs = max(_limiter_flux_value(params, psi, r, z, machine), psi_min + 1.0e-30)
    psi_shifted = psi - psi_min
    fields = _profile_fields_from_flux(params, psi_shifted, r, z, major_r, psi_edge=psi_lcfs - psi_min)
    axis_index = np.unravel_index(int(np.argmin(psi_shifted)), psi_shifted.shape)
    fields["inside"] = core_mask_from_axis(fields["psi_norm"], axis_index, limiter_mask)
    fields["core_shape"] = np.where(fields["inside"], np.maximum(1.0 - np.minimum(fields["psi_norm"], 1.0), 0.0), 0.0)
    f_flux = params.major_radius_m * params.b0_t * (1.0 + 0.04 * fields["core_shape"])
    fields["b_phi"] = f_flux / np.maximum(major_r, 1.0e-12)
    fields["b_total"] = np.sqrt(fields["br"] * fields["br"] + fields["bz"] * fields["bz"] + fields["b_phi"] * fields["b_phi"])
    p_axis = float(fields["p_axis"])
    fields["pressure_pa"] = np.where(fields["inside"], p_axis * pressure_shape_from_flux(params, fields["psi_norm"]), 0.0)
    fields["temperature_kev"] = params.temperature_axis_kev * (0.25 + 0.75 * fields["core_shape"] ** 0.55)
    fields["temperature_kev"] = np.where(fields["inside"], fields["temperature_kev"], 0.25 * params.temperature_axis_kev)
    fields["density"] = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * fields["core_shape"] ** 0.7
    )
    fields["density"] = np.where(
        fields["inside"],
        fields["density"],
        params.density_axis_1e19_m3 * params.density_edge_fraction * np.exp(-(fields["rho_pol"] - 1.0) ** 2 / 0.18),
    )
    j_phi, pprime, ffprime, _ = current_components_from_flux(
        params,
        fields["psi_norm"],
        fields["inside"],
        major_r,
        dr,
        dz,
        psi_lcfs - psi_min,
    )
    operator_residual = grad_shafranov_residual(psi, r, z, major_r, j_phi, fields["inside"])
    axis_r, axis_z, lcfs_value = magnetic_axis_and_lcfs(psi_shifted, fields["psi_norm"], r, z)
    x_points = estimate_x_points(psi_shifted, r, z)
    primary_x = x_points[0] if x_points else (0.0, 0.0, 0.0)
    strike_points = estimate_strike_points(machine, fields["psi_norm"], psi_shifted, r, z)
    primary_strike = strike_points[0] if strike_points else (0.0, 0.0, 0.0)
    topology = separatrix_topology_from_points(x_points, strike_points)

    q_profile_rho = np.linspace(0.0, 1.0, 80)
    q_profile = field_estimated_q_profile(params, r, z, fields["psi_norm"], fields["br"], fields["bz"], fields["b_phi"], fields["inside"])
    p_axis = float(fields["p_axis"])
    p_profile_pa = pressure_profile_1d(params, q_profile_rho)
    density_profile = params.density_axis_1e19_m3 * (
        params.density_edge_fraction + (1.0 - params.density_edge_fraction) * np.maximum(1.0 - q_profile_rho * q_profile_rho, 0.0) ** 0.7
    )
    _, final_shape_stats = shape_constraint_diagnostics(params, machine, r, z, psi_shifted, fields["psi_norm"])
    shape_stats = final_shape_stats if params.shape_constraint else shape_stats

    return EquilibriumState(
        r=r,
        z=z,
        major_r=major_r,
        psi=psi_shifted,
        psi_norm=fields["psi_norm"],
        rho_pol=fields["rho_pol"],
        br=fields["br"],
        bz=fields["bz"],
        b_phi=fields["b_phi"],
        b_total=fields["b_total"],
        density=fields["density"],
        pressure_pa=fields["pressure_pa"],
        temperature_kev=fields["temperature_kev"],
        j_phi=j_phi,
        q_profile_rho=q_profile_rho,
        q_profile=q_profile,
        p_profile_pa=p_profile_pa,
        density_profile=density_profile,
        inside=fields["inside"],
        solver_iterations=iterations,
        solver_residual=change,
        operator_residual=operator_residual,
        pprime=pprime,
        ffprime=ffprime,
        coil_flux=coil_flux,
        magnetic_axis_r_m=axis_r,
        magnetic_axis_z_m=axis_z,
        psi_lcfs=lcfs_value,
        x_point_count=len(x_points),
        primary_x_point_r_m=primary_x[0],
        primary_x_point_z_m=primary_x[1],
        primary_x_point_psi=primary_x[2],
        strike_point_count=len(strike_points),
        primary_strike_point_r_m=primary_strike[0],
        primary_strike_point_z_m=primary_strike[1],
        primary_strike_point_psi_norm=float(sample_grid(fields["psi_norm"], r, z, np.asarray([primary_strike[0]]), np.asarray([primary_strike[1]]))[0]) if strike_points else 0.0,
        separatrix_topology=str(topology["topology"]),
        divertor_balance=float(topology["divertor_balance"]),
        lower_strike_point_count=int(topology["lower_strike_count"]),
        upper_strike_point_count=int(topology["upper_strike_count"]),
        shape_control_rms_error=float(shape_stats.get("rms_error", 0.0)),
        shape_control_max_error=float(shape_stats.get("max_error", 0.0)),
        shape_control_rank=int(shape_stats.get("rank", 0.0)),
        machine=machine,
    )


def build_equilibrium(params: EquilibriumParams) -> EquilibriumState:
    if params.equilibrium_model == "free-boundary-gs":
        return build_free_boundary_gs_equilibrium(params)
    if params.equilibrium_model == "iterative-gs":
        return build_iterative_gs_equilibrium(params)
    return build_analytic_equilibrium(params)


def metrics(state: EquilibriumState, params: EquilibriumParams) -> dict[str, object]:
    inside = state.inside
    dr = float(state.r[1] - state.r[0]) if state.r.size > 1 else 0.0
    dz = float(state.z[1] - state.z[0]) if state.z.size > 1 else 0.0
    machine = state.machine or load_machine_geometry(params)
    axis_i = int(np.argmin(np.abs(state.r - state.magnetic_axis_r_m))) if state.r.size else 0
    axis_j = int(np.argmin(np.abs(state.z - state.magnetic_axis_z_m))) if state.z.size else 0
    total_abs_pf_ma_turn = sum(abs(coil.current_ma * coil.turns) for coil in machine.coils)
    return {
        "equilibrium_model": params.equilibrium_model,
        "machine_device": machine.device,
        "machine_config": params.machine_config,
        "geqdsk_input": params.geqdsk_input,
        "diagnostics_constraint": params.diagnostics_constraint,
        "reconstruction_mode": params.reconstruction_mode,
        "reconstruction_fit_params": params.reconstruction_fit_params,
        "reconstruction_regularization": params.reconstruction_regularization,
        "reconstruction_rms_error": state.reconstruction_rms_error,
        "reconstruction_max_error": state.reconstruction_max_error,
        "reconstruction_chi2_reduced": state.reconstruction_chi2_reduced,
        "reconstruction_constraint_count": state.reconstruction_constraint_count,
        "benchmark_geqdsk": params.benchmark_geqdsk,
        "benchmark_lcfs_rms_m": state.benchmark_lcfs_rms_m,
        "benchmark_q_rms": state.benchmark_q_rms,
        "profile_model": params.profile_model,
        "cocos_index": params.cocos_index,
        "psi_sign": params.psi_sign,
        "ip_sign": params.ip_sign,
        "btor_sign": params.btor_sign,
        "export_formats": params.export_formats,
        "shape_control": params.shape_control,
        "shape_control_rms_error": state.shape_control_rms_error,
        "shape_control_max_error": state.shape_control_max_error,
        "shape_control_rank": state.shape_control_rank,
        "pf_coil_count": len(machine.coils),
        "passive_structure_count": len(machine.passive_structures),
        "limiter_point_count": int(machine.limiter_r.size),
        "wall_point_count": int(machine.wall_r.size),
        "total_abs_pf_current_ma_turn": total_abs_pf_ma_turn,
        "n_r": params.n_r,
        "n_z": params.n_z,
        "minor_radius_m": params.minor_radius_m,
        "elongation": model_adjusted_params(params).elongation,
        "triangularity": model_adjusted_params(params).triangularity,
        "b_axis_t": float(state.b_total[axis_i, axis_j]),
        "b_min_t": float(np.min(state.b_total[inside])) if np.any(inside) else float(np.min(state.b_total)),
        "b_max_t": float(np.max(state.b_total[inside])) if np.any(inside) else float(np.max(state.b_total)),
        "density_axis_1e19_m3": params.density_axis_1e19_m3,
        "density_edge_1e19_m3": float(params.density_axis_1e19_m3 * params.density_edge_fraction),
        "pressure_axis_pa": float(np.max(state.pressure_pa)),
        "beta_percent": params.beta_percent,
        "plasma_current_ma": params.plasma_current_ma,
        "j_phi_peak_ma_m2": float(np.max(state.j_phi) / 1.0e6),
        "closed_surface_fraction": float(np.mean(inside)),
        "closed_surface_area_m2": float(np.sum(inside) * dr * dz),
        "magnetic_axis_minor_r_m": state.magnetic_axis_r_m,
        "magnetic_axis_z_m": state.magnetic_axis_z_m,
        "psi_lcfs": state.psi_lcfs,
        "x_point_count": state.x_point_count,
        "primary_x_point_minor_r_m": state.primary_x_point_r_m,
        "primary_x_point_z_m": state.primary_x_point_z_m,
        "primary_x_point_psi": state.primary_x_point_psi,
        "strike_point_count": state.strike_point_count,
        "primary_strike_point_minor_r_m": state.primary_strike_point_r_m,
        "primary_strike_point_z_m": state.primary_strike_point_z_m,
        "primary_strike_point_psi_norm": state.primary_strike_point_psi_norm,
        "separatrix_topology": state.separatrix_topology,
        "lower_strike_point_count": state.lower_strike_point_count,
        "upper_strike_point_count": state.upper_strike_point_count,
        "divertor_balance": state.divertor_balance,
        "q_axis": float(state.q_profile[0]) if state.q_profile.size else params.q_axis,
        "q_edge": float(state.q_profile[-1]) if state.q_profile.size else params.q_edge,
        "q_input_axis": params.q_axis,
        "q_input_edge": params.q_edge,
        "q_edge_error": abs(float(state.q_profile[-1]) - params.q_edge) if state.q_profile.size else 0.0,
        "solver_iterations": state.solver_iterations,
        "solver_residual": state.solver_residual,
        "solver_converged": int(state.solver_residual <= params.gs_tolerance),
        "gs_operator_residual": state.operator_residual,
        "boundary_relaxation_effective": min(params.boundary_relaxation, 0.005) if params.shape_control in {"isoflux", "inverse", "forward-inverse"} else params.boundary_relaxation,
    }


def map_rows(state: EquilibriumState) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, r_value in enumerate(state.r):
        for j, z_value in enumerate(state.z):
            rows.append(
                [
                    f"{r_value:.10e}",
                    f"{z_value:.10e}",
                    f"{state.major_r[i, j]:.10e}",
                    f"{state.psi_norm[i, j]:.10e}",
                    f"{state.psi[i, j]:.10e}",
                    f"{state.br[i, j]:.10e}",
                    f"{state.bz[i, j]:.10e}",
                    f"{state.b_phi[i, j]:.10e}",
                    f"{state.b_total[i, j]:.10e}",
                    f"{state.density[i, j]:.10e}",
                    f"{state.pressure_pa[i, j]:.10e}",
                    f"{state.temperature_kev[i, j]:.10e}",
                    f"{state.j_phi[i, j]:.10e}",
                    int(state.inside[i, j]),
                ]
            )
    return rows


def q_rows(state: EquilibriumState) -> list[list[object]]:
    return [
        [
            f"{rho:.10e}",
            f"{q:.10e}",
            f"{pressure:.10e}",
            f"{density:.10e}",
        ]
        for rho, q, pressure, density in zip(
            state.q_profile_rho,
            state.q_profile,
            state.p_profile_pa,
            state.density_profile,
        )
    ]


def gs_profile_rows(state: EquilibriumState) -> list[list[object]]:
    rho = state.q_profile_rho
    pprime_profile = np.zeros_like(rho)
    ffprime_profile = np.zeros_like(rho)
    if state.pprime is not None and state.ffprime is not None:
        for idx, target in enumerate(rho):
            band = np.abs(state.rho_pol - target) <= 0.02
            if np.any(band):
                pprime_profile[idx] = float(np.mean(state.pprime[band]))
                ffprime_profile[idx] = float(np.mean(state.ffprime[band]))
    return [
        [
            f"{rho_value:.10e}",
            f"{q:.10e}",
            f"{pressure:.10e}",
            f"{density:.10e}",
            f"{pprime:.10e}",
            f"{ffprime:.10e}",
        ]
        for rho_value, q, pressure, density, pprime, ffprime in zip(
            rho,
            state.q_profile,
            state.p_profile_pa,
            state.density_profile,
            pprime_profile,
            ffprime_profile,
        )
    ]


def machine_geometry_rows(machine: MachineGeometry, params: EquilibriumParams) -> list[list[object]]:
    rows: list[list[object]] = []
    for kind, r_values, z_values in (
        ("limiter", machine.limiter_r, machine.limiter_z),
        ("wall", machine.wall_r, machine.wall_z),
    ):
        for index, (r_minor, z_value) in enumerate(zip(r_values, z_values)):
            rows.append(
                [
                    kind,
                    index,
                    f"{float(r_minor):.10e}",
                    f"{float(params.major_radius_m + r_minor):.10e}",
                    f"{float(z_value):.10e}",
                ]
            )
    return rows


def pf_coil_rows(machine: MachineGeometry, params: EquilibriumParams) -> list[list[object]]:
    rows: list[list[object]] = []
    for index, coil in enumerate(machine.coils):
        rows.append(
            [
                index,
                coil.name,
                f"{float(coil.r_m):.10e}",
                f"{float(params.major_radius_m + coil.r_m):.10e}",
                f"{float(coil.z_m):.10e}",
                f"{float(coil.current_ma):.10e}",
                f"{float(coil.current_ma * 1.0e6):.10e}",
                f"{float(coil.turns):.10e}",
                int(coil.control),
                f"{float(coil.width_m):.10e}",
                f"{float(coil.height_m):.10e}",
                f"{float(coil.resistance_ohm):.10e}",
                f"{float(coil.voltage_v):.10e}",
            ]
        )
    return rows


def passive_structure_rows(machine: MachineGeometry, params: EquilibriumParams) -> list[list[object]]:
    rows: list[list[object]] = []
    for index, item in enumerate(machine.passive_structures):
        tau = item.resistance_ohm
        rows.append(
            [
                index,
                item.name,
                f"{float(item.r_m):.10e}",
                f"{float(params.major_radius_m + item.r_m):.10e}",
                f"{float(item.z_m):.10e}",
                f"{float(item.width_m):.10e}",
                f"{float(item.height_m):.10e}",
                f"{float(item.current_ma):.10e}",
                f"{float(item.current_ma * 1.0e6):.10e}",
                f"{float(item.turns):.10e}",
                f"{float(item.resistance_ohm):.10e}",
                f"{float(tau):.10e}",
            ]
        )
    return rows


def equilibrium_boundary_rows(state: EquilibriumState) -> list[list[object]]:
    rows: list[list[object]] = []
    cr, cz = sorted_lcfs_points(state, max_points=300)
    if cr.size:
        psi_values = sample_grid(state.psi, state.r, state.z, cr, cz)
        psi_norm_values = sample_grid(state.psi_norm, state.r, state.z, cr, cz)
        for rv, zv, psin, psiv in zip(cr, cz, psi_norm_values, psi_values):
            rows.append(
                [
                    "lcfs",
                    f"{float(rv):.10e}",
                    f"{float(state.major_r[0, 0] - state.r[0] + rv):.10e}",
                    f"{float(zv):.10e}",
                    f"{float(psin):.10e}",
                    f"{float(psiv):.10e}",
                ]
            )
    if state.x_point_count:
        rows.append(
            [
                "primary_x_point",
                f"{state.primary_x_point_r_m:.10e}",
                "",
                f"{state.primary_x_point_z_m:.10e}",
                "",
                f"{state.primary_x_point_psi:.10e}",
            ]
        )
    if state.strike_point_count:
        rows.append(
            [
                "primary_strike_point",
                f"{state.primary_strike_point_r_m:.10e}",
                "",
                f"{state.primary_strike_point_z_m:.10e}",
                f"{state.primary_strike_point_psi_norm:.10e}",
                "",
            ]
        )
    return rows


def separatrix_topology_rows(state: EquilibriumState) -> list[list[object]]:
    rows: list[list[object]] = [
        ["topology", state.separatrix_topology, "", "", "classified from X-point and strike-point signs"],
        ["x_point_count", state.x_point_count, "", "", "candidate nulls from |grad psi| minima with saddle Hessian"],
        ["strike_point_count", state.strike_point_count, "", "", "limiter samples closest to psi_norm=1"],
        ["lower_strike_point_count", state.lower_strike_point_count, "", "", "strike candidates below midplane"],
        ["upper_strike_point_count", state.upper_strike_point_count, "", "", "strike candidates above midplane"],
        ["divertor_balance", f"{state.divertor_balance:.10e}", "", "", "(upper-lower)/(upper+lower) strike count proxy"],
    ]
    if state.x_point_count:
        rows.append(["primary_x_point", state.primary_x_point_r_m, state.primary_x_point_z_m, state.primary_x_point_psi, "minor_r_m,z_m,psi_wb"])
    if state.strike_point_count:
        rows.append(["primary_strike_point", state.primary_strike_point_r_m, state.primary_strike_point_z_m, state.primary_strike_point_psi_norm, "minor_r_m,z_m,psi_norm"])
    return rows


def x_point_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    points = estimate_x_points(state.psi, state.r, state.z, max_points=8)
    if not points:
        return []
    dpsi_dr = np.gradient(state.psi, state.r, axis=0, edge_order=2)
    dpsi_dz = np.gradient(state.psi, state.z, axis=1, edge_order=2)
    d2_rr = np.gradient(dpsi_dr, state.r, axis=0, edge_order=2)
    d2_zz = np.gradient(dpsi_dz, state.z, axis=1, edge_order=2)
    d2_rz = np.gradient(dpsi_dr, state.z, axis=1, edge_order=2)
    rows: list[list[object]] = []
    for idx, (rv, zv, psiv) in enumerate(points, 1):
        br = float(sample_grid(state.br, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        bz = float(sample_grid(state.bz, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        psin = float(sample_grid(state.psi_norm, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        rr = int(np.argmin(np.abs(state.r - rv)))
        zz = int(np.argmin(np.abs(state.z - zv)))
        hdet = float(d2_rr[rr, zz] * d2_zz[rr, zz] - d2_rz[rr, zz] * d2_rz[rr, zz])
        rows.append([idx, f"{rv:.10e}", f"{params.major_radius_m + rv:.10e}", f"{zv:.10e}", f"{psiv:.10e}", f"{psin:.10e}", f"{math.hypot(br, bz):.10e}", f"{hdet:.10e}"])
    return rows


def strike_point_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    machine = state.machine or load_machine_geometry(params)
    points = estimate_strike_points(machine, state.psi_norm, state.psi, state.r, state.z, max_points=8)
    rows: list[list[object]] = []
    for idx, (rv, zv, psiv) in enumerate(points, 1):
        psin = float(sample_grid(state.psi_norm, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        rows.append([idx, f"{rv:.10e}", f"{params.major_radius_m + rv:.10e}", f"{zv:.10e}", f"{psiv:.10e}", f"{psin:.10e}", f"{abs(psin - 1.0):.10e}", "lower" if zv < -1.0e-3 else "upper" if zv > 1.0e-3 else "midplane"])
    return rows


def lcfs_topology_diagnostics_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    machine = state.machine or load_machine_geometry(params)
    lcfs_r, lcfs_z = sorted_lcfs_points(state, max_points=1000)
    wall_r, wall_z = resample_polygon(machine.wall_r, machine.wall_z, 6)
    min_wall_gap = 0.0
    if lcfs_r.size and wall_r.size:
        gaps = [float(np.min(np.hypot(wall_r - rv, wall_z - zv))) for rv, zv in zip(lcfs_r, lcfs_z)]
        min_wall_gap = float(np.min(gaps)) if gaps else 0.0
    if lcfs_r.size:
        inner_gap = float(np.min(lcfs_r) - np.min(machine.limiter_r))
        outer_gap = float(np.max(machine.limiter_r) - np.max(lcfs_r))
        top_gap = float(np.max(machine.limiter_z) - np.max(lcfs_z))
        bottom_gap = float(np.min(lcfs_z) - np.min(machine.limiter_z))
    else:
        inner_gap = outer_gap = top_gap = bottom_gap = 0.0
    return [
        ["lcfs_point_count", int(lcfs_r.size), "1", "number of extracted psi_norm=1 contour points"],
        ["topology", state.separatrix_topology, "label", "reduced topology classification"],
        ["primary_x_psi_norm_error", abs(float(sample_grid(state.psi_norm, state.r, state.z, np.asarray([state.primary_x_point_r_m]), np.asarray([state.primary_x_point_z_m]))[0]) - 1.0) if state.x_point_count else 0.0, "1", "proxy only; X point should lie near separatrix in high-fidelity solve"],
        ["primary_strike_psi_norm_error", abs(state.primary_strike_point_psi_norm - 1.0) if state.strike_point_count else 0.0, "1", "strike point proximity to psi_norm=1"],
        ["inner_limiter_gap_m", inner_gap, "m", "limiter inner gap proxy"],
        ["outer_limiter_gap_m", outer_gap, "m", "limiter outer gap proxy"],
        ["top_limiter_gap_m", top_gap, "m", "limiter top gap proxy"],
        ["bottom_limiter_gap_m", bottom_gap, "m", "limiter bottom gap proxy"],
        ["min_wall_gap_m", min_wall_gap, "m", "minimum sampled LCFS-to-wall distance"],
        ["passive_model", "static_prescribed_current_no_induction", "label", "passive structures are static current/geometry inputs, not inductive eddy-current evolution"],
    ]


def shape_constraint_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    machine = state.machine or load_machine_geometry(params)
    rows, _ = shape_constraint_diagnostics(params, machine, state.r, state.z, state.psi, state.psi_norm)
    table: list[list[object]] = []
    for row in rows:
        table.append(
            [
                row["constraint_index"],
                row["kind"],
                f"{float(row['r1_m']):.10e}",
                f"{float(row['z1_m']):.10e}",
                "" if row["r2_m"] == "" else f"{float(row['r2_m']):.10e}",
                "" if row["z2_m"] == "" else f"{float(row['z2_m']):.10e}",
                f"{float(row['weight']):.10e}",
                f"{float(row['psi_delta_wb']):.10e}",
                f"{float(row['psi_norm_delta']):.10e}",
                f"{float(row['weighted_residual_norm']):.10e}",
            ]
        )
    return table


def diagnostic_reconstruction_rows(state: EquilibriumState, params: EquilibriumParams) -> tuple[list[list[object]], dict[str, float]]:
    if not params.diagnostics_constraint:
        return [], {"count": 0.0, "rms": 0.0, "max": 0.0, "chi2": 0.0}
    raw = load_diagnostics_constraints(params.diagnostics_constraint)
    rows: list[list[object]] = []
    residuals: list[float] = []
    weighted: list[float] = []

    def add(kind: str, name: str, predicted: float, target: float, sigma: float, unit: str) -> None:
        residual = predicted - target
        safe_sigma = max(abs(float(sigma)), 1.0e-12)
        normalized = residual / safe_sigma
        residuals.append(float(residual))
        weighted.append(float(normalized))
        rows.append(
            [
                len(rows) + 1,
                kind,
                name,
                f"{predicted:.10e}",
                f"{target:.10e}",
                f"{residual:.10e}",
                f"{safe_sigma:.10e}",
                f"{normalized:.10e}",
                unit,
            ]
        )

    for item in raw.get("flux_loops", []):
        if not isinstance(item, Mapping):
            continue
        rv = float(item.get("r_m", item.get("R", params.major_radius_m))) - params.major_radius_m
        zv = float(item.get("z_m", item.get("Z", 0.0)))
        predicted = float(sample_grid(state.psi, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        target = float(item.get("psi_wb", item.get("target", predicted)))
        add("flux_loop", str(item.get("name", f"flux_loop_{len(rows)+1}")), predicted, target, float(item.get("sigma", item.get("sigma_wb", 1.0e-3))), "Wb")

    for item in raw.get("magnetic_probes", []):
        if not isinstance(item, Mapping):
            continue
        rv = float(item.get("r_m", item.get("R", params.major_radius_m))) - params.major_radius_m
        zv = float(item.get("z_m", item.get("Z", 0.0)))
        component = str(item.get("component", "b_total")).lower()
        field = {"br": state.br, "bz": state.bz, "bphi": state.b_phi, "b_phi": state.b_phi}.get(component, state.b_total)
        predicted = float(sample_grid(field, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        target = float(item.get("value_t", item.get("target", predicted)))
        add("magnetic_probe", str(item.get("name", f"probe_{len(rows)+1}")), predicted, target, float(item.get("sigma", item.get("sigma_t", 1.0e-3))), "T")

    for item in raw.get("q_points", []):
        if not isinstance(item, Mapping):
            continue
        rho = float(item.get("rho", item.get("rho_pol", 1.0)))
        predicted = float(np.interp(np.clip(rho, 0.0, 1.0), state.q_profile_rho, state.q_profile))
        target = float(item.get("q", item.get("target", predicted)))
        add("q_point", str(item.get("name", f"q_{rho:.2f}")), predicted, target, float(item.get("sigma", item.get("sigma_q", 0.05))), "1")

    for item in raw.get("pressure_points", []):
        if not isinstance(item, Mapping):
            continue
        rho = float(item.get("rho", item.get("rho_pol", 0.0)))
        predicted = float(np.interp(np.clip(rho, 0.0, 1.0), state.q_profile_rho, state.p_profile_pa))
        target = float(item.get("pressure_pa", item.get("target", predicted)))
        add("pressure_point", str(item.get("name", f"pressure_{rho:.2f}")), predicted, target, float(item.get("sigma", item.get("sigma_pa", max(0.05 * target, 1.0)))), "Pa")

    for item in raw.get("mse_points", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("r_m") is None and item.get("R") is None:
            rho = float(item.get("rho", item.get("rho_pol", 0.5)))
            rv_major = params.major_radius_m + params.minor_radius_m * np.clip(rho, 0.0, 1.0)
        else:
            rv_major = float(item.get("r_m", item.get("R", params.major_radius_m)))
        rv = rv_major - params.major_radius_m
        zv = float(item.get("z_m", item.get("Z", 0.0)))
        br = float(sample_grid(state.br, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        bz = float(sample_grid(state.bz, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        bphi = float(sample_grid(state.b_phi, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
        bpol = math.hypot(br, bz)
        predicted = math.degrees(math.atan2(bpol, max(abs(bphi), 1.0e-30)))
        target = float(item.get("pitch_angle_deg", item.get("gamma_deg", item.get("target", predicted))))
        add("mse_pitch", str(item.get("name", f"mse_{len(rows)+1}")), predicted, target, float(item.get("sigma", item.get("sigma_deg", 0.5))), "deg")

    for section in ("rogowski_current", "ip_constraints"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            target = item.get("plasma_current_ma", item.get("ip_ma", item.get("target", params.plasma_current_ma)))
            add("rogowski_current", str(item.get("name", f"ip_{len(rows)+1}")), float(params.plasma_current_ma), float(target), float(item.get("sigma", item.get("sigma_ma", 0.05))), "MA")

    for section in ("diamagnetic_loop", "beta_constraints"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            target = item.get("beta_percent", item.get("target", params.beta_percent))
            add("diamagnetic_beta", str(item.get("name", f"beta_{len(rows)+1}")), float(params.beta_percent), float(target), float(item.get("sigma", item.get("sigma_percent", 0.2))), "%")

    for section in ("lcfs_points", "boundary_points", "isoflux_points"):
        for item in raw.get(section, []):
            if not isinstance(item, Mapping):
                continue
            rv = float(item.get("r_m", item.get("R", params.major_radius_m))) - params.major_radius_m
            zv = float(item.get("z_m", item.get("Z", 0.0)))
            predicted = float(sample_grid(state.psi_norm, state.r, state.z, np.asarray([rv]), np.asarray([zv]))[0])
            target = float(item.get("psi_norm", item.get("target", 1.0)))
            add("lcfs_point", str(item.get("name", f"lcfs_{len(rows)+1}")), predicted, target, float(item.get("sigma", item.get("sigma_psi_norm", 0.03))), "1")

    residual_array = np.asarray(residuals, dtype=float)
    weighted_array = np.asarray(weighted, dtype=float)
    stats = {
        "count": float(len(rows)),
        "rms": float(np.sqrt(np.mean(residual_array * residual_array))) if residual_array.size else 0.0,
        "max": float(np.max(np.abs(residual_array))) if residual_array.size else 0.0,
        "chi2": float(np.mean(weighted_array * weighted_array)) if weighted_array.size else 0.0,
    }
    return rows, stats


def diagnostic_reconstruction_by_kind_rows(rows: Sequence[Sequence[object]]) -> list[list[object]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if len(row) < 8:
            continue
        kind = str(row[1])
        try:
            residual = float(row[5])
            normalized = float(row[7])
        except (TypeError, ValueError):
            continue
        grouped.setdefault(kind, []).append((residual, normalized))
    table: list[list[object]] = []
    for kind in sorted(grouped):
        values = np.asarray([item[0] for item in grouped[kind]], dtype=float)
        norm = np.asarray([item[1] for item in grouped[kind]], dtype=float)
        table.append(
            [
                kind,
                int(values.size),
                f"{float(np.sqrt(np.mean(values * values))):.10e}" if values.size else "0.0",
                f"{float(np.max(np.abs(values))):.10e}" if values.size else "0.0",
                f"{float(np.mean(norm * norm)):.10e}" if norm.size else "0.0",
            ]
        )
    return table


def benchmark_rows(state: EquilibriumState, params: EquilibriumParams) -> tuple[list[list[object]], dict[str, float]]:
    if not params.benchmark_geqdsk:
        return [], {"lcfs_rms_m": 0.0, "q_rms": 0.0}
    ref = read_geqdsk(params.benchmark_geqdsk)
    rows: list[list[object]] = []
    q_ref = np.asarray(ref.get("qpsi", []), dtype=float)
    q_rms = 0.0
    if q_ref.size >= 2 and state.q_profile.size >= 2:
        rho = np.linspace(0.0, 1.0, q_ref.size)
        q_here = np.interp(rho, state.q_profile_rho, state.q_profile)
        diff = q_here - q_ref
        q_rms = float(np.sqrt(np.mean(diff * diff)))
        rows.append(["q_profile", "rms", f"{q_rms:.10e}", "1", q_ref.size])
    r_ref = np.asarray(ref.get("rbdry", []), dtype=float)
    z_ref = np.asarray(ref.get("zbdry", []), dtype=float)
    r_here_minor, z_here = sorted_lcfs_points(state, max_points=max(50, int(r_ref.size)))
    lcfs_rms = 0.0
    if r_ref.size >= 3 and r_here_minor.size >= 3:
        count = min(r_ref.size, r_here_minor.size)
        theta = np.linspace(0.0, 1.0, count, endpoint=False)
        ref_theta = np.linspace(0.0, 1.0, r_ref.size, endpoint=False)
        here_theta = np.linspace(0.0, 1.0, r_here_minor.size, endpoint=False)
        r_ref_i = np.interp(theta, ref_theta, r_ref - params.major_radius_m)
        z_ref_i = np.interp(theta, ref_theta, z_ref)
        r_here_i = np.interp(theta, here_theta, r_here_minor)
        z_here_i = np.interp(theta, here_theta, z_here)
        dist2 = (r_here_i - r_ref_i) ** 2 + (z_here_i - z_ref_i) ** 2
        lcfs_rms = float(np.sqrt(np.mean(dist2)))
        rows.append(["lcfs", "rms_distance", f"{lcfs_rms:.10e}", "m", count])
    return rows, {"lcfs_rms_m": lcfs_rms, "q_rms": q_rms}


def cocos_report_rows(params: EquilibriumParams) -> list[list[object]]:
    return [
        ["cocos_index", params.cocos_index, "COCOS-like convention tag used by local export"],
        ["transform_status", "metadata_only", "No external COCOS remap is applied in this reduced exporter"],
        ["psi_sign", params.psi_sign, "Multiplier applied to exported poloidal flux sign metadata"],
        ["ip_sign", params.ip_sign, "Plasma current sign metadata"],
        ["btor_sign", params.btor_sign, "Toroidal field sign metadata"],
        ["q_sign_policy", "q profile left positive; signs are recorded separately", "Reduced convention report"],
        ["geqdsk_output", params.geqdsk_output, "EFIT-style g-file emitted by this run"],
    ]


def export_flux_surface_rows(state: EquilibriumState, params: EquilibriumParams) -> list[list[object]]:
    rows: list[list[object]] = []
    rho_values = np.linspace(0.0, 1.0, 64)
    pprime_profile = _mean_flux_profile(state, state.pprime if state.pprime is not None else np.zeros_like(state.psi), rho_values.size)
    ffprime_profile = _mean_flux_profile(state, state.ffprime if state.ffprime is not None else np.zeros_like(state.psi), rho_values.size)
    fpol_profile = params.major_radius_m * params.b0_t * (1.0 + 0.04 * np.maximum(1.0 - rho_values, 0.0))
    for rho in rho_values:
        idx = len(rows)
        band = np.abs(state.rho_pol - rho) <= 0.025
        if not np.any(band):
            band = np.abs(state.rho_pol - rho) <= 0.05
        pressure = float(np.mean(state.pressure_pa[band])) if np.any(band) else float(np.interp(rho, state.q_profile_rho, state.p_profile_pa))
        density = float(np.mean(state.density[band])) if np.any(band) else float(np.interp(rho, state.q_profile_rho, state.density_profile))
        j_phi = float(np.mean(state.j_phi[band])) if np.any(band) else 0.0
        q_value = float(np.interp(rho, state.q_profile_rho, state.q_profile))
        rows.append(
            [
                f"{float(rho):.10e}",
                f"{float(rho * rho):.10e}",
                f"{pressure:.10e}",
                f"{float(pprime_profile[idx]):.10e}",
                f"{float(ffprime_profile[idx]):.10e}",
                f"{float(fpol_profile[idx]):.10e}",
                f"{density:.10e}",
                f"{j_phi:.10e}",
                f"{q_value:.10e}",
            ]
        )
    return rows


def write_efit_sidecar_files(outdir: Path, state: EquilibriumState, params: EquilibriumParams) -> dict[str, str]:
    machine = state.machine or load_machine_geometry(params)
    summary = metrics(state, params)
    afile = outdir / "equilibrium.afile.csv"
    kfile = outdir / "equilibrium.kfile.csv"
    mfile = outdir / "equilibrium.mfile.csv"
    write_csv(
        afile,
        ["quantity", "value", "unit"],
        [
            ["shot", 0, "1"],
            ["time_ms", 0.0, "ms"],
            ["r_axis_m", params.major_radius_m + state.magnetic_axis_r_m, "m"],
            ["z_axis_m", state.magnetic_axis_z_m, "m"],
            ["b_axis_t", summary["b_axis_t"], "T"],
            ["plasma_current_ma", params.ip_sign * params.plasma_current_ma, "MA"],
            ["beta_percent", params.beta_percent, "%"],
            ["q_axis", summary["q_axis"], "1"],
            ["q_edge", summary["q_edge"], "1"],
            ["separatrix_topology", state.separatrix_topology, "label"],
            ["reconstruction_chi2_reduced", state.reconstruction_chi2_reduced, "1"],
            ["cocos_index", params.cocos_index, "1"],
        ],
    )
    write_csv(
        kfile,
        ["source", "index", "major_r_m", "z_m", "value", "sigma", "unit"],
        [
            ["pf_coil", idx + 1, params.major_radius_m + coil.r_m, coil.z_m, coil.current_ma, "", "MA"]
            for idx, coil in enumerate(machine.coils)
        ]
        + [
            ["passive_structure", idx + 1, params.major_radius_m + item.r_m, item.z_m, item.current_ma, "", "MA"]
            for idx, item in enumerate(machine.passive_structures)
        ],
    )
    lcfs_r, lcfs_z = sorted_lcfs_points(state, max_points=240)
    write_csv(
        mfile,
        ["kind", "index", "major_r_m", "z_m", "psi_norm", "description"],
        [
            ["lcfs", idx + 1, params.major_radius_m + rv, zv, 1.0, "exported separatrix contour"]
            for idx, (rv, zv) in enumerate(zip(lcfs_r, lcfs_z))
        ]
        + [
            ["limiter", idx + 1, params.major_radius_m + rv, zv, "", "machine limiter"]
            for idx, (rv, zv) in enumerate(zip(machine.limiter_r, machine.limiter_z))
        ]
        + [
            ["wall", idx + 1, params.major_radius_m + rv, zv, "", "machine wall"]
            for idx, (rv, zv) in enumerate(zip(machine.wall_r, machine.wall_z))
        ],
    )
    return {"afile": afile.name, "kfile": kfile.name, "mfile": mfile.name}


def write_export_files(outdir: Path, state: EquilibriumState, params: EquilibriumParams) -> None:
    formats = {item.strip().lower() for item in params.export_formats.split(",") if item.strip()}
    rows = export_flux_surface_rows(state, params)
    profile_header = ["rho_pol", "psi_norm", "pressure_pa", "pprime_pa_per_wb", "ffprime_t2m2_per_wb", "fpol_t_m", "density_1e19_m3", "j_phi_a_m2", "q"]
    if "profiles" in formats or "imas" in formats:
        write_csv(outdir / "flux_surface_averages.csv", profile_header, rows)
    if "chease" in formats:
        write_csv(outdir / "chease_profiles.csv", profile_header, rows)
        (outdir / "chease_input_summary.json").write_text(
            json.dumps(
                {
                    "schema": "xirong.chease_bridge.v1",
                    "n_profile": len(rows),
                    "major_radius_m": params.major_radius_m,
                    "b0_t": params.b0_t,
                    "plasma_current_ma": params.plasma_current_ma,
                    "cocos_index": params.cocos_index,
                    "source": "03A reduced Grad-Shafranov equilibrium",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if "helena" in formats:
        write_csv(outdir / "helena_profiles.csv", ["sqrt_psi_norm", "pressure_pa", "dp_dpsi", "q", "fpol_t_m", "ffprime_t2m2_per_wb"], [[row[0], row[2], row[3], row[8], row[5], row[4]] for row in rows])
        lcfs_r, lcfs_z = sorted_lcfs_points(state, max_points=300)
        write_csv(outdir / "helena_boundary.csv", ["index", "major_r_m", "z_m"], [[idx + 1, params.major_radius_m + rv, zv] for idx, (rv, zv) in enumerate(zip(lcfs_r, lcfs_z))])
    efit_sidecars: dict[str, str] = {}
    if {"efit", "afile", "kfile", "mfile"} & formats:
        efit_sidecars = write_efit_sidecar_files(outdir, state, params)
    report = {
        "schema": "xirong.equilibrium_export.v1",
        "cocos_index": params.cocos_index,
        "psi_sign": params.psi_sign,
        "ip_sign": params.ip_sign,
        "btor_sign": params.btor_sign,
        "formats": sorted(formats),
        "files": {
            "geqdsk": params.geqdsk_output,
            "profiles": "flux_surface_averages.csv",
            "chease": "chease_profiles.csv",
            "chease_summary": "chease_input_summary.json",
            "helena": "helena_profiles.csv",
            "helena_boundary": "helena_boundary.csv",
            **efit_sidecars,
        },
    }
    (outdir / "equilibrium_export_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _profile_to_length(x: Array, values: Array, count: int) -> Array:
    xp = np.asarray(x, dtype=float).reshape(-1)
    yp = np.asarray(values, dtype=float).reshape(-1)
    target = np.linspace(0.0, 1.0, count)
    if xp.size < 2 or yp.size < 2:
        return np.zeros(count, dtype=float)
    order = np.argsort(xp)
    xp = xp[order]
    yp = yp[order]
    unique = np.concatenate(([True], np.diff(xp) > 1.0e-9))
    xp = xp[unique]
    yp = yp[unique]
    if xp.size < 2:
        return np.full(count, float(yp[0]) if yp.size else 0.0, dtype=float)
    return np.interp(target, xp, yp)


def _mean_flux_profile(state: EquilibriumState, values: Array, count: int = 0) -> Array:
    target_count = count or state.r.size
    rho = np.sqrt(np.linspace(0.0, 1.0, target_count))
    profile = np.zeros(target_count, dtype=float)
    for idx, rho_value in enumerate(rho):
        band = np.abs(state.rho_pol - rho_value) <= 0.025
        if np.any(band):
            profile[idx] = float(np.mean(np.asarray(values)[band]))
        else:
            profile[idx] = float(np.nan)
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) >= 2:
        profile = np.interp(np.arange(target_count), np.flatnonzero(finite), profile[finite])
    else:
        profile = np.zeros(target_count, dtype=float)
    return profile


def _geqdsk_float(value: float) -> str:
    return f"{float(value):16.9E}"


def _write_geqdsk_values(fp: Any, values: Sequence[float]) -> None:
    for index, value in enumerate(values):
        fp.write(_geqdsk_float(float(value)))
        if (index + 1) % 5 == 0:
            fp.write("\n")
    if len(values) % 5:
            fp.write("\n")


def geqdsk_audit_tables(state: EquilibriumState, params: EquilibriumParams) -> tuple[list[list[object]], list[list[object]]]:
    nw = int(state.r.size)
    nh = int(state.z.size)
    major_axis = params.major_radius_m + state.magnetic_axis_r_m
    r_grid = params.major_radius_m + state.r
    rleft = float(np.min(r_grid))
    rdim = float(np.max(r_grid) - rleft)
    zmid = 0.5 * float(np.min(state.z) + np.max(state.z))
    zdim = float(np.max(state.z) - np.min(state.z))
    simag = float(np.min(state.psi))
    sibry = float(state.psi_lcfs)
    qpsi = _profile_to_length(state.q_profile_rho, state.q_profile, nw)
    pres = _profile_to_length(state.q_profile_rho, state.p_profile_pa, nw)
    pprime = _mean_flux_profile(state, state.pprime if state.pprime is not None else np.zeros_like(state.psi), nw)
    ffprime = _mean_flux_profile(state, state.ffprime if state.ffprime is not None else np.zeros_like(state.psi), nw)
    fpol = params.major_radius_m * params.b0_t * (1.0 + 0.04 * np.maximum(1.0 - np.linspace(0.0, 1.0, nw), 0.0))
    lcfs_r_minor, _ = sorted_lcfs_points(state, max_points=240)
    machine = state.machine or load_machine_geometry(params)
    header = [
        ["nw", nw, "radial grid count"],
        ["nh", nh, "vertical grid count"],
        ["rdim", rdim, "R grid span m"],
        ["zdim", zdim, "Z grid span m"],
        ["rcentr", params.major_radius_m, "reference major radius m"],
        ["rleft", rleft, "left R coordinate m"],
        ["zmid", zmid, "midplane Z m"],
        ["rmagx", major_axis, "magnetic axis R m"],
        ["zmagx", state.magnetic_axis_z_m, "magnetic axis Z m"],
        ["simagx", simag, "axis psi Wb"],
        ["sibdry", sibry, "boundary psi Wb"],
        ["bcentr", params.b0_t, "central toroidal field T"],
        ["cpasma", params.plasma_current_ma * 1.0e6, "plasma current A"],
        ["nbbbs", int(lcfs_r_minor.size), "boundary point count"],
        ["limitr", int(machine.limiter_r.size), "limiter point count"],
    ]
    profile = [
        [idx, f"{rho:.10e}", f"{fpol[idx]:.10e}", f"{pres[idx]:.10e}", f"{ffprime[idx]:.10e}", f"{pprime[idx]:.10e}", f"{qpsi[idx]:.10e}"]
        for idx, rho in enumerate(np.linspace(0.0, 1.0, nw))
    ]
    return header, profile


def write_geqdsk(path: Path, state: EquilibriumState, params: EquilibriumParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nw = int(state.r.size)
    nh = int(state.z.size)
    major_axis = params.major_radius_m + state.magnetic_axis_r_m
    r_grid = params.major_radius_m + state.r
    rleft = float(np.min(r_grid))
    rdim = float(np.max(r_grid) - rleft)
    zmid = 0.5 * float(np.min(state.z) + np.max(state.z))
    zdim = float(np.max(state.z) - np.min(state.z))
    simag = float(np.min(state.psi))
    sibry = float(state.psi_lcfs)
    qpsi = _profile_to_length(state.q_profile_rho, state.q_profile, nw)
    pres = _profile_to_length(state.q_profile_rho, state.p_profile_pa, nw)
    pprime = _mean_flux_profile(state, state.pprime if state.pprime is not None else np.zeros_like(state.psi), nw)
    ffprime = _mean_flux_profile(state, state.ffprime if state.ffprime is not None else np.zeros_like(state.psi), nw)
    fpol = params.major_radius_m * params.b0_t * (1.0 + 0.04 * np.maximum(1.0 - np.linspace(0.0, 1.0, nw), 0.0))
    lcfs_r_minor, lcfs_z = sorted_lcfs_points(state, max_points=240)
    if lcfs_r_minor.size < 3:
        machine = state.machine or load_machine_geometry(params)
        lcfs_r_minor, lcfs_z = machine.limiter_r, machine.limiter_z
    machine = state.machine or load_machine_geometry(params)
    limiter_r, limiter_z = resample_polygon(machine.limiter_r, machine.limiter_z, 1)
    with path.open("w", encoding="ascii", errors="replace") as fp:
        fp.write(f"{'XIRONG 03A EQUILIBRIUM':<48}{0:4d}{nw:4d}{nh:4d}\n")
        _write_geqdsk_values(fp, [rdim, zdim, params.major_radius_m, rleft, zmid])
        _write_geqdsk_values(fp, [major_axis, state.magnetic_axis_z_m, simag, sibry, params.b0_t])
        _write_geqdsk_values(fp, [params.plasma_current_ma * 1.0e6, simag, 0.0, major_axis, 0.0])
        _write_geqdsk_values(fp, [state.magnetic_axis_z_m, 0.0, sibry, 0.0, 0.0])
        _write_geqdsk_values(fp, fpol)
        _write_geqdsk_values(fp, pres)
        _write_geqdsk_values(fp, ffprime)
        _write_geqdsk_values(fp, pprime)
        _write_geqdsk_values(fp, state.psi.reshape(-1))
        _write_geqdsk_values(fp, qpsi)
        fp.write(f"{lcfs_r_minor.size:5d}{limiter_r.size:5d}\n")
        bdry_pairs = np.column_stack((params.major_radius_m + lcfs_r_minor, lcfs_z)).reshape(-1)
        lim_pairs = np.column_stack((params.major_radius_m + limiter_r, limiter_z)).reshape(-1)
        _write_geqdsk_values(fp, bdry_pairs)
        _write_geqdsk_values(fp, lim_pairs)


def save_state(outdir: Path, state: EquilibriumState, params: EquilibriumParams) -> None:
    resonance_hint = np.maximum(state.density / max(float(np.max(state.density)), 1.0e-12), 0.0)
    machine = state.machine or load_machine_geometry(params)
    write_module_imas_state(
        outdir,
        "equilibrium_mhd",
        {
            "equilibrium_state": {
                "model": np.asarray(params.equilibrium_model),
                "machine_device": np.asarray(machine.device),
                "coordinates": np.asarray("r,z"),
                "r": state.r.astype(np.float32),
                "z": state.z.astype(np.float32),
                "major_r": state.major_r.astype(np.float32),
                "psi": state.psi.astype(np.float32),
                "psi_norm": state.psi_norm.astype(np.float32),
                "rho_pol": state.rho_pol.astype(np.float32),
                "br_total": state.br.astype(np.float32),
                "bz_total": state.bz.astype(np.float32),
                "b_phi": state.b_phi.astype(np.float32),
                "b_total": state.b_total.astype(np.float32),
                "density": state.density.astype(np.float32),
                "pressure_pa": state.pressure_pa.astype(np.float32),
                "temperature_kev": state.temperature_kev.astype(np.float32),
                "j_phi": state.j_phi.astype(np.float32),
                "pprime": (state.pprime if state.pprime is not None else np.zeros_like(state.j_phi)).astype(np.float32),
                "ffprime": (state.ffprime if state.ffprime is not None else np.zeros_like(state.j_phi)).astype(np.float32),
                "pf_coil_flux": (state.coil_flux if state.coil_flux is not None else np.zeros_like(state.j_phi)).astype(np.float32),
                "q_profile_rho": state.q_profile_rho.astype(np.float32),
                "q_profile": state.q_profile.astype(np.float32),
                "resonance_hint": resonance_hint.astype(np.float32),
                "b0_t": np.asarray(params.b0_t),
                "major_radius_m": np.asarray(params.major_radius_m),
                "minor_radius_m": np.asarray(params.minor_radius_m),
                "elongation": np.asarray(model_adjusted_params(params).elongation),
                "triangularity": np.asarray(model_adjusted_params(params).triangularity),
                "plasma_current_ma": np.asarray(params.plasma_current_ma),
                "beta_percent": np.asarray(params.beta_percent),
                "pressure_current_fraction": np.asarray(params.pressure_current_fraction),
                "device_machine_state": np.asarray(params.device_machine_state),
                "machine_config": np.asarray(params.machine_config),
                "geqdsk_input": np.asarray(params.geqdsk_input),
                "geqdsk_output": np.asarray(params.geqdsk_output),
                "shape_constraint": np.asarray(params.shape_constraint),
                "diagnostics_constraint": np.asarray(params.diagnostics_constraint),
                "reconstruction_mode": np.asarray(params.reconstruction_mode),
                "reconstruction_fit_params": np.asarray(params.reconstruction_fit_params),
                "reconstruction_regularization": np.asarray(params.reconstruction_regularization),
                "reconstruction_rms_error": np.asarray(state.reconstruction_rms_error),
                "reconstruction_max_error": np.asarray(state.reconstruction_max_error),
                "reconstruction_chi2_reduced": np.asarray(state.reconstruction_chi2_reduced),
                "reconstruction_constraint_count": np.asarray(state.reconstruction_constraint_count),
                "profile_model": np.asarray(params.profile_model),
                "pressure_profile": np.asarray(params.pressure_profile),
                "current_profile": np.asarray(params.current_profile),
                "benchmark_geqdsk": np.asarray(params.benchmark_geqdsk),
                "benchmark_lcfs_rms_m": np.asarray(state.benchmark_lcfs_rms_m),
                "benchmark_q_rms": np.asarray(state.benchmark_q_rms),
                "cocos_index": np.asarray(params.cocos_index),
                "psi_sign": np.asarray(params.psi_sign),
                "ip_sign": np.asarray(params.ip_sign),
                "btor_sign": np.asarray(params.btor_sign),
                "export_formats": np.asarray(params.export_formats),
                "shape_control": np.asarray(params.shape_control),
                "shape_control_rms_error": np.asarray(state.shape_control_rms_error),
                "shape_control_max_error": np.asarray(state.shape_control_max_error),
                "shape_control_rank": np.asarray(state.shape_control_rank),
                "limiter_minor_r_m": machine.limiter_r.astype(np.float32),
                "limiter_major_r_m": (params.major_radius_m + machine.limiter_r).astype(np.float32),
                "limiter_z_m": machine.limiter_z.astype(np.float32),
                "wall_minor_r_m": machine.wall_r.astype(np.float32),
                "wall_major_r_m": (params.major_radius_m + machine.wall_r).astype(np.float32),
                "wall_z_m": machine.wall_z.astype(np.float32),
                "pf_coil_minor_r_m": np.asarray([coil.r_m for coil in machine.coils], dtype=np.float32),
                "pf_coil_major_r_m": np.asarray([params.major_radius_m + coil.r_m for coil in machine.coils], dtype=np.float32),
                "pf_coil_z_m": np.asarray([coil.z_m for coil in machine.coils], dtype=np.float32),
                "pf_coil_current_ma": np.asarray([coil.current_ma for coil in machine.coils], dtype=np.float32),
                "pf_coil_turns": np.asarray([coil.turns for coil in machine.coils], dtype=np.float32),
                "pf_coil_width_m": np.asarray([coil.width_m for coil in machine.coils], dtype=np.float32),
                "pf_coil_height_m": np.asarray([coil.height_m for coil in machine.coils], dtype=np.float32),
                "pf_coil_resistance_ohm": np.asarray([coil.resistance_ohm for coil in machine.coils], dtype=np.float32),
                "pf_coil_voltage_v": np.asarray([coil.voltage_v for coil in machine.coils], dtype=np.float32),
                "passive_minor_r_m": np.asarray([item.r_m for item in machine.passive_structures], dtype=np.float32),
                "passive_major_r_m": np.asarray([params.major_radius_m + item.r_m for item in machine.passive_structures], dtype=np.float32),
                "passive_z_m": np.asarray([item.z_m for item in machine.passive_structures], dtype=np.float32),
                "passive_current_ma": np.asarray([item.current_ma for item in machine.passive_structures], dtype=np.float32),
                "passive_turns": np.asarray([item.turns for item in machine.passive_structures], dtype=np.float32),
                "passive_width_m": np.asarray([item.width_m for item in machine.passive_structures], dtype=np.float32),
                "passive_height_m": np.asarray([item.height_m for item in machine.passive_structures], dtype=np.float32),
                "passive_resistance_ohm": np.asarray([item.resistance_ohm for item in machine.passive_structures], dtype=np.float32),
                "magnetic_axis_minor_r_m": np.asarray(state.magnetic_axis_r_m),
                "magnetic_axis_z_m": np.asarray(state.magnetic_axis_z_m),
                "psi_lcfs": np.asarray(state.psi_lcfs),
                "x_point_count": np.asarray(state.x_point_count),
                "primary_x_point_minor_r_m": np.asarray(state.primary_x_point_r_m),
                "primary_x_point_z_m": np.asarray(state.primary_x_point_z_m),
                "primary_x_point_psi": np.asarray(state.primary_x_point_psi),
                "strike_point_count": np.asarray(state.strike_point_count),
                "primary_strike_point_minor_r_m": np.asarray(state.primary_strike_point_r_m),
                "primary_strike_point_z_m": np.asarray(state.primary_strike_point_z_m),
                "primary_strike_point_psi_norm": np.asarray(state.primary_strike_point_psi_norm),
                "separatrix_topology": np.asarray(state.separatrix_topology),
                "lower_strike_point_count": np.asarray(state.lower_strike_point_count),
                "upper_strike_point_count": np.asarray(state.upper_strike_point_count),
                "divertor_balance": np.asarray(state.divertor_balance),
                "solver_iterations": np.asarray(state.solver_iterations),
                "solver_residual": np.asarray(state.solver_residual),
                "gs_operator_residual": np.asarray(state.operator_residual),
                "gs_tolerance": np.asarray(params.gs_tolerance),
                "free_boundary_extent": np.asarray(params.free_boundary_extent),
                "boundary_update_every": np.asarray(params.boundary_update_every),
                "boundary_relaxation": np.asarray(params.boundary_relaxation),
                "param_pf_coil_current_ma": np.asarray(params.pf_coil_current_ma),
                "param_pf_coil_r_offset_m": np.asarray(params.pf_coil_r_offset_m),
                "param_pf_coil_z_m": np.asarray(params.pf_coil_z_m),
                "source_rate": np.asarray(0.0),
                "energy_source_rate": np.asarray(0.0),
            }
        },
        native_filename="equilibrium_state.nc",
    )


def heatmap(values: Array, width: int = 330, height: int = 250) -> "Image.Image":
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros_like(arr)
    else:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        norm = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (35 + 200 * norm).astype(np.uint8)
    rgb[..., 1] = (75 + 130 * (1.0 - np.abs(norm - 0.55))).astype(np.uint8)
    rgb[..., 2] = (110 + 105 * (1.0 - norm)).astype(np.uint8)
    image = Image.fromarray(np.flipud(np.swapaxes(rgb, 0, 1)), "RGB")
    return image.resize((width, height))


def write_figure(outdir: Path, state: EquilibriumState, params: EquilibriumParams, summary: dict[str, object]) -> None:
    if Image is None or ImageDraw is None:
        return
    canvas = Image.new("RGB", (1180, 760), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default() if ImageFont is not None else None
    draw.text((36, 26), "MHD / Grad-Shafranov equilibrium", fill=(18, 28, 35), font=font)
    panels = [
        (state.psi_norm, "normalized flux psi_N", 36, 82),
        (state.b_total, "B total [T]", 410, 82),
        (state.j_phi / 1.0e6, "j_phi [MA/m2]", 784, 82),
        (state.density, "density [1e19 m-3]", 36, 410),
        (state.pressure_pa / 1.0e3, "pressure [kPa]", 410, 410),
        (state.q_profile[None, :], "q profile", 784, 410),
    ]
    for values, title, x, y in panels:
        draw.text((x, y - 22), title, fill=(52, 67, 76), font=font)
        img = heatmap(values, 320, 230)
        canvas.paste(img, (x, y))
        draw.rectangle((x, y, x + 320, y + 230), outline=(200, 212, 220))
    footer = (
        f"model={params.equilibrium_model}  "
        f"B_axis={float(summary['b_axis_t']):.3g} T  "
        f"Ip={params.plasma_current_ma:.3g} MA  "
        f"q={params.q_axis:.2g}->{params.q_edge:.2g}"
    )
    draw.text((36, 708), footer, fill=(52, 67, 76), font=font)
    canvas.save(outdir / "equilibrium_mhd.png")
    save_2d_projection_figure(
        outdir / "equilibrium_mhd_diagnostics.png",
        state.r,
        state.z,
        [
            {"name": "psi_N", "values": state.psi_norm, "unit": "1"},
            {"name": "|B|", "values": state.b_total, "unit": "T"},
            {"name": "density", "values": state.density, "unit": "1e19 m^-3"},
            {"name": "pressure", "values": state.pressure_pa / 1.0e3, "unit": "kPa"},
        ],
        title="Equilibrium maps and profile projections",
        x_label="minor radius r [m]",
        y_label="Z [m]",
        mid_y_label="mid-plane Z≈0",
        vertical_x_label="magnetic-axis R≈R0",
    )
    save_profile_figure(
        outdir / "equilibrium_flux_profiles.png",
        state.q_profile_rho,
        [
            ("q", state.q_profile),
            ("pressure [kPa]", state.p_profile_pa / 1.0e3),
            ("density [1e19 m^-3]", state.density_profile),
        ],
        title="Flux-surface radial profiles",
        x_label="normalized poloidal flux radius rho_pol",
        y_label="profile value",
    )
    if state.pprime is not None and state.ffprime is not None:
        save_2d_projection_figure(
            outdir / "equilibrium_gs_source_terms.png",
            state.r,
            state.z,
            [
                {"name": "j_phi", "values": state.j_phi / 1.0e6, "unit": "MA/m^2"},
                {"name": "pprime", "values": state.pprime, "unit": "Pa/Wb"},
                {"name": "ffprime", "values": state.ffprime, "unit": "T^2 m^2/Wb"},
                {"name": "PF coil psi", "values": state.coil_flux if state.coil_flux is not None else np.zeros_like(state.psi), "unit": "Wb"},
            ],
            title="Grad-Shafranov source terms",
            x_label="minor radius r [m]",
            y_label="Z [m]",
            mid_y_label="mid-plane Z≈0",
            vertical_x_label="magnetic-axis R≈R0",
        )
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="760" viewBox="0 0 1180 760">',
        '<rect width="1180" height="760" fill="#f8fafc"/>',
        '<text x="36" y="48" font-size="24" font-family="Arial" font-weight="bold" fill="#121c23">MHD / Grad-Shafranov equilibrium</text>',
        f'<text x="36" y="82" font-size="14" font-family="Arial" fill="#53656e">model={params.equilibrium_model}, B_axis={float(summary["b_axis_t"]):.4g} T, Ip={params.plasma_current_ma:.4g} MA</text>',
        '</svg>',
    ]
    (outdir / "equilibrium_mhd.svg").write_text("\n".join(svg))


def run(params: EquilibriumParams, outdir: Path) -> None:
    validate_params(params)
    params = apply_geqdsk_input(params)
    validate_params(params)
    params, reconstruction_parameter_rows, reconstruction_cost_rows = apply_reconstruction_constraints_with_report(params)
    validate_params(params)
    params = apply_device_machine_state(params)
    validate_params(params)
    params = apply_interface_state(params)
    validate_params(params)
    outdir.mkdir(parents=True, exist_ok=True)
    state = build_equilibrium(params)
    diagnostic_rows, diagnostic_stats = diagnostic_reconstruction_rows(state, params)
    benchmark_table, benchmark_stats = benchmark_rows(state, params)
    state = replace(
        state,
        reconstruction_constraint_count=int(diagnostic_stats.get("count", 0.0)),
        reconstruction_rms_error=float(diagnostic_stats.get("rms", 0.0)),
        reconstruction_max_error=float(diagnostic_stats.get("max", 0.0)),
        reconstruction_chi2_reduced=float(diagnostic_stats.get("chi2", 0.0)),
        benchmark_lcfs_rms_m=float(benchmark_stats.get("lcfs_rms_m", 0.0)),
        benchmark_q_rms=float(benchmark_stats.get("q_rms", 0.0)),
    )
    summary = metrics(state, params)
    machine = state.machine or load_machine_geometry(params)
    write_csv(outdir / "model_params.csv", ["parameter", "value"], [[key, value] for key, value in asdict(params).items()])
    write_csv(outdir / "final_summary.csv", ["quantity", "value"], [[key, value] for key, value in summary.items()])
    write_csv(
        outdir / "equilibrium_rz_map.csv",
        [
            "r_m",
            "z_m",
            "major_r_m",
            "psi_norm",
            "psi_wb",
            "br_t",
            "bz_t",
            "bphi_t",
            "b_total_t",
            "density_1e19_m3",
            "pressure_pa",
            "temperature_kev",
            "j_phi_a_m2",
            "inside_closed_surface",
        ],
        map_rows(state),
    )
    write_csv(outdir / "q_profile.csv", ["rho_pol", "q", "pressure_pa", "density_1e19_m3"], q_rows(state))
    write_csv(
        outdir / "gs_profiles.csv",
        ["rho_pol", "q_field_estimate", "pressure_pa", "density_1e19_m3", "pprime_pa_per_wb", "ffprime_t2m2_per_wb"],
        gs_profile_rows(state),
    )
    write_csv(
        outdir / "machine_geometry.csv",
        ["kind", "point_index", "minor_r_m", "major_r_m", "z_m"],
        machine_geometry_rows(machine, params),
    )
    write_csv(
        outdir / "pf_coils.csv",
        ["coil_index", "name", "minor_r_m", "major_r_m", "z_m", "current_ma", "current_a", "turns", "control", "width_m", "height_m", "resistance_ohm", "voltage_v"],
        pf_coil_rows(machine, params),
    )
    write_csv(
        outdir / "passive_structures.csv",
        ["structure_index", "name", "minor_r_m", "major_r_m", "z_m", "width_m", "height_m", "current_ma", "current_a", "turns", "resistance_ohm", "time_constant_proxy_s"],
        passive_structure_rows(machine, params),
    )
    write_csv(
        outdir / "equilibrium_boundary.csv",
        ["kind", "minor_r_m", "major_r_m", "z_m", "psi_norm", "psi_wb"],
        equilibrium_boundary_rows(state),
    )
    write_csv(
        outdir / "separatrix_topology.csv",
        ["quantity", "value", "z_or_aux", "psi_or_aux", "description"],
        separatrix_topology_rows(state),
    )
    write_csv(outdir / "x_points.csv", ["index", "minor_r_m", "major_r_m", "z_m", "psi_wb", "psi_norm", "bpol_t_proxy", "hessian_det"], x_point_rows(state, params))
    write_csv(outdir / "strike_points.csv", ["index", "minor_r_m", "major_r_m", "z_m", "psi_wb", "psi_norm", "psi_norm_error", "branch"], strike_point_rows(state, params))
    write_csv(outdir / "lcfs_topology_diagnostics.csv", ["quantity", "value", "unit", "description"], lcfs_topology_diagnostics_rows(state, params))
    write_csv(
        outdir / "shape_constraints.csv",
        [
            "constraint_index",
            "kind",
            "r1_m",
            "z1_m",
            "r2_m",
            "z2_m",
            "weight",
            "psi_delta_wb",
            "psi_norm_delta",
            "weighted_residual_norm",
        ],
        shape_constraint_rows(state, params),
    )
    write_csv(
        outdir / "shape_control_response.csv",
        ["coil_index", "name", "control", "base_current_ma", "final_current_ma", "delta_current_ma", "limit_ma", "margin_ma", "saturated"],
        shape_control_response_rows(params, machine),
    )
    write_csv(
        outdir / "pf_control_matrix.csv",
        ["constraint_index", "kind", "constraint_label", "coil_name", "coil_index", "weight", "response_wb_per_ma", "requested_delta_wb", "current_residual_wb", "unit"],
        pf_control_matrix_rows(state, params),
    )
    write_csv(
        outdir / "pf_forward_inverse_summary.csv",
        ["quantity", "value", "description"],
        pf_control_summary_rows(state, params),
    )
    write_csv(
        outdir / "diagnostic_reconstruction.csv",
        ["constraint_index", "kind", "name", "predicted", "target", "residual", "sigma", "normalized_residual", "unit"],
        diagnostic_rows,
    )
    write_csv(
        outdir / "diagnostic_reconstruction_by_kind.csv",
        ["kind", "count", "rms_residual", "max_abs_residual", "chi2_reduced"],
        diagnostic_reconstruction_by_kind_rows(diagnostic_rows),
    )
    write_csv(
        outdir / "reconstruction_parameter_updates.csv",
        ["parameter", "before", "delta", "after", "scale", "gain"],
        reconstruction_parameter_rows,
    )
    write_csv(
        outdir / "reconstruction_cost.csv",
        [
            "constraint_index",
            "kind",
            "name",
            "predicted_before",
            "predicted_after_linear",
            "target",
            "residual_before",
            "residual_after_linear",
            "sigma",
            "normalized_before",
            "normalized_after_linear",
            "unit",
            "sensitivities",
        ],
        reconstruction_cost_rows,
    )
    write_csv(outdir / "benchmark_metrics.csv", ["metric", "statistic", "value", "unit", "sample_count"], benchmark_table)
    write_csv(outdir / "equilibrium_cocos_report.csv", ["quantity", "value", "description"], cocos_report_rows(params))
    if params.geqdsk_output:
        geqdsk_path = Path(params.geqdsk_output).expanduser()
        if not geqdsk_path.is_absolute():
            geqdsk_path = outdir / geqdsk_path
        write_geqdsk(geqdsk_path, state, params)
        geqdsk_header, geqdsk_profiles = geqdsk_audit_tables(state, params)
        write_csv(outdir / "geqdsk_header.csv", ["quantity", "value", "description"], geqdsk_header)
        write_csv(outdir / "geqdsk_profiles.csv", ["index", "rho_pol", "fpol_t_m", "pressure_pa", "ffprime_t2m2_per_wb", "pprime_pa_per_wb", "q"], geqdsk_profiles)
    write_export_files(outdir, state, params)
    save_state(outdir, state, params)
    write_figure(outdir, state, params, summary)
    print(f"Output written to: {outdir.resolve()}")
    print(f"Model: {params.equilibrium_model}")
    print(f"B axis: {float(summary['b_axis_t']):.6g} T")
    print(f"Peak j_phi: {float(summary['j_phi_peak_ma_m2']):.6g} MA/m^2")
    print(
        f"Solver: {summary['solver_iterations']} iterations, "
        f"update residual {float(summary['solver_residual']):.6g}, "
        f"operator residual {float(summary['gs_operator_residual']):.6g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduced FreeGS-style Grad-Shafranov equilibrium generator.")
    parser.add_argument("--outdir", default="runs/equilibrium_mhd/grad_shafranov")
    parser.add_argument("--input-mode", choices=["manual", "interface"], default=EquilibriumParams.input_mode)
    parser.add_argument("--interface-state", default=EquilibriumParams.interface_state)
    parser.add_argument("--device-machine-state", default=EquilibriumParams.device_machine_state)
    parser.add_argument("--machine-config", default=EquilibriumParams.machine_config)
    parser.add_argument("--geqdsk-input", default=EquilibriumParams.geqdsk_input)
    parser.add_argument("--geqdsk-output", default=EquilibriumParams.geqdsk_output)
    parser.add_argument("--shape-constraint", default=EquilibriumParams.shape_constraint)
    parser.add_argument("--diagnostics-constraint", default=EquilibriumParams.diagnostics_constraint)
    parser.add_argument("--reconstruction-mode", choices=["off", "weighted", "least-squares"], default=EquilibriumParams.reconstruction_mode)
    parser.add_argument("--reconstruction-gain", type=float, default=EquilibriumParams.reconstruction_gain)
    parser.add_argument("--reconstruction-fit-params", default=EquilibriumParams.reconstruction_fit_params)
    parser.add_argument("--reconstruction-regularization", type=float, default=EquilibriumParams.reconstruction_regularization)
    parser.add_argument("--profile-model", choices=["power-law", "spline"], default=EquilibriumParams.profile_model)
    parser.add_argument("--pressure-profile", default=EquilibriumParams.pressure_profile)
    parser.add_argument("--current-profile", default=EquilibriumParams.current_profile)
    parser.add_argument("--benchmark-geqdsk", default=EquilibriumParams.benchmark_geqdsk)
    parser.add_argument("--cocos-index", type=int, default=EquilibriumParams.cocos_index)
    parser.add_argument("--psi-sign", type=float, choices=[-1.0, 1.0], default=EquilibriumParams.psi_sign)
    parser.add_argument("--ip-sign", type=float, choices=[-1.0, 1.0], default=EquilibriumParams.ip_sign)
    parser.add_argument("--btor-sign", type=float, choices=[-1.0, 1.0], default=EquilibriumParams.btor_sign)
    parser.add_argument("--export-formats", default=EquilibriumParams.export_formats)
    parser.add_argument("--shape-control", choices=["off", "forward", "isoflux", "inverse", "forward-inverse"], default=EquilibriumParams.shape_control)
    parser.add_argument("--shape-control-gain", type=float, default=EquilibriumParams.shape_control_gain)
    parser.add_argument("--shape-control-current-limit-ma", type=float, default=EquilibriumParams.shape_control_current_limit_ma)
    parser.add_argument("--shape-control-damping", type=float, default=EquilibriumParams.shape_control_damping)
    parser.add_argument("--equilibrium-model", choices=["free-boundary-gs", "iterative-gs", "circular-tokamak", "elongated-tokamak", "solovev"], default=EquilibriumParams.equilibrium_model)
    parser.add_argument("--n-r", type=int, default=EquilibriumParams.n_r)
    parser.add_argument("--n-z", type=int, default=EquilibriumParams.n_z)
    parser.add_argument("--major-radius-m", type=float, default=EquilibriumParams.major_radius_m)
    parser.add_argument("--minor-radius-m", type=float, default=EquilibriumParams.minor_radius_m)
    parser.add_argument("--elongation", type=float, default=EquilibriumParams.elongation)
    parser.add_argument("--triangularity", type=float, default=EquilibriumParams.triangularity)
    parser.add_argument("--b0-t", type=float, default=EquilibriumParams.b0_t)
    parser.add_argument("--plasma-current-ma", type=float, default=EquilibriumParams.plasma_current_ma)
    parser.add_argument("--beta-percent", type=float, default=EquilibriumParams.beta_percent)
    parser.add_argument("--q-axis", type=float, default=EquilibriumParams.q_axis)
    parser.add_argument("--q-edge", type=float, default=EquilibriumParams.q_edge)
    parser.add_argument("--pressure-alpha", type=float, default=EquilibriumParams.pressure_alpha)
    parser.add_argument("--current-alpha", type=float, default=EquilibriumParams.current_alpha)
    parser.add_argument("--pressure-current-fraction", type=float, default=EquilibriumParams.pressure_current_fraction)
    parser.add_argument("--density-axis-1e19-m3", type=float, default=EquilibriumParams.density_axis_1e19_m3)
    parser.add_argument("--density-edge-fraction", type=float, default=EquilibriumParams.density_edge_fraction)
    parser.add_argument("--temperature-axis-kev", type=float, default=EquilibriumParams.temperature_axis_kev)
    parser.add_argument("--poloidal-field-fraction", type=float, default=EquilibriumParams.poloidal_field_fraction)
    parser.add_argument("--gs-iterations", type=int, default=EquilibriumParams.gs_iterations)
    parser.add_argument("--gs-relaxation", type=float, default=EquilibriumParams.gs_relaxation)
    parser.add_argument("--gs-tolerance", type=float, default=EquilibriumParams.gs_tolerance)
    parser.add_argument("--free-boundary-extent", type=float, default=EquilibriumParams.free_boundary_extent)
    parser.add_argument("--boundary-update-every", type=int, default=EquilibriumParams.boundary_update_every)
    parser.add_argument("--boundary-relaxation", type=float, default=EquilibriumParams.boundary_relaxation)
    parser.add_argument("--pf-coil-current-ma", type=float, default=EquilibriumParams.pf_coil_current_ma)
    parser.add_argument("--pf-coil-r-offset-m", type=float, default=EquilibriumParams.pf_coil_r_offset_m)
    parser.add_argument("--pf-coil-z-m", type=float, default=EquilibriumParams.pf_coil_z_m)
    parser.add_argument("--pf-coil-turns", type=float, default=EquilibriumParams.pf_coil_turns)
    parser.add_argument("--limiter-points", type=int, default=EquilibriumParams.limiter_points)
    parser.add_argument("--wall-clearance-m", type=float, default=EquilibriumParams.wall_clearance_m)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = EquilibriumParams(
        input_mode=args.input_mode,
        interface_state=args.interface_state,
        device_machine_state=args.device_machine_state,
        machine_config=args.machine_config,
        geqdsk_input=args.geqdsk_input,
        geqdsk_output=args.geqdsk_output,
        shape_constraint=args.shape_constraint,
        diagnostics_constraint=args.diagnostics_constraint,
        reconstruction_mode=args.reconstruction_mode,
        reconstruction_gain=args.reconstruction_gain,
        reconstruction_fit_params=args.reconstruction_fit_params,
        reconstruction_regularization=args.reconstruction_regularization,
        profile_model=args.profile_model,
        pressure_profile=args.pressure_profile,
        current_profile=args.current_profile,
        benchmark_geqdsk=args.benchmark_geqdsk,
        cocos_index=args.cocos_index,
        psi_sign=args.psi_sign,
        ip_sign=args.ip_sign,
        btor_sign=args.btor_sign,
        export_formats=args.export_formats,
        shape_control=args.shape_control,
        shape_control_gain=args.shape_control_gain,
        shape_control_current_limit_ma=args.shape_control_current_limit_ma,
        shape_control_damping=args.shape_control_damping,
        equilibrium_model=args.equilibrium_model,
        n_r=args.n_r,
        n_z=args.n_z,
        major_radius_m=args.major_radius_m,
        minor_radius_m=args.minor_radius_m,
        elongation=args.elongation,
        triangularity=args.triangularity,
        b0_t=args.b0_t,
        plasma_current_ma=args.plasma_current_ma,
        beta_percent=args.beta_percent,
        q_axis=args.q_axis,
        q_edge=args.q_edge,
        pressure_alpha=args.pressure_alpha,
        current_alpha=args.current_alpha,
        pressure_current_fraction=args.pressure_current_fraction,
        density_axis_1e19_m3=args.density_axis_1e19_m3,
        density_edge_fraction=args.density_edge_fraction,
        temperature_axis_kev=args.temperature_axis_kev,
        poloidal_field_fraction=args.poloidal_field_fraction,
        gs_iterations=args.gs_iterations,
        gs_relaxation=args.gs_relaxation,
        gs_tolerance=args.gs_tolerance,
        free_boundary_extent=args.free_boundary_extent,
        boundary_update_every=args.boundary_update_every,
        boundary_relaxation=args.boundary_relaxation,
        pf_coil_current_ma=args.pf_coil_current_ma,
        pf_coil_r_offset_m=args.pf_coil_r_offset_m,
        pf_coil_z_m=args.pf_coil_z_m,
        pf_coil_turns=args.pf_coil_turns,
        limiter_points=args.limiter_points,
        wall_clearance_m=args.wall_clearance_m,
    )
    run(params, Path(args.outdir))


if __name__ == "__main__":
    main()
