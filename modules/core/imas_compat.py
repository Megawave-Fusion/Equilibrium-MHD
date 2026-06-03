#!/usr/bin/env python3
"""ITER IMAS compatibility helpers.

The platform treats IMAS netCDF files as the primary module exchange and
archive format.  Reduced solver arrays are written directly into an IMAS
``workflow`` IDS payload and are also projected into the closest standard IDS
fields when the installed Data Dictionary exposes them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import base64
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zlib

import numpy as np


ADAPTER_VERSION = "0.2"
PAYLOAD_PREFIX = "XIRONG_REDUCED_STATE_V1:"


IMAS_IDS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "data_model": ("summary",),
    "device_machine": ("pf_active", "wall"),
    "equilibrium_mhd": ("equilibrium", "core_profiles"),
    "icrh_antenna": ("ic_antennas", "waves"),
    "field": ("waves",),
    "waves_hcd": ("waves", "core_sources", "distribution_sources"),
    "rf_ql_diffusion": ("distribution_sources", "core_sources"),
    "local_fokker_planck": ("distributions", "core_profiles", "core_sources"),
    "kinetic_fast_ions": ("distributions", "core_profiles", "core_sources"),
    "core_transport": ("core_profiles", "core_transport", "core_sources"),
    "neutral_gas": ("gas_injection", "edge_sources", "core_sources"),
    "sources_fueling": ("nbi", "gas_injection", "pellets", "core_sources"),
    "fusion_reactions": ("core_sources", "distributions"),
    "atomic_impurity_radiation": ("radiation", "core_sources"),
    "turbulence_gyrokinetic": ("core_transport",),
}


IDS_PURPOSE: dict[str, str] = {
    "summary": "shot/run-level overview, scalar KPIs and data availability",
    "equilibrium": "axisymmetric equilibrium, flux geometry and magnetic field",
    "core_profiles": "time-dependent core density, temperature, q and current profiles",
    "core_sources": "heating, current-drive, particle and ash source terms",
    "waves": "RF/full-wave field, absorption and power deposition products",
    "ic_antennas": "ICRH antenna and launcher description",
    "distribution_sources": "quasilinear diffusion/source operators",
    "distributions": "fast-ion or kinetic distribution functions",
    "core_transport": "transport coefficients and 1.5D transport closures",
    "gas_injection": "gas fueling and neutral source description",
    "pellets": "pellet fueling source description",
    "nbi": "neutral beam injection and beam source description",
    "radiation": "radiated power and impurity radiation products",
    "edge_sources": "edge/SOL source terms",
    "pf_active": "active poloidal-field coil geometry and currents",
    "wall": "limiter, first-wall and machine boundary geometry",
}


STATE_IDS_MAP: dict[str, tuple[str, ...]] = {
    "plasma_state_index": ("summary",),
    "device_machine_state": ("pf_active", "wall"),
    "equilibrium_state": ("equilibrium", "core_profiles"),
    "antenna_design_state": ("ic_antennas", "waves"),
    "rf_transmitter_state": ("ic_antennas",),
    "matching_network_state": ("ic_antennas",),
    "icrh_system_state": ("ic_antennas", "waves"),
    "rf_full_wave_state": ("waves",),
    "rf_hcd_state": ("waves", "core_sources"),
    "ql_diffusion_state": ("distribution_sources", "core_sources"),
    "rf_ql_diffusion_state": ("distribution_sources", "core_sources"),
    "fp_distribution_state": ("distributions", "core_profiles", "core_sources"),
    "sympic_particles_final": ("distributions", "core_profiles", "core_sources"),
    "mirror_equilibrium_state": ("equilibrium", "core_profiles", "distributions"),
    "mirror_heated_state_final": ("distributions", "core_sources", "core_profiles"),
    "heating_equilibrium_state": ("equilibrium", "core_profiles", "distributions", "core_sources"),
    "core_transport_state": ("core_profiles", "core_transport", "core_sources"),
    "transport_coeff_state": ("core_transport",),
    "neutral_gas_state": ("gas_injection", "edge_sources", "core_sources"),
    "fueling_source_state": ("nbi", "gas_injection", "pellets", "core_sources"),
    "fusion_reaction_state": ("core_sources", "distributions"),
    "radiation_state": ("radiation", "core_sources"),
}


@dataclass(frozen=True)
class StateProduct:
    path: str
    ids: tuple[str, ...]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class IMASManifest:
    adapter: str
    adapter_version: str
    module_key: str
    native_uri: str
    native_status: str
    ids: tuple[str, ...]
    products: tuple[StateProduct, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = "xirong.imas_manifest.v1"
        return data


def imas_import_status() -> dict[str, str | bool]:
    """Return availability and version details for the official IMAS library."""

    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": str(getattr(imas, "__version__", "unknown")),
    }


def is_imas_reference(value: str | Path) -> bool:
    text = str(value)
    return text.startswith("imas:") or text.endswith(".nc")


def state_product_name(value: str | Path) -> str:
    path = Path(str(value))
    if path.suffix == ".nc":
        return path.stem
    return path.name


def product_ids(path: Path, module_key: str) -> tuple[str, ...]:
    return (
        STATE_IDS_MAP.get(state_product_name(path))
        or IMAS_IDS_BY_MODULE.get(module_key, ("workflow",))
    )


def write_module_imas_state(
    outdir: Path,
    module_key: str,
    states: Mapping[str, Mapping[str, Any]],
    *,
    native_filename: str | None = None,
) -> Path:
    """Write reduced module state directly to an IMAS netCDF file.

    ``states`` maps product names such as ``"equilibrium_state"`` or
    ``"rf_full_wave_state"`` to the arrays/scalars produced by the solver.  No
    intermediate state-cache file is created.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalized = {
        state_product_name(name): {key: np.asarray(value) for key, value in data.items()}
        for name, data in states.items()
    }
    products = tuple(
        StateProduct(
            str(outdir / f"{name}.nc"),
            product_ids(Path(name), module_key),
            tuple(str(key) for key in data.keys()),
        )
        for name, data in normalized.items()
    )
    ids = tuple(dict.fromkeys(ids_name for product in products for ids_name in product.ids))
    if "workflow" not in ids:
        ids = (*ids, "workflow")
    native_name = native_filename or f"imas_{module_key}.nc"
    native_uri = str(outdir / native_name)
    notes = [
        "IMAS .nc is the primary external state file.",
        "Reduced solver arrays are embedded in the workflow IDS payload; standard IDS projections are filled where possible.",
    ]
    if imas_import_status().get("available"):
        try:
            export_state_products_to_imas(normalized, native_uri, module_key)
            status = "written"
        except Exception as exc:  # pragma: no cover - local IMAS version dependent
            status = f"export_failed: {type(exc).__name__}: {exc}"
            notes.append("Native IMAS export failed.")
            try:
                Path(native_uri).unlink(missing_ok=True)
            except Exception:
                pass
    else:
        status = "imas_python_not_installed"
        notes.append("Install imas-python with netCDF support to emit native IMAS files.")

    manifest = IMASManifest(
        adapter="xirong-imas-compat",
        adapter_version=ADAPTER_VERSION,
        module_key=module_key,
        native_uri=native_uri,
        native_status=status,
        ids=ids,
        products=products,
        notes=tuple(notes),
    )
    path = outdir / "imas_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    return path


def export_state_products_to_imas(states: Mapping[str, Mapping[str, Any]], uri: str | Path, module_key: str) -> None:
    """Write in-memory reduced state arrays into an IMAS netCDF/URI."""

    import imas  # type: ignore

    loaded = _loaded_from_state_mapping(states)
    product_ids_by_name = {
        state_product_name(name): product_ids(Path(name), module_key)
        for name in states
    }
    ids_names = tuple(
        dict.fromkeys(
            ids_name
            for ids_tuple in product_ids_by_name.values()
            for ids_name in ids_tuple
        )
    )
    factory = imas.IDSFactory()
    ids_objects = []
    for ids_name in ids_names:
        creator = getattr(factory, ids_name, None)
        if creator is None:
            continue
        ids_obj = creator()
        _set_ids_comment(imas, ids_obj, module_key, ids_name, loaded)
        _fill_ids_from_state(imas, ids_obj, ids_name, loaded)
        ids_objects.append(ids_obj)

    workflow_creator = getattr(factory, "workflow", None)
    if workflow_creator is not None:
        workflow = workflow_creator()
        _set_ids_comment(imas, workflow, module_key, "workflow", loaded)
        _safe_set(workflow, "ids_properties.comment", _encode_state_payload(module_key, states))
        _safe_set(workflow, "time", _time_axis(loaded))
        ids_objects.append(workflow)

    with imas.DBEntry(str(uri), "w") as dbentry:
        for ids_obj in ids_objects:
            dbentry.put(ids_obj)


def load_imas_state(uri: str | Path, state_name: str | None = None) -> dict[str, Any]:
    """Load a reduced module state directly from an IMAS netCDF/URI.

    Files produced by :func:`write_module_imas_state` contain a complete
    reduced-state payload in the ``workflow`` IDS.  External IMAS files without
    that payload are read through a best-effort standard IDS projection.
    """

    text = str(uri)
    if not is_imas_reference(text):
        raise RuntimeError("Module interfaces must be IMAS .nc files or IMAS URIs.")
    products = load_imas_state_products(uri)
    if products:
        if state_name:
            wanted = state_product_name(state_name)
            for name, data in products.items():
                if state_product_name(name) == wanted:
                    return data
        if len(products) == 1:
            return next(iter(products.values()))
        if state_name is None:
            return next(iter(products.values()))
    return read_standard_imas_state(uri, state_name)


def load_imas_state_products(uri: str | Path) -> dict[str, dict[str, Any]]:
    """Return all embedded reduced-state products from an IMAS entry."""

    import imas  # type: ignore

    with imas.DBEntry(str(uri), "r") as dbentry:
        for ids_name in ("workflow", "summary", "equilibrium", "waves", "core_sources", "core_profiles"):
            ids_obj = _db_get_optional(dbentry, ids_name)
            comment = _safe_get(ids_obj, "ids_properties.comment")
            decoded = _decode_state_payload(comment)
            if decoded:
                return decoded
    return {}


def inspect_imas_entry(uri: str | Path, ids_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Inspect an external IMAS URI/netCDF file without changing it."""

    import imas  # type: ignore

    selected = tuple(ids_names or ("equilibrium", "core_profiles", "waves", "core_sources"))
    result: dict[str, Any] = {"uri": str(uri), "ids": {}}
    with imas.DBEntry(str(uri), "r") as dbentry:
        for ids_name in selected:
            try:
                ids_obj = dbentry.get(ids_name)
            except Exception as exc:
                result["ids"][ids_name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
                continue
            result["ids"][ids_name] = {
                "available": True,
                "non_empty": _ids_non_empty_paths(ids_obj),
            }
    return result


def read_standard_imas_state(uri: str | Path, state_name: str | None = None) -> dict[str, Any]:
    """Best-effort reduced array view of a standard external IMAS entry."""

    hint = state_product_name(state_name or "")
    if hint in {"rf_full_wave_state", "rf_hcd_state", "ql_diffusion_state", "rf_ql_diffusion_state"}:
        return read_imas_waves_state(uri)
    if hint in {"core_transport_state", "transport_coeff_state"}:
        data = read_imas_core_profiles_state(uri)
        if data:
            return data
    if hint in {"fp_distribution_state", "sympic_particles_final", "mirror_heated_state_final", "heating_equilibrium_state"}:
        data = read_imas_core_sources_state(uri)
        if data:
            return data
    if hint in {"antenna_design_state", "icrh_system_state"}:
        data = read_imas_antenna_state(uri)
        if data:
            return data
    # Equilibrium/core profiles is the safest default because it is the most
    # common vendor-supplied IMAS entry.
    return read_imas_equilibrium_state(uri)


def read_imas_equilibrium_state(uri: str | Path) -> dict[str, Any]:
    """Read equilibrium/core_profiles IDS into the platform's reduced arrays."""

    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional IMAS install
        raise RuntimeError("读取 IMAS .nc/URI 需要安装官方 IMAS-Python；可运行 pip install -r requirements-imas.txt") from exc

    with imas.DBEntry(str(uri), "r") as dbentry:
        equilibrium = _db_get_optional(dbentry, "equilibrium")
        core_profiles = _db_get_optional(dbentry, "core_profiles")

    rho = _ids_array(equilibrium, "time_slice.0.profiles_1d.rho_tor_norm")
    if rho is None:
        rho = _ids_array(core_profiles, "profiles_1d.0.grid.rho_tor_norm")
    if rho is None or rho.size < 2:
        rho = np.linspace(0.0, 1.0, 64)
    q_profile = _ids_array(equilibrium, "time_slice.0.profiles_1d.q")
    if q_profile is None:
        q_profile = _ids_array(core_profiles, "profiles_1d.0.q")
    if q_profile is None:
        q_profile = 1.0 + 3.0 * rho**2
    q_profile = _interp_like(q_profile, rho)

    pressure = _ids_array(equilibrium, "time_slice.0.profiles_1d.pressure")
    pressure = _interp_like(pressure, rho) if pressure is not None else 4.0e4 * (1.0 - rho**2) ** 1.5
    ne = _ids_array(core_profiles, "profiles_1d.0.electrons.density")
    ne_1e19 = _interp_like(ne / 1.0e19, rho) if ne is not None else 1.0 - 0.75 * rho**1.6
    te = _ids_array(core_profiles, "profiles_1d.0.electrons.temperature")
    te_kev = _interp_like(te / 1.0e3, rho) if te is not None else np.maximum(pressure / 8.0e3, 0.1)

    r = np.linspace(1.3, 2.4, rho.size)
    z = np.linspace(-0.8, 0.8, max(16, min(96, rho.size)))
    rho_2d = np.clip(np.abs((r[:, None] - np.mean(r)) / max(0.5 * (r[-1] - r[0]), 1.0e-12)), 0.0, 1.0)
    density_2d = np.interp(rho_2d[:, 0], rho, ne_1e19)[:, None] * np.ones((1, z.size))
    pressure_2d = np.interp(rho_2d[:, 0], rho, pressure)[:, None] * np.ones((1, z.size))
    temperature_2d = np.interp(rho_2d[:, 0], rho, te_kev)[:, None] * np.ones((1, z.size))
    b0_t = 3.0
    b_total = b0_t * (1.0 + 0.08 * rho_2d**2) * np.ones((1, z.size))
    psi_norm = rho_2d**2
    resonance_hint = np.maximum(density_2d / max(float(np.max(density_2d)), 1.0e-12), 0.0)
    return {
        "model": np.asarray("imas_import_adapter"),
        "coordinates": np.asarray("r,z"),
        "r": r.astype(np.float32),
        "z": z.astype(np.float32),
        "major_r": (r[:, None] * np.ones((1, z.size))).astype(np.float32),
        "psi": psi_norm.astype(np.float32),
        "psi_norm": psi_norm.astype(np.float32),
        "rho_pol": rho_2d.astype(np.float32),
        "br_total": np.zeros_like(b_total, dtype=np.float32),
        "bz_total": np.zeros_like(b_total, dtype=np.float32),
        "b_phi": b_total.astype(np.float32),
        "b_total": b_total.astype(np.float32),
        "density": density_2d.astype(np.float32),
        "pressure_pa": pressure_2d.astype(np.float32),
        "temperature_kev": temperature_2d.astype(np.float32),
        "j_phi": np.zeros_like(b_total, dtype=np.float32),
        "q_profile_rho": rho.astype(np.float32),
        "q_profile": q_profile.astype(np.float32),
        "resonance_hint": resonance_hint.astype(np.float32),
        "b0_t": np.asarray(b0_t),
        "source_imas_uri": np.asarray(str(uri)),
    }


def read_imas_waves_state(uri: str | Path) -> dict[str, Any]:
    """Read waves/core_sources IDS into the reduced RF state arrays."""

    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("读取 IMAS .nc/URI 需要安装官方 IMAS-Python；可运行 pip install -r requirements-imas.txt") from exc

    with imas.DBEntry(str(uri), "r") as dbentry:
        waves = _db_get_optional(dbentry, "waves")
        core_sources = _db_get_optional(dbentry, "core_sources")
    coherent = _first_struct(waves, "coherent_wave")
    p2 = _first_struct(coherent, "profiles_2d")
    fw = _first_struct(coherent, "full_wave")
    r_grid = _ids_array(p2, "grid.r")
    z_grid = _ids_array(p2, "grid.z")
    if r_grid is None or z_grid is None:
        r = np.linspace(0.0, 0.8, 64)
        z = np.linspace(-1.0, 1.0, 96)
    else:
        r = _axis_from_grid(r_grid, 0)
        z = _axis_from_grid(z_grid, 1)
    shape = (r.size, z.size)
    power_density = _ids_array(p2, "power_density")
    if power_density is None:
        power_density = _ids_array(p2, "electrons.power_density_thermal")
    rf_intensity = _shape_or_default(power_density, shape, 0.0) / 1.0e6
    e_parallel = _ids_array(fw, "e_field.parallel")
    e_real = _shape_or_default(e_parallel, shape, 0.0)
    absorption = np.ones(shape, dtype=float)
    if np.nanmax(np.abs(rf_intensity)) > 0.0:
        absorption = np.maximum(rf_intensity, 0.0) / max(float(np.nanmax(rf_intensity)), 1.0e-12)

    rho = _ids_array(core_sources, "source.0.profiles_1d.0.grid.rho_tor_norm")
    if rho is None:
        rho = np.linspace(0.0, 1.0, r.size)
    profile_power = _ids_array(core_sources, "source.0.profiles_1d.0.electrons.energy")
    if profile_power is not None:
        profile_power = np.asarray(profile_power, dtype=float) / 1.0e6
    return {
        "r": r.astype(np.float32),
        "z": z.astype(np.float32),
        "rho": np.asarray(rho, dtype=np.float32),
        "b_total": np.full(shape, 3.0, dtype=np.float32),
        "density": np.ones(shape, dtype=np.float32),
        "antenna_state": np.asarray(str(uri)),
        "rf_power": np.asarray(float(np.nanmean(np.maximum(rf_intensity, 0.0))) if rf_intensity.size else 0.0),
        "rf_frequency_norm": np.asarray(1.0),
        "harmonic": np.asarray(2.0),
        "antenna_r_m": np.asarray(0.12),
        "antenna_width_r_m": np.asarray(0.08),
        "antenna_z_m": np.asarray(0.0),
        "antenna_width_z_m": np.asarray(0.2),
        "resonance_hint": absorption.astype(np.float32),
        "rf_intensity": rf_intensity.astype(np.float32),
        "absorption_kernel": absorption.astype(np.float32),
        "power_deposition_mw_m3": rf_intensity.astype(np.float32),
        "rf_power_mw_m3": _radialize(profile_power, rho).astype(np.float32) if profile_power is not None else np.interp(np.linspace(0, 1, r.size), np.linspace(0, 1, r.size), np.nanmean(rf_intensity, axis=1)).astype(np.float32),
        "e_real": e_real.astype(np.float32),
        "e_imag": np.zeros(shape, dtype=np.float32),
        "source_imas_uri": np.asarray(str(uri)),
    }


def read_imas_core_profiles_state(uri: str | Path) -> dict[str, Any]:
    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("读取 IMAS .nc/URI 需要安装官方 IMAS-Python；可运行 pip install -r requirements-imas.txt") from exc
    with imas.DBEntry(str(uri), "r") as dbentry:
        core_profiles = _db_get_optional(dbentry, "core_profiles")
    rho = _ids_array(core_profiles, "profiles_1d.0.grid.rho_tor_norm")
    if rho is None:
        return {}
    ne = _ids_array(core_profiles, "profiles_1d.0.electrons.density")
    te = _ids_array(core_profiles, "profiles_1d.0.electrons.temperature")
    ti = _ids_array(core_profiles, "profiles_1d.0.t_i_average")
    q = _ids_array(core_profiles, "profiles_1d.0.q")
    j = _ids_array(core_profiles, "profiles_1d.0.j_tor")
    return {
        "rho": np.asarray(rho, dtype=np.float32),
        "ne_history_1e19_m3": np.asarray([_interp_like(ne / 1.0e19, rho) if ne is not None else np.ones_like(rho)], dtype=np.float32),
        "te_history_kev": np.asarray([_interp_like(te / 1.0e3, rho) if te is not None else np.ones_like(rho)], dtype=np.float32),
        "ti_history_kev": np.asarray([_interp_like(ti / 1.0e3, rho) if ti is not None else np.ones_like(rho)], dtype=np.float32),
        "q_profile": _interp_like(q, rho).astype(np.float32) if q is not None else np.ones_like(rho, dtype=np.float32),
        "j_total_a_m2": _interp_like(j, rho).astype(np.float32) if j is not None else np.zeros_like(rho, dtype=np.float32),
        "source_imas_uri": np.asarray(str(uri)),
    }


def read_imas_core_sources_state(uri: str | Path) -> dict[str, Any]:
    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("读取 IMAS .nc/URI 需要安装官方 IMAS-Python；可运行 pip install -r requirements-imas.txt") from exc
    with imas.DBEntry(str(uri), "r") as dbentry:
        core_sources = _db_get_optional(dbentry, "core_sources")
    rho = _ids_array(core_sources, "source.0.profiles_1d.0.grid.rho_tor_norm")
    if rho is None:
        return {}
    power = _ids_array(core_sources, "source.0.profiles_1d.0.electrons.energy")
    power_mw = np.asarray(power, dtype=float) / 1.0e6 if power is not None else np.zeros_like(rho)
    return {
        "rho": np.asarray(rho, dtype=np.float32),
        "rf_power_mw_m3": _interp_like(power_mw, rho).astype(np.float32),
        "power_to_electron_mw_m3": _interp_like(power_mw, rho).astype(np.float32),
        "power_to_ion_mw_m3": np.zeros_like(rho, dtype=np.float32),
        "current_drive_a_m2": np.zeros_like(rho, dtype=np.float32),
        "source_imas_uri": np.asarray(str(uri)),
    }


def read_imas_antenna_state(uri: str | Path) -> dict[str, Any]:
    try:
        import imas  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("读取 IMAS .nc/URI 需要安装官方 IMAS-Python；可运行 pip install -r requirements-imas.txt") from exc
    with imas.DBEntry(str(uri), "r") as dbentry:
        antennas = _db_get_optional(dbentry, "ic_antennas")
    frequency = _ids_array(antennas, "frequency")
    return {
        "frequency_mhz": np.asarray((_scalar(frequency, 55.0) / 1.0e6) if frequency is not None else 55.0),
        "launched_power_mw": np.asarray(6.0),
        "antenna_r_m": np.asarray(0.12),
        "antenna_z_m": np.asarray(0.0),
        "antenna_width_r_m": np.asarray(0.08),
        "antenna_width_z_m": np.asarray(0.2),
        "parallel_wavenumber_m": np.asarray(6.0),
        "source_imas_uri": np.asarray(str(uri)),
    }


def _loaded_from_state_mapping(states: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, state in states.items():
        product_name = state_product_name(name)
        arrays = {key: np.asarray(value) for key, value in state.items()}
        loaded[product_name] = arrays
    return loaded


def _encode_state_payload(module_key: str, states: Mapping[str, Mapping[str, Any]]) -> str:
    payload = {
        "schema": "xirong.reduced_state.v1",
        "adapter_version": ADAPTER_VERSION,
        "module_key": module_key,
        "products": {
            state_product_name(name): {
                "ids": list(product_ids(Path(name), module_key)),
                "arrays": {key: _encode_payload_value(value) for key, value in data.items()},
            }
            for name, data in states.items()
        },
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    packed = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    return PAYLOAD_PREFIX + packed


def _decode_state_payload(comment: Any) -> dict[str, dict[str, Any]]:
    if comment is None:
        return {}
    text = str(comment)
    if PAYLOAD_PREFIX not in text:
        return {}
    encoded = text.split(PAYLOAD_PREFIX, 1)[1].strip()
    try:
        raw = zlib.decompress(base64.b64decode(encoded.encode("ascii")))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    products: dict[str, dict[str, Any]] = {}
    for name, product in payload.get("products", {}).items():
        arrays = product.get("arrays", {})
        products[state_product_name(name)] = {
            key: _decode_payload_value(value) for key, value in arrays.items()
        }
    return products


def _encode_payload_value(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        return {"kind": "json", "value": arr.tolist()}
    arr = np.ascontiguousarray(arr)
    compressed = zlib.compress(arr.tobytes(order="C"), level=9)
    return {
        "kind": "ndarray",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_payload_value(value: Mapping[str, Any]) -> Any:
    if value.get("kind") == "json":
        raw = value.get("value")
        return np.asarray(raw)
    compressed = base64.b64decode(str(value.get("data", "")).encode("ascii"))
    raw = zlib.decompress(compressed)
    dtype = np.dtype(str(value.get("dtype", "float64")))
    shape = tuple(int(v) for v in value.get("shape", []))
    arr = np.frombuffer(raw, dtype=dtype).copy()
    return arr.reshape(shape)


def _db_get_optional(dbentry: Any, ids_name: str) -> Any:
    try:
        return dbentry.get(ids_name)
    except Exception:
        return None


def _ids_array(root: Any, path: str) -> np.ndarray | None:
    if root is None:
        return None
    cur = root
    for part in path.split("."):
        try:
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        except Exception:
            return None
    arr = np.asarray(cur, dtype=float)
    if arr.size == 0:
        return None
    return arr


def _first_struct(root: Any, name: str) -> Any:
    if root is None:
        return None
    try:
        arr = getattr(root, name)
        if arr.size < 1:
            return None
        return arr[0]
    except Exception:
        return None


def _axis_from_grid(grid: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(grid, dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    if arr.ndim == 1:
        return arr
    if axis == 0:
        return np.asarray(arr[:, 0], dtype=float)
    return np.asarray(arr[0, :], dtype=float)


def _shape_or_default(value: np.ndarray | None, shape: tuple[int, int], default: float) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == shape:
        return arr
    if arr.ndim == 1 and arr.size == shape[0]:
        return arr[:, None] * np.ones((1, shape[1]))
    if arr.ndim == 1 and arr.size == shape[1]:
        return np.ones((shape[0], 1)) * arr[None, :]
    return np.full(shape, float(np.nanmean(arr)) if arr.size else default, dtype=float)


def _interp_like(value: np.ndarray, rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    while arr.ndim > 1:
        arr = np.nanmean(arr, axis=-1)
    if arr.size == rho.size:
        return arr
    return np.interp(rho, np.linspace(0.0, 1.0, arr.size), arr)


def _set_ids_comment(imas: Any, ids_obj: Any, module_key: str, ids_name: str, loaded: dict[str, dict[str, Any]]) -> None:
    try:
        ids_obj.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    except Exception:
        pass
    try:
        ids_obj.ids_properties.comment = (
            f"Generated by Xirong Zhaobo module {module_key}; mapped to IMAS IDS {ids_name}; "
            f"source reduced states: {', '.join(sorted(loaded)) or 'none'}."
        )
    except Exception:
        pass
    _safe_set(ids_obj, "time", _time_axis(loaded))


def _fill_ids_from_state(imas: Any, ids_obj: Any, ids_name: str, loaded: dict[str, dict[str, Any]]) -> None:
    if ids_name == "core_profiles":
        _fill_core_profiles(ids_obj, loaded)
    elif ids_name == "equilibrium":
        _fill_equilibrium(ids_obj, loaded)
    elif ids_name == "core_transport":
        _fill_core_transport(ids_obj, loaded)
    elif ids_name == "waves":
        _fill_waves(ids_obj, loaded)
    elif ids_name == "core_sources":
        _fill_core_sources(ids_obj, loaded)
    elif ids_name == "distributions":
        _fill_distributions(ids_obj, loaded)
    elif ids_name == "distribution_sources":
        _fill_distribution_sources(ids_obj, loaded)
    elif ids_name == "ic_antennas":
        _fill_ic_antennas(ids_obj, loaded)


def _fill_core_profiles(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    rho = _first_array(loaded, "rho", "q_profile_rho")
    if rho is None:
        rho = _linspace_from(_first_array(loaded, "q_profile", "j_total_a_m2", "ne_history_1e19_m3"))
    if rho is None:
        return
    ne = _profile_history(loaded, "ne_history_1e19_m3", factor=1.0e19)
    te = _profile_history(loaded, "te_history_kev", factor=1.0e3)
    ti = _profile_history(loaded, "ti_history_kev", factor=1.0e3)
    q = _first_array(loaded, "q_profile")
    j = _first_array(loaded, "j_total_a_m2", "current_drive_a_m2", "j_phi")
    time = _time_axis(loaded)
    profiles = _safe_get(ids_obj, "profiles_1d")
    if profiles is None or not _safe_resize(profiles, len(time)):
        return
    for i, profile in enumerate(profiles):
        _safe_set(profile, "grid.rho_tor_norm", rho)
        _safe_set(profile, "q", q)
        _safe_set(profile, "electrons.density", _history_slice(ne, i))
        _safe_set(profile, "electrons.temperature", _history_slice(te, i))
        _safe_set(profile, "t_i_average", _history_slice(ti, i))
        _safe_set(profile, "j_tor", _radialize(j, rho))


def _fill_equilibrium(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    eq = loaded.get("equilibrium_state")
    if not eq:
        return
    time = _time_axis(loaded)
    rho = np.asarray(eq.get("q_profile_rho", []), dtype=float)
    psi_1d = rho * rho if rho.size else None
    slices = _safe_get(ids_obj, "time_slice")
    if slices is None or not _safe_resize(slices, len(time)):
        return
    for item in slices:
        _safe_set(item, "profiles_1d.psi", psi_1d)
        _safe_set(item, "profiles_1d.psi_norm", psi_1d)
        _safe_set(item, "profiles_1d.rho_tor_norm", rho)
        _safe_set(item, "profiles_1d.q", eq.get("q_profile"))
        _safe_set(item, "profiles_1d.pressure", _radialize(eq.get("pressure_pa"), eq.get("q_profile_rho")))
        _safe_set(item, "profiles_1d.j_tor", _radialize(eq.get("j_phi"), eq.get("q_profile_rho")))


def _fill_core_transport(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    rho = _first_array(loaded, "rho")
    if rho is None:
        return
    time = _time_axis(loaded)
    model = _safe_get(ids_obj, "model")
    _safe_set(ids_obj, "time", time)
    if model is not None:
        _safe_set(model, "description", "Xirong reduced 1.5D transport coefficients")


def _fill_waves(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    _safe_set(ids_obj, "identifier.name", "xirong_rf_wave_state")
    _safe_set(ids_obj, "identifier.description", "Reduced RF field, absorption and HCD state from Xirong platform")
    data = loaded.get("rf_full_wave_state") or loaded.get("rf_hcd_state") or {}
    r = data.get("r")
    z = data.get("z")
    if r is None or z is None:
        return
    rr, zz = np.meshgrid(np.asarray(r, dtype=float), np.asarray(z, dtype=float), indexing="ij")
    coherent = _safe_get(ids_obj, "coherent_wave")
    if coherent is None or not _safe_resize(coherent, 1):
        return
    wave = coherent[0]
    _safe_set(wave, "identifier.name", "xirong_reduced_rf")
    _safe_set(wave, "identifier.description", "Reduced RF full-wave/HCD state")
    global_q = _safe_get(wave, "global_quantities")
    if global_q is not None and _safe_resize(global_q, 1):
        _safe_set(global_q[0], "frequency", _scalar(data.get("frequency_hz"), _scalar(data.get("rf_frequency_norm"), 1.0)))
        _safe_set(global_q[0], "power", _scalar(data.get("rf_power"), _scalar(data.get("launched_power_mw"), 0.0)) * 1.0e6)
    profiles_2d = _safe_get(wave, "profiles_2d")
    if profiles_2d is not None and _safe_resize(profiles_2d, 1):
        p2 = profiles_2d[0]
        _safe_set(p2, "grid.r", rr)
        _safe_set(p2, "grid.z", zz)
        power = data.get("power_deposition_mw_m3")
        if power is None and "rf_intensity" in data:
            power = np.asarray(data["rf_intensity"], dtype=float) * np.asarray(data.get("absorption_kernel", 1.0), dtype=float)
        if power is not None:
            _safe_set(p2, "power_density", np.asarray(power, dtype=float) * 1.0e6)
            _safe_set(p2, "electrons.power_density_thermal", np.asarray(power, dtype=float) * 1.0e6)
    full_wave = _safe_get(wave, "full_wave")
    if full_wave is not None and _safe_resize(full_wave, 1):
        fw = full_wave[0]
        _safe_set(fw, "e_field.parallel", data.get("e_real"))
        _safe_set(fw, "grid.path", "profiles_2d[0]/grid")


def _fill_core_sources(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    _safe_set(ids_obj, "identifier.name", "xirong_core_sources")
    _safe_set(ids_obj, "identifier.description", "Reduced heating/current/particle source profiles from Xirong platform")
    data = loaded.get("rf_hcd_state") or loaded.get("core_transport_state") or loaded.get("fp_distribution_state") or {}
    rho = data.get("rho")
    source = _safe_get(ids_obj, "source")
    if rho is None or source is None or not _safe_resize(source, 1):
        return
    item = source[0]
    _safe_set(item, "identifier.name", "xirong_reduced_source")
    profiles = _safe_get(item, "profiles_1d")
    time = _time_axis(loaded)
    if profiles is not None and _safe_resize(profiles, len(time)):
        for i, p1 in enumerate(profiles):
            _safe_set(p1, "time", float(time[i]))
            _safe_set(p1, "grid.rho_tor_norm", rho)
            power = data.get("rf_power_mw_m3")
            if power is None:
                power = data.get("power_to_electron_mw_m3")
            if power is None:
                power = data.get("power_deposition_profile")
            if power is not None:
                _safe_set(p1, "electrons.energy", _radialize(_history_slice(np.asarray(power, dtype=float) * 1.0e6, i), rho))

def _fill_distributions(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    _safe_set(ids_obj, "identifier.name", "xirong_fast_ion_distribution")
    _safe_set(ids_obj, "identifier.description", "Reduced fast-ion distribution f(rho,E,xi)")


def _fill_distribution_sources(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    _safe_set(ids_obj, "identifier.name", "xirong_ql_diffusion")
    _safe_set(ids_obj, "identifier.description", "Reduced RF quasilinear diffusion tensor source")


def _fill_ic_antennas(ids_obj: Any, loaded: dict[str, dict[str, Any]]) -> None:
    antenna = loaded.get("antenna_design_state", {})
    _safe_set(ids_obj, "identifier.name", "xirong_icrh_antenna")
    _safe_set(ids_obj, "identifier.description", "ICRH transmitter, matching and antenna design state")
    _safe_set(ids_obj, "frequency", _scalar(antenna.get("frequency_mhz"), 0.0) * 1.0e6)


def _safe_get(root: Any, dotted_path: str) -> Any:
    cur = root
    for part in dotted_path.split("."):
        try:
            cur = getattr(cur, part)
        except Exception:
            return None
    return cur


def _safe_set(root: Any, dotted_path: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        value = np.asarray(value, dtype=float)
    parent_path, _, leaf = dotted_path.rpartition(".")
    parent = _safe_get(root, parent_path) if parent_path else root
    if parent is None:
        return False
    try:
        setattr(parent, leaf, value)
        return True
    except Exception:
        return False


def _safe_resize(value: Any, size: int) -> bool:
    try:
        value.resize(size)
        return True
    except Exception:
        return False


def _time_axis(loaded: dict[str, dict[str, Any]]) -> np.ndarray:
    arr = _first_array(loaded, "time_s")
    if arr is None or arr.ndim == 0:
        return np.asarray([0.0], dtype=float)
    return np.asarray(arr, dtype=float)


def _first_array(loaded: dict[str, dict[str, Any]], *keys: str) -> np.ndarray | None:
    for data in loaded.values():
        for key in keys:
            if key in data:
                return np.asarray(data[key])
    return None


def _profile_history(loaded: dict[str, dict[str, Any]], key: str, *, factor: float) -> np.ndarray | None:
    arr = _first_array(loaded, key)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float) * factor


def _history_slice(value: np.ndarray | None, index: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim >= 2:
        return arr[min(index, arr.shape[0] - 1)]
    return arr


def _radialize(value: Any, rho: Any) -> np.ndarray | None:
    if value is None or rho is None:
        return None
    arr = np.asarray(value, dtype=float)
    target = np.asarray(rho, dtype=float)
    if arr.ndim == 0:
        return np.full_like(target, float(arr))
    while arr.ndim > 1:
        arr = np.nanmean(arr, axis=-1)
    if arr.size == target.size:
        return arr
    return np.interp(np.linspace(0.0, 1.0, target.size), np.linspace(0.0, 1.0, arr.size), arr)


def _linspace_from(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    n = arr.shape[-1] if arr.ndim > 1 else arr.size
    if n <= 0:
        return None
    return np.linspace(0.0, 1.0, n)


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def _ids_non_empty_paths(ids_obj: Any) -> list[str]:
    try:
        iterator = ids_obj.iter_nonempty_(accept_lazy=True)
    except Exception:
        return []
    paths: list[str] = []
    for node in iterator:
        meta = getattr(node, "metadata", None)
        path = getattr(meta, "path_string", None) or getattr(meta, "path", None)
        if path:
            paths.append(str(path))
        elif len(paths) < 64:
            paths.append(str(node))
    return paths[:256]
