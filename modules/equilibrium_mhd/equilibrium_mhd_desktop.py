#!/usr/bin/env python3
"""Standalone desktop launcher for the Grad-Shafranov equilibrium module."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.equilibrium_mhd.module_spec import MODULE_SPEC


DEFAULTS = {str(key): str(value) for key, value in MODULE_SPEC["defaults"].items()}
PARAM_LABELS = {str(key): str(value) for key, value in MODULE_SPEC.get("param_labels", {}).items()}
PARAM_HELP = {str(key): str(value) for key, value in MODULE_SPEC.get("param_help", {}).items()}
PARAM_CHOICES = {str(key): tuple(str(item) for item in value) for key, value in MODULE_SPEC.get("param_choices", {}).items()}
OUTPUTS = {str(key): str(value) for key, value in MODULE_SPEC.get("outputs", {}).items()}
INTERFACE_FILES = tuple(str(key) for key in MODULE_SPEC.get("interface_files", ()))
REQUIRED_INTERFACE_FILES = tuple(str(key) for key in MODULE_SPEC.get("required_interface_files", ()))
INTERFACE_MODE_VISIBLE_PARAMS = tuple(
    str(key)
    for key in MODULE_SPEC.get("interface_mode_visible_params", ("input_mode", *INTERFACE_FILES))
)

FIELD_GROUPS = (
    (
        "输入来源",
        (
            "input_mode",
            "interface_state",
            "device_machine_state",
            "machine_config",
            "geqdsk_input",
            "shape_constraint",
            "diagnostics_constraint",
            "pressure_profile",
            "current_profile",
            "benchmark_geqdsk",
        ),
    ),
    ("网格", ("n_r", "n_z")),
    (
        "物理参数",
        (
            "equilibrium_model",
            "major_radius_m",
            "minor_radius_m",
            "elongation",
            "triangularity",
            "b0_t",
            "plasma_current_ma",
            "beta_percent",
            "q_axis",
            "q_edge",
            "pressure_alpha",
            "current_alpha",
            "pressure_current_fraction",
            "profile_model",
            "density_axis_1e19_m3",
            "density_edge_fraction",
            "temperature_axis_kev",
            "poloidal_field_fraction",
        ),
    ),
    (
        "求解器 / 边界",
        (
            "gs_iterations",
            "gs_relaxation",
            "gs_tolerance",
            "free_boundary_extent",
            "boundary_update_every",
            "boundary_relaxation",
            "reconstruction_mode",
            "reconstruction_gain",
            "reconstruction_fit_params",
            "reconstruction_regularization",
            "shape_control",
            "shape_control_gain",
            "shape_control_current_limit_ma",
            "shape_control_damping",
            "geqdsk_output",
            "cocos_index",
            "psi_sign",
            "ip_sign",
            "btor_sign",
            "export_formats",
            "pf_coil_current_ma",
            "pf_coil_r_offset_m",
            "pf_coil_z_m",
            "pf_coil_turns",
            "limiter_points",
            "wall_clearance_m",
        ),
    ),
)


def visible_field_groups(values: dict[str, str] | None = None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    values = values or DEFAULTS
    if values.get("input_mode") != "interface":
        groups: list[tuple[str, tuple[str, ...]]] = []
        for title, fields in FIELD_GROUPS:
            if title == "输入来源":
                groups.append((title, ("input_mode",)))
            else:
                groups.append((title, fields))
        return tuple(groups)
    allowed = set(INTERFACE_MODE_VISIBLE_PARAMS)
    groups: list[tuple[str, tuple[str, ...]]] = []
    for title, fields in FIELD_GROUPS:
        visible_fields = tuple(name for name in INTERFACE_MODE_VISIBLE_PARAMS if name in allowed and name in fields)
        if visible_fields:
            groups.append((title, visible_fields))
    return tuple(groups)

SUMMARY_ROWS = (
    ("equilibrium_model", "模型"),
    ("solver_converged", "收敛"),
    ("solver_iterations", "迭代步数"),
    ("solver_residual", "更新残差"),
    ("gs_operator_residual", "GS 算子残差"),
    ("boundary_relaxation_effective", "有效边界松弛"),
    ("b_axis_t", "轴上磁场 T"),
    ("plasma_current_ma", "等离子体电流 MA"),
    ("j_phi_peak_ma_m2", "峰值环向电流 MA/m2"),
    ("q_axis", "q0"),
    ("q_edge", "qa"),
    ("q_edge_error", "qa 误差"),
    ("pressure_axis_pa", "轴上压强 Pa"),
    ("closed_surface_fraction", "闭合磁面占比"),
    ("magnetic_axis_minor_r_m", "磁轴小半径 m"),
    ("magnetic_axis_z_m", "磁轴 Z m"),
    ("x_point_count", "X 点候选数"),
    ("strike_point_count", "strike 点候选数"),
    ("separatrix_topology", "分离面拓扑"),
    ("divertor_balance", "上下 strike 平衡"),
    ("reconstruction_mode", "重建模式"),
    ("reconstruction_constraint_count", "诊断约束数"),
    ("reconstruction_rms_error", "诊断 RMS 残差"),
    ("reconstruction_chi2_reduced", "诊断 reduced chi2"),
    ("benchmark_lcfs_rms_m", "Benchmark LCFS RMS m"),
    ("benchmark_q_rms", "Benchmark q RMS"),
    ("profile_model", "剖面模型"),
    ("cocos_index", "COCOS"),
    ("shape_control", "PF 形状控制"),
    ("shape_control_rms_error", "形状 RMS 残差"),
    ("shape_control_max_error", "形状最大残差"),
    ("pf_coil_count", "PF 线圈数"),
    ("passive_structure_count", "被动结构数"),
    ("total_abs_pf_current_ma_turn", "PF 总安匝 MA-turn"),
)


def resolve_path(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_command(values: dict[str, str], outdir: Path, python_runtime: str | None = None) -> list[str]:
    runtime = python_runtime or os.environ.get("XIRONG_PYTHON", sys.executable).strip('"')
    script = resolve_path(str(MODULE_SPEC["script"]))
    command = [runtime, str(script), "--outdir", str(outdir)]
    for flag, name in MODULE_SPEC["args"]:
        key = str(name)
        if key in values:
            command += [str(flag), values[key]]
    return command


class EquilibriumMHDApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GS 平衡模块启动界面")
        self.geometry("1220x780")
        self.minsize(1080, 700)
        self.vars: dict[str, tk.StringVar] = {}
        self.mode_var = tk.StringVar(value=DEFAULTS.get("input_mode", "manual"))
        self.value_cache = dict(DEFAULTS)
        self._rendering_controls = False
        self._pending_render_after_id: str | None = None
        self.outdir_var = tk.StringVar(value=str(resolve_path(str(MODULE_SPEC["outdir"]))))
        self.status_var = tk.StringVar(value="就绪")
        self.last_outdir: Path | None = None
        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Arial", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Arial", 11, "bold"))
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="GS / Grad-Shafranov 平衡模块", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status_var).pack(side=tk.RIGHT)

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        self._build_controls(left)
        self._build_results(right)

    def _build_controls(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, width=500, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_content = content
        self._build_mode_control(content)
        self.dynamic_controls_content = ttk.Frame(content)
        self.dynamic_controls_content.pack(fill=tk.X)
        self._render_controls_content()

    def _build_mode_control(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="输入来源", style="Section.TLabelframe", padding=10)
        group.pack(fill=tk.X, pady=(0, 10))
        group.columnconfigure(1, weight=1)
        ttk.Label(group, text=PARAM_LABELS.get("input_mode", "input_mode")).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        choices = PARAM_CHOICES.get("input_mode", ("manual", "interface"))
        widget = ttk.Combobox(group, textvariable=self.mode_var, values=choices, state="readonly")
        widget.grid(row=0, column=1, sticky="ew", pady=3)
        widget.bind("<<ComboboxSelected>>", lambda _event: self._on_input_mode_change())
        self.mode_var.trace_add("write", lambda *_args: self._on_input_mode_change())
        help_text = PARAM_HELP.get("input_mode", "")
        if help_text:
            ttk.Label(group, text=help_text, wraplength=340).grid(row=1, column=1, sticky="ew", pady=(0, 5))

    def _current_values(self) -> dict[str, str]:
        values = dict(DEFAULTS)
        values.update(self.value_cache)
        try:
            values["input_mode"] = self.mode_var.get().strip()
        except tk.TclError:
            pass
        for name, var in self.vars.items():
            try:
                values[name] = var.get().strip()
            except tk.TclError:
                continue
        return values

    def _cache_current_values(self) -> None:
        try:
            self.value_cache["input_mode"] = self.mode_var.get().strip()
        except tk.TclError:
            pass
        for name, var in list(self.vars.items()):
            try:
                self.value_cache[name] = var.get().strip()
            except tk.TclError:
                continue

    def _render_controls_content(self) -> None:
        if self._rendering_controls:
            return
        self._pending_render_after_id = None
        self._rendering_controls = True
        values = self._current_values()
        for child in self.dynamic_controls_content.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass
        self.vars.clear()
        try:
            for title, fields in visible_field_groups(values):
                fields = tuple(name for name in fields if name != "input_mode")
                if not fields:
                    continue
                group = ttk.LabelFrame(self.dynamic_controls_content, text=title, style="Section.TLabelframe", padding=10)
                group.pack(fill=tk.X, pady=(0, 10))
                group.columnconfigure(1, weight=1)
                for row, name in enumerate(fields):
                    self._add_param_row(group, row, name, values.get(name, DEFAULTS.get(name, "")))

            out = ttk.LabelFrame(self.dynamic_controls_content, text="输出目录", style="Section.TLabelframe", padding=10)
            out.pack(fill=tk.X, pady=(0, 10))
            out.columnconfigure(0, weight=1)
            ttk.Entry(out, textvariable=self.outdir_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
            ttk.Button(out, text="选择", command=self._choose_outdir).grid(row=0, column=1)

            buttons = ttk.Frame(self.dynamic_controls_content)
            buttons.pack(fill=tk.X, pady=(0, 10))
            self.run_button = ttk.Button(buttons, text="启动 GS 平衡", style="Accent.TButton", command=self._start_run)
            self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            ttk.Button(buttons, text="打开输出", command=self._open_output_dir).pack(side=tk.LEFT)
            ttk.Button(buttons, text="使用说明", command=self._open_manual).pack(side=tk.LEFT, padx=(8, 0))
        finally:
            self._rendering_controls = False

    def _on_input_mode_change(self) -> None:
        if self._rendering_controls:
            return
        self._cache_current_values()
        if self._pending_render_after_id is not None:
            try:
                self.after_cancel(self._pending_render_after_id)
            except tk.TclError:
                pass
        self._pending_render_after_id = self.after(150, self._render_controls_content)

    def _add_param_row(self, parent: ttk.Frame, row: int, name: str, value: str) -> None:
        label = PARAM_LABELS.get(name, name)
        ttk.Label(parent, text=label).grid(row=row * 2, column=0, sticky="w", padx=(0, 8), pady=3)
        var = tk.StringVar(value=value)
        if name == "input_mode":
            var.trace_add("write", lambda *_args: self._on_input_mode_change())
        self.vars[name] = var
        if name in PARAM_CHOICES:
            widget = ttk.Combobox(parent, textvariable=var, values=PARAM_CHOICES[name], state="readonly")
        else:
            widget = ttk.Entry(parent, textvariable=var)
        widget.grid(row=row * 2, column=1, sticky="ew", pady=3)
        if name in INTERFACE_FILES:
            ttk.Button(parent, text="选择", command=lambda key=name: self._choose_file(key)).grid(row=row * 2, column=2, padx=(8, 0))
        help_text = PARAM_HELP.get(name, "")
        if help_text:
            ttk.Label(parent, text=help_text, wraplength=340).grid(row=row * 2 + 1, column=1, columnspan=2, sticky="ew", pady=(0, 5))

    def _build_results(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        contract = ttk.LabelFrame(parent, text="本模块输入 / 输出", style="Section.TLabelframe", padding=10)
        contract.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        contract.columnconfigure(0, weight=1)
        self.contract = ttk.Treeview(contract, columns=("kind", "name", "path"), show="headings", height=7)
        self.contract.heading("kind", text="类型")
        self.contract.heading("name", text="名称")
        self.contract.heading("path", text="文件 / 说明")
        self.contract.column("kind", width=80, anchor="w")
        self.contract.column("name", width=180, anchor="w")
        self.contract.column("path", width=430, anchor="w")
        self.contract.grid(row=0, column=0, sticky="nsew")
        self._render_contract()

        summary_frame = ttk.LabelFrame(parent, text="运行摘要", style="Section.TLabelframe", padding=10)
        summary_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        summary_frame.columnconfigure(0, weight=1)
        self.summary = ttk.Treeview(summary_frame, columns=("metric", "value"), show="headings", height=11)
        self.summary.heading("metric", text="指标")
        self.summary.heading("value", text="数值")
        self.summary.column("metric", width=220, anchor="w")
        self.summary.column("value", width=300, anchor="w")
        self.summary.grid(row=0, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(parent, text="运行记录", style="Section.TLabelframe", padding=10)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap=tk.WORD, height=12)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)
        self._append_log("GS 平衡独立启动界面已就绪。")

    def _render_contract(self) -> None:
        for item in self.contract.get_children():
            self.contract.delete(item)
        inputs = MODULE_SPEC.get("inputs", {})
        if isinstance(inputs, dict):
            for key, info in inputs.items():
                description = ""
                if isinstance(info, dict):
                    description = str(info.get("description", ""))
                    default = info.get("default_path")
                    if default:
                        description = f"{description} 默认：{default}"
                self.contract.insert("", tk.END, values=("输入", key, description))
        for key, filename in OUTPUTS.items():
            self.contract.insert("", tk.END, values=("输出", key, filename))

    def _choose_outdir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(ROOT / "runs"))
        if path:
            self.outdir_var.set(path)

    def _choose_file(self, key: str) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(ROOT / "runs"),
            filetypes=(("Equilibrium / data files", "*.nc *.csv *.json *.geqdsk *.gfile g*"), ("All files", "*.*")),
        )
        if path:
            self.vars[key].set(path)

    def _collect_values(self) -> dict[str, str]:
        self._cache_current_values()
        values = self._current_values()
        visible_names = [name for _, fields in visible_field_groups(values) for name in fields]
        return {name: values.get(name, DEFAULTS.get(name, "")) for name in visible_names}

    def _validate_values(self, values: dict[str, str]) -> bool:
        if values.get("input_mode") == "interface":
            for key in REQUIRED_INTERFACE_FILES:
                if not values.get(key):
                    messagebox.showwarning("缺少输入文件", f"interface 模式需要选择：{PARAM_LABELS.get(key, key)}")
                    return False
        for key in INTERFACE_FILES:
            raw = values.get(key, "")
            if not raw:
                continue
            path = resolve_path(raw)
            if not path.exists():
                messagebox.showwarning("输入文件不存在", f"{PARAM_LABELS.get(key, key)} 不存在：\n{path}")
                return False
        return True

    def _start_run(self) -> None:
        values = self._collect_values()
        if not self._validate_values(values):
            return
        outdir = resolve_path(self.outdir_var.get())
        outdir.mkdir(parents=True, exist_ok=True)
        self.last_outdir = outdir
        command = build_command(values, outdir)
        self._clear_summary()
        self.log.delete("1.0", tk.END)
        self._append_log("运行命令：")
        self._append_log(" ".join(command))
        self.status_var.set("计算中")
        self.run_button.configure(state=tk.DISABLED)
        thread = threading.Thread(target=self._worker, args=(command, outdir), daemon=True)
        thread.start()

    def _worker(self, command: list[str], outdir: Path) -> None:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.after(0, self._append_log, line.rstrip("\n"))
            code = proc.wait()
        except Exception as exc:
            self.after(0, self._run_failed, str(exc))
            return
        if code == 0:
            self.after(0, self._run_finished, outdir)
        else:
            self.after(0, self._run_failed, f"进程退出码：{code}")

    def _run_finished(self, outdir: Path) -> None:
        self.status_var.set("完成")
        self.run_button.configure(state=tk.NORMAL)
        self._load_summary(outdir / str(MODULE_SPEC.get("summary", "final_summary.csv")))
        self._append_log("计算完成。")

    def _run_failed(self, message: str) -> None:
        self.status_var.set("失败")
        self.run_button.configure(state=tk.NORMAL)
        self._append_log(message)
        messagebox.showerror("运行失败", message)

    def _load_summary(self, path: Path) -> None:
        self._clear_summary()
        if not path.exists():
            self._append_log(f"未找到摘要文件：{path}")
            return
        data: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    data[row[0]] = row[1]
        for key, label in SUMMARY_ROWS:
            if key in data:
                self.summary.insert("", tk.END, values=(label, self._format_value(data[key])))

    def _clear_summary(self) -> None:
        for item in self.summary.get_children():
            self.summary.delete(item)

    def _open_output_dir(self) -> None:
        path = self.last_outdir or resolve_path(self.outdir_var.get())
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _open_manual(self) -> None:
        path = Path(__file__).resolve().with_name(f"{MODULE_SPEC.get('key', 'module')}_manual.pdf")
        if not path.exists():
            messagebox.showwarning("找不到说明", f"未找到 PDF 使用说明：\n{path}")
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _append_log(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def _format_value(self, value: str) -> str:
        if value in {"0", "1"}:
            return "是" if value == "1" else "否"
        try:
            number = float(value)
        except ValueError:
            return value
        return f"{number:.6g}"


def main() -> None:
    app = EquilibriumMHDApp()
    app.mainloop()


if __name__ == "__main__":
    main()
