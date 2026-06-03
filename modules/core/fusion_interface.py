#!/usr/bin/env python3
"""Common component interface for the Xirong Zhaobo fusion platform.

The design follows the same broad idea used by integrated modelling systems:
each physics code is a replaceable component with declared inputs, outputs,
configuration keys, and a small init/step/finalize lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionPort:
    name: str
    description: str
    units: str = ""


@dataclass(frozen=True)
class FusionModuleSpec:
    key: str
    group: str
    title: str
    status: str
    directory: str
    purpose: str
    inputs: tuple[FusionPort, ...] = ()
    outputs: tuple[FusionPort, ...] = ()
    config_keys: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()
    downstream: tuple[str, ...] = ()
    reference_codes: tuple[str, ...] = ()


@dataclass
class FusionRunResult:
    ok: bool
    message: str
    outdir: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)


class FusionComponent:
    """Base class for platform components."""

    spec: FusionModuleSpec

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, state: dict[str, Any], outdir: Path) -> FusionRunResult:
        return FusionRunResult(True, f"{self.spec.title}: initialized", outdir)

    def step(self, state: dict[str, Any], time_s: float, outdir: Path) -> FusionRunResult:
        return FusionRunResult(False, f"{self.spec.title}: interface only, solver not connected", outdir)

    def finalize(self, state: dict[str, Any], outdir: Path) -> FusionRunResult:
        return FusionRunResult(True, f"{self.spec.title}: finalized", outdir)


class PlaceholderComponent(FusionComponent):
    def __init__(self, spec: FusionModuleSpec, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.spec = spec


def port(name: str, description: str, units: str = "") -> FusionPort:
    return FusionPort(name=name, description=description, units=units)


FUSION_MODULE_SPECS: tuple[FusionModuleSpec, ...] = (
    FusionModuleSpec(
        key="device_machine",
        group="装置与数据底座",
        title="装置几何 / 线圈 / 网格",
        status="接口预留",
        directory="modules/device_machine",
        purpose="描述磁体、真空室、第一壁、限制器、加热端口、材料区域和计算网格。",
        inputs=(port("machine_config", "装置参数、线圈电流、壁面 CAD/STL/轮廓"),),
        outputs=(port("geometry", "R-Z/3D 几何与边界"), port("mesh", "场、粒子、输运网格")),
        config_keys=("device", "major_radius_m", "minor_radius_m", "coil_set", "wall_file"),
        downstream=("equilibrium_mhd", "wall_pwi", "diagnostics_synthetic"),
        reference_codes=("OMFIT COILS", "GRIDGEN", "IMAS machine description"),
    ),
    FusionModuleSpec(
        key="data_model",
        group="装置与数据底座",
        title="统一 Plasma State / IMAS 风格数据模型",
        status="已有可运行工程原型",
        directory="modules/data_model",
        purpose="统一实验数据、模拟数据、时间片、剖面、场、分布函数和文件交换。",
        inputs=(port("raw_data", "实验或模拟原始数据"),),
        outputs=(port("plasma_state", "设备无关的统一等离子体状态"),),
        config_keys=("shot", "time_slice_s", "state_format", "imas_mapping"),
        downstream=("workflow_coupling", "visualization", "validation_uq"),
        reference_codes=("IMAS IDS", "OMFIT trees", "TRANSP plasma state"),
    ),
    FusionModuleSpec(
        key="equilibrium_mhd",
        group="平衡与稳定",
        title="MHD / Grad-Shafranov 平衡",
        status="已有可运行原型",
        directory="modules/equilibrium_mhd",
        purpose="求解或重建轴对称 MHD 平衡，输出磁通、q 剖面、磁场、密度和压强几何量。",
        inputs=(port("profiles", "p, FF', j_phi 等剖面"), port("coils", "外部线圈和边界条件")),
        outputs=(port("equilibrium_state", "标准 IMAS 平衡态：psi, q, density, pressure"), port("fields", "B_R, B_phi, B_Z, |B|", "T")),
        config_keys=("equilibrium_model", "n_r", "n_z", "major_radius_m", "minor_radius_m", "elongation", "triangularity", "b0_t", "plasma_current_ma", "beta_percent"),
        upstream=("device_machine", "core_transport"),
        downstream=("mhd_stability", "core_transport", "field", "waves_hcd", "kinetic_fast_ions"),
        reference_codes=("EFIT", "CHEASE", "HELENA", "NICE"),
    ),
    FusionModuleSpec(
        key="mhd_stability",
        group="平衡与稳定",
        title="MHD 稳定性 / 响应 / 破裂",
        status="接口预留",
        directory="modules/mhd_stability",
        purpose="分析理想/电阻 MHD 稳定性、Alfven 模、NTM、ELM、VDE 和破裂响应。",
        inputs=(port("equilibrium", "MHD 平衡"), port("profiles", "压力、电流、快离子压强")),
        outputs=(port("stability_metrics", "增长率、模结构、稳定裕度"),),
        config_keys=("mode_numbers", "resistivity", "rotation", "wall_model"),
        upstream=("equilibrium_mhd", "kinetic_fast_ions"),
        downstream=("control_actuators", "validation_uq"),
        reference_codes=("ELITE", "GATO", "MARS", "M3D-C1", "JOREK"),
    ),
    FusionModuleSpec(
        key="core_transport",
        group="输运与边缘",
        title="07F 1.5D Core Transport / 剖面演化",
        status="已有可运行原型",
        directory="modules/core_transport",
        purpose="第 7 步 07F：汇总平衡、源项、辐射和输运系数，演化 ne、Te、Ti、phi 和 Wth。",
        inputs=(port("equilibrium", "03A 磁面几何与初始 phi/ne/Te"), port("sources", "RF/HCD、FP/CD、NBI、alpha、燃料源和辐射"), port("transport_coefficients", "07E chi_e/chi_i/D/V 闭合")),
        outputs=(port("profiles", "n_e, T_e, T_i, phi, q"), port("stored_energy", "Wth(t)", "MJ"), port("transport_state", "1.5D 输运状态")),
        config_keys=("input_mode", "interface_state", "hcd_state", "fp_state", "neutral_state", "fueling_state", "fusion_state", "radiation_state", "transport_coeff_state", "current_drive_model", "transport_model", "closure_model", "boundary_condition"),
        upstream=("equilibrium_mhd", "waves_hcd", "local_fokker_planck", "neutral_gas", "sources_fueling", "fusion_reactions", "atomic_impurity_radiation", "turbulence_gyrokinetic"),
        downstream=("equilibrium_mhd", "pedestal_edge_sol", "diagnostics_synthetic"),
        reference_codes=("ASTRA", "TGYRO", "ONETWO", "FASTRAN", "ETS"),
    ),
    FusionModuleSpec(
        key="pedestal_edge_sol",
        group="输运与边缘",
        title="Pedestal / Edge / SOL / Divertor",
        status="接口预留",
        directory="modules/pedestal_edge_sol",
        purpose="计算台基、刮削层、偏滤器热流、脱靶和边界条件。",
        inputs=(port("edge_geometry", "分离面、X 点、靶板"), port("core_boundary", "核心边界通量")),
        outputs=(port("pedestal", "台基高度/宽度"), port("divertor_loads", "靶板热流", "MW/m^2")),
        config_keys=("pedestal_model", "sol_model", "recycling", "detachment_model"),
        upstream=("equilibrium_mhd", "core_transport", "neutral_gas"),
        downstream=("wall_pwi", "diagnostics_synthetic", "control_actuators"),
        reference_codes=("EPED", "SOLPS", "UEDGE", "BOUT++", "COGENT"),
    ),
    FusionModuleSpec(
        key="turbulence_gyrokinetic",
        group="输运与边缘",
        title="07E 湍流 / 新经典输运系数闭合",
        status="已有可运行原型",
        directory="modules/turbulence_gyrokinetic",
        purpose="以剖面梯度、q/磁剪切和降阶湍流/新经典模型生成 chi_e、chi_i、D 和 pinch，作为 07F 核心输运闭合项。",
        inputs=(port("profile_state", "07F 上一轮 ne/Te/Ti/q 剖面"), port("equilibrium_state", "03A 平衡态 q/几何")),
        outputs=(port("transport_coeff_state", "chi_e, chi_i, D, Vpinch"),),
        config_keys=("input_mode", "profile_state", "equilibrium_state", "coefficient_model", "critical_gradient", "stiffness", "neoclassical_fraction"),
        upstream=("equilibrium_mhd", "core_transport"),
        downstream=("core_transport", "pedestal_edge_sol"),
        reference_codes=("GYRO", "CGYRO", "GENE", "TGLF", "COGENT", "NEO"),
    ),
    FusionModuleSpec(
        key="waves_hcd",
        group="低秩张量模拟方法 / 轨道跟踪",
        title="RF / HCD 波加热与电流驱动",
        status="已有可运行原型",
        directory="modules/waves_hcd",
        purpose="第 6 步 06B：读取 04B full-wave 场，输出 RF/HCD 功率沉积、电流驱动剖面和低秩可直接读取的 QL 扩散状态。",
        inputs=(port("rf_full_wave_state", "04B full-wave RF 场、吸收核和功率沉积代理"),),
        outputs=(port("power_deposition", "功率沉积分布", "MW/m^3"), port("current_drive", "驱动电流", "A/m^2"), port("rf_hcd_state", "RF/HCD 场和剖面状态"), port("ql_diffusion_state", "准线性扩散系数 D_EE/D_xixi/D_Exi")),
        config_keys=("wave_model", "field_solver", "frequency_hz", "launched_power_mw", "n_parallel", "antenna_geometry"),
        upstream=("field",),
        downstream=("core_transport", "rf_ql_diffusion", "low_rank_tensor"),
        reference_codes=("TORIC", "AORSA", "GENRAY", "CQL3D", "TRAVIS"),
    ),
    FusionModuleSpec(
        key="rf_ql_diffusion",
        group="低秩张量模拟方法 / 轨道跟踪",
        title="可选 QL 精细化 / RF 准线性扩散",
        status="已有可运行原型",
        directory="modules/rf_ql_diffusion",
        purpose="把 06B RF/HCD 场接口重新采样为更细的准线性扩散张量与源项；主流程可直接读取 06B 生成的 ql_diffusion_state。",
        inputs=(port("rf_hcd_state", "06B RF/HCD 场、吸收核和功率沉积状态"),),
        outputs=(port("ql_diffusion_state", "D_EE, D_xixi, D_Exi, rf_source"), port("power_deposition_profile", "径向功率沉积剖面", "MW/m^3")),
        config_keys=("input_mode", "interface_state", "n_rho", "n_energy", "n_pitch", "ql_strength"),
        upstream=("waves_hcd",),
        downstream=("low_rank_tensor",),
        reference_codes=("CQL3D", "TORIC QL operators", "AORSA/GENRAY coupling"),
    ),
    FusionModuleSpec(
        key="local_fokker_planck",
        group="低秩张量模拟方法 / 轨道跟踪",
        title="06G Local Fokker-Planck / 碰撞 / QL 扩散 / 电流驱动",
        status="已有可运行原型",
        directory="modules/kinetic_fast_ions",
        purpose="读取 06B QL 扩散状态，在局域 f(rho,E,xi) 上推进碰撞慢化、俯仰角散射、RF 准线性扩散和损失，输出电流驱动、快离子压强和电子/离子功率分配。",
        inputs=(port("ql_diffusion_state", "06B D_EE/D_xixi/D_Exi 与 RF source"),),
        outputs=(port("fp_distribution_state", "f(rho,E,xi)、Jcd、Pe/Pi、快离子压强和损失功率"),),
        config_keys=("input_mode", "ql_state", "fp_model", "cd_model", "cd_efficiency_ka_mw", "slowing_rate", "pitch_scatter", "energy_diffusion"),
        upstream=("waves_hcd", "rf_ql_diffusion"),
        downstream=("core_transport", "kinetic_fast_ions", "diagnostics_synthetic"),
        reference_codes=("CQL3D", "LUKE", "Fokker-Planck RF current-drive closures"),
    ),
    FusionModuleSpec(
        key="icrh_antenna",
        group="ICRH / RF 波场 / full-wave",
        title="ICRH 高功率发射与天线系统设计",
        status="已有原型",
        directory="modules/icrh_antenna",
        purpose="面向 ICRH 高功率发射器产品设计，在一个模块中计算发射机功率链、传输线、匹配网络、天线耦合器、S11/VSWR、端口电压电流和热负荷。",
        inputs=(port("machine_geometry", "天线端口、第一壁和等离子体边界几何"), port("rf_requirement", "频率、谐波、功率、阻抗、保护阈值和运行占空比")),
        outputs=(port("rf_transmitter_state", "发射机功率、效率、线损和保护状态"), port("matching_network_state", "匹配网络元件和传输线参数"), port("antenna_design_state", "给 full-wave 读取的天线设计 IMAS 接口"), port("icrh_system_state", "发射机-匹配-天线集成系统指标")),
        config_keys=("design_scope", "transmitter_architecture", "antenna_type", "frequency_mhz", "harmonic", "launched_power_mw", "target_impedance_ohm", "matching_topology"),
        upstream=("device_machine", "systems_design"),
        downstream=("field", "wall_pwi", "thermal_power_conversion"),
        reference_codes=("TOPICA", "TORIC antenna models", "AORSA", "ICRH test stand workflow"),
    ),
    FusionModuleSpec(
        key="field",
        group="ICRH / RF 波场 / full-wave",
        title="RF full-wave 场求解器",
        status="已有可运行原型",
        directory="modules/field",
        purpose="提供二维标量 cold-plasma full-wave RF 场强求解器，固定读取平衡态接口和天线设计接口，并把波场交给 05A PIC 和 06B RF/HCD 模块。",
        inputs=(port("equilibrium_state", "第 3 步平衡态接口：R-Z 网格、总磁场和密度"), port("antenna_state", "04A 天线设计接口：频率、功率、源位置、宽度和相位")),
        outputs=(port("rf_intensity", "RF 强度分布"), port("absorption_kernel", "吸收核"), port("power_deposition_proxy", "功率沉积代理"), port("field_residual", "频域迭代残差")),
        config_keys=("input_mode", "interface_state", "antenna_state", "rf_power", "rf_frequency_norm", "k0_norm", "density_coupling", "absorption_strength", "profile_model"),
        upstream=("icrh_antenna", "equilibrium_mhd", "data_model"),
        downstream=("waves_hcd", "low_rank_tensor", "sympic_pic", "validation_uq"),
        reference_codes=("TORIC", "AORSA", "cold-plasma full-wave prototypes"),
    ),
    FusionModuleSpec(
        key="sources_fueling",
        group="源汇与边界",
        title="07B NBI / 燃料 / 粒子源",
        status="已有可运行原型",
        directory="modules/sources_fueling",
        purpose="计算中性束沉积、燃料气、丸注入、粒子源、动量源和 NBI 电流驱动源，作为 07F 的外部源项。",
        inputs=(port("neutral_gas_state", "07A 中性气体离化源"), port("beam_geometry", "NBI 几何与能量组")),
        outputs=(port("fueling_source_state", "粒子源、NBI 功率、电流驱动和动量源"),),
        config_keys=("input_mode", "neutral_state", "source_model", "nbi_power_mw", "beam_energy_kev", "fueling_rate_1e20_s", "pellet_rate_1e20_s"),
        upstream=("device_machine", "neutral_gas"),
        downstream=("core_transport", "kinetic_fast_ions", "fusion_reactions"),
        reference_codes=("NUBEAM", "FREYA", "RABBIT"),
    ),
    FusionModuleSpec(
        key="kinetic_fast_ions",
        group="PIC 模拟方法 / 轨道跟踪",
        title="快离子 / Fokker-Planck / 轨道跟踪",
        status="已有可运行原型",
        directory="modules/kinetic_fast_ions",
        purpose="推进快离子 Fokker-Planck 分布函数，计算慢化、俯仰角散射、径向输运、轨道损失和导心/磁镜轨道。",
        inputs=(port("equilibrium", "磁场与磁镜比"), port("sources", "ICRH/NBI/alpha 源"), port("collisions", "慢化、散射和扩散系数")),
        outputs=(port("fast_distribution", "f(rho,E,xi)"), port("radial_moments", "径向快离子库存、平均能量和压强矩"), port("orbit_traces", "样本导心/磁镜轨道"), port("fast_pressure", "快离子压强矩", "a.u.")),
        config_keys=("source_model", "phase_space", "slowing_rate", "pitch_scatter", "radial_diffusion", "orbit_model"),
        upstream=("waves_hcd", "sources_fueling", "fusion_reactions"),
        downstream=("mhd_stability", "diagnostics_synthetic", "core_transport"),
        reference_codes=("ASCOT", "NUBEAM", "CQL3D", "ORBIT", "SPIRAL"),
    ),
    FusionModuleSpec(
        key="fusion_reactions",
        group="源汇与边界",
        title="07C 聚变反应 / Alpha 加热 / 中子源",
        status="已有可运行原型",
        directory="modules/fusion_reactions",
        purpose="计算 DD/DT/DHe3 反应率、alpha 加热、中子源、灰分、DD 副产物和反应功率径向剖面。",
        inputs=(port("thermal_profiles", "n_D, n_T, n_He3, T_i"), port("fast_distribution", "快离子增强或 alpha 沉积剖面")),
        outputs=(port("fusion_power", "聚变功率", "MW"), port("neutron_source", "中子源", "1/s/m^3"), port("alpha_heating", "alpha 出生/沉积加热", "MW/m^3"), port("ash_source", "He4 灰分源", "1/s/m^3")),
        config_keys=("reaction_set", "reactivity_fit", "alpha_model", "ash_model", "profile_state"),
        upstream=("core_transport", "kinetic_fast_ions"),
        downstream=("core_transport", "diagnostics_synthetic", "wall_pwi"),
        reference_codes=("TRANSP fusion products", "NUBEAM alpha model"),
    ),
    FusionModuleSpec(
        key="atomic_impurity_radiation",
        group="源汇与边界",
        title="07D 原子过程 / 杂质 / 辐射",
        status="已有可运行原型",
        directory="modules/atomic_impurity_radiation",
        purpose="处理电离、复合、换电荷、杂质输运、辐射冷却和谱线发射，给 07F 提供能量损失项。",
        inputs=(port("profile_state", "07F 上一轮 n_e/T_e 剖面"), port("neutral_gas_state", "07A 换电荷损失")),
        outputs=(port("radiation_state", "辐射功率、线辐射、Zeff 和平均电荷态"),),
        config_keys=("input_mode", "profile_state", "neutral_state", "impurity_species", "radiation_model", "impurity_fraction"),
        upstream=("neutral_gas", "core_transport"),
        downstream=("core_transport", "diagnostics_synthetic", "pedestal_edge_sol"),
        reference_codes=("ADAS", "STRAHL", "Aurora", "ImpRad"),
    ),
    FusionModuleSpec(
        key="neutral_gas",
        group="源汇与边界",
        title="07A 中性气体 / 换电荷 / 再循环",
        status="已有可运行原型",
        directory="modules/neutral_gas",
        purpose="模拟边界中性气体、再循环、气阀、丸注入和换电荷损失，为 07B 粒子源、07D 原子辐射和边缘模块提供输入。",
        inputs=(port("wall_fluxes", "壁面粒子通量"), port("gas_sources", "气阀/丸源")),
        outputs=(port("neutral_gas_state", "中性粒子密度、离化源、换电荷损失和再循环源"),),
        config_keys=("input_mode", "neutral_model", "recycling_coeff", "gas_puff_rate_1e20_s", "wall_flux_1e20_s", "pellet_rate_1e20_s"),
        upstream=("wall_pwi", "sources_fueling"),
        downstream=("pedestal_edge_sol", "atomic_impurity_radiation", "sources_fueling"),
        reference_codes=("DEGAS2", "KN1D", "EIRENE"),
    ),
    FusionModuleSpec(
        key="wall_pwi",
        group="源汇与边界",
        title="第一壁 / PWI / 热负荷",
        status="接口预留",
        directory="modules/wall_pwi",
        purpose="计算壁面粒子与热负荷、溅射、沉积、材料响应和寿命指标。",
        inputs=(port("particle_fluxes", "离子/中性/中子通量"), port("wall_geometry", "壁面几何和材料")),
        outputs=(port("heat_loads", "热负荷", "MW/m^2"), port("erosion", "侵蚀/沉积", "m/s")),
        config_keys=("wall_materials", "sputtering_model", "thermal_model"),
        upstream=("pedestal_edge_sol", "fusion_reactions", "kinetic_fast_ions"),
        downstream=("neutral_gas", "validation_uq", "visualization"),
        reference_codes=("IonOrb", "FEM thermal solvers", "PWI material models"),
    ),
    FusionModuleSpec(
        key="diagnostics_synthetic",
        group="诊断控制验证",
        title="合成诊断 / 实验对比",
        status="接口预留",
        directory="modules/diagnostics_synthetic",
        purpose="生成磁探针、MSE、ECE、BES、FIDA、中子、软 X 射线、干涉仪等合成观测。",
        inputs=(port("plasma_state", "统一 plasma state"), port("diagnostic_geometry", "诊断视线/响应函数")),
        outputs=(port("synthetic_signals", "合成诊断信号"), port("residuals", "实验-模拟残差")),
        config_keys=("diagnostic_set", "noise_model", "instrument_response"),
        upstream=("data_model", "core_transport", "kinetic_fast_ions", "atomic_impurity_radiation"),
        downstream=("validation_uq", "control_actuators"),
        reference_codes=("CHERAB", "FIDASIM", "EFIT constraints", "OMFIT diagnostics"),
    ),
    FusionModuleSpec(
        key="control_actuators",
        group="诊断控制验证",
        title="控制 / 执行器 / 放电场景",
        status="接口预留",
        directory="modules/control_actuators",
        purpose="描述线圈、电源、加热器、气阀和反馈控制，用于场景设计和运行优化。",
        inputs=(port("diagnostics", "实时或合成诊断"), port("targets", "形状/功率/密度/稳定性目标")),
        outputs=(port("actuator_commands", "线圈、电源、加热、气阀命令"),),
        config_keys=("controller", "target_shape", "constraints", "optimization_method"),
        upstream=("diagnostics_synthetic", "mhd_stability"),
        downstream=("device_machine", "waves_hcd", "sources_fueling"),
        reference_codes=("PCS models", "Waveform-Editor", "scenario optimization"),
    ),
    FusionModuleSpec(
        key="workflow_coupling",
        group="装置与数据底座",
        title="工作流耦合 / 任务调度 / Checkpoint",
        status="接口预留",
        directory="modules/workflow_coupling",
        purpose="管理组件依赖、时间推进、并行任务、checkpoint/restart、数据交换和报告生成。",
        inputs=(port("scenario", "放电或计算场景"), port("component_specs", "模块接口描述")),
        outputs=(port("run_manifest", "运行清单"), port("checkpoints", "可恢复状态")),
        config_keys=("driver", "time_loop", "checkpoint_every", "parallel_backend"),
        upstream=("data_model",),
        downstream=("all_modules",),
        reference_codes=("IPS", "OMFIT workflows", "AToM"),
    ),
    FusionModuleSpec(
        key="validation_uq",
        group="诊断控制验证",
        title="验证 / Benchmark / 不确定性量化",
        status="已有原型",
        directory="modules/validation_uq",
        purpose="统一 benchmark、代码对比、实验验证、误差条、不确定性传播和回归测试。",
        inputs=(port("simulation_outputs", "模拟输出"), port("experimental_data", "实验数据或基准解")),
        outputs=(port("metrics", "误差、漂移、守恒量、置信区间"), port("reports", "报告和图表")),
        config_keys=("metric_set", "reference_case", "uq_samples", "regression_threshold"),
        upstream=("diagnostics_synthetic", "workflow_coupling"),
        downstream=("visualization",),
        reference_codes=("OMFIT Regression", "TRANSP validation", "platform benchmark"),
    ),
    FusionModuleSpec(
        key="visualization",
        group="诊断控制验证",
        title="可视化 / 报告 / 数字孪生展示",
        status="接口预留",
        directory="modules/visualization",
        purpose="统一图像、PPT、PDF、交互式剖面、3D 几何和运行看板。",
        inputs=(port("plasma_state", "统一状态"), port("metrics", "验证指标")),
        outputs=(port("figures", "PNG/SVG/HTML 图"), port("reports", "PPT/PDF/Markdown")),
        config_keys=("theme", "figure_set", "report_template", "export_format"),
        upstream=("validation_uq", "workflow_coupling"),
        downstream=(),
        reference_codes=("IMAS-ParaView", "OMFIT plotting", "custom desktop shell"),
    ),
    FusionModuleSpec(
        key="systems_design",
        group="系统工程与电站",
        title="系统设计 / 参数优化 / 成本约束",
        status="接口预留",
        directory="modules/systems_design",
        purpose="从 0D/1D 约束、装置尺寸、功率目标、技术假设和经济指标生成自洽设计点。",
        inputs=(port("requirements", "Q 值、净电功率、脉冲长度、成本和技术边界"), port("technology_assumptions", "磁体、包层、热工、材料和维护假设")),
        outputs=(port("design_point", "主半径、小半径、磁场、功率、库存和约束余量"), port("trade_space", "参数扫描和多目标优化结果")),
        config_keys=("objective", "optimizer", "constraint_set", "fidelity_level"),
        upstream=("data_model",),
        downstream=("device_machine", "scenario_operations", "neutronics_blanket", "thermal_power_conversion", "plant_systems_safety"),
        reference_codes=("PROCESS", "FUSE", "OpenMDAO", "systems code"),
    ),
    FusionModuleSpec(
        key="scenario_operations",
        group="系统工程与电站",
        title="放电场景 / 脉冲调度 / 运行边界",
        status="接口预留",
        directory="modules/scenario_operations",
        purpose="规划启动、爬升、平顶、降功率和终止阶段，给控制和物理模块提供时间线与执行器波形。",
        inputs=(port("design_point", "系统设计点"), port("plasma_targets", "电流、形状、密度、温度、功率和安全因子目标")),
        outputs=(port("scenario", "分阶段放电计划和时间网格"), port("waveforms", "线圈、电源、加热、燃料和缓解系统波形")),
        config_keys=("scenario_type", "ramp_rate_limits", "pulse_length_s", "operating_envelope"),
        upstream=("systems_design", "control_actuators"),
        downstream=("workflow_coupling", "equilibrium_mhd", "core_transport", "control_actuators"),
        reference_codes=("DINA", "METIS", "NICE", "Waveform-Editor", "OMFIT STEP"),
    ),
    FusionModuleSpec(
        key="disruption_runaway",
        group="平衡与稳定",
        title="破裂缓解 / 逃逸电子 / 快速终止",
        status="接口预留",
        directory="modules/disruption_runaway",
        purpose="模拟热淬灭、电流淬灭、注入缓解、逃逸电子产生和终止阶段载荷，服务安全边界和控制设计。",
        inputs=(port("equilibrium", "破裂前 MHD 平衡"), port("stability_metrics", "不稳定性和触发指标"), port("mitigation_commands", "MGI/SPI/线圈等缓解命令")),
        outputs=(port("runaway_distribution", "逃逸电子分布和损失通量"), port("mitigation_metrics", "热负荷、电磁载荷、电流淬灭时间和残余电流")),
        config_keys=("trigger_model", "mitigation_model", "runaway_source", "radiation_loss_model"),
        upstream=("mhd_stability", "control_actuators", "diagnostics_synthetic"),
        downstream=("wall_pwi", "materials_lifecycle", "plant_systems_safety", "validation_uq"),
        reference_codes=("JOREK", "M3D-C1", "NIMROD", "KORC", "SOFT"),
    ),
    FusionModuleSpec(
        key="neutronics_blanket",
        group="核工程与电站",
        title="中子学 / 增殖包层 / 活化屏蔽",
        status="接口预留",
        directory="modules/neutronics_blanket",
        purpose="计算中子输运、核热沉积、氚增殖比、屏蔽、材料活化、衰变热和剂量率。",
        inputs=(port("neutron_source", "中子源强和能谱"), port("geometry", "真空室、包层、屏蔽、磁体和端口几何"), port("materials", "材料组成、温度和核数据")),
        outputs=(port("neutron_flux", "空间/能量分辨中子通量"), port("tbr", "氚增殖比"), port("activation", "活化、衰变热和剂量率"), port("nuclear_heating", "核热沉积", "MW/m^3")),
        config_keys=("transport_code", "nuclear_data", "blanket_concept", "tally_set"),
        upstream=("fusion_reactions", "device_machine", "systems_design"),
        downstream=("tritium_fuel_cycle", "thermal_power_conversion", "materials_lifecycle", "plant_systems_safety"),
        reference_codes=("OpenMC", "MCNP", "Shift", "FISPACT-II", "Paramak"),
    ),
    FusionModuleSpec(
        key="tritium_fuel_cycle",
        group="核工程与电站",
        title="氚燃料循环 / 库存 / 处理回收",
        status="接口预留",
        directory="modules/tritium_fuel_cycle",
        purpose="跟踪 D/T 注入、燃耗、抽气、同位素分离、储存、滞留、去氚和包层产氚闭环。",
        inputs=(port("fuel_consumption", "等离子体燃耗和灰分排出"), port("blanket_breeding", "包层产氚率"), port("exhaust_streams", "抽气和杂质/氦灰流")),
        outputs=(port("tritium_inventory", "分系统氚库存和滞留"), port("recycle_flow", "回收、补给和损失流量"), port("accountancy", "氚平衡、倍增裕量和安全约束")),
        config_keys=("fuel_cycle_topology", "separation_model", "storage_model", "detritiation_model"),
        upstream=("sources_fueling", "fusion_reactions", "neutronics_blanket", "wall_pwi"),
        downstream=("sources_fueling", "plant_systems_safety", "systems_design"),
        reference_codes=("H3AT", "IAEA D-T fuel cycle", "PROCESS", "FUSE"),
    ),
    FusionModuleSpec(
        key="thermal_power_conversion",
        group="核工程与电站",
        title="热工水力 / 冷却回路 / 发电转换",
        status="接口预留",
        directory="modules/thermal_power_conversion",
        purpose="把第一壁、偏滤器、包层和屏蔽热源转换为冷却剂状态、热效率、净电功率和瞬态安全余量。",
        inputs=(port("surface_heat_loads", "等离子体侧热负荷"), port("nuclear_heating", "中子核热沉积"), port("coolant_config", "冷却剂、通道、换热器和涡轮循环参数")),
        outputs=(port("coolant_state", "温度、压降、流量和换热裕量"), port("gross_electric_power", "毛电功率", "MW"), port("net_electric_power", "净电功率", "MW")),
        config_keys=("coolant", "cycle_model", "heat_exchanger_model", "transient_solver"),
        upstream=("wall_pwi", "neutronics_blanket", "systems_design"),
        downstream=("plant_systems_safety", "systems_design", "visualization"),
        reference_codes=("PROCESS", "FUSE", "RELAP", "SAM", "Modelica"),
    ),
    FusionModuleSpec(
        key="magnets_cryogenics",
        group="核工程与电站",
        title="超导磁体 / 电源 / 低温系统",
        status="接口预留",
        directory="modules/magnets_cryogenics",
        purpose="描述 TF/PF/CS 线圈、电源、结构载荷、淬火保护、低温负荷和脉冲能量管理。",
        inputs=(port("coil_geometry", "线圈几何、匝数和材料"), port("magnetic_loads", "电磁力、核热和交流损耗"), port("scenario", "电流波形和脉冲计划")),
        outputs=(port("coil_limits", "临界电流、温度裕量、应力和疲劳指标"), port("cryogenic_load", "低温热负荷和冷功率"), port("power_supply", "电源电压、电流和储能需求")),
        config_keys=("conductor_model", "quench_model", "cryoplant_model", "power_supply_model"),
        upstream=("device_machine", "neutronics_blanket", "scenario_operations"),
        downstream=("systems_design", "plant_systems_safety", "control_actuators"),
        reference_codes=("PROCESS magnet models", "FUSE", "ITER magnet system", "Cryo plant models"),
    ),
    FusionModuleSpec(
        key="materials_lifecycle",
        group="核工程与电站",
        title="材料寿命 / 远程维护 / 部件更换",
        status="接口预留",
        directory="modules/materials_lifecycle",
        purpose="评估等离子体面对材料、包层、屏蔽和真空室的辐照损伤、氚滞留、疲劳寿命、检查和远程维护计划。",
        inputs=(port("heat_loads", "热循环和瞬态热负荷"), port("neutron_flux", "中子通量、dpa 和氦/氢产生"), port("tritium_inventory", "材料中氚滞留和渗透")),
        outputs=(port("lifetime_limits", "寿命、损伤、腐蚀/侵蚀和更换阈值"), port("maintenance_plan", "检查、退役、部件更换和遥操作窗口")),
        config_keys=("material_database", "damage_model", "inspection_interval", "remote_handling_model"),
        upstream=("wall_pwi", "neutronics_blanket", "tritium_fuel_cycle", "disruption_runaway"),
        downstream=("plant_systems_safety", "systems_design", "visualization"),
        reference_codes=("FISPACT-II", "FESTIM", "UKAEA Materials", "RAICo", "remote handling models"),
    ),
    FusionModuleSpec(
        key="plant_systems_safety",
        group="核工程与电站",
        title="全厂系统 / 安全边界 / 成本与 RAMI",
        status="接口预留",
        directory="modules/plant_systems_safety",
        purpose="汇总物理、电站工程、库存、可靠性、可维护性、可用率、安全包络和成本，形成全厂级设计约束。",
        inputs=(port("design_point", "系统设计点"), port("inventories", "氚、活化、能量和热库存"), port("failure_modes", "失效模式、保护动作和维护事件")),
        outputs=(port("plant_kpis", "Q、净电、可用率、效率、成本和寿命指标"), port("safety_envelope", "安全边界、源项、剂量和许可相关指标"), port("cost_model", "CAPEX/OPEX 和不确定性")),
        config_keys=("rami_model", "safety_case", "cost_basis", "availability_target"),
        upstream=("systems_design", "neutronics_blanket", "tritium_fuel_cycle", "thermal_power_conversion", "materials_lifecycle", "magnets_cryogenics"),
        downstream=("systems_design", "validation_uq", "visualization"),
        reference_codes=("PROCESS", "FUSE", "RAMI", "safety case", "costing models"),
    ),
)


SPEC_BY_KEY = {spec.key: spec for spec in FUSION_MODULE_SPECS}


def get_spec(key: str) -> FusionModuleSpec:
    try:
        return SPEC_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown fusion module spec: {key}") from exc


def build_placeholder_component(key: str, config: dict[str, Any] | None = None) -> PlaceholderComponent:
    return PlaceholderComponent(get_spec(key), config)
