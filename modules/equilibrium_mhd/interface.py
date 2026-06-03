from modules.core.fusion_interface import build_placeholder_component, get_spec

SPEC = get_spec("equilibrium_mhd")


def build_component(config=None):
    return build_placeholder_component("equilibrium_mhd", config)

