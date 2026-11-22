import os
import sys
import importlib.util

# Forward everything from root architectures.common_utils
_root_arch_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../architectures/common_utils.py")
)
if os.path.isfile(_root_arch_path):
    _spec = importlib.util.spec_from_file_location("root_common_utils", _root_arch_path)
    _root_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_root_mod)
    for _k, _v in _root_mod.__dict__.items():
        if not _k.startswith("__"):
            globals()[_k] = _v


def identity(x):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x


