"""Load vLLM 0.30's real CPU-offload manager (pure-Python scheduler side) without
importing torch/CUDA: packages are registered with their real __path__ but their
__init__.py is skipped, and the few heavy leaf modules are stubbed.
Usage: VLLM_SITE=<site-packages> python tests/test_real_manager.py
"""
import importlib, logging, os, sys, types

SITE = os.environ["VLLM_SITE"]
VLLM = os.path.join(SITE, "vllm")


def _pkg(name):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(SITE, *name.split("."))]
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_pkg(parent), child, m)
    return m


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_pkg(parent), child, m)
    return m


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()


def setup():
    for p in ["vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu",
              "vllm.v1.kv_offload.cpu.policies", "vllm.distributed",
              "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
              "vllm.distributed.kv_transfer.kv_connector.v1",
              "vllm.distributed.kv_transfer.kv_connector.v1.offloading"]:
        _pkg(p)
    _stub("torch", Tensor=object, dtype=object, device=object)
    def init_logger(n):
        lg = logging.getLogger(n)
        lg.warning_once = lg.warning  # vLLM logger extensions
        lg.info_once = lg.info
        lg.debug_once = lg.debug
        return lg
    _stub("vllm.logger", init_logger=init_logger)
    # metrics module pulls in vllm.config (pydantic, transformers...): only a
    # type name is needed by the manager's scheduler-side logic.
    _stub("vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics",
          OffloadingConnectorStats=_Any)


def load(modname):
    """Import a vLLM module, stubbing any missing leaf module on the way."""
    for _ in range(40):
        try:
            return importlib.import_module(modname)
        except ModuleNotFoundError as e:
            missing = e.name
            if not missing or missing in sys.modules:
                raise
            print(f"  [harness] stubbing missing module {missing}")
            _stub(missing, __getattr__=lambda n: _Any)
        except ImportError as e:
            # "cannot import name X from module M" where M is one of our stubs
            msg = str(e)
            raise
    raise RuntimeError("too many stubs")
