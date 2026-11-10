"""Minimal numpy stand-in for the parts of torch that MagneticOracle's renderer uses (checkpoint load + tiny MLP forward).
Only for running the renderer where torch is not installed. Not a general torch replacement."""
import io, pickle, zipfile, contextlib
import numpy as np
from . import nn
float32 = np.float32

class Tensor(np.ndarray):
    def unsqueeze(self, d): return np.expand_dims(np.asarray(self), d).view(Tensor)
    def squeeze(self, d=None): return np.squeeze(np.asarray(self), d).view(Tensor)
    def cpu(self): return self
    def numpy(self): return np.asarray(self)

def _t(a): return np.asarray(a).view(Tensor)
def tensor(data, dtype=None): return _t(np.array(data, dtype=dtype or np.float32))
def from_numpy(a): return _t(a)
def sigmoid(x): x = np.asarray(x, dtype=np.float32); return _t(1.0 / (1.0 + np.exp(-x)))
@contextlib.contextmanager
def no_grad(): yield

class _Storage:
    def __init__(self, dtype): self.dtype = dtype
_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16}
class _StorageType:
    def __init__(self, name): self.dtype = _DT[name]

def load(path, map_location=None, weights_only=None):
    z = zipfile.ZipFile(path); names = z.namelist(); prefix = names[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *rest):
        a = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=tuple(s * storage.itemsize for s in stride)); return a.copy()
    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if module == "torch" and name in _DT: return _StorageType(name)
            if module == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        def persistent_load(self, pid):
            _, st, key, loc, numel = pid; return np.frombuffer(z.read(f"{prefix}/data/{key}"), dtype=st.dtype)
    return U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()

def manual_seed(s): return None   # the renderer's torch RNG is not used for sampling in this path
