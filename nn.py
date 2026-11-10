import numpy as np
class Module:
    def __init__(self): self._modules = {}
    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        if isinstance(v, Module): self._modules[k] = v
    def __call__(self, x): return self.forward(x)
    def eval(self): return self
    def load_state_dict(self, sd, strict=True):
        own = dict(self._named())
        if strict and set(own) != set(sd): raise RuntimeError(f"key mismatch: {sorted(set(own) ^ set(sd))}")
        for k, v in sd.items(): own[k][0].__dict__[own[k][1]] = np.asarray(v, dtype=np.float32)
    def _named(self, prefix=""):
        for n, m in self._modules.items():
            yield from m._named(prefix + n + ".")
    def _own(self): return []
class Linear(Module):
    def __init__(self, i, o): super().__init__(); self.weight = np.zeros((o, i), np.float32); self.bias = np.zeros(o, np.float32)
    def _named(self, prefix=""): yield prefix + "weight", (self, "weight"); yield prefix + "bias", (self, "bias")
    def forward(self, x): return (np.asarray(x, np.float32) @ self.weight.T + self.bias).astype(np.float32).view(__import__("torch").Tensor)
class ReLU(Module):
    def _named(self, prefix=""): return iter(())
    def forward(self, x): return np.maximum(np.asarray(x), 0).astype(np.float32).view(__import__("torch").Tensor)
class Sequential(Module):
    def __init__(self, *mods):
        super().__init__()
        for i, m in enumerate(mods): self._modules[str(i)] = m
    def forward(self, x):
        for m in self._modules.values(): x = m(x)
        return x
