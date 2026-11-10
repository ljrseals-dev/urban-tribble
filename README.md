torch_shim — A Minimal NumPy Stand-In for PyTorch

What this is: A pure-NumPy reimplementation of the subset of PyTorch that render_magnetic_clavecin.py uses to load a checkpoint and run a forward pass. It allows the renderer to run on machines where PyTorch is not installed.

What this is not: A general PyTorch replacement. It does not support training, autograd, optimizers, GPU tensors, or any operation not used by the renderer.

What works:

· Loading magnetic-oracle-v1 checkpoints via torch.load(..., weights_only=True).
· Reconstructing Sequential(Linear, ReLU, Linear, ReLU, Linear) models.
· Running forward passes via model(x).
· torch.sigmoid, torch.tensor, torch.from_numpy, torch.no_grad.
· Tensor.unsqueeze(0), Tensor.squeeze(0), Tensor.cpu(), Tensor.numpy(), Tensor.tolist().

What does not work:

· Training. train_magnetic_oracle.py requires real PyTorch. Under this shim, it will fail with NameError: optim (there is no torch.optim in the shim).
· Autograd. There is no .backward(), no .grad, no computation graph.
· GPU tensors. Everything is on CPU.
· torch.use_deterministic_algorithms, torch.manual_seed (the latter is a no-op).
· Any torch API not listed in "What works."

Security: The shim's checkpoint loader uses a restricted pickle.Unpickler with an allowlist of classes (_rebuild_tensor_v2, storage types, OrderedDict). It raises UnpicklingError for any other class. This is stricter than torch.load(weights_only=False) and approximately equivalent to torch.load(weights_only=True). Do not use this shim to load checkpoints from untrusted sources.

Testing: The shim should be tested against real PyTorch on the actual MagneticOracle checkpoint. The test is: run render_magnetic_clavecin.py --checkpoint ... --fast under real PyTorch and under the shim, and compare the output WAV and diagnostics.json. If they match, the shim is correct for the renderer's use case. If they do not match, the shim has a bug and should not be used.

Scope: This shim exists only so the renderer can run without PyTorch. It should not be used for any other purpose. If you find yourself needing a torch API that isn't listed in "What works," install real PyTorch instead of extending the shim.

---

The honest verdict

The shim has a one-sentence docstring and no README. That is not enough for a piece of code that is meant to be a substitute for a major dependency. The shim is correct for its intended use case (the renderer), but a user who picks it up without context will not know what it does, what it doesn't do, or how to test it.

The fix is a README like the one above. It is a small piece of work (maybe thirty minutes to write and review), and it makes the difference between a shim that is used correctly and a shim that is misused.

If you want me to draft the actual README file (in markdown, ready to drop into the repository), say so and I'll write it out in full.
