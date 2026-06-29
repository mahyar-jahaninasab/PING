
import contextlib
import torch
import torch.nn as nn
import time
import pynvml
import matplotlib.pyplot as plt
import os 
import matplotlib as mpl
import threading
import time as _time
import numpy as np

from model import MLP
from dotenv import load_dotenv
import json 

load_dotenv()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


problem_path = os.getenv('PROBLEM')
with open(problem_path) as f:
    data = json.load(f)

GENERATOR = data["GENERATOR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

class NVMLSampler:
    def __init__(self, handle, interval_s=0.05):
        self.handle = handle
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []  # (t_rel_s, util_pct, mem_used_MiB)

    def start(self):
        self.t0 = _time.time()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def stop(self):
        self._stop.set()
        if hasattr(self, "th"):
            self.th.join()

    def _run(self):
        while not self._stop.is_set():
            t_rel = _time.time() - self.t0
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)  # MiB
            except Exception:
                util, mem = None, None
            self.samples.append((t_rel, util, mem))
            _time.sleep(self.interval_s)

def _save(fig, path_base):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{path_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

def plot_util_series(meta, out_dir, run_id):
    ts = np.array(meta["nvml"]["t"])
    util = np.array(meta["nvml"]["util"])
    mem = np.array(meta["nvml"]["mem"])
    fig, ax1 = plt.subplots(figsize=(3.4, 2.2))
    l1, = ax1.plot(ts, util, color="tab:blue", lw=1.5, label="GPU Util (%)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Utilization (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    l2, = ax2.plot(ts, mem, color="tab:red", lw=1.2, alpha=0.8, label="VRAM (MiB)")
    ax2.set_ylabel("Memory (MiB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("GPU Utilization and Memory vs Time")
    ax1.legend(handles=[l1, l2], loc="upper right")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{run_id}_gpu_timeseries"))

def plot_latency(meta, out_dir, run_id):
    lat_ms = np.array(meta["per_model_ms"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(lat_ms, bins=40, color="0.3", alpha=0.9)
    ax.set_xlabel("Per‑model latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per‑model Latency Distribution")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{run_id}_latency_hist"))
    # ECDF
    x = np.sort(lat_ms)
    y = np.arange(1, x.size + 1) / x.size
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(x, y, lw=1.5)
    ax.set_xlabel("Per‑model latency (ms)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per‑model Latency ECDF")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{run_id}_latency_ecdf"))

def plot_outputs(outputs_list, t_tensor, out_dir, run_id):
    with torch.no_grad():
        O = torch.stack(outputs_list, dim=0).detach().cpu().numpy()
        t = t_tensor.detach().cpu().numpy().squeeze()
    M, T, D = O.shape
    for d in range(D):
        mu = O[:, :, d].mean(axis=0)
        sd = O[:, :, d].std(axis=0, ddof=1)
        lo = mu - 1.96 * sd
        hi = mu + 1.96 * sd
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ax.plot(t, mu, color="tab:blue", lw=1.5, label=f"mean (D={d})")
        ax.fill_between(t, lo, hi, color="tab:blue", alpha=0.2, label="95% CI")
        ax.set_xlabel("t")
        ax.set_ylabel("output")
        ax.set_title("Ensemble Mean ± 95% CI")
        ax.legend()
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"{run_id}_mean_ci_d{d}"))
    idx = 0
    Z = O[:, :, idx]
    vmax = np.percentile(np.abs(Z), 99)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[t.min(), t.max(), 0, M],
                   cmap="viridis", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("t")
    ax.set_ylabel("model index")
    ax.set_title("Model Outputs Heatmap")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("output")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{run_id}_outputs_heatmap"))


class ForwardPhase:
    def __init__(self):
        self.meta_data = {}
        self.outputs = []

        batch_size = GENERATOR["batch_size"]
        if batch_size == 1:
            try:
                print('For fixed time domain the batch size of 1 will be used only')
                upper_limits = torch.full((1,), GENERATOR["upper_bound_time"])
            except KeyError:
                raise ValueError(
                    "Batch size of 1 requires 'upper_limit' in GENERATOR configuration"
                )
        else:
            upper_limits = torch.distributions.Exponential(
                rate=GENERATOR["rate"]
            ).sample((batch_size,))

        self.t = torch.stack([
            torch.linspace(0., float(limit), GENERATOR["num_points"]).view(-1, 1)
            for limit in upper_limits
        ]).to(DEVICE)

        if batch_size > 1:
            _, sorted_indices = torch.sort(self.t[:, -1, 0], descending = GENERATOR["descending"], stable=True)
            self.t = self.t[sorted_indices]
            
            max_time = self.t[-1, -1, 0].item()
            if GENERATOR["descending"]:
                print(
                    f"Minimum time: {max_time:.4f}\n"
                    f"Tip: Change the rate to change this limit."
                )
            else:
                print(
                    f"Maximum time: {max_time:.4f}\n"
                    f"Tip: Change the rate to change this limit."
                )

    def generate(self):
        num_models = int(GENERATOR["numbers"])
        models = []
        t0 = _time.time()
        for i in range(num_models):
            torch.manual_seed(i)
            m = MLP(
                input_dim=int(GENERATOR["input"]),
                output_dim=int(GENERATOR["output"]),
                hidden_sizes=GENERATOR["layers"],
                activation=nn.Tanh
            ).to(DEVICE).eval()
            models.append(m)
        stream = torch.cuda.Stream(device=DEVICE) if DEVICE.type == "cuda" else None
        with torch.inference_mode():
            if stream is not None:
                with torch.cuda.stream(stream):
                    for m in models:
                        _ = m(self.t )
            else:
                for m in models:
                    _ = m(self.t )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        sampler = NVMLSampler(handle, interval_s=0.05)
        sampler.start()
        stride = max(1, num_models // 200)  
        if DEVICE.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
        self.outputs = []
        with torch.inference_mode():
            ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx:
                for i, m in enumerate(models):
                    if DEVICE.type == "cuda" and (i % stride == 0):
                        start_ev.record()
                        out = m(self.t)
                        end_ev.record()
                        torch.cuda.synchronize(DEVICE)
                        self.outputs.append(out)
                    else:
                        out = m(self.t)
                        self.outputs.append(out)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        execution = _time.time() - t0
        sampler.stop()
        nvml_t, nvml_util, nvml_mem = zip(*sampler.samples) if sampler.samples else ([], [], [])
        util_snapshot = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        self.meta_data = {
            "execution": execution,
            "util_snapshot_pct": util_snapshot,
            "num_models": num_models,
            "seq_len": int(self.t.numel()),
        }
       
        return self.outputs, self.t, self.meta_data

    def plot_run(self, out_dir="figs", run_id="run1", outputs=None, t=None, meta_data=None):
        outs = outputs if outputs is not None else self.outputs
        tt = t if t is not None else self.t
        md = meta_data if meta_data is not None else self.meta_data
        os.makedirs(out_dir, exist_ok=True)
        plot_util_series(md, out_dir, run_id)
        if md.get("per_model_ms"):
            plot_latency(md, out_dir, run_id)
        plot_outputs(outs, tt, out_dir, run_id)

def tensor_to_column_tuples(x: torch.Tensor):
    """
    Convert a tensor [B, R, C] into a list of length B,
    where each element is a tuple of column tensors.

    Example:
        x[b] -> (col1, col2, ..., colC), 
        each col is shape [R].
    """
    B, R, C = x.shape
    result = []
    for b in range(B):
        cols = tuple(x[b, :, c] for c in range(C))
        result.append(cols)
    return result


