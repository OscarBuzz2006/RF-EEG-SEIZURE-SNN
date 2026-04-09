"""
rf_eeg_seizure_snn.py
=====================
Neuromorphic Edge Computing for Neuromodulation Feedback Systems
Supervisor: Omid Kavehei — Biomedical Engineering, University of Sydney

Architecture:
  TUSZ EDF loader → Delta Modulation encoding → Conv2d → RFSpike (40 Hz) →
  Conv2d → RFSpike (10 Hz) → Flatten → Linear → Leaky (output)

RFSpike neuron follows Izhikevich (2001) complex-state formulation, with
ZOH discretization as used in Huber et al. S5-RF (2024) and consistent with
JCPappo/Resonate-and-Fire-neurons (PyTorch/snnTorch). The frequency parameter
is set to gamma band (40 Hz) for the first layer and alpha band (10 Hz) for the
second, matching seizure-relevant EEG oscillations.

Dataset: TUSZ v2.0.0 evaluation subset
  Structure: [patient_id] / [session_id] / [montage] / [filename].edf
  Sampling rate: 256 Hz | Window: 4 s (1024 samples) | Channels: up to 34

References:
  - Izhikevich (2001) Neural Networks 14:883-894
  - Huber et al. (2024) S5-RF: Scaling Up Resonate-and-Fire Networks
  - Kavehei Lab (2024) APL Machine Learning 2(2):026114 [LTC-SNN on TUH]
  - Zhang et al. (2024) Frontiers in Neuroscience [EESNN cross-patient]
"""

import os
import glob
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import snntorch as snn
from snntorch import surrogate, utils
from snntorch import functional as SF

# ── Optional MNE import (graceful fallback) ─────────────────────────────────
try:
    import mne
    mne.set_log_level("WARNING")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn(
        "MNE not found. Install with: pip install mne\n"
        "Falling back to dummy data for testing pipeline.",
        UserWarning
    )

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Global EEG constants ─────────────────────────────────────────────────────
SFREQ          = 256          # Hz — TUSZ recording sample rate
WINDOW_SECS    = 4            # seconds per sample window
WINDOW_SAMPLES = SFREQ * WINDOW_SECS   # 1024 samples per window
BANDPASS_LOW   = 0.5          # Hz
BANDPASS_HIGH  = 128.0        # Hz (Nyquist for 256 Hz)
MAX_CHANNELS   = 34           # TUSZ can have up to 34 channels
MIN_CHANNELS   = 19           # standard 10-20 montage minimum
NUM_CLASSES    = 2            # 0=background (interictal), 1=seizure (ictal)
BATCH_SIZE     = 32
NUM_EPOCHS     = 10
NUM_STEPS      = 25           # SNN temporal steps (matches JCPappo MNIST results)
DELTA_THRESHOLD = 0.15        # delta modulation threshold (normalised units)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TUSZ EDF Data Loader
# ════════════════════════════════════════════════════════════════════════════

def parse_tse_bi_labels(tse_path, total_duration_sec, window_start_sec, window_end_sec):
    """
    Parse a TUSZ .tse_bi annotation file to determine whether a 4-second
    window contains a seizure.

    .tse_bi format (space-separated):
        version = tse_v1.0.0
        start_time  end_time  label  confidence
        e.g.:  0.0000  30.0000  bckg  1.0
               30.0000 34.0000  seiz  1.0

    Returns:
        1 if any seizure ('seiz') overlaps the window, else 0.
    """
    label = 0
    if not os.path.exists(tse_path):
        return label
    with open(tse_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                seg_start = float(parts[0])
                seg_end   = float(parts[1])
                seg_label = parts[2].lower()
            except ValueError:
                continue
            # Check for overlap with our window
            if seg_label == "seiz":
                overlap = min(seg_end, window_end_sec) - max(seg_start, window_start_sec)
                if overlap > 0:
                    label = 1
                    break
    return label


class TUSZDataset(Dataset):
    """
    Loads TUSZ v2.0.0 EDF files from the hierarchical directory structure:
        base_path / [patient_id] / [session_id] / [montage] / [filename].edf

    Each EDF is sliced into non-overlapping 4-second windows. Seizure labels
    are read from the paired .tse_bi annotation file.

    If MNE is not available, generates dummy tensors so the rest of the
    pipeline can still be tested.

    Args:
        base_path (str): Root path of the TUSZ dataset.
        max_channels (int): Cap on number of EEG channels to load.
                            If a file has fewer channels, it is zero-padded.
                            If more, only the first max_channels are taken.
    """
    def __init__(self, base_path, max_channels=MAX_CHANNELS):
        self.samples = []   # list of (edf_path, window_start_sec, label)
        self.max_channels = max_channels

        if not MNE_AVAILABLE:
            print("[TUSZDataset] MNE not available — generating dummy data.")
            self._generate_dummy()
            return

        edf_files = glob.glob(
            os.path.join(base_path, "**", "*.edf"), recursive=True
        )
        if len(edf_files) == 0:
            print(f"[TUSZDataset] No EDF files found at {base_path}. "
                  "Generating dummy data.")
            self._generate_dummy()
            return

        print(f"[TUSZDataset] Found {len(edf_files)} EDF files. Indexing windows...")
        for edf_path in edf_files:
            try:
                # Read header only to get duration without loading signal
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                duration_sec = raw.n_times / raw.info["sfreq"]
                n_windows = int(duration_sec // WINDOW_SECS)

                # Find paired annotation file (.tse_bi same name, different ext)
                tse_path = edf_path.replace(".edf", ".tse_bi")

                for w in range(n_windows):
                    t_start = w * WINDOW_SECS
                    t_end   = t_start + WINDOW_SECS
                    label   = parse_tse_bi_labels(tse_path, duration_sec, t_start, t_end)
                    self.samples.append((edf_path, t_start, label))

            except Exception as e:
                warnings.warn(f"Skipping {edf_path}: {e}")
                continue

        n_seiz = sum(s[2] for s in self.samples)
        print(f"[TUSZDataset] {len(self.samples)} windows indexed. "
              f"Seizure: {n_seiz} | Background: {len(self.samples) - n_seiz}")

    def _generate_dummy(self, n_samples=200):
        """Fallback dummy data when MNE is absent or no EDFs found."""
        for i in range(n_samples):
            self.samples.append(("__dummy__", 0.0, i % 2))

    def _load_window(self, edf_path, t_start):
        """
        Load a single 4-second window from an EDF file.

        Returns:
            np.ndarray of shape (max_channels, WINDOW_SAMPLES), normalised.
        """
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Band-pass filter to 0.5–128 Hz
        actual_high = min(BANDPASS_HIGH, raw.info["sfreq"] / 2 - 1)
        raw.filter(BANDPASS_LOW, actual_high, verbose=False)

        # Resample to 256 Hz if needed
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ, verbose=False)

        # Extract time window
        t_end    = t_start + WINDOW_SECS
        start_ix = int(t_start * SFREQ)
        end_ix   = start_ix + WINDOW_SAMPLES
        data     = raw.get_data()          # (n_ch, n_times)
        n_ch     = data.shape[0]

        # Conditionally handle channel count — no hard limit on input channels
        out = np.zeros((self.max_channels, WINDOW_SAMPLES), dtype=np.float32)
        ch_to_use = min(n_ch, self.max_channels)
        segment   = data[:ch_to_use, start_ix:end_ix]

        # Pad time axis if segment is shorter than expected
        actual_len = segment.shape[1]
        out[:ch_to_use, :actual_len] = segment[:, :min(actual_len, WINDOW_SAMPLES)]

        # Z-score normalise per channel (avoid division by zero)
        for c in range(ch_to_use):
            std = out[c].std()
            if std > 1e-8:
                out[c] = (out[c] - out[c].mean()) / std

        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        edf_path, t_start, label = self.samples[idx]

        if edf_path == "__dummy__":
            data = np.random.randn(self.max_channels, WINDOW_SAMPLES).astype(np.float32)
        else:
            data = self._load_window(edf_path, t_start)

        # Shape: (1, max_channels, WINDOW_SAMPLES) — channel-first for Conv2d
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Encoding Functions
# ════════════════════════════════════════════════════════════════════════════

def delta_modulation_encoding(data, num_steps, threshold=DELTA_THRESHOLD):
    """
    Delta Modulation (DM) encoding — the active encoding method.

    Converts a continuous EEG tensor into a sparse spike train by firing
    when the signal changes by more than `threshold` since the last spike.
    This is biologically plausible and directly mirrors asynchronous delta
    modulation used in neuromorphic sensors (e.g. Dynamic Vision Sensors).

    Args:
        data     : Tensor of shape (batch, 1, n_channels, n_timepoints)
        num_steps: Number of SNN time steps (temporal resolution)
        threshold: Minimum normalised change to trigger a spike

    Returns:
        Tensor of shape (num_steps, batch, 1, n_channels, step_width)
        — a spike train ready for time-stepped SNN forward pass.
    """
    batch, _, n_ch, n_time = data.shape
    step_width = n_time // num_steps
    spikes = torch.zeros(num_steps, batch, 1, n_ch, step_width, device=data.device)

    prev = data[:, :, :, :step_width].mean(dim=-1, keepdim=True)  # (B,1,C,1)

    for t in range(num_steps):
        segment = data[:, :, :, t * step_width:(t + 1) * step_width]
        current = segment.mean(dim=-1, keepdim=True)               # (B,1,C,1)
        delta   = current - prev
        # Spike where absolute change exceeds threshold
        spike_t = (delta.abs() > threshold).float()
        spikes[t] = spike_t.expand_as(spikes[t])
        prev = current

    return spikes


def stft_encoding(data, num_steps, n_fft=64, hop_length=None):
    """
    Short-Time Fourier Transform (STFT) encoding — stored for future use.

    Converts EEG to a time-frequency representation (spectrogram), then
    rate-encodes the power in each frequency bin as spike probability.
    This is the approach used by Kavehei Lab's LTC-SNN on TUH (2024).

    NOT currently called in the training loop — use delta_modulation_encoding.
    Swap in here when ready to experiment with frequency-domain input.

    Args:
        data      : Tensor (batch, 1, n_channels, n_timepoints)
        num_steps : SNN time steps
        n_fft     : FFT window size
        hop_length: Hop between windows (defaults to n_fft // 4)

    Returns:
        Tensor (num_steps, batch, 1, n_channels, n_fft//2+1) — spike train
        from thresholded spectrogram power.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    batch, _, n_ch, n_time = data.shape
    spikes_list = []

    for b in range(batch):
        ch_spikes = []
        for c in range(n_ch):
            signal = data[b, 0, c, :]                      # (n_time,)
            # Compute STFT power spectrum
            stft   = torch.stft(signal, n_fft=n_fft, hop_length=hop_length,
                                return_complex=True)        # (freq_bins, frames)
            power  = stft.abs().pow(2)                     # (freq_bins, frames)
            # Normalise to [0,1] and use as spike probability
            p_max  = power.max()
            if p_max > 1e-8:
                power = power / p_max
            # Resample frames to num_steps via interpolation
            power  = F.interpolate(
                power.unsqueeze(0).unsqueeze(0),           # (1,1,freq,frames)
                size=(power.shape[0], num_steps),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)                        # (freq_bins, num_steps)
            ch_spikes.append(power)
        spikes_list.append(torch.stack(ch_spikes))        # (n_ch, freq_bins, steps)

    # For now return a placeholder — full integration requires architecture change
    # to accommodate freq_bins as the spatial dimension
    raise NotImplementedError(
        "STFT encoding requires a matching architecture change. "
        "Use delta_modulation_encoding for now."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RFSpike Neuron (True Resonate-and-Fire)
# ════════════════════════════════════════════════════════════════════════════

class RFSpike(nn.Module):
    """
    True Resonate-and-Fire (RF) spiking neuron — Izhikevich (2001).

    The state is a complex number z = x + iy. Each timestep, z is multiplied
    by a complex decay factor (beta * e^{i*omega}), equivalent to a damped
    rotation in the complex plane. The neuron spikes when the real part x
    exceeds the threshold, and a soft reset subtracts the threshold from x.

    Discretization follows the Zero-Order Hold (ZOH) method used in
    Huber et al. S5-RF (2024): omega = 2*pi*frequency / sfreq, so the
    resonant frequency is physically meaningful in Hz.

    Compared to a standard LIF neuron, this model:
      - Has two coupled state variables (x, y) rather than one membrane
      - Oscillates at a preferred frequency even without input
      - Acts as a band-pass filter — preferentially responds to inputs near
        its resonant frequency (critical for EEG band selection)
      - Can fire in response to inhibitory inputs (post-inhibitory rebound)

    Args:
        frequency  (float): Resonant frequency in Hz (default 40 Hz = gamma band)
        beta       (float): Envelope decay per timestep (default 0.99)
        threshold  (float): Spike threshold on real state x (default 1.0)
        sfreq      (int)  : Sample/SNN step rate for ZOH discretization
        spike_grad        : Surrogate gradient (default fast_sigmoid)
        output     (bool) : If True, return (spike, x_real); else return spike only
    """
    def __init__(
        self,
        frequency=40.0,
        beta=0.99,
        threshold=1.0,
        sfreq=NUM_STEPS,
        spike_grad=None,
        output=False,
    ):
        super().__init__()
        self.beta      = beta
        self.threshold = threshold
        self.output    = output
        self.spike_grad = spike_grad if spike_grad is not None else surrogate.fast_sigmoid()

        # ZOH discretization: omega = 2*pi*f / sfreq
        # This ensures the resonant frequency is exactly `frequency` Hz
        # when the SNN runs at `sfreq` steps per second (Huber et al., 2024)
        omega = 2.0 * math.pi * frequency / sfreq
        self.register_buffer("cos_w", torch.tensor(math.cos(omega), dtype=torch.float32))
        self.register_buffer("sin_w", torch.tensor(math.sin(omega), dtype=torch.float32))

        # Hidden complex state: real (x = membrane) and imaginary (y = oscillator)
        self.x = None
        self.y = None

    def _init_state(self, like):
        self.x = torch.zeros_like(like)
        self.y = torch.zeros_like(like)

    def forward(self, input_current):
        """
        One timestep of the RF neuron.

        Update equations (discrete ZOH):
            x_new = beta * (cos_w * x  -  sin_w * y) + I(t)
            y_new = beta * (sin_w * x  +  cos_w * y)

        Spike condition: x_new >= threshold
        Soft reset:      x = x_new - spike * threshold

        The imaginary update has no input injection — only the real part
        receives the synaptic current, consistent with Izhikevich (2001).
        """
        if self.x is None or self.x.shape != input_current.shape:
            self._init_state(input_current)

        # Complex rotation with envelope decay
        x_new = self.beta * (self.cos_w * self.x - self.sin_w * self.y) + input_current
        y_new = self.beta * (self.sin_w * self.x + self.cos_w * self.y)

        # Spike via surrogate gradient (backprop-compatible)
        spike = self.spike_grad(x_new - self.threshold)

        # Soft reset on real part only
        self.x = x_new - spike * self.threshold
        self.y = y_new

        if self.output:
            return spike, self.x
        return spike

    def reset_state(self):
        """Call between batches to clear hidden state."""
        self.x = None
        self.y = None


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SNN Architecture
# ════════════════════════════════════════════════════════════════════════════

class EEGSeizureSNN(nn.Module):
    """
    Convolutional SNN for binary seizure detection on TUSZ EEG.

    Input shape: (batch, 1, n_channels, n_timepoints)
      e.g. (32, 1, 34, 1024) for 34-channel, 4-second, 256 Hz windows

    Architecture (matching Kavehei Lab LTC-SNN style with RF neurons):

      Conv2d(1→16, kernel=(1,25), stride=(1,5))   — temporal feature extraction
        ↓ RFSpike(40 Hz, gamma band)               — first RF layer
      Conv2d(16→32, kernel=(1,10), stride=(1,2))  — further temporal compression
        ↓ RFSpike(10 Hz, alpha band)               — second RF layer (alpha resonance)
      Conv2d(32→64, kernel=(1,5),  stride=(1,2))  — spatial-temporal compression
        ↓ Leaky LIF                                — standard leaky layer
      AdaptiveAvgPool2d(1,1) → Flatten
      Linear(64, 128) → Leaky LIF
      Linear(128, 2)  → Leaky LIF (output, binary classification)

    Two stacked RF layers allow the network to jointly resonate at gamma (40 Hz,
    associated with ictal high-frequency activity) and alpha (10 Hz, associated
    with normal background rhythms), giving the classifier direct spectral
    discriminability between seizure and non-seizure states.
    """
    def __init__(self, n_channels=MAX_CHANNELS, beta=0.99, num_classes=NUM_CLASSES):
        super().__init__()
        grad = surrogate.fast_sigmoid()

        # ── Convolutional feature extraction ────────────────────────────────
        # kernel (1, 25) operates only along time axis — preserves channel topology
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 25), stride=(1, 5), padding=(0, 12))
        self.bn1   = nn.BatchNorm2d(16)
        # RF layer 1 — gamma band (40 Hz): sensitive to ictal high-freq bursts
        self.rf1   = RFSpike(frequency=40.0, beta=beta, sfreq=NUM_STEPS, spike_grad=grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 10), stride=(1, 2), padding=(0, 4))
        self.bn2   = nn.BatchNorm2d(32)
        # RF layer 2 — alpha band (10 Hz): captures background rhythm disruption
        self.rf2   = RFSpike(frequency=10.0, beta=beta, sfreq=NUM_STEPS, spike_grad=grad)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.bn3   = nn.BatchNorm2d(64)
        # Standard Leaky LIF for third convolutional stage
        self.lif3  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)

        # ── Classifier head ─────────────────────────────────────────────────
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1   = nn.Linear(64, 128)
        self.lif4  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
        self.fc2   = nn.Linear(128, num_classes)
        self.lif5  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)

        self.dropout = nn.Dropout(p=0.3)

    def reset_all(self):
        """Reset all hidden states before each new sequence."""
        self.rf1.reset_state()
        self.rf2.reset_state()
        utils.reset(self)   # resets all snntorch Leaky hidden states

    def forward(self, x):
        """
        Forward pass for one SNN timestep.

        Args:
            x: (batch, 1, n_channels, step_width) — one slice of the spike train

        Returns:
            spk : (batch, num_classes) spike output
            mem : (batch, num_classes) membrane potential
        """
        # Conv block 1 → RF (gamma)
        x = self.bn1(self.conv1(x))
        x = self.rf1(x)                        # returns spike tensor

        # Conv block 2 → RF (alpha)
        x = self.bn2(self.conv2(x))
        x = self.rf2(x)

        # Conv block 3 → Leaky LIF
        x = self.bn3(self.conv3(x))
        x = self.lif3(x)

        # Pool + flatten
        x = self.pool(x)                       # (batch, 64, 1, 1)
        x = x.flatten(1)                       # (batch, 64)
        x = self.dropout(x)

        # FC layers
        x = self.fc1(x)
        x = self.lif4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        spk, mem = self.lif5(x)

        return spk, mem


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Forward Pass and Training Utilities
# ════════════════════════════════════════════════════════════════════════════

def forward_pass(net, spike_train):
    """
    Run the SNN forward pass across all NUM_STEPS timesteps.

    Args:
        net        : EEGSeizureSNN instance
        spike_train: (num_steps, batch, 1, n_channels, step_width) from DM encoding

    Returns:
        spk_rec: (num_steps, batch, num_classes) — spike output at each step
        mem_rec: (num_steps, batch, num_classes) — membrane potential at each step
    """
    spk_rec, mem_rec = [], []
    net.reset_all()

    for t in range(spike_train.shape[0]):
        spk, mem = net(spike_train[t])
        spk_rec.append(spk)
        mem_rec.append(mem)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def compute_accuracy(spk_rec, targets):
    """
    Binary classification accuracy from spike counts.
    Prediction = argmax of total spikes over all timesteps.

    Args:
        spk_rec: (num_steps, batch, num_classes)
        targets: (batch,)
    """
    spike_counts = spk_rec.sum(dim=0)          # (batch, num_classes)
    preds        = spike_counts.argmax(dim=1)  # (batch,)
    return (preds == targets).float().mean().item()


def test_accuracy(net, test_loader):
    """Evaluate model on test set, returning mean accuracy."""
    net.eval()
    accs = []
    with torch.no_grad():
        for data, targets in test_loader:
            data    = data.to(device)
            targets = targets.to(device)

            spike_train = delta_modulation_encoding(data, num_steps=NUM_STEPS)
            spk_rec, _ = forward_pass(net, spike_train)
            accs.append(compute_accuracy(spk_rec, targets))

    net.train()
    return float(np.mean(accs))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Training Loop
# ════════════════════════════════════════════════════════════════════════════

def train(net, train_loader, test_loader):
    """
    Training loop with cross-entropy loss on spike counts.

    Uses mean spike count across timesteps as logits — this is the standard
    approach for binary classification SNNs (Zhang et al., 2024; Li et al., 2024).
    """
    print("=" * 60)
    print("EEG Seizure SNN — Training")
    print(f"  Architecture : EEGSeizureSNN (RF x2 + LIF x3)")
    print(f"  RF Layer 1   : 40 Hz (gamma band)")
    print(f"  RF Layer 2   : 10 Hz (alpha band)")
    print(f"  Encoding     : Delta Modulation (threshold={DELTA_THRESHOLD})")
    print(f"  Time steps   : {NUM_STEPS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Classes      : {NUM_CLASSES} (0=background, 1=seizure)")
    print("=" * 60)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        n_batches  = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data    = data.to(device)
            targets = targets.to(device)

            # Delta modulation encoding: (B,1,C,T) → (steps,B,1,C,step_width)
            spike_train = delta_modulation_encoding(data, num_steps=NUM_STEPS)

            # Forward pass across all timesteps
            spk_rec, mem_rec = forward_pass(net, spike_train)

            # Loss on mean spike count (sum over time, softmax-like logits)
            spike_counts = spk_rec.sum(dim=0)          # (batch, num_classes)
            loss = loss_fn(spike_counts, targets)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — important for stability with RF neurons
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        test_acc = test_accuracy(net, test_loader)
        print(f"\nEpoch {epoch+1} Summary — "
              f"Avg Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}%\n"
              + "-" * 60)

    return net


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Comparison: LIF-only baseline vs RF SNN
# ════════════════════════════════════════════════════════════════════════════

class EEGSeizureSNN_LIFBaseline(nn.Module):
    """
    Identical architecture to EEGSeizureSNN but replaces both RFSpike layers
    with standard snn.Leaky (LIF) neurons. Used as baseline comparison to
    quantify the benefit of the RF resonance mechanism.
    """
    def __init__(self, n_channels=MAX_CHANNELS, beta=0.99, num_classes=NUM_CLASSES):
        super().__init__()
        grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 25), stride=(1, 5), padding=(0, 12))
        self.bn1   = nn.BatchNorm2d(16)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 10), stride=(1, 2), padding=(0, 4))
        self.bn2   = nn.BatchNorm2d(32)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.bn3   = nn.BatchNorm2d(64)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1     = nn.Linear(64, 128)
        self.lif4    = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True)
        self.fc2     = nn.Linear(128, num_classes)
        self.lif5    = snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
        self.dropout = nn.Dropout(p=0.3)

    def reset_all(self):
        """Reset all hidden states before each new sequence."""
        utils.reset(self)

    def forward(self, x):
        x = self.lif1(self.bn1(self.conv1(x)))
        x = self.lif2(self.bn2(self.conv2(x)))
        x = self.lif3(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.lif4(self.fc1(x))
        x = self.dropout(x)
        spk, mem = self.lif5(self.fc2(x))
        return spk, mem


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Dataset ─────────────────────────────────────────────────────────────
    # Set TUSZ_PATH to your local TUSZ v2.0.0 root directory.
    # Structure expected: TUSZ_PATH/[patient_id]/[session_id]/[montage]/*.edf
    TUSZ_PATH = "./tusz_data"   # ← UPDATE THIS PATH

    print(f"Loading TUSZ dataset from: {TUSZ_PATH}")
    full_dataset = TUSZDataset(base_path=TUSZ_PATH, max_channels=MAX_CHANNELS)

    # 80/20 train-test split
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_test  = n_total - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {n_train} | Test samples: {n_test}")

    # ── RF SNN ──────────────────────────────────────────────────────────────
    print("\n--- Training RF SNN (gamma 40 Hz + alpha 10 Hz) ---")
    rf_net = EEGSeizureSNN(n_channels=MAX_CHANNELS).to(device)
    print(rf_net)
    rf_net = train(rf_net, train_loader, test_loader)
    rf_acc = test_accuracy(rf_net, test_loader)

    # ── LIF Baseline ────────────────────────────────────────────────────────
    print("\n--- Training LIF Baseline SNN ---")
    lif_net = EEGSeizureSNN_LIFBaseline(n_channels=MAX_CHANNELS).to(device)
    lif_net = train(lif_net, train_loader, test_loader)
    lif_acc = test_accuracy(lif_net, test_loader)

    # ── Results ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON RESULTS")
    print("=" * 60)
    print(f"RF SNN  (40 Hz + 10 Hz resonance) : {rf_acc  * 100:.2f}%")
    print(f"LIF SNN (baseline, no resonance)   : {lif_acc * 100:.2f}%")
    print("=" * 60)

    # Save both models
    torch.save(rf_net.state_dict(),  "rf_eeg_snn.pt")
    torch.save(lif_net.state_dict(), "lif_eeg_snn.pt")
    print("Models saved: rf_eeg_snn.pt | lif_eeg_snn.pt")