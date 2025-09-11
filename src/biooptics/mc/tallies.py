import numpy as np

class Tallies:
    def __init__(self, track_trajectories=False, max_tracks=100):
        self.absorbed = 0.0
        self.reflected = 0.0   # S1 目前为 0，占位
        self.N = 0
        self.track_trajectories = track_trajectories
        self.max_tracks = max_tracks
        self.tracks = []  # 每条轨迹是 [(x,z), (x,z), ...]
        # 方向/步长采样展示
        self.sampled_uz = []
        self.sampled_steps = []

    def add_absorb(self, a): self.absorbed += float(a)
    def add_reflect(self, r): self.reflected += float(r)
    def bump(self): self.N += 1

    def log_step(self, s): self.sampled_steps.append(float(s))
    def log_uz(self, uz): self.sampled_uz.append(float(uz))

    def start_track(self):
        if self.track_trajectories and len(self.tracks) < self.max_tracks:
            self.tracks.append([])

    def log_pos(self, x, z):
        if self.track_trajectories and self.tracks:
            self.tracks[-1].append((float(x), float(z)))

    def results(self):
        R_d = self.reflected / max(self.N, 1)
        A   = self.absorbed / max(self.N, 1)
        return R_d, A


# --- S2 energy tally for small-run checks ---
class EnergyTally:
    def __init__(self, n_layers: int):
        import numpy as np
        self.R_d = 0.0  # 顶出射
        self.T_d = 0.0  # 底出射
        self.A_layers = np.zeros(int(n_layers), dtype=float)

# 一个轻量的可视化采样器，字段名和 S1 的 Tallies 一致
# src/biooptics/mc/tallies.py 末尾

class VizTallies:
    def __init__(self, track_trajectories=True, max_tracks=1, max_points=20000, step_stride=3):
        self.sampled_steps = []
        self.sampled_uz = []
        self.tracks = [[]] if track_trajectories else None
        self._track_on = track_trajectories
        self._max_points = int(max_points)
        self._step_stride = int(step_stride)
        self._n_points = 0
        self._step_count = 0

    def _ok(self):
        return self._n_points < self._max_points

    def log_step(self, s: float):
        self._step_count += 1
        if self._ok() and (self._step_count % self._step_stride == 0):
            self.sampled_steps.append(float(s))
            self._n_points += 1

    def log_uz(self, uz: float):
        if self._ok():
            self.sampled_uz.append(float(uz))
            self._n_points += 1

    def log_pos(self, x: float, z: float):
        if self._track_on and self.tracks is not None and self._ok():
            self.tracks[0].append((float(x), float(z)))
            self._n_points += 1
