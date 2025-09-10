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
