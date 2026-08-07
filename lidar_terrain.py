# -*- coding: utf-8 -*-
"""
lidar_terrain.py - LiDAR 저장 파일에서 지형 고도맵 복원

배경
    시뮬레이터의 Save LiDAR Data 기능은 약 0.2초마다
    LidarData_t{초}_{소수}.csv 를 폴더에 저장한다.
    파일 하나가 반경 최대 149 m, 유효점 1,500여 개를 담으므로
    주행 궤적 샘플링(1회 1점)보다 압도적으로 효율적이다.

실측으로 확인한 규약
    - vertical_angle 은 **양수가 하향**이다 (+22.5 = 아래로 22.5도)
    - 8채널 중 하향 4개만 지면을 찍는다 (상향 4개는 하늘)
    - 점군에서 센서 원점을 역산할 수 있다 (잔차 0.0001 m)
    - 지면 도달거리 = 센서 지상고 / tan(3.214도) ~= 지상고 x 17.8
      Max Distance 설정으로 추가 제한됨

사용
    python lidar_terrain.py <폴더> [--cell 2.0] [--out heightmap]

    from lidar_terrain import LidarTerrain
    lt = LidarTerrain(cell=2.0)
    lt.load_dir("C:/.../LidarData")
    h = lt.height(150.0, 120.0)
    lt.save("heightmap")
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
from typing import Optional, Tuple, List

import numpy as np

MAP_MIN, MAP_MAX = 0.0, 300.0


# ══════════════════════════════════════════════════════════
# 파일 하나 읽기
# ══════════════════════════════════════════════════════════
def read_scan(path: str) -> Optional[dict]:
    """
    LiDAR CSV 한 개를 읽어 점군과 센서 원점을 돌려준다.
    원점은 |점 - 원점| = distance 를 만족하는 값을 최소제곱으로 구한다.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r.get("isDetected") == "True"]
    except (OSError, csv.Error):
        return None
    if len(rows) < 20:
        return None

    try:
        P = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        D = np.array([float(r["distance"]) for r in rows])
        V = np.array([float(r["vertical_angle"]) for r in rows])
    except (KeyError, ValueError):
        return None

    o = solve_origin(P, D)
    return {"path": path, "P": P, "D": D, "V": V, "origin": o,
            "t": parse_time(path)}


def solve_origin(P: np.ndarray, D: np.ndarray, iters: int = 40) -> np.ndarray:
    """
    가우스-뉴턴으로 |P_i - o| = D_i 를 만족하는 o 를 찾는다.
    scipy 없이 동작하도록 직접 구현했다.
    """
    o = np.array([P[:, 0].mean(), P[:, 1].mean() + 2.0, P[:, 2].mean()])
    for _ in range(iters):
        d = P - o
        r = np.linalg.norm(d, axis=1)
        r = np.maximum(r, 1e-9)
        f = r - D                      # 잔차
        J = -d / r[:, None]            # d(residual)/d(o)
        try:
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        o = o + step
        if np.linalg.norm(step) < 1e-9:
            break
    return o


def parse_time(path: str) -> float:
    """LidarData_t0030_48.csv -> 30.48"""
    b = os.path.basename(path)
    try:
        core = b.replace("LidarData_t", "").rsplit(".", 1)[0]
        a, c = core.split("_")
        return float(a) + float(c) / 100.0
    except (ValueError, IndexError):
        return 0.0


# ══════════════════════════════════════════════════════════
# 고도맵 누적
# ══════════════════════════════════════════════════════════
class LidarTerrain:
    """
    여러 스캔을 격자에 누적해 지형 고도맵을 만든다.

    cell        격자 크기 [m]. 작을수록 세밀하나 미관측 셀이 늘어난다
    max_slope   인접 셀 대비 이 기울기를 넘는 점은 물체로 보고 지형에서 제외
    """

    def __init__(self, cell: float = 2.0, size: float = MAP_MAX,
                 keep_lowest: bool = True):
        self.cell = cell
        self.size = size
        self.n = int(size / cell)
        self.keep_lowest = keep_lowest      # 같은 셀이면 낮은 값 채택 (물체 배제)
        self._sum = np.zeros((self.n, self.n))
        self._cnt = np.zeros((self.n, self.n))
        self._min = np.full((self.n, self.n), np.inf)
        self._max = np.full((self.n, self.n), -np.inf)
        self.scans = 0
        self.points = 0
        self.origins: List[np.ndarray] = []

    # ── 누적 ──────────────────────────────────────────────
    def add_scan(self, scan: dict):
        P = scan["P"]
        ix = (P[:, 0] / self.cell).astype(int)
        iz = (P[:, 2] / self.cell).astype(int)
        m = (ix >= 0) & (ix < self.n) & (iz >= 0) & (iz < self.n)
        for x, y, z in zip(ix[m], P[m, 1], iz[m]):
            self._sum[z, x] += y
            self._cnt[z, x] += 1
            if y < self._min[z, x]:
                self._min[z, x] = y
            if y > self._max[z, x]:
                self._max[z, x] = y
        self.scans += 1
        self.points += int(m.sum())
        self.origins.append(scan["origin"])

    def load_dir(self, folder: str, pattern: str = "LidarData_*.csv",
                 verbose: bool = True) -> int:
        files = sorted(glob.glob(os.path.join(folder, pattern)), key=parse_time)
        if not files:
            if verbose:
                print(f"[없음] {folder} 에 {pattern} 파일이 없습니다")
            return 0
        ok = 0
        for i, f in enumerate(files):
            s = read_scan(f)
            if s is None:
                continue
            self.add_scan(s)
            ok += 1
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} 처리  커버리지 {self.coverage*100:.1f}%")
        if verbose:
            print(f"스캔 {ok}개 반영, 점 {self.points:,}개, "
                  f"커버리지 {self.coverage*100:.1f}%")
        return ok

    # ── 조회 ──────────────────────────────────────────────
    @property
    def coverage(self) -> float:
        return float((self._cnt > 0).sum()) / (self.n * self.n)

    def raw_grid(self) -> np.ndarray:
        """
        관측 격자. 미관측은 NaN.
        같은 셀에 여러 점이 있으면 최저값을 쓴다
        (전차·바위 등 물체 상단이 아니라 지면을 얻기 위함).
        """
        g = np.full((self.n, self.n), np.nan)
        m = self._cnt > 0
        if self.keep_lowest:
            g[m] = self._min[m]
        else:
            g[m] = self._sum[m] / self._cnt[m]
        return g

    def object_grid(self) -> np.ndarray:
        """셀별 (최고 - 최저). 값이 크면 물체가 서 있을 가능성"""
        g = np.full((self.n, self.n), np.nan)
        m = self._cnt > 0
        g[m] = self._max[m] - self._min[m]
        return g

    def filled_grid(self, max_r: int = 8, power: float = 2.0) -> np.ndarray:
        """
        미관측 셀을 역거리가중(IDW)으로 보간한 격자.
        max_r 안에 관측 셀이 없으면 NaN 으로 남긴다.
        """
        g = self.raw_grid()
        known = ~np.isnan(g)
        if known.sum() < 3:
            return g
        kz, kx = np.nonzero(known)
        kv = g[known]
        out = g.copy()
        uz, ux = np.nonzero(~known)
        for iz, ix in zip(uz, ux):
            dz = kz - iz
            dx = kx - ix
            d2 = dx * dx + dz * dz
            sel = d2 <= max_r * max_r
            if not sel.any():
                continue
            w = 1.0 / np.maximum(d2[sel], 0.25) ** (power / 2)
            out[iz, ix] = float(np.sum(kv[sel] * w) / np.sum(w))
        return out

    def height(self, x: float, z: float, grid: Optional[np.ndarray] = None
               ) -> Optional[float]:
        g = self.raw_grid() if grid is None else grid
        ix = int(x / self.cell)
        iz = int(z / self.cell)
        if not (0 <= ix < self.n and 0 <= iz < self.n):
            return None
        v = g[iz, ix]
        return None if np.isnan(v) else float(v)

    # ── 품질 평가 ─────────────────────────────────────────
    def holdout_error(self, frac: float = 0.25, seed: int = 0,
                      max_r: int = 8) -> dict:
        """
        관측 셀 일부를 가리고 나머지로 보간해 오차를 추정한다.

        주의: 무작위 셀 홀드아웃은 이웃 셀이 바로 옆에 남아 있어
        오차를 크게 과소평가한다. 여기서는 **블록 단위**로 가려
        실제 보간 상황에 가깝게 만든다.
        """
        rng = np.random.default_rng(seed)
        g = self.raw_grid()
        known = ~np.isnan(g)
        if known.sum() < 50:
            return {"n": 0}
        blk = max(2, max_r // 2)
        nb = int(np.ceil(self.n / blk))          # 격자를 완전히 덮도록 올림
        mask = rng.random((nb, nb)) < frac
        hold = np.kron(mask, np.ones((blk, blk), bool))[:self.n, :self.n]
        hold = hold & known

        train = g.copy()
        train[hold] = np.nan
        kz, kx = np.nonzero(~np.isnan(train))
        if len(kz) < 3:
            return {"n": 0}
        kv = train[kz, kx]

        errs = []
        hz, hx = np.nonzero(hold)
        idx = rng.permutation(len(hz))[:600]
        for i in idx:
            iz, ix = hz[i], hx[i]
            d2 = (kx - ix) ** 2 + (kz - iz) ** 2
            sel = d2 <= max_r * max_r
            if not sel.any():
                continue
            w = 1.0 / np.maximum(d2[sel], 0.25)
            est = float(np.sum(kv[sel] * w) / np.sum(w))
            errs.append(est - g[iz, ix])
        if not errs:
            return {"n": 0}
        e = np.array(errs)
        return {"n": len(e), "rms": float(np.sqrt((e ** 2).mean())),
                "mae": float(np.abs(e).mean()), "max": float(np.abs(e).max()),
                "block_m": blk * self.cell}

    # ── 저장 / 불러오기 ───────────────────────────────────
    def save(self, prefix: str = "heightmap"):
        np.save(f"{prefix}_raw.npy", self.raw_grid())
        np.save(f"{prefix}_filled.npy", self.filled_grid())
        np.save(f"{prefix}_count.npy", self._cnt)
        np.save(f"{prefix}_object.npy", self.object_grid())
        with open(f"{prefix}_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"cell={self.cell}\nn={self.n}\nsize={self.size}\n"
                    f"scans={self.scans}\npoints={self.points}\n"
                    f"coverage={self.coverage:.4f}\n")
        print(f"저장: {prefix}_raw.npy / _filled.npy / _count.npy / _object.npy")

    @staticmethod
    def load_grid(path: str) -> np.ndarray:
        return np.load(path)

    # ── terrain.TerrainMemory 로 이식 ─────────────────────
    def to_terrain_memory(self, filled: bool = True):
        """
        threat.py / auto_aim_bot 이 쓰는 TerrainMemory 객체로 변환한다.
        """
        try:
            from terrain import TerrainMemory
        except ImportError:
            print("[경고] terrain.py 를 찾을 수 없습니다")
            return None
        tm = TerrainMemory(size=self.size, cell=self.cell)
        g = self.filled_grid() if filled else self.raw_grid()
        for iz in range(self.n):
            for ix in range(self.n):
                v = g[iz, ix]
                if not np.isnan(v):
                    tm.add((ix + 0.5) * self.cell, float(v),
                           (iz + 0.5) * self.cell)
        return tm


# ══════════════════════════════════════════════════════════
# 진단
# ══════════════════════════════════════════════════════════
def inspect_dir(folder: str, pattern: str = "LidarData_*.csv", limit: int = 30):
    """스캔 원점이 실제로 이동하는지 확인한다 (갱신 여부 진단)"""
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=parse_time)
    if not files:
        print(f"[없음] {folder}")
        return
    print(f"{'파일':>26s} {'t':>8s} {'유효':>6s} {'원점 (x, z)':>20s} {'이동':>8s}")
    prev = None
    shown = 0
    total_move = 0.0
    for f in files:
        s = read_scan(f)
        if s is None:
            continue
        o = s["origin"]
        mv = 0.0 if prev is None else math.dist((o[0], o[2]), (prev[0], prev[2]))
        total_move += mv
        if shown < limit:
            print(f"{os.path.basename(f):>26s} {s['t']:8.2f} {len(s['P']):6d}  "
                  f"({o[0]:7.2f},{o[2]:8.2f}) {mv:7.2f}m")
            shown += 1
        prev = o
    print(f"\n총 이동 거리 {total_move:.1f} m")
    if total_move < 1.0:
        print("→ 원점이 거의 고정. 전차가 정지해 있었거나 갱신되지 않는 상태")
    else:
        print("→ 원점이 이동하고 있다. 스캔이 정상 갱신됨")


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    folder = sys.argv[1]
    cell = 2.0
    prefix = "heightmap"
    if "--cell" in sys.argv:
        cell = float(sys.argv[sys.argv.index("--cell") + 1])
    if "--out" in sys.argv:
        prefix = sys.argv[sys.argv.index("--out") + 1]
    if "--inspect" in sys.argv:
        inspect_dir(folder)
        sys.exit(0)

    lt = LidarTerrain(cell=cell)
    lt.load_dir(folder)
    if lt.scans == 0:
        sys.exit(1)

    O = np.array(lt.origins)
    print(f"\n스캔 원점 범위  x {O[:,0].min():.1f}~{O[:,0].max():.1f}  "
          f"z {O[:,2].min():.1f}~{O[:,2].max():.1f}")
    g = lt.raw_grid()
    v = g[~np.isnan(g)]
    if len(v):
        print(f"고도 {v.min():.2f} ~ {v.max():.2f} m")
    f = lt.filled_grid()
    print(f"커버리지  관측 {lt.coverage*100:.1f}%  →  보간 후 "
          f"{(~np.isnan(f)).sum()/(lt.n*lt.n)*100:.1f}%")

    e = lt.holdout_error()
    if e.get("n"):
        print(f"\n보간 오차 (블록 {e['block_m']:.0f} m 홀드아웃, n={e['n']})")
        print(f"  RMS {e['rms']:.3f} m   MAE {e['mae']:.3f} m   최대 {e['max']:.3f} m")

    lt.save(prefix)
