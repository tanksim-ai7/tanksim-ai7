# -*- coding: utf-8 -*-
"""
terrain.py - 지형 고도 기억 및 탄도 차폐 판정

왜 필요한가
    평지에서는 탄도 역해만으로 충분하지만, 굴곡 지형에서는
      (1) 표적이 언덕 위/아래에 있어 고도차가 생기고
      (2) 중간의 언덕이 탄을 가로막는다
    두 번째가 "터무니없는 곳에 쏘는" 현상의 원인이다.
    포구에서 표적까지의 포물선이 지형을 통과하는지 검사해야 한다.

지형 고도를 어디서 얻는가
    LiDAR 점군이 갱신되지 않는 문제가 있어(초기 스캔 고정) 대신
    주행하며 관측되는 값들을 누적한다.
      - 아군 위치        playerPos (x, y, z)  -> y 가 그 지점의 지면고
      - 적 위치          enemyPos
      - 착탄점           /update_bullet 의 (x, y, z)
      - LiDAR 점군       사용 가능하면 함께 누적
    관측되지 않은 셀은 주변에서 IDW 보간하고, 그마저 없으면 '미지'로 둔다.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

Vec3 = Tuple[float, float, float]


class TerrainMemory:
    """
    (x, z) 격자에 관측 고도를 누적한다.
    cell 이 크면 커버리지가 빨리 차지만 언덕 형상이 뭉개진다.
    """

    def __init__(self, size: float = 300.0, cell: float = 4.0,
                 search_r: int = 4):
        self.size = size
        self.cell = cell
        self.n = int(size / cell)
        self.search_r = search_r          # 보간 시 탐색 반경(셀)
        self._sum = [0.0] * (self.n * self.n)
        self._cnt = [0] * (self.n * self.n)
        self.samples = 0

    # ── 기록 ──────────────────────────────────────────────
    def _idx(self, x: float, z: float) -> Optional[int]:
        ix = int(x / self.cell)
        iz = int(z / self.cell)
        if 0 <= ix < self.n and 0 <= iz < self.n:
            return iz * self.n + ix
        return None

    def add(self, x: float, y: float, z: float, w: float = 1.0):
        i = self._idx(x, z)
        if i is None:
            return
        self._sum[i] += y * w
        self._cnt[i] += w
        self.samples += 1

    def add_point(self, p: Vec3, w: float = 1.0):
        self.add(p[0], p[1], p[2], w)

    def add_lidar(self, points: List[dict]):
        for p in points:
            if not p.get("isDetected"):
                continue
            q = p.get("position") or {}
            if q.get("x") is not None:
                self.add(q["x"], q["y"], q["z"], 0.5)

    # ── 조회 ──────────────────────────────────────────────
    def raw(self, x: float, z: float) -> Optional[float]:
        i = self._idx(x, z)
        if i is None or self._cnt[i] == 0:
            return None
        return self._sum[i] / self._cnt[i]

    def height(self, x: float, z: float) -> Optional[float]:
        """관측값이 없으면 주변에서 역거리가중 보간. 그래도 없으면 None."""
        v = self.raw(x, z)
        if v is not None:
            return v
        ix, iz = int(x / self.cell), int(z / self.cell)
        num = den = 0.0
        for dz in range(-self.search_r, self.search_r + 1):
            for dx in range(-self.search_r, self.search_r + 1):
                jx, jz = ix + dx, iz + dz
                if not (0 <= jx < self.n and 0 <= jz < self.n):
                    continue
                j = jz * self.n + jx
                if self._cnt[j] == 0:
                    continue
                d2 = dx * dx + dz * dz
                if d2 == 0:
                    continue
                w = 1.0 / d2
                num += (self._sum[j] / self._cnt[j]) * w
                den += w
        return num / den if den > 0 else None

    @property
    def coverage(self) -> float:
        c = sum(1 for v in self._cnt if v > 0)
        return c / (self.n * self.n)

    def export_grid(self):
        """(n, n) 리스트. 미관측은 None"""
        out = []
        for iz in range(self.n):
            row = []
            for ix in range(self.n):
                i = iz * self.n + ix
                row.append(self._sum[i] / self._cnt[i] if self._cnt[i] else None)
            out.append(row)
        return out


# ══════════════════════════════════════════════════════════
# 탄도 차폐 판정
# ══════════════════════════════════════════════════════════
class TrajectoryCheck:
    """
    포구에서 표적까지의 포물선이 지형에 막히는지 검사한다.

    반환값
        ok        통과 가능
        blocked   지형에 막힘 (막히는 지점과 여유고 포함)
        unknown   경로상 지형 정보가 부족
    """

    def __init__(self, terrain: TerrainMemory, samples: int = 24,
                 clearance: float = 1.5, min_known: float = 0.45,
                 tail_clearance: float = 0.05):
        self.terrain = terrain
        self.samples = samples
        self.clearance = clearance          # 경로 초반에 요구하는 여유 [m]
        self.tail_clearance = tail_clearance  # 표적 근처에서 요구하는 여유 [m]
        # 근거리 사격은 부각이라 탄이 지면에 바싹 붙어 날아간다.
        # 꼬리 여유를 크게 잡으면 평지에서도 막힘으로 오판한다.
        self.min_known = min_known          # 이 비율 이상 관측되어야 판정한다

    def _required(self, frac: float) -> float:
        """
        요구 여유고는 표적에 가까울수록 작아진다.
        탄은 표적에 닿기 직전 자연히 지면 가까이 내려오므로,
        고정 여유를 쓰면 평지에서도 '막힘'으로 오판한다.
        """
        return max(self.tail_clearance, self.clearance * (1.0 - frac))

    def check_along(self, fire_pos: Vec3, muzzle_h: float,
                    bearing_deg: float, theta_deg: float,
                    v: float, g: float, target_dist: float) -> dict:
        x0, y0, z0 = fire_pos
        muzzle_y = y0 + muzzle_h
        th = math.radians(theta_deg)
        vx = v * math.cos(th)
        vy = v * math.sin(th)
        if vx < 1e-6:
            return {"status": "unknown", "min_clear": None, "at": None}

        b = math.radians(bearing_deg)
        ux, uz = math.sin(b), math.cos(b)

        known = 0
        min_clear = None      # 실제 여유고 (표시용)
        min_margin = None     # 요구치 대비 여유 (판정용)
        min_at = None
        # 발사 직후 근거리는 차체 주변이라 제외 (0.1 ~ 0.95 구간)
        for i in range(self.samples + 1):
            frac = 0.10 + (0.95 - 0.10) * i / self.samples
            d = target_dist * frac
            t = d / vx
            y_traj = muzzle_y + vy * t - 0.5 * g * t * t
            gx, gz = x0 + ux * d, z0 + uz * d
            gy = self.terrain.height(gx, gz)
            if gy is None:
                continue
            known += 1
            clear = y_traj - gy
            margin = clear - self._required(frac)
            if min_margin is None or margin < min_margin:
                min_margin, min_clear, min_at = margin, clear, d

        ratio = known / (self.samples + 1)
        if ratio < self.min_known:
            return {"status": "unknown", "min_clear": min_clear,
                    "margin": min_margin, "at": min_at, "known": ratio}
        if min_margin is not None and min_margin < 0.0:
            return {"status": "blocked", "min_clear": min_clear,
                    "margin": min_margin, "at": min_at, "known": ratio}
        return {"status": "ok", "min_clear": min_clear,
                "margin": min_margin, "at": min_at, "known": ratio}


# ══════════════════════════════════════════════════════════
# 차체 기울기 보정
# ══════════════════════════════════════════════════════════
def hull_tilt_in_direction(body_pitch: float, body_roll: float,
                           rel_azimuth_deg: float) -> float:
    """
    차체 피치/롤이 특정 방위(차체 기준 상대각)에서 만드는 기울기.

    포탑 앙각이 '차체 기준'으로 측정된다면
        실제 월드 앙각 = turret_y + hull_tilt_in_direction(...)
    포탑 앙각이 '월드 기준'이라면 보정이 필요 없다.
    어느 쪽인지는 경사지 실측(P5)으로 확인해야 한다.
    """
    r = math.radians(rel_azimuth_deg)
    return body_pitch * math.cos(r) + body_roll * math.sin(r)
