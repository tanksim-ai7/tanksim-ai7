# -*- coding: utf-8 -*-
"""
fire_module.py - 사격통제 모듈 (팀 통합용)

역할 분담
    경로팀   moveWS / moveAD  (차체 이동·선회)
    인식팀   표적 식별
    사격팀   turretQE / turretRF / fire   <- 이 모듈

    이 모듈은 이동 명령을 내지 않는다. 포탑과 사격만 담당한다.

auto_aim_bot_v6.py 에서 추출한 것
    Flask 서버, 기동(Maneuver), 대시보드, 적봇 서버는 제외했다.
    남긴 것은 사격에 필요한 최소 경로다.
        표적 추적(TargetTracker) -> 조준해(FireControl) -> 포탑 명령
        착탄 되먹임(BiasEstimator) -> 잔여 모형오차 보정
        사격 로그(ShotLog 축소판)

필요 파일
    fire_control.py   (팀원 버전, 수정 없이 그대로)

사용법
    from fire_module import FireModule

    fm = FireModule()

    # 매 /info 마다
    fm.on_info(info_json)

    # 매 /get_action 마다 - 포탑/사격 명령만 반환
    cmd = fm.get_turret_command(
        my_vel=(vx, 0.0, vz),        # 경로팀이 내는 이동에 따른 속도
        body_rate_dps=body_rate,     # 경로팀이 내는 차체 선회 각속도
        hull_settled=(my_speed < 0.3 and abs(body_rate) < 1e-6),
    )
    # cmd = {"turretQE": {...}, "turretRF": {...}, "fire": bool}

    # 매 /update_bullet 마다
    fm.on_impact(bullet_json)

주의
    body_rate_dps 와 my_vel 을 경로팀에서 정확히 받아야 한다.
    포탑이 비안정화라 차체가 움직이면 발사 순간 포탑 월드각이 틀어지는데,
    이 두 값으로 그만큼을 미리 보정한다. 넘기지 않으면 명중률이 떨어진다.
"""
import math
from typing import Optional, Tuple, Dict

from fire_control import (Ballistics, FireControl, TurretParams, TargetSize,
                          TargetTracker, MotionLimits, BiasEstimator,
                          impact_aspect, aspect_zone, bearing, ang_diff,
                          dist2d, optimal_range)

Vec3 = Tuple[float, float, float]

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
CFG = {
    "reload_s": 6.9,          # 재장전 [s]
    "moving_fire": True,      # 차체 이동 중 사격 허용
    "use_bias": True,         # 착탄 되먹임 보정 사용
    "p_hit_min": 0.0,         # 0 이면 확률 게이트 미사용
    "halt_speed": 0.30,       # 이 속도 이하를 정지로 본다 [m/s]
    "min_target_speed": 0.40,  # 이보다 빠르면 표적 방위를 리드에 반영
}


def _norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Telemetry:
    """/info 원문에서 사격에 필요한 값만 뽑는다"""

    def __init__(self):
        self.t = None
        self.my = None
        self.enemy = None
        self.body_x = 0.0
        self.turret_x = 0.0
        self.turret_y = 0.0
        self.enemy_body_x = 0.0
        self.body_y = 0.0          # 차체 피치
        self.body_z = 0.0          # 차체 롤
        self.my_speed = 0.0
        self.enemy_speed = 0.0
        self.my_hp = None
        self.enemy_hp = None

    def update(self, r: dict):
        if not r:
            return
        t = r.get("time")
        if t is not None:
            self.t = float(t)
        p = r.get("playerPos")
        if isinstance(p, dict):
            self.my = (float(p.get("x", 0)), float(p.get("y", 0)),
                       float(p.get("z", 0)))
        e = r.get("enemyPos")
        if isinstance(e, dict):
            self.enemy = (float(e.get("x", 0)), float(e.get("y", 0)),
                          float(e.get("z", 0)))
        f = lambda k, d: float(r.get(k, d) or 0.0)
        self.body_x = f("playerBodyX", self.body_x)
        self.turret_x = f("playerTurretX", self.turret_x)
        self.turret_y = f("playerTurretY", self.turret_y)
        self.enemy_body_x = f("enemyBodyX", self.enemy_body_x)
        self.body_y = _norm180(f("playerBodyY", self.body_y))
        self.body_z = _norm180(f("playerBodyZ", self.body_z))
        self.my_speed = f("playerSpeed", 0.0)
        self.enemy_speed = f("enemySpeed", 0.0)
        if r.get("playerHealth") is not None:
            self.my_hp = r.get("playerHealth")
        if r.get("enemyHealth") is not None:
            self.enemy_hp = r.get("enemyHealth")

    @property
    def dist(self) -> Optional[float]:
        if self.my is None or self.enemy is None:
            return None
        return dist2d(self.my, self.enemy)

    @property
    def ready(self) -> bool:
        return (self.my is not None and self.enemy is not None
                and self.t is not None)


class ShotLog:
    """발사와 착탄을 짝지어 기록한다. 분석과 되먹임에 쓰인다."""

    def __init__(self):
        self.records = []
        self.pending = None
        self.fired = 0
        self.hits = 0
        self.unmatched = 0

    def on_fire(self, tm: Telemetry, sol, trk=None):
        if sol is None or not getattr(sol, "valid", False):
            return
        if self.pending is not None:
            self.unmatched += 1
        self.fired += 1
        self.pending = {
            "id": self.fired,
            "t": round(tm.t, 2) if tm.t is not None else None,
            "fire_pos": tm.my,
            "target_pos": tm.enemy,
            "aim_point": getattr(sol, "aim_point", None),
            "dist": round(getattr(sol, "distance", 0.0), 2),
            "tof": round(getattr(sol, "flight", 0.0), 3),
            "p_hit": round(getattr(sol, "p_hit", 0.0), 3),
            "bearing": round(getattr(sol, "bearing", 0.0), 3),
            "elev": round(getattr(sol, "elevation", 0.0), 3),
            "enemy_speed": round(tm.enemy_speed, 2),
            "my_speed": round(tm.my_speed, 2),
            "hull_pitch": round(tm.body_y, 2),
            "hull_roll": round(tm.body_z, 2),
            "est_speed": round(getattr(trk, "speed", 0.0), 2) if trk else None,
            "impact": None, "kind": None, "zone": None,
            "miss": None, "drift_deg": None,
        }

    def on_impact(self, d: dict, tm: Telemetry, bias=None):
        rec = self.pending
        if rec is None:
            return
        self.pending = None
        x, y, z = d.get("x"), d.get("y"), d.get("z")
        if x is None:
            return
        ip = (float(x), float(y or 0.0), float(z or 0.0))
        raw = str(d.get("hit", "")).lower()
        kind = "tank" if ("enemy" in raw or "tank" in raw) else "terrain"
        rec["impact"] = ip
        rec["kind"] = kind
        if kind == "tank":
            self.hits += 1

        # 비행 중 표적이 얼마나 방향을 바꿨는가 (예측 실패의 직접 지표)
        if tm.enemy and rec.get("target_pos"):
            dx = tm.enemy[0] - rec["target_pos"][0]
            dz = tm.enemy[2] - rec["target_pos"][2]
            moved = math.hypot(dx, dz)
            if moved > 0.3:
                rec["drift_deg"] = round(_norm180(
                    math.degrees(math.atan2(dx, dz)) - tm.enemy_body_x), 2)

        if tm.enemy:
            rec["miss"] = round(math.hypot(ip[0] - tm.enemy[0],
                                           ip[2] - tm.enemy[2]), 3)
        if rec.get("fire_pos"):
            asp = impact_aspect(rec["fire_pos"], ip, tm.enemy_body_x)
            rec["aspect"] = round(asp, 1)
            rec["zone"] = aspect_zone(asp)

        # 지면 착탄은 조준점과 직접 비교되므로 되먹임에 쓴다
        if (bias is not None and CFG["use_bias"] and kind == "terrain"
                and rec.get("aim_point") and rec.get("fire_pos")):
            try:
                # expected_long 은 '일부러 길게 조준한 양'이다.
                # 이 모듈은 lon_shift 를 쓰지 않으므로 0 을 넘긴다.
                bias.observe(rec["fire_pos"], rec["aim_point"], ip, 0.0)
            except Exception:
                pass

        self.records.insert(0, rec)
        del self.records[500:]

    @property
    def hit_rate(self) -> float:
        n = sum(1 for r in self.records if r["kind"] in ("tank", "terrain"))
        return (100.0 * self.hits / n) if n else 0.0


class FireModule:
    """
    포탑 조준과 사격만 담당한다.

    경로팀이 차체를 움직이므로, 그 운동을 my_vel 과 body_rate_dps 로
    받아 발사 시점 보정에 쓴다.
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg:
            CFG.update(cfg)
        self.bal = Ballistics()
        self.bias = BiasEstimator()
        self.fc = FireControl(self.bal, TurretParams(),
                              target=TargetSize(), bias=self.bias,
                              reload_s=CFG["reload_s"])
        self.trk = TargetTracker(limits=MotionLimits())
        self.tm = Telemetry()
        self.log = ShotLog()
        self.ctrl_dt = 0.14
        self._last_t = None
        self._last_sol = None

    # ── /info ────────────────────────────────────────────
    def on_info(self, info: dict):
        self.tm.update(info)
        t = self.tm.t
        if t is not None:
            if self._last_t is not None:
                dt = t - self._last_t
                if 0.02 < dt < 1.0:
                    self.ctrl_dt = 0.2 * dt + 0.8 * self.ctrl_dt
            self._last_t = t
        # 표적 추적기 갱신
        if self.tm.enemy is not None and t is not None:
            self.trk.update(t, self.tm.enemy)

    # ── /get_action ──────────────────────────────────────
    def get_turret_command(self,
                           my_vel: Vec3 = (0.0, 0.0, 0.0),
                           body_rate_dps: float = 0.0,
                           hull_settled: Optional[bool] = None,
                           inhibit_fire: bool = False) -> Dict:
        """
        포탑 명령과 사격 여부만 반환한다. 이동 명령은 포함하지 않는다.

        my_vel        경로팀 이동에 따른 자기 속도 벡터 [m/s]
        body_rate_dps 경로팀 선회에 따른 차체 각속도 [deg/s]
        hull_settled  차체 정지 여부. None 이면 속도로 자동 판정
        inhibit_fire  상위 판단으로 사격을 막을 때 True
        """
        idle = {"turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False}
        if not self.tm.ready:
            return idle

        if hull_settled is None:
            hull_settled = (self.tm.my_speed <= CFG["halt_speed"]
                            and abs(body_rate_dps) < 1e-6)

        tgt_head = (self.tm.enemy_body_x
                    if self.tm.enemy_speed > CFG["min_target_speed"] else None)

        prev = self.fc.last_fire_t
        out = self.fc.update(
            my_pos=self.tm.my,
            turret_x=self.tm.turret_x, turret_y=self.tm.turret_y,
            target_pos=self.tm.enemy, tracker=self.trk,
            target_heading=tgt_head,
            sim_time=self.tm.t or 0.0,
            hull_settled=hull_settled,
            my_vel=my_vel, body_rate_dps=body_rate_dps,
            allow_moving_fire=CFG["moving_fire"],
            inhibit_fire=inhibit_fire)

        sol = self.fc.last_solution
        self._last_sol = sol
        # 예측 오차 학습 예약
        if sol is not None and getattr(sol, "valid", False):
            try:
                self.trk.enqueue_prediction(sol.flight + self.ctrl_dt)
            except Exception:
                pass

        if out.get("fire") and self.fc.last_fire_t != prev:
            self.log.on_fire(self.tm, sol, self.trk)

        return {"turretQE": out.get("turretQE", idle["turretQE"]),
                "turretRF": out.get("turretRF", idle["turretRF"]),
                "fire": bool(out.get("fire"))}

    # ── /update_bullet ───────────────────────────────────
    def on_impact(self, bullet: dict):
        self.log.on_impact(bullet or {}, self.tm, self.bias)

    # ── 조회 ─────────────────────────────────────────────
    def status(self) -> dict:
        sol = self._last_sol
        d = self.tm.dist
        rmin = self.bal.min_range()
        rmax = self.bal.max_range()[0] if self.tm.my else None
        return {
            "state": self.fc.state,
            "dist": round(d, 1) if d else None,
            "in_envelope": (d is not None and rmax is not None
                            and rmin <= d <= rmax),
            "envelope": [round(rmin, 1), round(rmax, 1)] if rmax else None,
            "aim_point": getattr(sol, "aim_point", None) if sol else None,
            "p_hit": round(getattr(sol, "p_hit", 0.0), 3) if sol else None,
            "tof": round(getattr(sol, "flight", 0.0), 2) if sol else None,
            "fired": self.log.fired, "hits": self.log.hits,
            "hit_rate": round(self.log.hit_rate, 1),
            "bias": {"range": round(self.bias.range_bias, 3),
                     "bearing": round(self.bias.bearing_bias, 4),
                     "n": self.bias.n},
            "reload_left": round(max(0.0, CFG["reload_s"] -
                                     ((self.tm.t or 0) -
                                      (self.fc.last_fire_t or -99))), 1),
        }

    def suggest_range(self) -> Optional[float]:
        """
        경로팀에 넘길 권장 교전 거리 [m].

        상충 관계
            가까울수록  비행시간이 짧아 예측오차가 작다 (오차 ~ t^2)
            멀수록      필요한 포탑 각속도가 작아 추적 여유가 생긴다
        표적 횡단 속도로 그 절충점을 찾는다.
        """
        try:
            v_cross = self._cross_speed()
            tgt = TargetSize()
            r, _ = optimal_range(self.bal, self.trk,
                                 tgt.width * 0.5, tgt.length * 0.5,
                                 v_cross, self.fc.t)
            return round(r, 1)
        except Exception:
            return None

    def _cross_speed(self) -> float:
        """표적 속도 중 시선에 수직한 성분 [m/s]. 포탑 추적 부담을 결정한다."""
        if self.tm.my is None or self.tm.enemy is None:
            return 0.0
        v = self.tm.enemy_speed
        if v <= 0.05:
            return 0.0
        los = bearing(self.tm.my, self.tm.enemy)
        rel = math.radians(_norm180(self.tm.enemy_body_x - los))
        return abs(v * math.sin(rel))

    def export_csv(self) -> str:
        import csv
        import io
        cols = ["id", "t", "dist", "tof", "p_hit", "miss", "kind", "zone",
                "drift_deg", "enemy_speed", "my_speed",
                "hull_pitch", "hull_roll", "est_speed"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        for r in reversed(self.log.records):
            w.writerow([r.get(c) for c in cols])
        return buf.getvalue()
