# -*- coding: utf-8 -*-
"""
viz3d.py — 전술 3D 상황도  (주소는 아래 VIEW_PREFIX 가 정한다)

설계 원칙
    · 다른 팀 코드를 한 줄도 고치지 않는다.
    · fm / drive_controller / detect 객체에서 값을 "읽기만" 한다.
      사격 판정, 경로 계획, 객체 인식에 어떤 영향도 주지 않는다.
    · 이 파일에서 예외가 나도 서버가 죽지 않는다. 전부 감싼다.
    · Blueprint 이름을 'viz3d' 으로 고유하게 두어
      다른 사람의 /view3d 와 엔드포인트가 충돌하지 않는다.

붙이는 방법 (ally-controller.py 맨 아래)

    try:
        from viz3d import attach_viz_taek
        attach_viz_taek(app, fm=fm, drive=drive_controller, detect=tskijun)
    except ImportError:
        pass
"""

import base64
import json
import math
import os
import threading
import time
import urllib.request
import zlib

from flask import Blueprint, Response, jsonify, request

# ── 화면 주소 ────────────────────────────────────────────
#  2026-09-08  주소를 바꾸려면 이 한 줄만 고치면 된다.
#    예)  "/view3d"   ->  http://localhost:5000/view3d
#         "/fcs"      ->  http://localhost:5000/fcs
#  화면 JS 는 자기가 열린 주소(location.pathname)에서 접두어를 읽으므로
#  여기만 바꾸면 /state · /static-data · /reset-fire 호출도 같이 따라간다.
#  Blueprint 이름("viz3d")은 내부 식별자다. 주소와 무관하니 그대로 둔다.
#
#  2026-09-08  "/view3d/taek" -> "/view3d" 로 줄였다.
VIEW_PREFIX = "/view3d"

bp = Blueprint("viz3d", __name__, url_prefix=VIEW_PREFIX)

_REF = {"fm": None, "drive": None, "detect": None}
_CACHE = {}
_LOCK = threading.Lock()
_HERE = os.path.dirname(os.path.abspath(__file__))

MAP_SPAN = 300.0            # 맵 한 변 [m]
TRAIL_MAX = 500             # 이동 궤적 보관 점 수
SHOT_MAX = 40               # 화면에 남기는 사격 이력


def attach_viz_taek(app, fm=None, drive=None, detect=None):
    """Flask app 에 뷰를 붙인다. 이미 붙어 있으면 참조만 갱신한다."""
    _REF["fm"] = fm
    _REF["drive"] = drive
    _REF["detect"] = detect
    if "viz3d" not in app.blueprints:
        app.register_blueprint(bp)
        print("[viz3d] 전술 3D 상황도 준비 완료"
              "  ->  http://localhost:5000%s" % VIEW_PREFIX)
        # 2026-09-09  적 사격을 5100 번에서 가져오는 스레드
        threading.Thread(target=_foe_poll, daemon=True).start()
        print("[viz3d] 적 사격 수집 시작  <-  %s" % FOE_URL)
    return bp
attach_viz = attach_viz_taek


# ============================================================
# 공통 보조
# ============================================================

def _g(obj, name, default=None):
    try:
        v = getattr(obj, name, default)
        return default if v is None else v
    except Exception:
        return default


def _num(v, nd=2):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, nd)
    except Exception:
        return None


def _xz(p):
    try:
        if p is None:
            return None
        if len(p) >= 3:
            return [_num(p[0]), _num(p[2])]
        if len(p) == 2:
            return [_num(p[0]), _num(p[1])]
    except Exception:
        pass
    return None


def _pack(a, dtype):
    """넘파이 배열을 zlib + base64 로 압축. 300x300 도 가볍게 보낸다."""
    return base64.b64encode(
        zlib.compress(a.astype(dtype).tobytes(), 9)).decode()


# ============================================================
# 이동 궤적
# ============================================================

class _Trail:
    def __init__(self):
        self.my = []
        self.enemy = []
        self._last_t = None

    def push(self, t, my, enemy):
        try:
            if t is None or (self._last_t is not None and t < self._last_t):
                self.my.clear()
                self.enemy.clear()          # 에피소드 재시작 감지
            self._last_t = t
            for buf, p in ((self.my, my), (self.enemy, enemy)):
                if not p:
                    continue
                q = [round(float(p[0]), 1), round(float(p[2]), 1)]
                if buf and abs(buf[-1][0] - q[0]) < 0.4 \
                        and abs(buf[-1][1] - q[1]) < 0.4:
                    continue
                buf.append(q)
                del buf[:-TRAIL_MAX]
        except Exception:
            pass


_trail = _Trail()


# ============================================================
# 정적 데이터 — 지형 / 맵 장애물
# ============================================================

def _load_terrain():
    """
    risk_layers.npz 를 300x300 그대로 압축해 보낸다.
    height 는 uint16 정규화, slope/exposure/blocked 는 uint8.
    """
    if "terrain" in _CACHE:
        return _CACHE["terrain"]

    out = {"ok": False, "n": 0, "span": MAP_SPAN,
           "lo": 0.0, "hi": 1.0, "water": 0.0,
           "hm": "", "sl": "", "ex": "", "bl": "", "ft": ""}
    try:
        import numpy as np
        d = np.load(os.path.join(_HERE, "move", "risk_layers.npz"))
        h = np.nan_to_num(d["height"].astype("float32"))
        n = int(h.shape[0])
        lo, hi = float(h.min()), float(h.max())
        rng = max(1e-6, hi - lo)

        out.update({
            "ok": True, "n": n, "lo": lo, "hi": hi,
            "water": lo + rng * 0.30,          # 수면 기준선
            "hm": _pack((h - lo) / rng * 65535.0, "<u2"),
        })
        if "slope_cost" in d.files:
            out["sl"] = _pack(np.clip(d["slope_cost"], 0, 1) * 255, "u1")
        if "exposure" in d.files:
            e = d["exposure"].astype("float32")
            e = (e - e.min()) / max(1e-6, e.max() - e.min())
            out["ex"] = _pack(e * 255, "u1")
        if "blocked" in d.files:
            out["bl"] = _pack(d["blocked"], "u1")

        # ── 2026-08-26  사격 진지 적합도 (ft) ──────────────────────
        #
        #   "그 자리에 전차를 세우면 얼마나 기우는가" 를 도 단위로 담는다.
        #   점의 경사가 아니라 차체가 놓이는 5 m 반경 안의 최대 경사다.
        #   전차는 6.3 x 3.3 m 면이라 면 전체가 평평해야 안 기운다.
        #
        #   깃발을 마우스로 찍기 때문에 눈으로 보이는 안내가 필요하다.
        #   ally-controller 의 자동 평지 보정과 같은 기준을 쓴다
        #   (flat_snap.py — FOOT_R 5 m).
        try:
            span = MAP_SPAN / (n - 1)
            gz, gx = np.gradient(h, span, span)
            slope = np.degrees(np.arctan(np.hypot(gx, gz)))
            r = max(1, int(round(5.0 / span)))
            foot = slope.copy()
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dz * dz > r * r:
                        continue
                    foot = np.maximum(foot,
                                      np.roll(np.roll(slope, dz, 0), dx, 1))
            out["ft"] = _pack(np.clip(foot, 0, 255), "u1")
        except Exception as e:
            print("[viz3d] 진지 적합도 계산 실패:", e)

        print("[viz3d] 지형 %dx%d  %.1f~%.1f m  적재" % (n, n, lo, hi))
    except Exception as e:
        print("[viz3d] 지형 적재 실패:", e)
    _CACHE["terrain"] = out
    return out


def _load_map_obstacles():
    """DemoMap.map(JSON) 원본 장애물. '정답지'이므로 레이어로 분리해 둔다."""
    if "obstacles" in _CACHE:
        return _CACHE["obstacles"]
    out = []
    try:
        for fn in ("DemoMap.map", "test_Mine_pathfind_1 (1).map"):
            path = os.path.join(_HERE, fn)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for o in data.get("obstacles", []):
                p = o.get("position", {})
                name = str(o.get("prefabName", "?"))
                kind = "".join(c for c in name if c.isalpha()) or "?"
                out.append({"kind": kind,
                            "x": _num(p.get("x"), 1),
                            "y": _num(p.get("y"), 1),
                            "z": _num(p.get("z"), 1)})
            break
    except Exception:
        out = []
    _CACHE["obstacles"] = out
    return out


# ============================================================
# 실시간 상태
# ============================================================

# 전차 차체 크기 (dstar_lite_planner_cost.py 의 get_bb_corners 인자와 동일)
TANK_W, TANK_L = 3.303, 6.339


def _static_index():
    """맵 원본의 정지 오브젝트 좌표 목록. 적 전차 판별에 쓴다."""
    if "static_idx" in _CACHE:
        return _CACHE["static_idx"]
    idx = [(o["x"], o["z"], o["kind"]) for o in _load_map_obstacles()
           if o["x"] is not None and o["z"] is not None]
    _CACHE["static_idx"] = idx
    return idx


def _planner_obstacles(drive):
    """
    /update_obstacle 로 들어와 플래너가 '실제로 아는' 장애물.

    적 전차 판별
        dstar_lite_planner_cost.py 의 enemy_tank_list 등이 전부 비어 있어
        모든 장애물이 'nature' 로 분류된다. 그래서 여기서 따로 추정한다.

        조건 ① 크기가 전차 차체(3.30 x 6.34 m)의 회전 AABB 범위 안
              ② 맵 원본(DemoMap.map)의 정지 오브젝트와 겹치지 않음
        둘을 함께 만족하면 '움직이는 전차 크기 물체' = 적 전차로 본다.
    """
    try:
        planner = _g(drive, "planner") or _g(drive, "path_planner")
        rects = _g(planner, "obstacle_rectangles", []) or []
        static = _static_index()

        # 회전에 따른 AABB 한계
        dmin = TANK_W * 0.85
        dmax = (TANK_W + TANK_L) / math.sqrt(2) * 1.12
        amin, amax = TANK_W * TANK_L * 0.80, (dmax ** 2) * 1.05

        out = []
        for r in rects[:500]:
            try:
                x0, x1 = float(r.x_min), float(r.x_max)
                z0, z1 = float(r.z_min), float(r.z_max)
                w, dep = x1 - x0, z1 - z0
                cx, cz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
                t = _g(r, "type", "nature")

                # ── 적 전차 추정 ──
                guess = None
                if (dmin <= w <= dmax and dmin <= dep <= dmax
                        and amin <= w * dep <= amax):
                    near = min((math.hypot(sx - cx, sz - cz)
                                for sx, sz, _k in static), default=999.0)
                    if near > 4.0:            # 맵 원본에 없는 물체
                        guess = "enemy_tank"

                out.append({
                    "x0": round(x0, 1), "x1": round(x1, 1),
                    "z0": round(z0, 1), "z1": round(z1, 1),
                    "cx": round(cx, 1), "cz": round(cz, 1),
                    "t": guess or t,
                    "g": bool(guess),          # 추정 여부
                })
            except Exception:
                continue
        return out
    except Exception:
        return []


def _shots(fm):
    """사격 이력 — 발사점 · 조준점 · 탄착점."""
    try:
        recs = list(_g(_g(fm, "log"), "records", []) or [])[:SHOT_MAX]
        out = []
        for r in recs:
            fp, ap, ip = r.get("fire_pos"), r.get("aim_point"), r.get("impact")

            # ── 탄착오차를 사선 기준 '앞뒤(range)' 와 '좌우(cross)' 로 분해 ──
            #   앞뒤가 크면  탄도해·고저차·앙각 문제
            #   좌우가 크면  방위·차체 롤·포탑 yaw 문제
            rng_e = crs_e = None
            try:
                if fp and ap and ip:
                    dx, dz = ap[0] - fp[0], ap[2] - fp[2]
                    L = math.hypot(dx, dz)
                    if L > 1e-6:
                        ux, uz = dx / L, dz / L          # 사선 단위벡터
                        ex, ez = ip[0] - ap[0], ip[2] - ap[2]
                        rng_e = ex * ux + ez * uz         # + 길게 / - 짧게
                        crs_e = ex * uz - ez * ux         # + 오른쪽 / - 왼쪽
            except Exception:
                pass

            out.append({
                "rng_err": _num(rng_e, 2), "crs_err": _num(crs_e, 2),
                "id": r.get("id"),
                "t": _num(r.get("t"), 1),
                "fire": _xz(fp), "aim": _xz(ap), "imp": _xz(ip),
                "fy": _num(fp[1], 1) if fp else None,
                "iy": _num(ip[1], 1) if ip else None,
                "dist": _num(r.get("dist"), 1),
                "tof": _num(r.get("tof"), 2),
                "p_hit": _num(r.get("p_hit"), 3),
                "kind": r.get("kind"),
                "zone": r.get("zone"),
                "miss": _num(r.get("miss"), 2),
                "hit": r.get("kind") == "tank",
            })
        return out
    except Exception:
        return []


def _collect_fire():
    fm = _REF["fm"]
    if fm is None:
        return {"available": False, "reason": "fm 참조 없음"}

    out = {"available": True}
    try:
        out.update(fm.status() or {})
    except Exception as e:
        out["status_error"] = str(e)

    tm, fc, trk, log = _g(fm, "tm"), _g(fm, "fc"), _g(fm, "trk"), _g(fm, "log")

    if tm is not None:
        _trail.push(_g(tm, "t"), _g(tm, "my"), _g(tm, "enemy"))
        out["tm"] = {
            "t": _num(_g(tm, "t"), 2),
            "my": _g(tm, "my"), "enemy": _g(tm, "enemy"),
            "body_x": _num(_g(tm, "body_x", 0.0)),
            "turret_x": _num(_g(tm, "turret_x", 0.0)),
            "turret_y": _num(_g(tm, "turret_y", 0.0)),
            "body_y": _num(_g(tm, "body_y", 0.0)),
            "body_z": _num(_g(tm, "body_z", 0.0)),
            "my_speed": _num(_g(tm, "my_speed", 0.0)),
            "enemy_speed": _num(_g(tm, "enemy_speed", 0.0)),
            "enemy_body_x": _num(_g(tm, "enemy_body_x", 0.0)),
            "my_hp": _g(tm, "my_hp"), "enemy_hp": _g(tm, "enemy_hp"),
        }

    if fc is not None:
        sol = _g(fc, "last_solution") or _g(fm, "_last_sol")
        # 2026-09-08  해가 없을 때 사유를 그대로 내보낸다.
        #   "탄도해없음" 까지만 알면 왜 없는지(사거리 초과/너무 가까움/예측 발산)
        #   를 바깥에서 추측해야 했다. Solution.reason 에 이미 적혀 있다.
        if sol is not None and not _g(sol, "valid", False):
            out["sol_reason"] = _g(sol, "reason", "") or "사유 없음"
        if sol is not None and _g(sol, "valid", False):
            out["aim3"] = _g(sol, "aim_point")
            out["sol"] = {
                "elev": _num(_g(sol, "elevation"), 3),
                "bearing": _num(_g(sol, "bearing"), 3),
                "flight": _num(_g(sol, "flight"), 3),
                "drdt": _num(_g(sol, "drdt"), 2),
                "half_lat": _num(_g(sol, "half_lat"), 2),
                "sig_lat": _num(_g(sol, "sig_lat"), 3),
            }
        # 조준 오차와 허용 데드밴드 — SLEW 원인 진단의 핵심
        out["aim"] = {
            "yaw_err":   _num(_g(fc, "yaw_err", 0.0), 3),
            "pitch_err": _num(_g(fc, "pitch_err", 0.0), 3),
            "yaw_db":    _num(_g(fc, "yaw_db", 0.0), 3),
            "pitch_db":  _num(_g(fc, "pitch_db", 0.0), 3),
            "track_duty":_num(_g(fc, "track_duty", 0.0), 3),
            "duty_max":  _num(_g(fc, "track_duty_max", 0.0), 3),
        }
        # ── 시각 어긋남 감지 ──────────────────────────────
        # 에피소드를 Restart 하면 sim_time 이 0 으로 되감기는데
        # FireControl.last_fire_t 는 이전 판의 값을 그대로 들고 있다.
        # 그러면 reload_left 가 거대한 양수가 되어 영원히 사격하지 못한다.
        _lf, _now = _g(fc, "last_fire_t"), _g(tm, "t") if tm is not None else None
        if _lf is not None and _now is not None and _lf > _now + 0.5:
            out["desync"] = {"last_fire": _num(_lf, 1), "now": _num(_now, 1),
                             "gap": _num(_lf - _now, 1)}

        out["gates"] = {
            "p_hit_min": _num(_g(fc, "p_hit_min"), 3),
            "p_hit_floor": _num(_g(fc, "p_hit_floor"), 3),
            "body_rate_max": _num(_g(fc, "body_rate_max"), 2),
            "tof_max": _num(_g(fc, "tof_max"), 3),
            "pitch_db_max": _num(_g(fc, "pitch_db_max"), 3),
            "lon_gain": _num(_g(fc, "lon_gain"), 3),
            "db_safety": _num(_g(fc, "db_safety"), 3),
            "reload_s": _num(_g(fc, "reload_s"), 3),
            "state": _g(fc, "state"),
            # 2026-09-09  최소 교전거리. ally-controller 가 fm 에 달아 둔다.
            #   fire/ 안의 게이트가 아니라 상위(inhibit_fire)에서 막는 값이라
            #   fc 가 아니라 fm 에서 읽는다.
            "min_engage": _num(_g(fm, "min_engage"), 1),
        }

    if trk is not None:
        # 2026-08-25  속성 이름을 실제 TargetTracker 에 맞춰 바로잡았다.
        #   duty 는 트래커가 아니라 FireControl.track_duty 에 있다 (out["aim"]).
        #   n 은 존재하지 않는다. n_scored 가 맞다.
        #   k_err 이 기동사격의 핵심 지표다: 예측오차 ≈ k_err x 비과시간^2
        tof_now = _num(_g(_g(fm, "_last_sol"), "flight", 0.0), 3) or 0.0
        kerr = _g(trk, "k_err", 0.0)
        out["track"] = {
            "speed":   _num(_g(trk, "speed"), 2),
            "heading": _num(_g(trk, "heading"), 1),
            "omega":   _num(math.degrees(_g(trk, "omega", 0.0)), 2),  # 선회 각속도 [deg/s]
            "a_long":  _num(_g(trk, "a_long"), 2),                    # 접선 가속 [m/s^2]
            "k_err":   _num(kerr, 3),
            "n_scored": _g(trk, "n_scored", 0),
            # 지금 비과시간에서 기대되는 예측오차 [m]
            "pred_err": _num(kerr * tof_now * tof_now, 2),
        }
        try:
            lat, lon = trk.sigma(tof_now)
            out["track"]["sig_lat"] = _num(lat, 2)
            out["track"]["sig_lon"] = _num(lon, 2)
        except Exception:
            pass

    if log is not None:
        out["unmatched"] = _g(log, "unmatched", 0)
        out["recorded"] = len(_g(log, "records", []) or [])
        out["lost"] = list(_g(log, "lost", []) or [])

    try:
        out["suggest"] = fm.suggest_range()
    except Exception:
        out["suggest"] = None

    # 2026-08-25  말이 안 되는 표적 좌표를 버린 횟수.
    #   0 이 아니면 적이 월드 밖으로 떨어졌거나 좌표가 깨진 것이다.
    out["bad_target"] = _g(fm, "bad_target", 0)

    out["shots"] = _shots(fm)
    out["trail"] = {"my": _trail.my[-TRAIL_MAX:],
                    "enemy": _trail.enemy[-TRAIL_MAX:]}
    # 2026-09-09  적이 쏜 것도 같이 내보낸다. _FOE 스레드가 채운다.
    out["foe"] = dict(_FOE)
    return out


# ============================================================
# 적 사격  (2026-09-09)
# ============================================================
#
#  왜 여기서 가져오나
#      적 전차는 5100 번 서버가 조종하고, 적이 쏜 기록도 거기에만
#      있다. 이 상황도는 아군 FireModule 만 읽으므로 적 사격이
#      화면에 한 번도 안 나왔다 (9/09 46번 판: 적이 8발을 쐈는데
#      3D 화면에는 아무것도 없었다).
#
#  왜 스레드인가
#      /state 요청 안에서 HTTP 를 부르면 5100 이 느릴 때 이 화면이
#      통째로 멈춘다. 백그라운드로 돌리고 마지막 값만 읽는다.
FOE_URL = "http://127.0.0.1:5100/state"
FOE_POLL_S = 0.5
_FOE = {"ok": False, "shots": [], "fires": 0, "hits": 0,
        "aim": "", "enemy_fire": None, "err": "적 서버 응답 없음"}


def _foe_poll():
    while True:
        try:
            with urllib.request.urlopen(FOE_URL, timeout=2.0) as r:
                j = json.loads(r.read().decode("utf-8"))
            _FOE.update({
                "ok": True,
                "shots": j.get("shots") or [],
                "fires": j.get("fires") or 0,
                "hits": j.get("hits") or 0,
                "aim": j.get("aim") or "",
                "enemy_fire": j.get("enemy_fire"),
                "reload_left": j.get("reload_left"),
                "err": "",
            })
        except Exception as e:
            _FOE["ok"] = False
            _FOE["err"] = f"{type(e).__name__}: {e}"
        time.sleep(FOE_POLL_S)


# 주행 이력 추적 — 목적지가 몇 번 바뀌었고 경로를 몇 번 다시 짰는가
_DRIVE = {"dest_n": 0, "replan_n": 0, "last_dest": None, "last_sig": None,
          "dests": []}


def _collect_drive():
    dv = _REF["drive"]
    if dv is None:
        return {"available": False, "reason": "drive 참조 없음"}
    path = _g(dv, "current_path", []) or []
    try:
        pts = [p for p in (_xz(q) for q in path) if p]
    except Exception:
        pts = []

    pos = _xz(_g(dv, "current_pos"))
    dest = _xz(_g(dv, "dest"))

    # ── 목적지 변경 감지 ──
    if dest is not None:
        d0 = _DRIVE["last_dest"]
        if d0 is None or math.hypot(dest[0] - d0[0], dest[1] - d0[1]) > 1.0:
            _DRIVE["dest_n"] += 1
            _DRIVE["last_dest"] = dest
            _DRIVE["dests"].append(dest)
            del _DRIVE["dests"][:-12]

    # ── 재계획 감지 ──
    sig = (len(pts), round(pts[-1][0], 1) if pts else 0,
           round(pts[-1][1], 1) if pts else 0)
    if sig != _DRIVE["last_sig"]:
        _DRIVE["replan_n"] += 1
        _DRIVE["last_sig"] = sig

    # ── 경로 길이 · 우회율 ──
    plen = 0.0
    for i in range(len(pts) - 1):
        plen += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    straight = (math.hypot(dest[0] - pos[0], dest[1] - pos[1])
                if (pos and dest) else None)

    return {
        "available": True,
        "pos": pos,
        "dest": dest,
        "path": pts,
        "path_len": _num(plen, 1),
        "straight": _num(straight, 1),
        "detour": _num(plen / straight, 2) if (straight and straight > 1) else None,
        "speed_kmh": _num(_g(dv, "info_speed_kmh"), 2),
        "arrived": bool(_g(dv, "arrival_latched", False)),
        "stop_flag": bool(_g(dv, "stop_flag", False)),
        "dest_n": _DRIVE["dest_n"],
        "replan_n": _DRIVE["replan_n"],
        "dests": _DRIVE["dests"][-6:],
        "known": _planner_obstacles(dv),
        # 2026-08-26  목적지가 '전차가 기울지 않는 자리' 인가.
        #   터미널 메시지가 /info 로그에 묻혀 안 보인다는 지적이 있어
        #   화면에도 같은 정보를 띄운다.
        "dest_tilt": _dest_tilt(dest),
    }


def _dest_tilt(dest):
    """목적지에 전차를 세우면 얼마나 기우는가 [deg]. 모르면 None."""
    if not dest:
        return None
    try:
        import flat_snap
        flat_snap.init(_HERE)
        return flat_snap.tilt_at(dest[0], dest[1])
    except Exception:
        return None


def _collect_detect():
    dt = _REF["detect"]
    if dt is None:
        return {"available": False, "objects": [], "reason": "detect 참조 없음"}
    try:
        raw = getattr(dt, "DETECTED_OBJECTS_INFO", []) or []
        objs = []
        for o in raw:
            if isinstance(o, dict):
                objs.append({k: (_num(v, 2) if isinstance(v, (int, float)) else v)
                             for k, v in o.items()})
            else:
                objs.append({"raw": str(o)})
        return {"available": True, "objects": objs[:60], "count": len(raw)}
    except Exception as e:
        return {"available": False, "objects": [], "reason": str(e)}


# ============================================================
# 라우트
# ============================================================

@bp.route("/state")
def state():
    try:
        return jsonify({"ok": True,
                        "fire": _collect_fire(),
                        "drive": _collect_drive(),
                        "detect": _collect_detect()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/gap")
def gap():
    """
    두 전차의 현재 좌표만 돌려준다.  gap.py 가 쓴다.

    2026-09-09  42번 판에서 gap.py 가 /state 를 0.25 초 주기로 찔렀는데
    800 초 동안 표본이 348 개밖에 안 남았다.  평균 2.3 초에 한 개다.
    /state 는 한 번 부를 때마다 궤적 500점 x 2, 사격 이력 40발,
    주행 계획까지 전부 새로 만든다.  그 시간이 폴링 주기보다 길었다.
    표본 간격이 2.3 초면 5 m/s 로 접근하는 적이 11 m 를 건너뛴다.
    최소 접근 거리를 재는 자로는 쓸 수 없다.

    여기서는 tm 의 좌표 두 개만 읽는다.  응답이 200 바이트 안쪽이라
    0.25 초 주기를 그대로 지킬 수 있다.
    """
    try:
        fm = _REF["fm"]
        tm = _g(fm, "tm")
        if tm is None:
            return jsonify({"ok": False, "error": "tm 없음"})
        return jsonify({"ok": True,
                        "t": _num(_g(tm, "t"), 2),
                        "my": _g(tm, "my"),
                        "enemy": _g(tm, "enemy"),
                        # 2026-09-09  적 서버(_poll_ally)가 이 값을 쓴다.
                        #   /state 를 대신 부르던 것을 여기로 옮겼다.
                        "enemy_body_x": _num(_g(tm, "enemy_body_x", 0.0)),
                        "my_speed": _num(_g(tm, "my_speed", 0.0))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/static-data")
def static_data():
    try:
        with _LOCK:
            return jsonify({"ok": True,
                            "obstacles": _load_map_obstacles(),
                            "terrain": _load_terrain()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/reset-fire", methods=["GET", "POST"])
def reset_fire():
    """
    에피소드 재시작으로 어긋난 사격 모듈의 '시각' 상태만 되돌린다.

    왜 필요한가
        시뮬레이터를 Restart 하면 sim_time 이 0 으로 되감긴다.
        그런데 FireControl.last_fire_t 는 이전 판의 큰 값을 그대로 들고 있어
        reload_left = reload_s - (now - last_fire_t) 가 거대한 양수가 된다.
        결과적으로 영원히 RELOAD 상태에 머물며 한 발도 쏘지 못한다.

    무엇을 건드리는가
        last_fire_t, _t0, state 세 개뿐이다.
        조준 파라미터 · 보정값 · 사격 기록은 그대로 둔다.

    근본 해결
        ally-controller.py 의 /init 에서 사격 모듈을 초기화해야 한다.
        (FireModule 에 reset() 이 없으므로 fire_module.py 에 추가 필요)
    """
    try:
        fm = _REF["fm"]
        fc = _g(fm, "fc")
        if fc is None:
            return jsonify({"ok": False, "error": "FireControl 참조 없음"})
        before = _g(fc, "last_fire_t")
        fc.last_fire_t = None
        try:
            fc._t0 = None
        except Exception:
            pass
        try:
            fc.state = "IDLE"
        except Exception:
            pass
        print("[viz3d] 사격 시각 상태 초기화  last_fire_t %s -> None" % before)
        return jsonify({"ok": True, "before": _num(before, 1)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route("/shots.csv")
def shots_csv():
    try:
        fm = _REF["fm"]
        text = fm.export_csv() if fm else ""
        return Response("﻿" + text, mimetype="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=shots.csv"})
    except Exception as e:
        return Response("error,%s" % e, mimetype="text/csv")


@bp.route("/", strict_slashes=False)
def page():
    return Response(_CLASSIC_HTML if request.args.get("style") == "classic" else _HTML, mimetype="text/html; charset=utf-8")


# ============================================================
# 화면
# ============================================================

_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>전술 3D 상황도</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
:root{
  --bg:#05070b; --glass:rgba(11,16,24,.82); --line:rgba(120,160,210,.16);
  --fg:#e6edf7; --dim:#7d8ba1; --acc:#38bdf8;
  --friend:#4b9cf5; --hostile:#ef4444; --route:#f5a524;
  --ok:#4ade80; --no:#f87171; --warn:#fbbf24;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--fg);
  font:12.5px/1.5 "Malgun Gothic",-apple-system,system-ui,sans-serif;
  user-select:none;-webkit-font-smoothing:antialiased}
#gl{position:fixed;inset:0}
.pane{position:fixed;background:var(--glass);border:1px solid var(--line);
  border-radius:10px;backdrop-filter:blur(14px) saturate(1.3);
  box-shadow:0 10px 34px rgba(0,0,0,.55);z-index:10}
.pane h3{margin:0;padding:9px 13px;font-size:9.5px;letter-spacing:1.7px;
  color:var(--dim);text-transform:uppercase;font-weight:700;
  border-bottom:1px solid var(--line);display:flex;
  justify-content:space-between;align-items:center}
.pane h3 em{font-style:normal;font:10px Consolas,monospace;color:var(--acc)}

/* ── 상단 판독 띠 ── */
#top{top:0;left:0;right:0;height:40px;border-radius:0;border-width:0 0 1px;
  display:flex;align-items:center;padding:0 14px;gap:0}
#top .b{font-weight:800;letter-spacing:2px;font-size:11px;color:var(--acc);
  padding-right:15px;margin-right:4px;border-right:1px solid var(--line)}
#top .r{padding:0 13px;border-right:1px solid var(--line);min-width:88px;
  display:flex;flex-direction:column;line-height:1.25}
#top .r b{font:700 12.5px Consolas,monospace}
#top .r i{font-style:normal;font-size:9px;color:var(--dim);letter-spacing:.9px}
#top .sp{margin-left:auto;display:flex;gap:5px;border:0;padding:0}
.btn{background:rgba(20,28,40,.9);border:1px solid var(--line);color:var(--dim);
  padding:5px 13px;border-radius:5px;cursor:pointer;font:600 11px inherit;
  transition:.12s}
.btn:hover{color:var(--fg);border-color:rgba(120,160,210,.4)}
.btn.on{background:var(--acc);border-color:var(--acc);color:#04121c}

/* ── 좌 패널 ── */
#left{top:52px;left:12px;width:210px;max-height:calc(100% - 66px);overflow:auto}
.body{padding:9px 13px 12px}
.ck{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer;font-size:11.5px}
.ck input{accent-color:var(--acc);width:12px;height:12px;cursor:pointer;margin:0}
.sl{padding:6px 0 2px}
.sl label{display:flex;justify-content:space-between;font-size:11px;color:var(--dim)}
.sl label b{color:var(--acc);font:700 11px Consolas,monospace}
.sl input{width:100%;accent-color:var(--acc);height:3px;margin:5px 0 0}
select{width:100%;background:rgba(20,28,40,.95);color:var(--fg);
  border:1px solid var(--line);border-radius:5px;padding:5px 7px;font:inherit;font-size:11.5px}
.mk{display:flex;align-items:center;gap:9px;padding:6px 13px;cursor:pointer;
  border-bottom:1px solid rgba(255,255,255,.035)}
.mk:hover{background:rgba(56,189,248,.09)}
.mk b{display:block;font-size:11.5px}
.mk span{display:block;font:10px Consolas,monospace;color:var(--dim)}

/* ── 우 패널 ── */
/* 2026-08-26  우측 패널이 화면 아래로 넘쳐 마지막 줄들이 잘렸다.
   (1σ 횡/종 · 차체 피치/롤 · 포탑/앙각 · 인지 장애물 등)
   높이를 화면 안으로 제한하고 넘치면 스크롤되게 한다. */
#right{top:52px;right:12px;width:238px;
  max-height:calc(100vh - 64px);overflow-y:auto;overscroll-behavior:contain}
#right::-webkit-scrollbar{width:7px}
#right::-webkit-scrollbar-thumb{background:rgba(125,139,161,.4);border-radius:4px}
#right::-webkit-scrollbar-track{background:transparent}
.hp{padding:8px 13px 4px}
.hp .t{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
.hp .t b{font:700 12px Consolas,monospace}
.bar{height:5px;border-radius:3px;background:rgba(255,255,255,.09);overflow:hidden}
.bar i{display:block;height:100%;border-radius:3px;transition:width .3s}
.row{display:flex;justify-content:space-between;padding:3px 13px;font-size:11.5px;
  gap:8px}
/* 2026-08-26  라벨이 '비과시' / '간' 으로 쪼개지던 것을 막는다.
   flex 자식은 기본이 축소 허용이라 좁아지면 글자 단위로 줄이 바뀐다. */
.row>span{white-space:nowrap;flex:0 0 auto}
.row>b{text-align:right;min-width:0}
.row span{color:var(--dim)}
.row b{font:600 11.5px Consolas,monospace}
.sep{height:1px;background:var(--line);margin:6px 0}
.pill{padding:1px 7px;border-radius:8px;font:700 10px inherit}
.pass{background:rgba(74,222,128,.17);color:var(--ok)}
.fail{background:rgba(248,113,113,.17);color:var(--no)}
.off{background:rgba(125,139,161,.17);color:var(--dim)}

/* ── 축소 지도 ──
   2026-08-26  접기만으로는 부족했다. 접어도 자리가 우측 사격 패널
   위라서 맨 아랫줄(차체 피치/롤 · 포탑/앙각 · 인지 장애물)을 계속
   가렸다. 화면 아래 가운데로 옮겨 어느 패널과도 겹치지 않게 한다.
     왼쪽 범례      12 ~ 312 px
     오른쪽 사격판  화면 우측 끝
   그 사이 324 px 부터가 빈 자리다. */
#mini{left:324px;bottom:12px;width:210px;padding:0;overflow:hidden}
#mmh{display:flex;align-items:center;justify-content:space-between;
  padding:5px 9px;font-size:10.5px;color:var(--dim);cursor:pointer;
  user-select:none;letter-spacing:.04em}
#mmh:hover{color:var(--fg)}
#mmh b{font-weight:600}
#mmc{width:210px;height:210px;display:block;border-radius:0 0 10px 10px}
#mini.fold #mmc{display:none}
#mini.fold{width:96px}

/* ── 나침반 · 범례 ── */
#comp{position:fixed;top:56px;left:236px;width:58px;height:58px;z-index:11;
  pointer-events:none}
#leg{left:12px;bottom:12px;padding:8px 13px;font-size:10.5px;color:var(--dim);
  display:flex;flex-direction:column;gap:5px;max-width:300px}
#leg .r1{display:flex;gap:12px;flex-wrap:wrap}
#leg s{text-decoration:none;display:flex;align-items:center;gap:4px}
#leg u{text-decoration:none;width:9px;height:9px;border-radius:2px;display:inline-block}
#toast{position:fixed;left:50%;top:52px;transform:translateX(-50%);z-index:30;
  background:var(--glass);border:1px solid var(--line);border-radius:8px;
  padding:7px 16px;font-size:11.5px;opacity:0;transition:.3s;pointer-events:none}
#toast.on{opacity:1}

/* ── 피격 효과 ── */
@keyframes shake{
  0%,100%{transform:translate(0,0)}   15%{transform:translate(-5px,3px)}
  30%{transform:translate(5px,-3px)}  45%{transform:translate(-4px,-2px)}
  60%{transform:translate(4px,2px)}   80%{transform:translate(-2px,1px)}}
body.shk #gl{animation:shake .42s ease-out}
#vig{position:fixed;inset:0;z-index:25;pointer-events:none;opacity:0;
  transition:opacity .12s;
  box-shadow:inset 0 0 190px 55px rgba(255,40,30,.62)}
#vig.on{opacity:1;transition:opacity 0s}
#vig.hitok{box-shadow:inset 0 0 190px 55px rgba(80,255,150,.5)}
.hp .bar i.flash{animation:hpf .45s ease-out 3}
@keyframes hpf{0%,100%{filter:none}50%{filter:brightness(3)}}
::-webkit-scrollbar{width:7px;height:7px}
::-webkit-scrollbar-thumb{background:#22304a;border-radius:4px}
::-webkit-scrollbar-track{background:transparent}
</style></head><body>

<div id="gl"></div>

<div class="pane" id="top">
  <span class="b">전술 3D 상황도</span>
  <div class="r" style="min-width:172px"><b id="t-xy">–</b><i>격자 좌표</i></div>
  <div class="r"><b id="t-el" style="color:#ffd23f">–</b><i>표고</i></div>
  <div class="r"><b id="t-rg">–</b><i>교전거리</i></div>
  <div class="r"><b id="t-br">–</b><i>방위</i></div>
  <div class="r"><b id="t-sp">–</b><i>아군속도</i></div>
  <div class="r"><b id="t-st">–</b><i>사격통제</i></div>
  <div class="r sp">
    <button class="btn" id="b-follow">아군 추적</button>
    <button class="btn" id="b-top">부감</button>
    <button class="btn" id="b-reset">시점 초기화</button>
  </div>
</div>

<div class="pane" id="left">
  <h3>표시 설정</h3>
  <div class="body">
    <div class="sl"><label>표면 <b id="v-surf"></b></label>
      <select id="surf">
        <option value="terrain">위성 지도</option>
        <option value="slope">경사도</option>
        <option value="expo">피탐 노출도</option>
        <option value="block">통행 불가</option>
        <option value="flat">사격 진지 적합</option>
      </select></div>
    <div class="sl"><label>수직 과장 <b id="v-vs">2.2×</b></label>
      <input type="range" id="vs" min="1" max="5" step="0.1" value="2.2"></div>
    <div class="sl"><label>수면 높이 <b id="v-wl">–</b></label>
      <input type="range" id="wl" min="0" max="100" step="1" value="30"></div>
  </div>
  <h3>레이어</h3>
  <div class="body" id="layers"></div>
  <h3>마커 <em id="mk-n"></em></h3>
  <div id="markers"></div>
</div>

<div class="pane" id="right">
  <h3>전투 현황 <em id="r-t"></em></h3>
  <div class="hp">
    <div class="t"><span>아군 HP</span><b id="h-my">–</b></div>
    <div class="bar"><i id="h-myb" style="width:100%;background:linear-gradient(90deg,#22c55e,#4ade80)"></i></div>
  </div>
  <div class="hp">
    <div class="t"><span>적 HP</span><b id="h-en">–</b></div>
    <div class="bar"><i id="h-enb" style="width:100%;background:linear-gradient(90deg,#dc2626,#f87171)"></i></div>
  </div>
  <div class="sep"></div>
  <div class="row"><span>교전거리</span><b id="r-d">–</b></div>
  <div class="row"><span>권장거리</span><b id="r-sg">–</b></div>
  <div class="row"><span>유효사거리</span><b id="r-ev">–</b></div>
  <div class="row"><span>명중확률</span><b id="r-ph">–</b></div>
  <div class="row"><span>비과시간</span><b id="r-tf">–</b></div>
  <div class="row"><span>재장전</span><b id="r-rl">–</b></div>
  <div class="sep"></div>
  <div class="row"><span>사격 / 명중</span><b id="r-fh">–</b></div>
  <div class="row"><span>명중률</span><b id="r-hr">–</b></div>
  <div class="sep"></div>
  <div id="gates"></div>
  <div class="sep"></div>
  <div id="diag"></div>
  <div class="sep"></div>
  <div id="track"></div>
  <div class="sep"></div>
  <div class="row"><span>차체 피치 / 롤</span><b id="r-hull">–</b></div>
  <div class="row"><span>포탑 / 앙각</span><b id="r-tur">–</b></div>
  <div class="row"><span>인지 장애물</span><b id="r-ko">–</b></div>
  <div class="row"><span>탐지 객체</span><b id="r-do">–</b></div>
  <div class="sep"></div>
  <div id="drive"></div>
</div>

<div class="pane fold" id="mini">
  <div id="mmh"><span>축소 지도</span><b id="mmt">펴기</b></div>
  <canvas id="mmc"></canvas></div>
<canvas id="comp" width="116" height="116"></canvas>

<div class="pane" id="leg">
  <div class="r1">
    <s><u style="background:#4b9cf5"></u>아군</s>
    <s><u style="background:#ef4444"></u>적</s>
    <s><u style="background:#f5a524"></u>계획경로</s>
    <s><u style="background:#4ade80"></u>사선개통</s>
    <s><u style="background:#a78bfa"></u>탄도</s>
    <s><u style="background:#fbbf24"></u>탄착</s>
    <s><u style="background:#f97316"></u>적 탄착</s>
  </div>
  <div style="opacity:.75">좌클릭 회전 · 우클릭 이동 · 휠 확대 · 마커 클릭 시 해당 지점으로</div>
</div>

<div id="vig"></div>
<div id="toast"></div>

<script>
"use strict";
// 2026-09-08  API 접두어를 이 페이지가 열린 주소에서 읽는다.
//   서버의 VIEW_PREFIX 를 바꿔도 여기를 따라 고칠 필요가 없다.
//   (끝 슬래시 유무 둘 다 대응)
const API=location.pathname.replace(/\/+$/,'');
const $=s=>document.querySelector(s);
const SPAN=300;
let ST=null, S=null, TR=null;          // 정적 / 상태 / 지형 디코드
let VS=2.2, WATERP=30, SURF='terrain';
let follow=false;

const LAYERS=[
 ['route','계획 경로',1],['trail','이동 궤적',1],['shots','탄도 · 탄착',1],
 ['foeshots','적 사격',1],
 ['los','사선',1],['rings','사거리 링',1],['aim','조준점',1],['ray','포신 지향선',1],
 ['known','인지 장애물',0],['foes','적 전차 (추정)',1],['mapobs','맵 원본 장애물',1],
 ['det','탐지 객체',1],['label','라벨 · HP',1],['grid','격자',0],['water','수면',1]];
const L={}; LAYERS.forEach(([k,,v])=>L[k]=!!v);
$('#layers').innerHTML=LAYERS.map(([k,n])=>
 `<label class="ck"><input type="checkbox" data-l="${k}" ${L[k]?'checked':''}>${n}</label>`).join('');
$('#layers').onchange=e=>{const k=e.target.dataset.l; if(k){L[k]=e.target.checked; applyLayers();}};

function toast(t){const e=$('#toast');e.textContent=t;e.classList.add('on');
  clearTimeout(e._t);e._t=setTimeout(()=>e.classList.remove('on'),1600);}

/** 에피소드 재시작으로 어긋난 사격 시각 상태를 되돌린다. */
async function fixDesync(){
  try{
    const r=await fetch(API+'/reset-fire',{method:'POST'});
    const j=await r.json();
    toast(j.ok ? `사격 시각 초기화됨 (이전 ${j.before} s)` : '실패: '+j.error);
  }catch(e){ toast('실패: '+e); }
}
window.fixDesync=fixDesync;

// ══════════════════════════════════════════
//  지형 디코드
// ══════════════════════════════════════════
function unpack(b64,Type){
  const bin=atob(b64), u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  const raw=pako.inflate(u);
  // 정렬 문제를 피하려고 항상 자기 버퍼로 복사한다.
  const copy=new Uint8Array(raw.length); copy.set(raw);
  return new Type(copy.buffer);
}
function decodeTerrain(t){
  if(!t||!t.ok) return null;
  const n=t.n, rng=t.hi-t.lo;
  const hm=unpack(t.hm,Uint16Array);
  const H=new Float32Array(n*n);
  for(let i=0;i<n*n;i++) H[i]=t.lo+hm[i]/65535*rng;
  return {n, lo:t.lo, hi:t.hi, H,
    sl: t.sl?unpack(t.sl,Uint8Array):null,
    ex: t.ex?unpack(t.ex,Uint8Array):null,
    bl: t.bl?unpack(t.bl,Uint8Array):null,
    ft: t.ft?unpack(t.ft,Uint8Array):null};
}
// 월드(x,z) → 격자 인덱스.  3D 메시와 같은 매핑.
function idx(i,j){ return j*TR.n+i; }
function hAt(wx,wz){
  if(!TR) return 0;
  const n=TR.n, d=SPAN/(n-1);
  let fi=wx/d, fj=(SPAN-wz)/d;
  fi=Math.max(0,Math.min(n-1.001,fi)); fj=Math.max(0,Math.min(n-1.001,fj));
  const i=fi|0, j=fj|0, u=fi-i, v=fj-j, H=TR.H;
  return (H[idx(i,j)]*(1-u)+H[idx(i+1,j)]*u)*(1-v)
       + (H[idx(i,j+1)]*(1-u)+H[idx(i+1,j+1)]*u)*v;
}
const W2S=(x,z)=>[x-SPAN/2, SPAN/2-z];
const S2W=(x,z)=>[x+SPAN/2, SPAN/2-z];
// 월드(x,z) → 지형에 붙인 씬 좌표
function V(x,z,dy){const s=W2S(x,z);
  return new THREE.Vector3(s[0], hAt(x,z)*VS+(dy||0), s[1]);}

/**
 * 시뮬레이터 방위각[deg] → three.js 의 Y축 회전[rad]
 *
 *   시뮬레이터: 0° = 월드 +Z(북), 시계방향으로 증가
 *   씬 좌표   : scene_z = 150 - world_z  (Z축이 뒤집혀 있다)
 *   모형      : 정면이 로컬 +Z
 *
 *   방위 θ 의 월드 방향은 (sinθ, cosθ).
 *   씬에서는 (sinθ, -cosθ) 가 되어야 하므로 회전각 φ 는
 *   sinφ = sinθ, cosφ = -cosθ  →  φ = π - θ
 *
 *   -θ 로 두면 정확히 반대를 향한다.
 */
const yawOf = deg => Math.PI - (deg||0)*Math.PI/180;
const dist2=(a,b)=>Math.hypot(b[0]-a[0],b[1]-a[1]);
/** MGRS 풍 격자 좌표 표기. 100 m 방안을 문자로, 나머지를 5자리 숫자로. */
function mgrs(x,z){
  const L='ABCDEFGHJKLMNPQRSTUVWXYZ';
  const gx=L[Math.max(0,Math.min(23,Math.floor(x/100)))]||'A';
  const gz=L[Math.max(0,Math.min(23,Math.floor(z/100)+8))]||'A';
  const e=String(Math.round((x%100)*100)).padStart(5,'0');
  const nn=String(Math.round((z%100)*100)).padStart(5,'0');
  return `52S ${gx}${gz} ${e} ${nn}`;
}
const brg=(a,b)=>(Math.atan2(b[0]-a[0],b[1]-a[1])*180/Math.PI+360)%360;
const waterLevel=()=>TR? TR.lo+(TR.hi-TR.lo)*(WATERP/100) : 0;

// ══════════════════════════════════════════
//  전차 모형
// ══════════════════════════════════════════
function mat(c,r,m){return new THREE.MeshStandardMaterial(
  {color:c,roughness:r===undefined?.78:r,metalness:m===undefined?.22:m});}

function buildTank(P){
  // P: {hull, turret, track, metal, accent, ring, emis}
  const g=new THREE.Group();
  const mHull=mat(P.hull,.72,.25), mTur=mat(P.turret,.7,.28),
        mTrk=mat(P.track,.92,.12), mMet=mat(P.metal,.45,.75),
        mAcc=mat(P.accent,.5,.4);
  // 진영이 멀리서도 구분되도록 은은한 자체발광을 넣는다.
  if(P.emis){ mHull.emissive=new THREE.Color(P.emis); mHull.emissiveIntensity=.55;
              mTur.emissive =new THREE.Color(P.emis); mTur.emissiveIntensity=.55; }
  // 발밑 진영 링
  const fring=new THREE.Mesh(new THREE.RingGeometry(3.7,4.5,40),
    new THREE.MeshBasicMaterial({color:P.ring,transparent:true,opacity:.6,
      side:THREE.DoubleSide,depthWrite:false}));
  fring.rotation.x=-Math.PI/2; fring.position.y=0.10;
  fring.userData={base:P.ring}; g.add(fring);
  const add=(p,mm,x,y,z,rx,ry,rz)=>{const o=new THREE.Mesh(p,mm);
    o.position.set(x,y,z); if(rx)o.rotation.x=rx; if(ry)o.rotation.y=ry;
    if(rz)o.rotation.z=rz; o.castShadow=true; o.receiveShadow=true;
    g.add(o); return o;};

  // ── 차체 ──
  add(new THREE.BoxGeometry(3.0,0.85,6.1), mHull, 0,0.95,0);
  // 경사 전면장갑
  const gl=add(new THREE.BoxGeometry(2.96,0.16,2.35), mHull, 0,1.32,2.34, -0.62);
  // 상부 구조
  add(new THREE.BoxGeometry(2.9,0.5,3.5), mHull, 0,1.60,-0.55);
  // 후면판
  add(new THREE.BoxGeometry(2.9,0.75,0.18), mHull, 0,1.35,-3.02);
  // 펜더
  add(new THREE.BoxGeometry(3.85,0.09,5.9), mMet, 0,1.42,-0.1);

  // ── 궤도 · 전륜 ──
  [-1.42,1.42].forEach(sx=>{
    add(new THREE.BoxGeometry(0.62,1.20,6.30), mTrk, sx,0.72,0);
    add(new THREE.BoxGeometry(0.70,0.20,6.34), mTrk, sx,1.30,0);   // 상부 궤도
    for(let k=0;k<6;k++)
      add(new THREE.CylinderGeometry(0.42,0.42,0.50,14), mMet,
          sx,0.55,-2.35+k*0.94, 0,0,Math.PI/2);
    add(new THREE.CylinderGeometry(0.50,0.50,0.52,12), mMet, sx,0.80, 3.02,0,0,Math.PI/2);
    add(new THREE.CylinderGeometry(0.50,0.50,0.52,12), mMet, sx,0.80,-3.02,0,0,Math.PI/2);
  });

  // ── 포탑 (독립 회전) ──
  const T=new THREE.Group(); T.position.set(0,1.88,-0.35); g.add(T);
  const tadd=(p,mm,x,y,z,rx,ry,rz)=>{const o=new THREE.Mesh(p,mm);
    o.position.set(x,y,z); if(rx)o.rotation.x=rx; if(ry)o.rotation.y=ry;
    if(rz)o.rotation.z=rz; o.castShadow=true; T.add(o); return o;};

  tadd(new THREE.CylinderGeometry(1.42,1.62,0.92,8), mTur, 0,0.46,0, 0,Math.PI/8);
  tadd(new THREE.BoxGeometry(2.05,0.72,1.55), mTur, 0,0.50,1.05);   // 전면 확장
  tadd(new THREE.BoxGeometry(1.75,0.55,1.05), mTur, 0,0.42,-1.42);  // 후방 수납
  tadd(new THREE.BoxGeometry(1.55,0.42,0.55), mMet, 0,0.80,-1.55);  // 잡물함
  tadd(new THREE.CylinderGeometry(0.44,0.44,0.34,12), mTur, -0.55,0.99,-0.32); // 큐폴라
  tadd(new THREE.BoxGeometry(0.30,0.16,0.72), mMet, -0.55,1.22,0.05);          // 기관총
  tadd(new THREE.CylinderGeometry(0.30,0.30,0.24,10), mTur, 0.60,0.99,-0.30);  // 해치
  [-1.05,1.05].forEach(sx=>
    tadd(new THREE.CylinderGeometry(0.035,0.035,2.2,5), mMet, sx,1.60,-1.20)); // 안테나
  // 연막탄 발사기
  [-1.12,1.12].forEach(sx=>{ for(let k=0;k<3;k++)
    tadd(new THREE.CylinderGeometry(0.10,0.10,0.42,7), mAcc,
         sx,0.62,0.30+k*0.24, Math.PI/2.6); });

  // ── 주포 (앙각) ──
  const Gun=new THREE.Group(); Gun.position.set(0,0.46,1.35); T.add(Gun);
  const gadd=(p,mm,z,r)=>{const o=new THREE.Mesh(p,mm);
    o.rotation.x=Math.PI/2; o.position.set(0,0,z); o.castShadow=true;
    Gun.add(o); return o;};
  gadd(new THREE.SphereGeometry(0.62,12,10), mTur, 0.10);          // 방수포
  gadd(new THREE.CylinderGeometry(0.155,0.175,4.9,12), mMet, 2.55);// 포신
  gadd(new THREE.CylinderGeometry(0.235,0.235,2.0,12), mTur, 1.75);// 열차폐막
  gadd(new THREE.CylinderGeometry(0.255,0.255,0.62,12), mMet, 4.85);// 포구제퇴기

  // 포구 섬광
  const flash=new THREE.Mesh(new THREE.SphereGeometry(1.0,10,8),
    new THREE.MeshBasicMaterial({color:0xffd27a,transparent:true,opacity:0}));
  flash.position.set(0,0,5.3); Gun.add(flash);

  // 피격 점멸에 쓸 재질 목록. 원래 색을 기억해 둔다.
  const mats=[mHull,mTur,mMet,mAcc,mTrk];
  mats.forEach(m=>{ if(!m.emissive) m.emissive=new THREE.Color(0x000000);
    m.userData={emis:m.emissive.getHex(), inten:m.emissiveIntensity||0}; });

  g.userData={turret:T, gun:Gun, flash:flash, ring:fring, mats:mats};
  return g;
}

// ══════════════════════════════════════════
//  포탄 비행 · 폭발 효과
// ══════════════════════════════════════════
let SHELLS=[], BOOMS=[], HITFX=[];

/** 피격 점멸 — 맞은 전차가 하얗게 번쩍이며 흔들린다. */
function hitFlash(tank, base, mine){
  if(!tank) return;
  HITFX.push({tank, base:base.clone(), t:0, dur:1.25});
  document.body.classList.add('shk');
  setTimeout(()=>document.body.classList.remove('shk'),430);
  // 화면 가장자리 섬광 — 적 명중은 녹색, 아군 피격은 적색
  const v=$('#vig');
  v.classList.toggle('hitok', !mine);
  v.classList.add('on');
  setTimeout(()=>v.classList.remove('on'),90);
  setTimeout(()=>{v.classList.add('on');
    setTimeout(()=>v.classList.remove('on'),80);},190);
  // HP 바 점멸
  const b=$(mine?'#h-myb':'#h-enb');
  b.classList.remove('flash'); void b.offsetWidth; b.classList.add('flash');
  toast(mine?'⚠ 아군 피격':'✦ 적 명중');
}
function stepHit(dt){
  for(let i=HITFX.length-1;i>=0;i--){
    const h=HITFX[i]; h.t+=dt;
    const u=Math.min(1,h.t/h.dur);
    // 6 회 깜빡이며 점점 약해진다
    const on=Math.sin(u*Math.PI*12)>0 ? 1 : 0;
    const k=on*(1-u)*(1-u);
    h.tank.userData.mats.forEach(m=>{
      m.emissive.setHex(k>0.25 ? 0xffffff : m.userData.emis);
      m.emissiveIntensity=m.userData.inten + k*5.0;
    });
    // 충격으로 흔들림
    const sh=(1-u)*(1-u)*0.9;
    h.tank.position.set(h.base.x+(Math.random()-0.5)*sh,
                        h.base.y+Math.abs(Math.random()-0.5)*sh*0.7,
                        h.base.z+(Math.random()-0.5)*sh);
    const rg=h.tank.userData.ring;
    if(rg) rg.material.color.setHex(k>0.25 ? 0xffffff : rg.userData.base);
    if(u>=1){
      h.tank.userData.mats.forEach(m=>{
        m.emissive.setHex(m.userData.emis); m.emissiveIntensity=m.userData.inten;});
      if(rg) rg.material.color.setHex(rg.userData.base);
      HITFX.splice(i,1);
    }
  }
}
function glowTex(){
  if(glowTex._t) return glowTex._t;
  const s=64,c=document.createElement('canvas'); c.width=c.height=s;
  const g=c.getContext('2d'), gr=g.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
  gr.addColorStop(0,'rgba(255,240,190,1)'); gr.addColorStop(.35,'rgba(255,180,80,.55)');
  gr.addColorStop(1,'rgba(255,140,40,0)');
  g.fillStyle=gr; g.fillRect(0,0,s,s);
  return glowTex._t=new THREE.CanvasTexture(c);
}

/** 발사 → 탄착 포물선을 따라 날아가는 탄 하나를 만든다. */
function spawnShell(sh){
  if(!RN||!sh.fire||!sh.imp) return;
  const core=new THREE.Mesh(new THREE.SphereGeometry(0.5,10,8),
    new THREE.MeshBasicMaterial({color:0xfff0c0}));
  const halo=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:0xffb347,transparent:true,opacity:.95,depthTest:false}));
  halo.scale.set(5,5,1); core.add(halo);
  const tail=new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({color:0xffc98a,transparent:true,opacity:.75}));
  tail.frustumCulled=false;
  RN.fxG.add(core); RN.fxG.add(tail);
  const a=sh.fire, b=sh.imp;
  SHELLS.push({core, tail, a, b, pts:[], t:0,
    dur: Math.max(0.55, Math.min(2.6, (sh.tof||0.9)*1.7)),
    h0: hAt(a[0],a[1])*VS+2.7, h1: hAt(b[0],b[1])*VS+0.5,
    lift: Math.min(30, dist2(a,b)*0.22), hit: !!sh.hit});
  // 포구 섬광
  if(RN.me.userData.flash) RN.me.userData.flash.material.opacity=1.0;
}

/** 탄착 폭발 */
function boom(x,z,hit){
  if(!RN) return;
  const p=V(x,z,0.8);
  const core=new THREE.Mesh(new THREE.SphereGeometry(1,14,10),
    new THREE.MeshBasicMaterial({color:hit?0xfff4b0:0xffd79a,transparent:true,opacity:1}));
  core.position.copy(p);
  const ring=new THREE.Mesh(new THREE.RingGeometry(0.6,1.15,40),
    new THREE.MeshBasicMaterial({color:hit?0x7dfba0:0xffc46b,transparent:true,
      opacity:.95,side:THREE.DoubleSide,depthWrite:false}));
  ring.rotation.x=-Math.PI/2; ring.position.copy(p).setY(p.y+0.35);
  const smoke=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:hit?0xc9f7d6:0x8d8578,transparent:true,opacity:.85,depthTest:false}));
  smoke.scale.set(7,7,1); smoke.position.copy(p).setY(p.y+2.5);
  RN.fxG.add(core); RN.fxG.add(ring); RN.fxG.add(smoke);
  BOOMS.push({core, ring, smoke, t:0, dur:1.15, hit});
}

/** 매 프레임 진행 */
function stepFX(dt){
  if(!RN) return;
  for(let i=SHELLS.length-1;i>=0;i--){
    const s=SHELLS[i]; s.t+=dt;
    const u=Math.min(1,s.t/s.dur);
    const x=s.a[0]+(s.b[0]-s.a[0])*u, z=s.a[1]+(s.b[1]-s.a[1])*u;
    const y=s.h0+(s.h1-s.h0)*u + s.lift*Math.sin(Math.PI*u);
    const w=W2S(x,z);
    s.core.position.set(w[0],y,w[1]);
    s.pts.push(new THREE.Vector3(w[0],y,w[1]));
    if(s.pts.length>26) s.pts.shift();
    s.tail.geometry.setFromPoints(s.pts);
    if(u>=1){
      boom(s.b[0],s.b[1],s.hit);
      // destoryObstaclesOnHit — 탄착 지점의 오브젝트를 무너뜨린다
      destroyAt(s.b[0], s.b[1]);
      // 명중이면 적 전차가 피격 점멸한다
      if(s.hit) hitFlash(RN.foe, RN.foe.position);
      RN.fxG.remove(s.core); RN.fxG.remove(s.tail);
      s.core.geometry.dispose(); s.tail.geometry.dispose();
      SHELLS.splice(i,1);
    }
  }
  for(let i=BOOMS.length-1;i>=0;i--){
    const b=BOOMS[i]; b.t+=dt;
    const u=Math.min(1,b.t/b.dur);
    b.core.scale.setScalar(1+u*5.5);
    b.core.material.opacity=Math.max(0,1-u*1.6);
    b.ring.scale.setScalar(1+u*13);
    b.ring.material.opacity=Math.max(0,0.95-u);
    b.smoke.scale.setScalar(7+u*16);
    b.smoke.position.y+=dt*5.5;
    b.smoke.material.opacity=Math.max(0,0.85-u*0.95);
    if(u>=1){[b.core,b.ring,b.smoke].forEach(o=>{RN.fxG.remove(o);
      if(o.geometry)o.geometry.dispose();}); BOOMS.splice(i,1);}
  }
}

// ══════════════════════════════════════════
//  씬 구성
// ══════════════════════════════════════════
let RN=null;
function boot(){
  if(RN||!ST||!window.THREE) return;
  TR=decodeTerrain(ST.terrain);
  if(!TR){ toast('지형 데이터를 읽지 못했다'); return; }
  $('#v-wl').textContent=waterLevel().toFixed(1)+' m';

  const sc=new THREE.Scene();
  sc.background=new THREE.Color(0x0a1018);
  sc.fog=new THREE.FogExp2(0x243a52,0.0021);      // 푸른 대기 산란
  sc.add(skyDome());

  const cam=new THREE.PerspectiveCamera(46,innerWidth/innerHeight,0.5,4000);
  const rd=new THREE.WebGLRenderer({antialias:true});
  rd.setPixelRatio(Math.min(2,devicePixelRatio));
  rd.setSize(innerWidth,innerHeight);
  rd.shadowMap.enabled=true; rd.shadowMap.type=THREE.PCFSoftShadowMap;
  rd.outputEncoding=THREE.sRGBEncoding;
  rd.toneMapping=THREE.ACESFilmicToneMapping; rd.toneMappingExposure=0.82;
  $('#gl').appendChild(rd.domElement);

  // ── 조명 ──
  // 지형 음영은 이미 정점색에 힐셰이드로 구워져 있다.
  // 그래서 3D 조명은 거의 평평하게 두고, 전차·나무만 입체로 살린다.
  sc.add(new THREE.HemisphereLight(0x93aac4,0x191710,1.05));
  const sun=new THREE.DirectionalLight(0xffeed2,0.62);
  sun.position.set(150,240,110); sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);
  const sh=sun.shadow.camera; sh.near=1; sh.far=900;
  sh.left=sh.bottom=-190; sh.right=sh.top=190; sh.updateProjectionMatrix();
  sun.shadow.bias=-0.0012; sc.add(sun);
  const fill=new THREE.DirectionalLight(0x7ea8d8,0.30);
  fill.position.set(-160,90,-120); sc.add(fill);

  // ── 지형 ──
  const n=TR.n;
  const geo=new THREE.PlaneGeometry(SPAN,SPAN,n-1,n-1); geo.rotateX(-Math.PI/2);
  geo.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(n*n*3),3));
  const terrain=new THREE.Mesh(geo, new THREE.MeshStandardMaterial(
    {vertexColors:true, roughness:.95, metalness:.03, flatShading:false}));
  terrain.receiveShadow=true; terrain.castShadow=true; sc.add(terrain);

  // ── 수면 ──
  const water=new THREE.Mesh(new THREE.PlaneGeometry(SPAN,SPAN),
    new THREE.MeshStandardMaterial({color:0x1d4a5e,transparent:true,opacity:0.86,
      roughness:0.22,metalness:0.42}));
  water.rotation.x=-Math.PI/2; sc.add(water);

  // ── 격자 ──
  const grid=new THREE.GridHelper(SPAN,6,0x36506e,0x22354c);
  grid.material.transparent=true; grid.material.opacity=.5; sc.add(grid);

  // ── 수목 · 구조물 ──
  const mapG=new THREE.Group(); sc.add(mapG);
  const knownG=new THREE.Group(); sc.add(knownG);
  const detG=new THREE.Group(); sc.add(detG);
  const foeG=new THREE.Group(); sc.add(foeG);   // 추정 적 전차 표식

  // ── 전차 ──
  const me =buildTank({hull:0x2b5a91,turret:0x356ba6,track:0x121924,
                       metal:0x7a92ab,accent:0x1b3757,
                       ring:0x4b9cf5, emis:0x0b2140});
  const foe=buildTank({hull:0xb01410,turret:0xcc1a13,track:0x1d0d0b,
                       metal:0xd2837a,accent:0x6d0b07,
                       ring:0xff2b21, emis:0x460604});
  sc.add(me); sc.add(foe);

  // ── 선 요소 ──
  const mkL=(c,o,w)=>{const l=new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color:c,transparent:true,opacity:o===undefined?1:o}));
    l.frustumCulled=false; sc.add(l); return l;};
  const routeL=mkL(0xf5a524), losL=mkL(0x4ade80,.9),
        trailMe=mkL(0x4b9cf5,.55), trailFoe=mkL(0xef4444,.55),
        ringMin=mkL(0x38bdf8,.55), ringMax=mkL(0x38bdf8,.34),
        ringSug=mkL(0xa78bfa,.5);
  const shotG=new THREE.Group(); sc.add(shotG);
  // 2026-09-09  적이 쏜 탄도 · 탄착. 아군 것과 색으로 구분한다.
  const foeShotG=new THREE.Group(); sc.add(foeShotG);
  const fxG=new THREE.Group(); sc.add(fxG);          // 비행 탄 · 폭발 효과
  // 포신 지향선 — 주포가 실제로 겨누는 방향
  const aimRay=new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({color:0x9fe8ff,dashSize:3,gapSize:2.4,
      transparent:true,opacity:.65}));
  aimRay.frustumCulled=false; sc.add(aimRay);

  // ── 빌보드 ──
  const spr=(canvas,scale,order)=>{const s=new THREE.Sprite(new THREE.SpriteMaterial(
      {map:new THREE.CanvasTexture(canvas),depthTest:false,transparent:true}));
    s.scale.set(canvas.width*scale,canvas.height*scale,1);
    s.renderOrder=order||20; sc.add(s); return s;};
  const meTag=spr(tagCanvas('아군','#4b9cf5',1,1),0.052),
        foeTag=spr(tagCanvas('적','#ef4444',1,1),0.052);
  const aimSp=spr(aimCanvas(),0.10,21);

  RN={sc,cam,rd,sun,terrain,water,grid,mapG,knownG,detG,me,foe,
      routeL,losL,trailMe,trailFoe,ringMin,ringMax,ringSug,shotG,foeShotG,fxG,aimRay,foeG,
      meTag,foeTag,aimSp,geo,
      cam3:{yaw:.62,pit:.72,dis:340,tgt:new THREE.Vector3(0,0,0)}};

  paintTerrain(); buildMapObstacles(); applyLayers();

  // ── 조작 ──
  const el=rd.domElement, C=RN.cam3;
  el.oncontextmenu=e=>e.preventDefault();
  let drag=0,px=0,py=0;
  el.onmousedown=e=>{drag=e.button===2?2:1;px=e.clientX;py=e.clientY;};
  addEventListener('mouseup',()=>drag=0);
  addEventListener('mousemove',e=>{
    if(drag===1){C.yaw-=(e.clientX-px)*.0055;
      C.pit=Math.max(.10,Math.min(1.52,C.pit-(e.clientY-py)*.0045)); follow=false; syncBtn();}
    else if(drag===2){const s=C.dis*.0015;
      C.tgt.x-=Math.cos(C.yaw)*(e.clientX-px)*s-Math.sin(C.yaw)*(e.clientY-py)*s;
      C.tgt.z+=Math.sin(C.yaw)*(e.clientX-px)*s+Math.cos(C.yaw)*(e.clientY-py)*s;
      follow=false; syncBtn();}
    px=e.clientX;py=e.clientY;});
  el.onwheel=e=>{e.preventDefault();
    C.dis=Math.max(18,Math.min(1000,C.dis*(1+Math.sign(e.deltaY)*.1)));};

  const ray=new THREE.Raycaster(), m2=new THREE.Vector2();
  el.addEventListener('mousemove',e=>{
    m2.x=(e.clientX/innerWidth)*2-1; m2.y=-(e.clientY/innerHeight)*2+1;
    ray.setFromCamera(m2,cam);
    const h=ray.intersectObject(RN.terrain);
    if(h.length){const w=S2W(h[0].point.x,h[0].point.z);
      $('#t-xy').textContent=mgrs(w[0],w[1]);
      $('#t-el').textContent=hAt(w[0],w[1]).toFixed(0)+' m';}});

  addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;
    cam.updateProjectionMatrix(); rd.setSize(innerWidth,innerHeight); sizeMini();});

  let tt=0, prev=performance.now();
  (function loop(){requestAnimationFrame(loop);
    const now=performance.now(), dt=Math.min(0.05,(now-prev)/1000); prev=now; tt+=dt;
    if(follow&&S&&S.fire&&S.fire.tm&&S.fire.tm.my){
      const m=S.fire.tm.my, s=W2S(m[0],m[2]);
      C.tgt.lerp(new THREE.Vector3(s[0],hAt(m[0],m[2])*VS,s[1]),0.12);}
    cam.position.set(C.tgt.x+Math.sin(C.yaw)*Math.cos(C.pit)*C.dis,
                     C.tgt.y+Math.sin(C.pit)*C.dis,
                     C.tgt.z+Math.cos(C.yaw)*Math.cos(C.pit)*C.dis);
    cam.lookAt(C.tgt);
    RN.water.material.opacity=0.70+Math.sin(tt*1.1)*0.05;
    // 진영 링 맥동
    const pu=0.45+Math.sin(tt*2.2)*0.18;
    [me,foe].forEach(t=>{if(t.userData.ring) t.userData.ring.material.opacity=pu;});
    // 포구 섬광 감쇠
    [me,foe].forEach(t=>{const f=t.userData.flash;
      if(f.material.opacity>0) f.material.opacity=Math.max(0,f.material.opacity-dt*7);});
    stepFX(dt); stepHit(dt); stepFall(dt);
    rd.render(sc,cam); drawCompass(C.yaw);})();
  sizeMini();
  $('#mmh').addEventListener('click', toggleMini);   // 축소 지도 접기/펴기
}

function skyDome(){
  // 안쪽을 보는 큰 구. 정점 색으로 하늘 그라디언트를 만든다.
  const g=new THREE.SphereGeometry(1600,24,16);
  const pos=g.attributes.position, col=[];
  const top=new THREE.Color(0x14314f), mid=new THREE.Color(0x3d648a),
        bot=new THREE.Color(0x1b2b3c), c=new THREE.Color();
  for(let i=0;i<pos.count;i++){
    const u=Math.max(0,Math.min(1,(pos.getY(i)/1600+1)/2));   // 0 아래 ~ 1 위
    if(u>0.52) c.copy(mid).lerp(top,(u-0.52)/0.48);
    else       c.copy(bot).lerp(mid,u/0.52);
    col.push(c.r,c.g,c.b);
  }
  g.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
  const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial(
    {vertexColors:true,side:THREE.BackSide,fog:false,depthWrite:false}));
  m.renderOrder=-1;
  return m;
}
/** ATAK 방식 라벨판 — 회색 둥근 판에 흰 글씨, 아래로 지시선 */
function tagCanvas(text,color){
  const f=17, padX=13, padY=8, stem=15, R=7;
  const m=document.createElement('canvas').getContext('2d');
  m.font=`700 ${f}px "Malgun Gothic",sans-serif`;
  const w=Math.ceil(m.measureText(text).width)+padX*2;
  const hh=f+padY*2;
  const c=document.createElement('canvas');
  c.width=w+6; c.height=hh+stem+6;
  const g=c.getContext('2d');
  g.font=`700 ${f}px "Malgun Gothic",sans-serif`;
  const x0=3, y0=3;
  // 그림자
  g.shadowColor='rgba(0,0,0,.75)'; g.shadowBlur=6; g.shadowOffsetY=2;
  // 둥근 판
  g.beginPath();
  g.moveTo(x0+R,y0); g.lineTo(x0+w-R,y0); g.quadraticCurveTo(x0+w,y0,x0+w,y0+R);
  g.lineTo(x0+w,y0+hh-R); g.quadraticCurveTo(x0+w,y0+hh,x0+w-R,y0+hh);
  g.lineTo(x0+R,y0+hh); g.quadraticCurveTo(x0,y0+hh,x0,y0+hh-R);
  g.lineTo(x0,y0+R); g.quadraticCurveTo(x0,y0,x0+R,y0); g.closePath();
  g.fillStyle='rgba(58,64,74,.94)'; g.fill();
  g.shadowColor='transparent';
  g.strokeStyle=color; g.lineWidth=1.8; g.stroke();
  // 지시선
  g.beginPath(); g.moveTo(x0+w/2,y0+hh); g.lineTo(x0+w/2,y0+hh+stem);
  g.strokeStyle=color; g.lineWidth=1.6; g.stroke();
  // 글씨
  g.fillStyle='#f2f6fb'; g.textBaseline='middle'; g.textAlign='center';
  g.fillText(text,x0+w/2,y0+hh/2+1);
  return c;
}
function aimCanvas(){
  const s=64,c=document.createElement('canvas');c.width=c.height=s;
  const g=c.getContext('2d');
  g.strokeStyle='#fbbf24'; g.lineWidth=3;
  g.beginPath();g.arc(s/2,s/2,18,0,7);g.stroke();
  g.lineWidth=2.4;
  [[0,-26,0,-11],[0,11,0,26],[-26,0,-11,0],[11,0,26,0]].forEach(v=>{
    g.beginPath();g.moveTo(s/2+v[0],s/2+v[1]);g.lineTo(s/2+v[2],s/2+v[3]);g.stroke();});
  g.fillStyle='#fbbf24';g.beginPath();g.arc(s/2,s/2,3,0,7);g.fill();
  return c;
}

// ── 절차적 잡음 (위성사진의 얼룩덜룩한 질감을 만든다) ──
function hash2(i,j){
  let h=(i*374761393+j*668265263)|0;
  h=(h^(h>>>13))*1274126177|0;
  return ((h^(h>>>16))>>>0)/4294967295;
}
function noise2(x,y){
  const i=Math.floor(x), j=Math.floor(y), fx=x-i, fy=y-j;
  const u=fx*fx*(3-2*fx), v=fy*fy*(3-2*fy);
  const a=hash2(i,j), b=hash2(i+1,j), c=hash2(i,j+1), d=hash2(i+1,j+1);
  return (a*(1-u)+b*u)*(1-v)+(c*(1-u)+d*u)*v;
}
function fbm(x,y){
  let s=0,a=0.5,f=1;
  for(let k=0;k<4;k++){ s+=a*noise2(x*f,y*f); a*=0.5; f*=2.03; }
  return s;                      // 0 ~ 1 근처
}
const LERP3=(A,B,t)=>[A[0]+(B[0]-A[0])*t, A[1]+(B[1]-A[1])*t, A[2]+(B[2]-A[2])*t];

// ── 지형 채색 ──
function paintTerrain(){
  if(!RN||!TR) return;
  const n=TR.n, pos=RN.geo.attributes.position, col=RN.geo.attributes.color;
  const rng=(TR.hi-TR.lo)||1, wl=waterLevel();
  const C=new THREE.Color(), H=TR.H;

  // ── 위성사진 팔레트 ──────────────────────────────
  // 전부 어둡고 채도가 낮다. 실제 항공/위성 영상은 이 대역에 있다.
  // 실제 지형 데이터로 렌더 시험한 값. 평균 밝기 0.277 / 대비 0.091 —
  // 참조로 삼은 전술지도(ATAK) 화면과 같은 대역이다.
  const P_SHOAL=[0.026,0.062,0.082];   // 얕은 물
  const P_SHORE=[0.118,0.108,0.082];   // 물가 모래톱
  const P_FOR_D=[0.026,0.046,0.022];   // 짙은 침엽수림
  const P_FOR_L=[0.046,0.070,0.033];   // 활엽수림
  const P_SCRUB=[0.076,0.092,0.046];   // 관목지
  const P_GRASS=[0.104,0.104,0.060];   // 초지
  const P_DRY  =[0.142,0.128,0.082];   // 마른 풀 · 황토
  const P_SOIL =[0.156,0.132,0.096];   // 나지
  const P_ROCK =[0.132,0.128,0.118];   // 암반
  const P_RIDGE=[0.192,0.186,0.174];   // 능선 노출암

  // 힐셰이드용 태양 (북서 315°, 고도 45°) — 위성 영상 관례
  const SAZ=315*Math.PI/180, SEL=45*Math.PI/180;
  const SUNV=[Math.sin(SAZ)*Math.cos(SEL), Math.sin(SEL), -Math.cos(SAZ)*Math.cos(SEL)];
  const cell=SPAN/(n-1);

  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const k=idx(i,j), h=H[k];
    pos.setY(k,h*VS);
    let r,g,b;

    if(SURF==='slope'){ const s=TR.sl?TR.sl[k]/255:0;
      C.setHSL(0.34-0.34*s,0.55,0.20+0.26*s); r=C.r;g=C.g;b=C.b; }
    else if(SURF==='expo'){ const e=TR.ex?TR.ex[k]/255:0;
      C.setHSL(0.62-0.62*e,0.62,0.17+0.30*e); r=C.r;g=C.g;b=C.b; }
    else if(SURF==='block'){ const q=TR.bl?TR.bl[k]:0;
      r=q?0.44:0.10; g=q?0.09:0.17; b=q?0.09:0.12; }
    // ── 2026-08-26  사격 진지 적합도 ─────────────────────────────
    //
    //   전차가 그 자리에 섰을 때 얼마나 기우는지를 색으로 보여준다.
    //   '점' 의 경사가 아니라 차체가 놓이는 5 m 반경 안의 최대 경사다.
    //   전차는 6.3 x 3.3 m 면이라 면 전체가 평평해야 안 기운다.
    //
    //   실측 근거 (8/26 기동사격 27발)
    //       평평 (0°)        9발 9명중  100 %
    //       6.3° / -10.8°    9발 8명중   89 %
    //      12.8° / -11.2°    9발 6명중   67 %
    //
    //   초록 = 여기 찍으면 된다.  붉은색 = 기운다.
    else if(SURF==='flat'){ const t=TR.ft?TR.ft[k]:255;
      if(t<=2)      { r=0.13; g=0.62; b=0.31; }   // 2° 이하 — 적합
      else if(t<=5) { r=0.42; g=0.55; b=0.20; }   // 5° 이하 — 무난
      else if(t<=10){ r=0.55; g=0.42; b=0.14; }   // 10° 이하 — 주의
      else          { r=0.45; g=0.13; b=0.13; }   // 그 이상 — 부적합
    }
    else {
      const u=(h-TR.lo)/rng;
      const s=TR.sl?TR.sl[k]/255:0;
      const i0=Math.max(1,Math.min(n-2,i)), j0=Math.max(1,Math.min(n-2,j));

      // ── 지형 기울기 벡터 (동쪽 · 남쪽 방향 성분) ──
      const dzdx=(H[idx(i0+1,j0)]-H[idx(i0-1,j0)])/(2*cell);
      const dzdy=(H[idx(i0,j0+1)]-H[idx(i0,j0-1)])/(2*cell);
      const nl=Math.hypot(dzdx,dzdy,1);
      const nrm=[-dzdx/nl, 1/nl, -dzdy/nl];
      // 사면 방향: +1 이면 남향(볕이 잘 듦) → 마른 황토, -1 이면 북향 → 짙은 숲
      const aspect=Math.max(-1,Math.min(1,dzdy/Math.max(0.05,Math.hypot(dzdx,dzdy))));

      // ── ① 고도 + 사면 방향으로 식생을 정한다 ──
      const dryness=Math.max(0,Math.min(1, u*0.95 + aspect*0.25 + s*0.35 - 0.10));
      let c1;
      if(dryness<0.24)      c1=LERP3(P_FOR_D,P_FOR_L, dryness/0.24);
      else if(dryness<0.44) c1=LERP3(P_FOR_L,P_SCRUB,(dryness-0.24)/0.20);
      else if(dryness<0.62) c1=LERP3(P_SCRUB,P_GRASS,(dryness-0.44)/0.18);
      else if(dryness<0.80) c1=LERP3(P_GRASS,P_DRY  ,(dryness-0.62)/0.18);
      else                  c1=LERP3(P_DRY  ,P_SOIL ,Math.min(1,(dryness-0.80)/0.20));

      // ── ② 급경사 · 능선은 암반이 드러난다 ──
      const rk=Math.max(0,Math.min(1,(s-0.20)/0.42));
      c1=LERP3(c1,P_ROCK,rk*0.80);
      if(u>0.86) c1=LERP3(c1,P_RIDGE,(u-0.86)/0.14*0.55);

      // ── ③ 물가 · 얕은 물 ──
      const sh=Math.max(0,Math.min(1,(h-wl)/1.8));
      c1=LERP3(P_SHORE,c1,sh);
      if(h<wl) c1=LERP3(P_SHOAL,P_SHORE,Math.max(0,Math.min(1,(h-wl+2.4)/2.4)));

      // ── ④ 수관 질감 — 숲일수록 오돌토돌하게 ──
      const nx=i*0.11, ny=j*0.11;
      const m1=fbm(nx,ny)-0.5;                      // 큰 임상 얼룩
      const m2=fbm(nx*5.1+31,ny*5.1-17)-0.5;        // 우듬지 잔결
      const m3=fbm(nx*13.0-5,ny*13.0+9)-0.5;        // 미세 노이즈
      const forest=Math.max(0,1-dryness*1.5)*Math.max(0,1-rk);
      const tex=1 + m1*0.30 + m2*(0.18+forest*0.34) + m3*0.12;
      c1=[c1[0]*tex, c1[1]*tex*(1+forest*m2*0.10), c1[2]*tex];

      // ── ⑤ 힐셰이드 — 음영을 색에 굽는다 (위성영상의 핵심) ──
      const dot=Math.max(0, nrm[0]*SUNV[0]+nrm[1]*SUNV[1]+nrm[2]*SUNV[2]);
      const hs=0.34+1.10*dot;                        // 0.34 ~ 1.44, 대비 강함
      c1=[c1[0]*hs, c1[1]*hs, c1[2]*hs];

      // ── ⑥ 침식골 — 오목한 곳을 어둡게 (물길이 드러난다) ──
      const mean=(H[idx(i0-1,j0)]+H[idx(i0+1,j0)]+
                  H[idx(i0,j0-1)]+H[idx(i0,j0+1)])*0.25;
      const curv=Math.max(-1,Math.min(1,(h-mean)*1.1));
      const ak=1+curv*0.30;
      c1=[c1[0]*ak,c1[1]*ak,c1[2]*ak];

      r=Math.max(0,c1[0]); g=Math.max(0,c1[1]); b=Math.max(0,c1[2]);
    }
    col.setXYZ(k,r,g,b);
  }
  pos.needsUpdate=true; col.needsUpdate=true;
  RN.geo.computeVertexNormals();
  RN.water.position.y=wl*VS;
  RN.grid.position.y=TR.hi*VS+20;
  drawMini._bg=null;             // 미니맵 배경도 다시 그린다
}

// ══════════════════════════════════════════
//  맵 오브젝트 — 시뮬레이터 실물에 맞춘 모형
// ══════════════════════════════════════════
const rnd=(i,a,b)=>a+((((i*2654435761)>>>0)%1000)/1000)*(b-a);

/** 세로 판자벽 텍스처 */
function plankTex(dark){
  const key='pk'+(dark?1:0); if(plankTex[key]) return plankTex[key];
  const w=128,h=128,c=document.createElement('canvas'); c.width=w;c.height=h;
  const g=c.getContext('2d');
  g.fillStyle=dark?'#38291d':'#8f7d5e'; g.fillRect(0,0,w,h);
  for(let i=0;i<16;i++){
    const x=i*8, k=Math.abs((Math.sin(i*12.9898)*43758.5453)%1);
    g.fillStyle=dark ? `rgb(${44+k*26|0},${32+k*20|0},${23+k*14|0})`
                     : `rgb(${132+k*40|0},${116+k*34|0},${88+k*26|0})`;
    g.fillRect(x,0,7,h);
    g.fillStyle='rgba(0,0,0,.30)'; g.fillRect(x+7,0,1,h);
  }
  const t=new THREE.CanvasTexture(c);
  t.wrapS=t.wrapT=THREE.RepeatWrapping;
  return plankTex[key]=t;
}

/** 수목 — 큰 키 활엽수.
 *
 *  2026-08-26  실제 숲 사진에 맞춰 되돌렸다.
 *    줄기 높이   3 m  ->  9~14 m
 *    줄기 굵기   가늘게 · 색 더 어둡게
 *    수관        원뿔 한 덩어리  ->  성긴 잎뭉치 10 개
 *    배치        줄기 위쪽 40 % 구간에만 (아래는 맨 줄기)
 *    색          자홍 + 황록을 한 나무 안에 섞는다
 */
const BLOBS=10;                       // 나무 한 그루당 잎뭉치 수
function addTrees(list){
  if(!list.length) return;
  const N=list.length;

  // 줄기 — 가늘고 어둡게. 높이 1 로 만들고 인스턴스마다 늘린다.
  const trunk=new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.10,0.20,1,5),
    mat(0x2b2018,.98,.01), N);

  // 수관 — 저폴리 잎뭉치. N x BLOBS 개를 한 InstancedMesh 로 그린다.
  const crown=new THREE.InstancedMesh(
    new THREE.IcosahedronGeometry(1,0),
    mat(0xffffff,.94,.02), N*BLOBS);

  trunk.castShadow=crown.castShadow=true; crown.receiveShadow=true;

  const d=new THREE.Object3D();
  const rec=[];                       // 파괴 처리에 쓸 나무별 정보
  const cMag=new THREE.Color(0xc2417f);   // 자홍
  const cLim=new THREE.Color(0xa8bf46);   // 황록
  const tmp=new THREE.Color();

  // 결정론적 난수 — 새로고침해도 숲 모양이 같아야 비교가 된다
  const rnd=(a,b)=>{ const t=Math.sin(a*127.1+b*311.7)*43758.5453;
                     return t-Math.floor(t); };

  list.forEach((o,i)=>{
    const s=W2S(o.x,o.z), y=hAt(o.x,o.z)*VS;
    const H=(9+rnd(i,1)*5)*VS;        // 줄기 높이 9~14 m
    const R=0.9+rnd(i,2)*0.5;         // 수관 반경 계수

    d.position.set(s[0], y+H/2, s[1]);
    d.scale.set(1, H, 1);
    d.rotation.set(0, i*1.1, 0);
    d.updateMatrix(); trunk.setMatrixAt(i, d.matrix);

    for(let b=0;b<BLOBS;b++){
      // 위쪽 40 % 구간에만 흩뿌린다
      const f=0.60+rnd(i,10+b)*0.40;
      const ang=rnd(i,30+b)*Math.PI*2;
      const rad=(0.4+rnd(i,50+b)*1.5)*R;
      const bs=(0.75+rnd(i,70+b)*0.95)*R;
      d.position.set(s[0]+Math.cos(ang)*rad, y+H*f, s[1]+Math.sin(ang)*rad);
      d.scale.set(bs, bs*0.8, bs);
      d.rotation.set(rnd(i,90+b)*3, rnd(i,110+b)*3, 0);
      d.updateMatrix(); crown.setMatrixAt(i*BLOBS+b, d.matrix);
      // 한 나무 안에서도 자홍과 황록이 섞이게
      tmp.copy(cMag).lerp(cLim, rnd(i,130+b));
      crown.setColorAt(i*BLOBS+b, tmp);
    }
    rec.push({x:o.x, z:o.z, i:i, sx:s[0], sz:s[1], y:y, sc:1, H:H, R:R,
              dead:false});
  });
  if(crown.instanceColor) crown.instanceColor.needsUpdate=true;
  RN.mapG.add(trunk); RN.mapG.add(crown);
  RN.trees={trunk, crown, rec, d:new THREE.Object3D()};
}

// ══════════════════════════════════════════
//  파괴 — destoryObstaclesOnHit 반영
//  시뮬레이터가 맞은 오브젝트를 없애므로 화면에서도 무너뜨린다.
// ══════════════════════════════════════════
const KILL_R={House:5.0, Tent:3.2, Car:3.0, Human:1.6, Tank:4.0, Tree:2.2};
let FALLING=[];

/** 탄착 지점 주변 오브젝트를 무너뜨린다. */
function destroyAt(wx,wz){
  if(!RN) return 0;
  let n=0;
  // 개별 오브젝트 (헛간 · 천막 · 차량 · 사람 · 전차 소품)
  (RN.mapObjs||[]).forEach(o=>{
    if(o.dead) return;
    if(Math.hypot(o.x-wx,o.z-wz) <= (KILL_R[o.kind]||2.5)){
      o.dead=true; n++;
      FALLING.push({obj:o.obj, t:0, dur:0.85,
        y0:o.obj.position.y, tilt:(Math.random()-0.5)*1.6});
      dustPuff(o.x,o.z, o.kind==='House'?2.2:1.3);
    }
  });
  // 수목 — 인스턴스를 0 으로 줄여 없앤다
  const T=RN.trees;
  if(T){
    T.rec.forEach(r=>{
      if(r.dead) return;
      if(Math.hypot(r.x-wx,r.z-wz) <= KILL_R.Tree){
        r.dead=true; n++;
        FALLING.push({tree:r, t:0, dur:0.7});
        dustPuff(r.x,r.z,1.0);
      }
    });
  }
  return n;
}

/** 먼지 · 파편 */
function dustPuff(wx,wz,scale){
  const p=V(wx,wz,1.0);
  const s=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:glowTex(),color:0x9a8f7c,transparent:true,opacity:.8,depthTest:false}));
  s.scale.set(6*scale,6*scale,1); s.position.copy(p);
  RN.fxG.add(s);
  BOOMS.push({core:s, ring:s, smoke:s, t:0, dur:1.0, dustOnly:true});
}

/** 무너지는 중인 것들을 매 프레임 진행 */
function stepFall(dt){
  if(!RN) return;
  const T=RN.trees;
  for(let i=FALLING.length-1;i>=0;i--){
    const f=FALLING[i]; f.t+=dt;
    const u=Math.min(1,f.t/f.dur);
    if(f.obj){
      f.obj.rotation.z=f.tilt*u;
      f.obj.position.y=f.y0-u*u*3.2;
      f.obj.scale.setScalar(Math.max(0.01,1-u*0.55));
      f.obj.traverse(o=>{ if(o.material&&!o.material._t){
        o.material.transparent=true; o.material._t=1; }
        if(o.material) o.material.opacity=1-u; });
      if(u>=1){ RN.mapG.remove(f.obj); FALLING.splice(i,1); }
    }else if(f.tree&&T){
      // 2026-08-26  수관이 잎뭉치 BLOBS 개로 바뀌어 함께 줄인다.
      const r=f.tree, k=Math.max(0.001,1-u);
      T.d.position.set(r.sx, r.y+r.H*k/2, r.sz);
      T.d.scale.set(k, r.H*k, k); T.d.rotation.set(u*1.3, r.i*1.1, 0);
      T.d.updateMatrix(); T.trunk.setMatrixAt(r.i,T.d.matrix);
      for(let b=0;b<BLOBS;b++){
        T.d.position.set(r.sx, r.y+r.H*k*0.7, r.sz);
        T.d.scale.setScalar(0.001);
        T.d.updateMatrix(); T.crown.setMatrixAt(r.i*BLOBS+b, T.d.matrix);
      }
      T.trunk.instanceMatrix.needsUpdate=true;
      T.crown.instanceMatrix.needsUpdate=true;
      if(u>=1) FALLING.splice(i,1);
    }else FALLING.splice(i,1);
  }
}

/** 목조 헛간 — 어두운 판자벽 + 박공지붕 + 밝은 서까래 + 흰 문 */
function buildHouse(i){
  const g=new THREE.Group();
  const W=6.4, D=6.0, Hh=3.4, RH=3.0;
  const wallM=new THREE.MeshStandardMaterial(
    {map:plankTex(true),roughness:.96,metalness:.02});
  wallM.map.repeat.set(2,1.2);
  const beamM=mat(0x9c8a68,.92,.03);
  const body=new THREE.Mesh(new THREE.BoxGeometry(W,Hh,D), wallM);
  body.position.y=Hh/2; body.castShadow=body.receiveShadow=true; g.add(body);
  const gable=new THREE.Shape();
  gable.moveTo(-W/2,0); gable.lineTo(W/2,0); gable.lineTo(0,RH); gable.closePath();
  [D/2,-D/2].forEach(z=>{
    const m=new THREE.Mesh(new THREE.ShapeGeometry(gable),
      new THREE.MeshStandardMaterial({map:plankTex(true),roughness:.96,metalness:.02}));
    m.position.set(0,Hh,z); if(z<0) m.rotation.y=Math.PI;
    m.castShadow=true; g.add(m);
  });
  const slope=Math.atan2(RH,W/2), len=Math.hypot(W/2,RH);
  [1,-1].forEach(sx=>{
    const m=new THREE.Mesh(new THREE.BoxGeometry(len,0.14,D+0.5), beamM);
    m.position.set(sx*W/4, Hh+RH/2, 0); m.rotation.z=-sx*slope;
    m.castShadow=true; g.add(m);
  });
  for(let k=-2;k<=2;k++){
    const m=new THREE.Mesh(new THREE.BoxGeometry(W+0.3,0.13,0.13), beamM);
    m.position.set(0, Hh+0.15, k*D/5); m.castShadow=true; g.add(m);
  }
  const ridge=new THREE.Mesh(new THREE.BoxGeometry(0.18,0.18,D+0.6), beamM);
  ridge.position.set(0,Hh+RH,0); g.add(ridge);
  const white=mat(0xe8e4da,.85,.02);
  const door=new THREE.Mesh(new THREE.BoxGeometry(1.1,1.9,0.12), white);
  door.position.set(-0.9,0.95,D/2+0.02); g.add(door);
  const win=new THREE.Mesh(new THREE.BoxGeometry(0.9,0.8,0.12), white);
  win.position.set(1.2,2.1,D/2+0.02); g.add(win);
  // DemoMap.map 의 rotation 이 전부 (0,0,0,1) 이므로 회전을 주지 않는다.
  return g;
}

/** 개방형 A형 천막 골조 */
function buildTent(i){
  const g=new THREE.Group();
  const W=3.6, D=4.4, Hh=2.6;
  const woodM=mat(0x4a4238,.95,.03), barM=mat(0x6d6252,.94,.03);
  [1,-1].forEach(sx=>{
    const len=Math.hypot(W/2,Hh), slope=Math.atan2(Hh,W/2);
    for(let k=0;k<7;k++){
      const t=(k+0.5)/7;
      const m=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.12,D), barM);
      m.position.set(sx*(W/2)*(1-t), Hh*t, 0);
      m.castShadow=true; g.add(m);
    }
    const p=new THREE.Mesh(new THREE.BoxGeometry(len,0.09,D), woodM);
    p.position.set(sx*W/4, Hh/2, 0); p.rotation.z=-sx*slope;
    p.castShadow=p.receiveShadow=true; g.add(p);
  });
  const rdg=new THREE.Mesh(new THREE.BoxGeometry(0.14,0.14,D+0.3), barM);
  rdg.position.y=Hh; g.add(rdg);
  [W/2,-W/2].forEach(x=>{
    const m=new THREE.Mesh(new THREE.BoxGeometry(0.16,0.16,D+0.3), barM);
    m.position.set(x,0.08,0); g.add(m);
  });
  return g;
}

/** 차량 — 승합 · 스포츠 · 픽업 · 세단 */
const CAR_COL=[{body:0x1f9c72,roof:0x1f9c72,kind:'van'},
               {body:0x2331b8,roof:0x14161f,kind:'coupe'},
               {body:0xe8b423,roof:0xf2efe6,kind:'pickup'},
               {body:0xc9b489,roof:0xc9b489,kind:'sedan'}];
function buildCar(i){
  const g=new THREE.Group();
  const C=CAR_COL[i%CAR_COL.length];
  const bodyM=mat(C.body,.42,.55), roofM=mat(C.roof,.42,.55);
  const glass=new THREE.MeshStandardMaterial(
    {color:0x1b2530,roughness:.12,metalness:.85});
  const tire=mat(0x14161a,.95,.05);
  const van=C.kind==='van', pick=C.kind==='pickup', coupe=C.kind==='coupe';
  const L=van?4.9:pick?5.2:4.4, W=1.95, Hb=van?1.5:coupe?0.62:0.78;
  const b=new THREE.Mesh(new THREE.BoxGeometry(W,Hb,L), bodyM);
  b.position.y=0.62+Hb/2; b.castShadow=b.receiveShadow=true; g.add(b);
  if(!van){
    const cl=pick?1.9:coupe?2.1:2.4;
    const cab=new THREE.Mesh(new THREE.BoxGeometry(W*0.92,coupe?0.62:0.78,cl), roofM);
    cab.position.set(0,0.62+Hb+(coupe?0.31:0.39), pick?0.9:0.15);
    cab.castShadow=true; g.add(cab);
    const win=new THREE.Mesh(
      new THREE.BoxGeometry(W*0.94,coupe?0.42:0.52,cl*0.94), glass);
    win.position.copy(cab.position); g.add(win);
    if(pick){
      const bed=new THREE.Mesh(new THREE.BoxGeometry(W*0.94,0.55,2.0), bodyM);
      bed.position.set(0,0.62+Hb+0.27,-1.5); g.add(bed);
    }
  }else{
    const win=new THREE.Mesh(new THREE.BoxGeometry(W*1.01,0.62,L*0.62), glass);
    win.position.set(0,0.62+Hb*0.72,0.35); g.add(win);
  }
  [[-1,1],[1,1],[-1,-1],[1,-1]].forEach(v=>{
    const w=new THREE.Mesh(new THREE.CylinderGeometry(0.36,0.36,0.28,14), tire);
    w.rotation.z=Math.PI/2;
    w.position.set(v[0]*(W/2-0.06), 0.36, v[1]*(L/2-1.05));
    w.castShadow=true; g.add(w);
  });
  return g;
}

/**
 * 사람 — 어두운 지형 위에서 형태가 읽히도록 밝은 카키로 올렸다.
 * 머리 · 몸통 · 팔 · 다리를 나눠 실루엣이 사람으로 보이게 한다.
 */
function buildHuman(i){
  const g=new THREE.Group();
  const skin =mat(0xd8b189,.88,.02);          // 밝은 피부
  const cloth=mat(0xb9ad84,.90,.03);          // 밝은 카키 상의
  const pants=mat(0x7d7458,.92,.03);          // 하의
  const add=(geo,m,x,y,z,rz)=>{const o=new THREE.Mesh(geo,m);
    o.position.set(x,y,z); if(rz)o.rotation.z=rz;
    o.castShadow=true; o.receiveShadow=true; g.add(o); return o;};

  add(new THREE.CylinderGeometry(0.19,0.23,0.62,10), cloth, 0,1.20,0);   // 몸통
  add(new THREE.SphereGeometry(0.135,12,10), skin, 0,1.42,0);            // 어깨선
  add(new THREE.SphereGeometry(0.155,12,10), skin, 0,1.66,0);            // 머리
  add(new THREE.CylinderGeometry(0.055,0.05,0.58,7), skin, -0.245,1.16,0, 0.13);  // 팔
  add(new THREE.CylinderGeometry(0.055,0.05,0.58,7), skin,  0.245,1.16,0,-0.13);
  [-0.095,0.095].forEach(x=>
    add(new THREE.CylinderGeometry(0.085,0.075,0.90,7), pants, x,0.45,0));       // 다리
  [-0.095,0.095].forEach(x=>
    add(new THREE.BoxGeometry(0.13,0.08,0.26), mat(0x3a3428,.95,.03), x,0.04,0.04));

  return g;
}

/** 맵에 놓인 정지 전차 (소품 — 살아 있는 적이 아니다) */
function buildPropTank(i){
  const g=buildTank({hull:0x3a4038,turret:0x434a40,track:0x14170f,
                     metal:0x6d7466,accent:0x252a20,
                     ring:0x8d9099, emis:0x000000});
  if(g.userData.ring) g.userData.ring.visible=false;
  g.userData.turret.rotation.y=rnd(i+59,0,6.28);
  return g;
}

function buildMapObstacles(){
  if(!RN||!ST.obstacles) return;
  RN.mapG.clear(); RN.mapObjs=[]; FALLING=[];
  addTrees(ST.obstacles.filter(o=>o.kind==='Tree'));
  const BUILD={House:buildHouse, Tent:buildTent, Car:buildCar,
               Human:buildHuman, Tank:buildPropTank};
  let idx=0;
  ST.obstacles.filter(o=>o.kind!=='Tree').forEach(o=>{
    const f=BUILD[o.kind];
    let m;
    if(f) m=f(idx++);
    else { m=new THREE.Mesh(new THREE.BoxGeometry(2,2,2), mat(0x5b6470,.9,.05));
           m.castShadow=true; }
    const s=W2S(o.x,o.z);
    m.position.set(s[0], hAt(o.x,o.z)*VS, s[1]);
    RN.mapG.add(m);
    RN.mapObjs.push({kind:o.kind, x:o.x, z:o.z, obj:m, dead:false});
  });
}

function applyLayers(){
  if(!RN) return;
  RN.mapG.visible=L.mapobs; RN.knownG.visible=L.known; RN.detG.visible=L.det;
  if(RN.foeG) RN.foeG.visible=L.foes;
  RN.routeL.visible=L.route; RN.losL.visible=L.los;
  RN.trailMe.visible=RN.trailFoe.visible=L.trail;
  RN.ringMin.visible=RN.ringMax.visible=RN.ringSug.visible=L.rings;
  RN.shotG.visible=L.shots; RN.grid.visible=L.grid; RN.water.visible=L.water;
  RN.foeShotG.visible=L.foeshots;
  RN.meTag.visible=RN.foeTag.visible=L.label; RN.aimSp.visible=L.aim;
  if(RN.aimRay) RN.aimRay.visible=L.ray;
}

// ── 컨트롤 ──
$('#vs').oninput=e=>{VS=+e.target.value; $('#v-vs').textContent=VS.toFixed(1)+'×';
  paintTerrain(); if(S)update3D();};
$('#wl').oninput=e=>{WATERP=+e.target.value;
  $('#v-wl').textContent=waterLevel().toFixed(1)+' m'; paintTerrain();};
$('#surf').onchange=e=>{SURF=e.target.value; paintTerrain();};
$('#b-reset').onclick=()=>{if(!RN)return; const C=RN.cam3;
  C.yaw=.62;C.pit=.72;C.dis=340;C.tgt.set(0,0,0);follow=false;syncBtn();};
$('#b-top').onclick=()=>{if(!RN)return; const C=RN.cam3;
  C.pit=1.50;C.yaw=0;C.dis=330;C.tgt.set(0,0,0);follow=false;syncBtn();};
$('#b-follow').onclick=()=>{follow=!follow; if(follow&&RN)RN.cam3.dis=Math.min(RN.cam3.dis,90);
  syncBtn(); toast(follow?'아군 추적 켬':'아군 추적 끔');};
function syncBtn(){$('#b-follow').classList.toggle('on',follow);}

// ══════════════════════════════════════════
//  데이터
// ══════════════════════════════════════════
fetch(API+'/static-data').then(r=>r.json()).then(d=>{ST=d;boot();});
async function poll(){
  try{const r=await fetch(API+'/state',{cache:'no-store'});
    S=await r.json(); paint();}catch(e){}
}
setInterval(poll,220); poll();

const T=(v,u='')=>(v===null||v===undefined)?'–':(v+u);
function gateRow(n,pass,note){
  const c=pass===null?'off':(pass?'pass':'fail');
  const t=pass===null?'꺼짐':(pass?'통과':'차단');
  return `<div class="row"><span>${n}</span><b><span style="color:var(--dim);
    margin-right:6px;font-size:10px">${note||''}</span>
    <span class="pill ${c}">${t}</span></b></div>`;
}

function paint(){
  if(!S||!S.fire) return;
  const f=S.fire, tm=f.tm||{}, g=f.gates||{}, tk=f.track||{},
        dv=S.drive||{}, dt=S.detect||{};

  $('#r-t').textContent = tm.t!=null?('t '+tm.t.toFixed(1)):'';
  $('#t-st').textContent= T(f.state);
  $('#t-sp').textContent= tm.my_speed!=null?tm.my_speed.toFixed(1)+' m/s':'–';

  const hp=(v)=>v==null?100:Math.max(0,Math.min(100,+v));
  $('#h-my').textContent=T(tm.my_hp); $('#h-myb').style.width=hp(tm.my_hp)+'%';
  $('#h-en').textContent=T(tm.enemy_hp); $('#h-enb').style.width=hp(tm.enemy_hp)+'%';
  // HP 가 줄면 피격 점멸 (적탄에 맞은 경우도 잡힌다)
  if(paint._mhp!=null && tm.my_hp!=null && tm.my_hp<paint._mhp && RN)
    hitFlash(RN.me, RN.me.position, true);
  if(paint._ehp!=null && tm.enemy_hp!=null && tm.enemy_hp<paint._ehp && RN
     && !HITFX.some(h=>h.tank===RN.foe))
    hitFlash(RN.foe, RN.foe.position, false);
  paint._mhp=tm.my_hp; paint._ehp=tm.enemy_hp;

  if(tm.my&&tm.enemy){
    const a=[tm.my[0],tm.my[2]], e=[tm.enemy[0],tm.enemy[2]];
    $('#t-rg').textContent=dist2(a,e).toFixed(0)+' m';
    $('#t-br').textContent=brg(a,e).toFixed(0)+'°';
  }
  $('#r-d').textContent =T(f.dist,' m');
  $('#r-sg').textContent=T(f.suggest,' m');
  $('#r-ev').textContent=f.envelope?f.envelope[0]+'–'+f.envelope[1]+' m':'–';
  $('#r-ph').textContent=T(f.p_hit);
  $('#r-tf').textContent=T(f.tof,' s');
  $('#r-rl').textContent=f.reload_left>0?f.reload_left+' s':'준비';
  $('#r-fh').textContent=T(f.fired)+' / '+T(f.hits);
  $('#r-hr').textContent=T(f.hit_rate,' %');
  // 2026-08-26  차체 자세 경고
  //
  //   8/26 기동사격 27발 실측 — 차체 기울기가 명중률을 그대로 가른다.
  //       평평 (0°)        9발 9명중  100 %   평균오차 0.60 m
  //       6.3° / -10.8°    9발 8명중   89 %   평균오차 0.73 m
  //      12.8° / -11.2°    9발 6명중   67 %   평균오차 0.89 m
  //   빗나간 4발이 전부 기운 상태였다. 알고리즘이 아니라 선 자리 문제다.
  //   그래서 값만 보여주지 않고 색으로 경고한다.
  {
    const py=Math.abs(+tm.body_y||0), rz=Math.abs(+tm.body_z||0);
    const tilt=Math.max(py,rz);
    const e=$('#r-hull');
    e.textContent=T(tm.body_y,'°')+' / '+T(tm.body_z,'°');
    e.style.color = tilt>=8 ? 'var(--no)' : (tilt>=4 ? 'var(--warn)' : '');
    e.title = tilt>=8 ? '차체가 많이 기울었다. 실측 명중률 67 %'
            : (tilt>=4 ? '차체가 기울었다. 실측 명중률 89 %'
                       : '평평하다. 실측 명중률 100 %');
  }
  $('#r-tur').textContent=T(tm.turret_x,'°')+' / '+T(tm.turret_y,'°');
  $('#r-ko').textContent=(dv.known||[]).length+' 개';
  $('#r-do').textContent=T(dt.count)+' 개';

  const pm=g.p_hit_min,br2=g.body_rate_max,tf=g.tof_max;
  // 비과시간 게이트는 표적이 0.6 m/s 넘게 움직일 때만 적용된다.
  //   fire_control.py:  if (tof_max is not None and tracker.speed > 0.6 and ...)
  const tofActive = (tf!=null && (tk.speed||0) > 0.6);
  $('#gates').innerHTML=
    gateRow('탄도해',f.in_envelope===undefined?null:!!f.in_envelope,'')
  + gateRow('명중확률',pm==null?null:(f.p_hit!=null&&f.p_hit>=pm),
            (f.p_hit!=null?f.p_hit:'–')+'≥'+(pm==null?'—':pm))
  + gateRow('차체각속도',br2==null?null:true,br2==null?'제한없음':'≤'+br2)
  + gateRow('비과시간', !tofActive ? null : (f.tof!=null&&f.tof<=tf),
            (f.tof!=null?f.tof:'–')+'≤'+(tf==null?'—':tf)
            + (tofActive?'':'  정지표적 → 미적용'));

  $('#diag').innerHTML=diagnose(f, dv);
  $('#track').innerHTML=trackDiag(f);
  $('#drive').innerHTML=driveDiag(dv, tm);
  reconcileObstacles(dv);
  buildMarkers(); update3D(); drawMini();
}

/**
 * 플래너가 아는 장애물 목록과 대조해 이미 사라진 것을 지운다.
 *
 * pid_controller 의 set_obstacles() 는 목록을 통째로 교체하므로,
 * planner.obstacle_rectangles 가 현재 남아 있는 장애물의 정답이다.
 * 다만 시뮬레이터가 일부만 보내는 경우 오판할 수 있어,
 * 전체의 절반 이상이 들어왔을 때만 보정한다.
 */
function reconcileObstacles(dv){
  if(!RN||!RN.mapObjs) return;
  const kn=dv.known||[];
  const total=(ST&&ST.obstacles)?ST.obstacles.length:0;
  if(!total || kn.length < total*0.5) return;      // 부분 수신이면 건너뛴다
  const covered=(x,z,pad)=>kn.some(r=>
    x>=r.x0-pad && x<=r.x1+pad && z>=r.z0-pad && z<=r.z1+pad);
  RN.mapObjs.forEach(o=>{
    if(o.dead) return;
    if(!covered(o.x,o.z,1.5)){ o.dead=true;
      FALLING.push({obj:o.obj,t:0,dur:0.6,y0:o.obj.position.y,
                    tilt:(Math.random()-0.5)*1.4}); }
  });
  const T=RN.trees;
  if(T) T.rec.forEach(r=>{
    if(r.dead) return;
    if(!covered(r.x,r.z,1.2)){ r.dead=true;
      FALLING.push({tree:r,t:0,dur:0.5}); }
  });
}

/**
 * 기동표적 진단 — 목표 2 의 핵심 지표.
 *
 * 예측오차 ≈ k_err × 비과시간²  이다.
 * 비과시간이 제곱으로 들어가므로 거리를 좁히는 것이 가장 강력한 대응이다.
 */
function trackDiag(f){
  const tk=f.track||{}, sol=f.sol||{};
  const T=(v,u='')=>(v==null?'–':v+u);
  const moving=(tk.speed||0)>0.6;

  let html=`<div class="row"><span style="color:var(--acc);font-weight:600">표적 추적</span>
    <b>${moving?'<span class="pill fail">기동 중</span>'
               :'<span class="pill off">정지</span>'}</b></div>`;
  html+=`<div class="row"><span>표적 속도</span><b>${T(tk.speed,' m/s')}</b></div>`;
  if(moving){
    html+=`<div class="row"><span>진행 방위</span><b>${T(tk.heading,'°')}</b></div>`;
    html+=`<div class="row"><span>선회 각속도</span><b>${T(tk.omega,' °/s')}</b></div>`;
    html+=`<div class="row"><span>접선 가속</span><b>${T(tk.a_long,' m/s²')}</b></div>`;
  }
  html+=`<div class="row"><span>예측 채점</span><b>${T(tk.n_scored,' 회')}</b></div>`;

  // 예측 품질 — 이것이 명중률을 좌우한다
  if(tk.k_err!=null){
    const tof=f.tof||0;
    const pe=tk.pred_err;
    html+=`<div class="row"><span>k_err</span><b>${T(tk.k_err,' m/s²')}</b></div>`;
    if(pe!=null){
      const bad=pe>1.8;   // 표적 반폭
      html+=`<div class="row"><span>예상 예측오차</span>
        <b style="color:${bad?'var(--no)':'var(--ok)'}">${pe} m</b></div>`;
      html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--dim);
        line-height:1.55">${tk.k_err} × ${tof}² = <b style="color:var(--fg)">${pe} m</b>
        &nbsp;·&nbsp; 표적 반폭 1.8 m<br>`;
      if(bad && tof>0.6){
        // 반폭 안에 들어오려면 비과시간이 얼마여야 하나
        const tofNeed=Math.sqrt(1.8/Math.max(0.001,tk.k_err));
        html+=`<b style="color:var(--warn)">비과시간을 ${tofNeed.toFixed(2)} s 이하로 줄여야
          반폭 안에 들어온다.</b><br>지금 ${tof} s 다. 예측오차는 비과시간의
          <b>제곱</b>에 비례하므로 거리를 좁히는 것이 가장 효과적이다`;
      }else if(!bad){
        html+=`반폭 안에 들어온다. 예측은 문제없다`;
      }
      html+=`</div>`;
    }
  }
  if(tk.sig_lat!=null)
    html+=`<div class="row"><span>1σ 횡 / 종</span>
      <b>${T(tk.sig_lat)} / ${T(tk.sig_lon)} m</b></div>`;

  // 추적기가 죽어 있으면 그것부터 알려준다
  if(moving && (tk.n_scored||0)===0)
    html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--no);
      line-height:1.55">⚠ 표적이 움직이는데 예측 채점이 0 이다.
      추적기가 동작하지 않는다 — 리드 계산이 안 되므로 기동표적은 못 맞춘다</div>`;
  return html;
}

/**
 * 주행 진단 — 왜 저 경로로 가는지 답한다.
 *
 * 핵심: 전차는 '적'이 아니라 '목적지'를 향해 간다.
 *       목적지는 시뮬레이터가 /set_destination 으로 정해준다.
 *       그래서 적과 엉뚱한 방향으로 가는 것처럼 보일 수 있다.
 */
function driveDiag(dv, tm){
  if(!dv||!dv.available) return '';
  const T=(v,u='')=>(v==null?'–':v+u);
  let html=`<div class="row"><span style="color:var(--acc);font-weight:600">주행</span>
    <b>${dv.arrived?'<span class="pill pass">도착</span>'
                   :(dv.stop_flag?'<span class="pill off">정지</span>':'이동중')}</b></div>`;
  html+=`<div class="row"><span>목적지</span><b>${
    dv.dest?`(${dv.dest[0]}, ${dv.dest[1]})`:'없음'}</b></div>`;
  // 2026-08-26  목적지가 '전차가 기울지 않는 자리' 인가.
  //   기울면 명중률이 100 % -> 67 % 로 떨어진다 (8/26 27발 실측).
  //   터미널 [평지보정] 메시지가 /info 로그에 묻혀 안 보인다는
  //   지적이 있어 같은 정보를 여기에도 띄운다.
  if(dv.dest_tilt!=null){
    const t=dv.dest_tilt;
    const g = t<=2 ? ['적합','pass'] : t<=5 ? ['무난','pass']
            : t<=10 ? ['주의','off'] : ['부적합','fail'];
    html+=`<div class="row"><span>진지 기울기</span>
      <b><span style="color:var(--dim);margin-right:6px;font-size:10px">
      ${t.toFixed(1)}°</span><span class="pill ${g[1]}">${g[0]}</span></b></div>`;
  }
  html+=`<div class="row"><span>남은 직선거리</span><b>${T(dv.straight,' m')}</b></div>`;
  html+=`<div class="row"><span>계획 경로 길이</span><b>${T(dv.path_len,' m')}</b></div>`;
  if(dv.detour!=null){
    const bad=dv.detour>1.6;
    html+=`<div class="row"><span>우회율</span>
      <b style="color:${bad?'var(--no)':'var(--fg)'}">${dv.detour} 배</b></div>`;
    if(bad) html+=`<div style="padding:2px 13px 6px;font-size:10.5px;color:var(--warn);
      line-height:1.5">직선의 ${dv.detour}배로 돌아간다. 장애물 여유폭
      (clearance_radius 8 m)이 넓어 좁은 길을 피하는 것이다</div>`;
  }
  html+=`<div class="row"><span>목적지 변경</span><b>${T(dv.dest_n,' 회')}</b></div>`;
  html+=`<div class="row"><span>경로 재계산</span><b>${T(dv.replan_n,' 회')}</b></div>`;

  // 목적지와 적이 다른 방향이면 그걸 명시한다 — 가장 흔한 오해다
  if(dv.dest && tm && tm.my && tm.enemy){
    const me=[tm.my[0],tm.my[2]], en=[tm.enemy[0],tm.enemy[2]];
    const bDest=brg(me,dv.dest), bEn=brg(me,en);
    const gap=Math.abs(((bDest-bEn+540)%360)-180);
    const dEn=dist2(me,en);
    html+=`<div class="row"><span>목적지 / 적 방위차</span>
      <b style="color:${gap>45?'var(--no)':'var(--fg)'}">${gap.toFixed(0)}°</b></div>`;
    if(gap>45)
      html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--warn);
        line-height:1.55">목적지(${bDest.toFixed(0)}°)와 적(${bEn.toFixed(0)}°, ${dEn.toFixed(0)} m)이
        <b>${gap.toFixed(0)}° 다른 방향</b>이다.<br>
        전차는 적이 아니라 목적지를 향해 간다 — 시뮬레이터가
        /set_destination 으로 준 좌표다. 사격은 그 이동 중에 곁다리로 이뤄진다.</div>`;
  }
  if(dv.dests && dv.dests.length>1){
    html+=`<div style="padding:2px 13px 7px;font-size:10px;color:var(--dim);
      line-height:1.5">최근 목적지: ${dv.dests.map(d=>`(${d[0]},${d[1]})`).join(' → ')}</div>`;
  }
  return html;
}

/**
 * 사격 진단 — 지금 무엇이 발사를 막고 있는지 한 줄로 답한다.
 * 판단 순서는 fire_control 의 게이트 순서를 그대로 따른다.
 */
function diagnose(f, dv){
  const g=f.gates||{}, tm=f.tm||{}, tk=f.track||{}, A=f.aim||{};

  // FireControl.state 가 곧 게이트 판정 결과다. 그대로 해석한다.
  const st=(f.state||'').toUpperCase();
  const yE=Math.abs(A.yaw_err||0),   yD=A.yaw_db||0;
  const pE=Math.abs(A.pitch_err||0), pD=A.pitch_db||0;

  const D={
    FIRE:    ['발사','pass','모든 게이트 통과'],
    RELOAD:  ['재장전 중','off', (f.reload_left||0)+' s 남음'],
    SLEW:    ['포탑 정렬 중','fail',
      `조준오차 yaw ${yE.toFixed(2)}° / 허용 ${yD.toFixed(2)}°` +
      (yE>yD?' <b style="color:var(--no)">← 초과</b>':'') +
      ` &nbsp;·&nbsp; pitch ${pE.toFixed(2)}° / 허용 ${pD.toFixed(2)}°` +
      (pE>pD?' <b style="color:var(--no)">← 초과</b>':'') +
      (yD<0.3||pD<0.2 ? '<br>데드밴드가 매우 좁다. 거리가 멀수록 더 좁아진다 — '
        +'db_safety 를 올리면 완화된다' : '')],
    TOFCAP:  ['비과시간 초과','fail',
      `비행 ${f.tof} s > 상한 ${g.tof_max} s. 표적이 ${tk.speed} m/s 로 움직일 때만 걸리는 게이트다. `
      +'거리를 좁히거나 tof_max 를 0.9 로 올려야 한다'],
    TURNING: ['차체 선회 중','fail',
      `차체 각속도가 ${g.body_rate_max} °/s 를 넘었다. 멎기를 기다린다`],
    SETTLE:  ['차체 정지 대기','off','moving_fire 가 꺼져 있다'],
    HOLD:    ['명중확률 미달','fail',
      `p_hit ${f.p_hit} < 기준 ${g.p_hit_min}`
      +(tk.speed>1.0?` — 표적이 ${tk.speed} m/s 로 움직여 예측이 흔들린다`:'')
      +`<br>${g.patience} s 기다리면 기준이 ${g.p_hit_floor} 까지 내려간다`],
    NOTRACK: ['포탑 추적 한계','fail',
      `표적 각속도가 포탑보다 빠르다 (duty ${A.track_duty} > ${A.duty_max}). 거리를 벌려야 한다`],
    INHIBIT: ['사격 억제','fail','외부에서 발사를 막고 있다'],
    IDLE:    ['대기','off','탄도해가 없거나 표적이 없다'],
  };

  let why,cls,fix;
  if(D[st]){ why=D[st][0]; cls=D[st][1]; fix=D[st][2]; }
  else if(!tm.enemy){ why='적 위치 없음'; cls='fail'; fix='텔레메트리에 enemyPos 가 안 온다'; }
  else if(f.dist!=null && f.envelope && f.dist>f.envelope[1]){
    why='최대 사거리 밖'; cls='fail';
    fix=`적까지 ${f.dist} m — ${f.envelope[1]} m 안으로 접근해야 한다`; }
  else { why=st||'—'; cls='off'; fix=''; }

  // ── 시각 어긋남 — 이게 있으면 다른 진단은 의미가 없다 ──
  let html='';
  if(f.desync){
    html+=`<div style="margin:2px 9px 8px;padding:8px 10px;border-radius:7px;
      background:rgba(248,113,113,.14);border:1px solid rgba(248,113,113,.45)">
      <div style="color:var(--no);font-weight:800;font-size:11px;margin-bottom:4px">
        ⚠ 사격 모듈 시각 어긋남</div>
      <div style="font-size:10.5px;color:var(--dim);line-height:1.55">
        마지막 발사 시각이 <b style="color:var(--fg)">${f.desync.last_fire} s</b> 인데
        현재는 <b style="color:var(--fg)">${f.desync.now} s</b> 다
        (${f.desync.gap} s 미래).<br>
        에피소드를 Restart 해서 sim_time 이 되감겼는데
        FireControl.last_fire_t 가 이전 판 값을 들고 있다.<br>
        <b style="color:var(--no)">이 상태로는 한 발도 못 쏜다.</b></div>
      <button class="btn" style="margin-top:7px;width:100%" onclick="fixDesync()">
        시각 상태 초기화</button></div>`;
  }
  html+=`<div class="row"><span>사격 진단</span>
    <b><span class="pill ${cls}">${why}</span></b></div>`;
  if(fix) html+=`<div style="padding:2px 13px 7px;font-size:10.5px;color:var(--dim);
    line-height:1.55">${fix}</div>`;
  // 조준 정렬 현황은 항상 보여준다
  if(A.yaw_db!==undefined){
    const bar=(e,d)=>{const r=d>0?Math.min(1,e/d):1;
      const c=e<=d?'var(--ok)':'var(--no)';
      return `<span style="display:inline-block;width:52px;height:4px;border-radius:2px;
        background:rgba(255,255,255,.12);vertical-align:middle;overflow:hidden">
        <i style="display:block;height:100%;width:${(r*100).toFixed(0)}%;background:${c}"></i></span>`;};
    html+=`<div class="row"><span>방위 정렬</span><b>${bar(yE,yD)}
      <span style="margin-left:6px">${yE.toFixed(2)} / ${yD.toFixed(2)}°</span></b></div>
      <div class="row"><span>앙각 정렬</span><b>${bar(pE,pD)}
      <span style="margin-left:6px">${pE.toFixed(2)} / ${pD.toFixed(2)}°</span></b></div>`;
  }

  // ── 탄착오차 분해 진단 ────────────────────────────
  // 조준 오차가 작은데 탄착이 벌어지면 원인은 데드밴드가 아니다.
  // 오차를 사선 기준 앞뒤/좌우로 쪼개야 범인이 갈린다.
  const sh=(f.shots||[]).filter(s=>s.imp);
  if(sh.length>=2){
    const avgOf=k=>{const v=sh.map(s=>s[k]).filter(x=>x!=null);
      return v.length? v.reduce((a,b)=>a+b,0)/v.length : null;};
    const absOf=k=>{const v=sh.map(s=>s[k]).filter(x=>x!=null).map(Math.abs);
      return v.length? v.reduce((a,b)=>a+b,0)/v.length : null;};
    const mAll=absOf('miss'), R=avgOf('rng_err'), C=avgOf('crs_err');
    const aR=absOf('rng_err'), aC=absOf('crs_err');
    const hits=sh.filter(s=>s.hit).length;
    const terr=sh.filter(s=>s.kind==='terrain').length;

    const sgn=(v,pos,neg)=>v==null?'':(v>0?pos:neg);
    html+=`<div class="row"><span>평균 탄착오차</span>
      <b>${mAll!=null?mAll.toFixed(1)+' m':'–'}</b></div>`;
    if(R!=null) html+=`<div class="row"><span>앞뒤 오차</span>
      <b style="color:${aR>aC?'var(--no)':'var(--fg)'}">${R>0?'+':''}${R.toFixed(1)} m
      <span style="font-size:10px;color:var(--dim)">${sgn(R,'길게','짧게')}</span></b></div>`;
    if(C!=null) html+=`<div class="row"><span>좌우 오차</span>
      <b style="color:${aC>aR?'var(--no)':'var(--fg)'}">${C>0?'+':''}${C.toFixed(1)} m
      <span style="font-size:10px;color:var(--dim)">${sgn(C,'오른쪽','왼쪽')}</span></b></div>`;

    if(hits===0 && aR!=null && aC!=null){
      const d=f.dist||1, sol=f.sol||{}, drdt=sol.drdt||6.8;
      let note;
      const aimTight = yE<=yD*1.5 && pE<=pD*1.5;
      if(aR > aC*2){
        // 앞뒤가 지배 → 앙각 계열
        const degEq=(aR/drdt);
        note=`<b style="color:var(--fg)">앞뒤 오차가 좌우의 ${(aR/Math.max(0.01,aC)).toFixed(1)}배</b>다. `
          +`앙각 계열 문제다.<br>거리오차 ${aR.toFixed(1)} m ÷ drdt ${drdt} = 앙각 `
          +`<b style="color:var(--fg)">${degEq.toFixed(2)}°</b> 어긋난 셈이다.<br>`
          +(aimTight
            ? `조준 오차는 ${pE.toFixed(2)}° 뿐인데 탄착은 ${degEq.toFixed(2)}° 만큼 벌어졌다 — `
              +`<b style="color:var(--warn)">데드밴드가 아니라 탄도해 자체가 틀렸다.</b> `
              +`차체 피치 ${tm.body_y}° 가 앙각 기준에 반영되지 않았을 가능성이 크다`
            : `조준이 아직 안 잡혔다. 정렬을 먼저 확인할 것`)
          + (R<0 ? '<br>일관되게 <b>짧게</b> 떨어진다 — 앙각을 올려야 한다'
                 : '<br>일관되게 <b>길게</b> 떨어진다 — 앙각을 내려야 한다');
      }else if(aC > aR*2){
        const degEq=Math.atan2(aC,d)*180/Math.PI;
        note=`<b style="color:var(--fg)">좌우 오차가 앞뒤의 ${(aC/Math.max(0.01,aR)).toFixed(1)}배</b>다. `
          +`방위 계열 문제다.<br>${aC.toFixed(1)} m / ${d} m = 방위 `
          +`<b style="color:var(--fg)">${degEq.toFixed(2)}°</b> 어긋난 셈이다.<br>`
          +`차체 롤 ${tm.body_z}° 가 포탑 yaw 를 월드 기준에서 틀어놓고 있을 수 있다`;
      }else{
        note=`앞뒤 ${aR.toFixed(1)} m · 좌우 ${aC.toFixed(1)} m 로 비슷하다. `
          +`차체 자세 전반(피치 ${tm.body_y}° / 롤 ${tm.body_z}°)이 원인일 가능성이 크다`;
      }
      html+=`<div style="padding:3px 13px 7px;font-size:10.5px;color:var(--dim);
        line-height:1.55">${sh.length}발 중 명중 ${hits} · 지면착탄 ${terr}<br>${note}</div>`;
    }
  }
  return html;
}

// ── 마커 ──
let mkSig='';
function buildMarkers(){
  const f=S.fire||{},tm=f.tm||{},dv=S.drive||{},dt=S.detect||{};
  const it=[];
  const push=(c,n,s,x,z)=>it.push({c,n,s,x,z});
  if(tm.my)   push('#4b9cf5','아군 기갑',
    `(${tm.my[0].toFixed(0)}, ${tm.my[2].toFixed(0)})  ▲${hAt(tm.my[0],tm.my[2]).toFixed(0)}m`,
    tm.my[0],tm.my[2]);
  if(tm.enemy)push('#ef4444','적 기갑',
    `(${tm.enemy[0].toFixed(0)}, ${tm.enemy[2].toFixed(0)})  ${tm.my?
      dist2([tm.my[0],tm.my[2]],[tm.enemy[0],tm.enemy[2]]).toFixed(0)+'m':''}`,
    tm.enemy[0],tm.enemy[2]);
  if(dv.dest) push('#f5a524','목적지',
    `(${dv.dest[0].toFixed(0)}, ${dv.dest[1].toFixed(0)})  ${dv.arrived?'도착':'이동중'}`,
    dv.dest[0],dv.dest[1]);
  (dt.objects||[]).slice(0,10).forEach(o=>{
    const x=o.x!=null?o.x:o.posX, z=o.z!=null?o.z:o.posZ;
    if(x==null||z==null)return;
    push('#eab308',(o.className||o.name||'미상'),
      `(${(+x).toFixed(0)}, ${(+z).toFixed(0)})  탐지`,+x,+z);});
  const sig=it.map(i=>i.n+i.s).join('|'); if(sig===mkSig)return; mkSig=sig;
  $('#mk-n').textContent=it.length;
  const b=$('#markers'); b.innerHTML='';
  it.forEach(i=>{const d=document.createElement('div'); d.className='mk';
    d.innerHTML=`<u style="width:9px;height:9px;border-radius:2px;background:${i.c};
      display:inline-block;flex:none"></u><div><b>${i.n}</b><span>${i.s}</span></div>`;
    d.onclick=()=>{if(!RN)return; const s=W2S(i.x,i.z);
      RN.cam3.tgt.set(s[0],hAt(i.x,i.z)*VS,s[1]);
      RN.cam3.dis=Math.min(RN.cam3.dis,110); follow=false; syncBtn();};
    b.appendChild(d);});
}

// ── 사선 차폐 ──
function losBlocked(a,b){
  if(!TR)return false;
  const ha=hAt(a[0],a[1])+2.5,hb=hAt(b[0],b[1])+2.5,N=72;
  for(let i=1;i<N;i++){const t=i/N;
    if(hAt(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)>ha+(hb-ha)*t+0.7)return true;}
  return false;
}

// ── 3D 갱신 ──
let lastShot=-1;
function update3D(){
  if(!RN||!S||!S.fire)return;
  const f=S.fire,tm=f.tm||{},dv=S.drive||{};

  if(tm.my){
    const p=V(tm.my[0],tm.my[2],0);
    const shakeMe=HITFX.some(h=>h.tank===RN.me);
    if(shakeMe) HITFX.forEach(h=>{if(h.tank===RN.me) h.base.copy(p);});
    else RN.me.position.copy(p);
    RN.me.rotation.order='YXZ';
    RN.me.rotation.y=yawOf(tm.body_x);                     // 차체 방위
    RN.me.rotation.z=(tm.body_z||0)*Math.PI/180;           // 롤
    RN.me.rotation.x=(tm.body_y||0)*Math.PI/180;           // 피치
    // playerTurretX 는 월드 절대 방위다(차체각 포함).
    // 포탑은 차체의 자식이므로 차체 회전을 상쇄해야 실제 방위를 향한다.
    RN.me.userData.turret.rotation.y = yawOf(tm.turret_x) - yawOf(tm.body_x);
    RN.me.userData.gun.rotation.x = -(tm.turret_y||0)*Math.PI/180;   // 앙각(+가 위)
    RN.meTag.position.copy(p).setY(p.y+12);
    // 포신 지향선 — 실제로 어디를 겨누는지 눈으로 확인한다
    if(RN.aimRay){
      const th=(tm.turret_x||0)*Math.PI/180, el=(tm.turret_y||0)*Math.PI/180;
      const R0=dist2([tm.my[0],tm.my[2]],
        tm.enemy?[tm.enemy[0],tm.enemy[2]]:[tm.my[0],tm.my[2]+60])||60;
      const pts=[];
      for(let k=0;k<=16;k++){
        const d=R0*1.15*k/16;
        const x=tm.my[0]+Math.sin(th)*d, z=tm.my[2]+Math.cos(th)*d;
        const w=W2S(x,z);
        pts.push(new THREE.Vector3(w[0], p.y+3.0+Math.tan(el)*d*VS*0.35, w[1]));
      }
      RN.aimRay.geometry.setFromPoints(pts);
      RN.aimRay.computeLineDistances();     // 점선 재계산 (필수)
    }
  }
  if(tm.enemy){
    const p=V(tm.enemy[0],tm.enemy[2],0);
    // 피격 점멸 중에는 흔들림이 덮어쓰지 않도록 기준 위치만 갱신한다.
    const shaking=HITFX.some(h=>h.tank===RN.foe);
    if(shaking) HITFX.forEach(h=>{if(h.tank===RN.foe) h.base.copy(p);});
    else RN.foe.position.copy(p);
    RN.foe.rotation.y=yawOf(tm.enemy_body_x);
    // 적 포탑은 우리 쪽을 향하게 둔다 (적 포탑각은 텔레메트리에 없다)
    if(tm.my){
      const bk=brg([tm.enemy[0],tm.enemy[2]],[tm.my[0],tm.my[2]]);
      RN.foe.userData.turret.rotation.y=yawOf(bk)-yawOf(tm.enemy_body_x);
    }
    RN.foeTag.position.copy(p).setY(p.y+12);
  }
  // 조준점
  if(f.aim3){ RN.aimSp.visible=L.aim;
    RN.aimSp.position.copy(V(f.aim3[0],f.aim3[2],3.4)); }
  else RN.aimSp.visible=false;

  // 경로 — 지형에 밀착
  if(dv.path&&dv.path.length>1){
    const pts=[];
    for(let i=0;i<dv.path.length-1;i++){
      const a=dv.path[i],b=dv.path[i+1],seg=Math.max(2,Math.round(dist2(a,b)/3));
      for(let k=0;k<=seg;k++){const t=k/seg;
        pts.push(V(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,0.9));}}
    RN.routeL.geometry.setFromPoints(pts);
  }
  // 이동 궤적
  const tr=f.trail||{};
  [['my',RN.trailMe],['enemy',RN.trailFoe]].forEach(([k,ln])=>{
    const q=tr[k]||[]; if(q.length>1)
      ln.geometry.setFromPoints(q.map(p=>V(p[0],p[1],0.7)));});

  // 사선
  if(tm.my&&tm.enemy){
    const a=[tm.my[0],tm.my[2]],e=[tm.enemy[0],tm.enemy[2]];
    const bl=losBlocked(a,e);
    RN.losL.material.color.setHex(bl?0xf87171:0x4ade80);
    RN.losL.geometry.setFromPoints([V(a[0],a[1],2.6),V(e[0],e[1],2.6)]);
  }
  // 사거리 링
  if(tm.my&&f.envelope){
    const ring=r=>{const p=[];
      for(let d=0;d<=360;d+=3){const t=d*Math.PI/180;
        const x=tm.my[0]+Math.sin(t)*r,z=tm.my[2]+Math.cos(t)*r;
        if(x<0||x>SPAN||z<0||z>SPAN)continue; p.push(V(x,z,0.6));}
      return p;};
    RN.ringMin.geometry.setFromPoints(ring(f.envelope[0]));
    RN.ringMax.geometry.setFromPoints(ring(f.envelope[1]));
    if(f.suggest) RN.ringSug.geometry.setFromPoints(ring(f.suggest));
  }
  // ── 추정 적 전차 — 크기·맵대조로 찾아낸 것. 항상 표시한다. ──
  const foes=(dv.known||[]).filter(o=>o.g);
  if(RN.foeG.children.length!==foes.length){
    RN.foeG.clear();
    foes.forEach(()=>{
      const grp=new THREE.Group();
      const box=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),
        new THREE.MeshBasicMaterial({color:0xff2b21,wireframe:true,
          transparent:true,opacity:.95}));
      const ring=new THREE.Mesh(new THREE.RingGeometry(4.0,4.9,40),
        new THREE.MeshBasicMaterial({color:0xff2b21,transparent:true,
          opacity:.55,side:THREE.DoubleSide,depthWrite:false}));
      ring.rotation.x=-Math.PI/2;
      const sp=new THREE.Sprite(new THREE.SpriteMaterial(
        {map:new THREE.CanvasTexture(tagCanvas('적 전차 (추정)','#ff2b21')),
         depthTest:false,transparent:true}));
      const im=sp.material.map.image;
      sp.scale.set(im.width*0.052,im.height*0.052,1); sp.renderOrder=22;
      grp.add(box); grp.add(ring); grp.add(sp);
      grp.userData={box,ring,sp};
      RN.foeG.add(grp);
    });
  }
  foes.forEach((o,i)=>{
    const g2=RN.foeG.children[i]; if(!g2)return;
    const p=V(o.cx,o.cz,0);
    const w=Math.max(2,o.x1-o.x0), dd=Math.max(2,o.z1-o.z0);
    g2.userData.box.scale.set(w,3.0,dd);
    g2.userData.box.position.copy(p).setY(p.y+1.6);
    g2.userData.ring.position.copy(p).setY(p.y+0.25);
    g2.userData.sp.position.copy(p).setY(p.y+13);
    g2.userData.ring.material.opacity=0.35+Math.sin(performance.now()*0.004)*0.22;
  });

  // 인지 장애물
  const kn=dv.known||[];
  if(RN.knownG.children.length!==kn.length){
    RN.knownG.clear();
    const C={enemy_tank:0xef4444,enemy:0xf97316,team_tank:0x4b9cf5,
             team:0x60a5fa,unknown:0xeab308,nature:0x5b6b7d};
    // 자연물은 수가 많아 아주 흐리게, 위협 대상만 또렷하게 그린다.
    const OP={enemy_tank:.85,enemy:.75,unknown:.60,team_tank:.6,team:.5,nature:.14};
    kn.forEach(o=>{
      const w=Math.max(1,o.x1-o.x0),d2=Math.max(1,o.z1-o.z0);
      const m=new THREE.Mesh(new THREE.BoxGeometry(w,3.2,d2),
        new THREE.MeshBasicMaterial({color:C[o.t]||0x5b6b7d,wireframe:true,
          transparent:true,opacity:OP[o.t]!==undefined?OP[o.t]:.3,
          depthWrite:false}));
      const cx=(o.x0+o.x1)/2,cz=(o.z0+o.z1)/2;
      m.position.copy(V(cx,cz,1.8)); RN.knownG.add(m);});
  }
  // 탐지 객체
  const objs=(S.detect.objects||[]).filter(o=>o.x!=null||o.posX!=null);
  if(RN.detG.children.length!==objs.length){
    RN.detG.clear();
    objs.forEach(()=>{const m=new THREE.Mesh(new THREE.OctahedronGeometry(1.6),
      new THREE.MeshBasicMaterial({color:0xeab308,wireframe:true}));
      RN.detG.add(m);});
  }
  objs.forEach((o,i)=>{const x=+(o.x!=null?o.x:o.posX),z=+(o.z!=null?o.z:o.posZ);
    if(RN.detG.children[i]) RN.detG.children[i].position.copy(V(x,z,4));});

  // ── 탄도 · 탄착 ──
  const shots=(f.shots||[]).filter(s=>s.fire&&s.imp);
  if(RN.shotG.children.length!==shots.length*2){
    RN.shotG.clear();
    shots.forEach(s=>{
      const arc=new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({color:s.hit?0x4ade80:0xa78bfa,
          transparent:true,opacity:.55}));
      const mk=new THREE.Mesh(new THREE.SphereGeometry(s.hit?1.5:1.0,10,8),
        new THREE.MeshBasicMaterial({color:s.hit?0x4ade80:0xfbbf24,
          transparent:true,opacity:.85}));
      RN.shotG.add(arc); RN.shotG.add(mk);
    });
  }
  shots.forEach((s,i)=>{
    const arc=RN.shotG.children[i*2], mk=RN.shotG.children[i*2+1];
    if(!arc||!mk)return;
    const a=s.fire,b=s.imp,N=28,pts=[];
    const h0=hAt(a[0],a[1])*VS+2.6, h1=hAt(b[0],b[1])*VS+0.5;
    const lift=Math.min(28,dist2(a,b)*0.22);
    for(let k=0;k<=N;k++){const t=k/N;
      const x=a[0]+(b[0]-a[0])*t, z=a[1]+(b[1]-a[1])*t;
      const y=h0+(h1-h0)*t + lift*Math.sin(Math.PI*t);
      const w=W2S(x,z);
      pts.push(new THREE.Vector3(w[0],y,w[1]));}
    arc.geometry.setFromPoints(pts);
    mk.position.copy(V(b[0],b[1],1.0));
  });
  // ── 적 사격 (2026-09-09) ──
  //   9/09 46번 판에서 적이 8발을 쐈는데 3D 화면에 하나도 안 나왔다.
  //   이 상황도는 아군 FireModule 만 읽는데, 적 사격 기록은 5100 번
  //   서버에만 있었기 때문이다. 이제 _foe_poll() 이 가져온다.
  //
  //   색으로 가른다.  아군 초록/보라  <->  적 주황
  const fs=((f.foe||{}).shots||[]).filter(s=>s.fire&&s.imp);
  if(RN.foeShotG.children.length!==fs.length*2){
    RN.foeShotG.clear();
    fs.forEach(s=>{
      const arc=new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({color:s.hit?0xf97316:0x9a3412,
          transparent:true,opacity:.5}));
      const mk=new THREE.Mesh(new THREE.SphereGeometry(s.hit?1.5:1.0,10,8),
        new THREE.MeshBasicMaterial({color:s.hit?0xf97316:0xea580c,
          transparent:true,opacity:.8}));
      RN.foeShotG.add(arc); RN.foeShotG.add(mk);
    });
  }
  fs.forEach((s,i)=>{
    const arc=RN.foeShotG.children[i*2], mk=RN.foeShotG.children[i*2+1];
    if(!arc||!mk)return;
    const a=s.fire,b=s.imp,N=28,pts=[];
    const h0=hAt(a[0],a[1])*VS+2.6, h1=hAt(b[0],b[1])*VS+0.5;
    const lift=Math.min(28,dist2(a,b)*0.22);
    for(let k=0;k<=N;k++){const t=k/N;
      const x=a[0]+(b[0]-a[0])*t, z=a[1]+(b[1]-a[1])*t;
      const y=h0+(h1-h0)*t + lift*Math.sin(Math.PI*t);
      const w=W2S(x,z);
      pts.push(new THREE.Vector3(w[0],y,w[1]));}
    arc.geometry.setFromPoints(pts);
    mk.position.copy(V(b[0],b[1],1.0));
  });

  // 새 사격 → 포탄 발사 (비행 + 탄착 폭발)
  if(shots.length){
    if(lastShot<0){ lastShot=shots[0].id; }        // 첫 폴링은 과거분 무시
    else if(shots[0].id!==lastShot){
      // 마지막으로 본 것 이후의 새 사격을 오래된 것부터 재생
      const fresh=[];
      for(const s of shots){ if(s.id===lastShot) break; fresh.push(s); }
      fresh.reverse().forEach((s,k)=>setTimeout(()=>spawnShell(s),k*180));
      lastShot=shots[0].id;
    }
  }
}

// ── 나침반 ──
function drawCompass(yaw){
  const c=$('#comp'),g=c.getContext('2d'),S2=116,R=44;
  g.clearRect(0,0,S2,S2); g.save(); g.translate(S2/2,S2/2);
  g.fillStyle='rgba(11,16,24,.85)';g.beginPath();g.arc(0,0,R+7,0,7);g.fill();
  g.strokeStyle='rgba(120,160,210,.22)';g.lineWidth=1.5;g.stroke();
  g.rotate(yaw);
  for(let a=0;a<360;a+=15){const t=a*Math.PI/180,big=a%90===0;
    g.strokeStyle=big?'#38bdf8':'rgba(120,160,210,.3)';g.lineWidth=big?2.4:1.2;
    g.beginPath();g.moveTo(Math.sin(t)*(R-(big?12:6)),-Math.cos(t)*(R-(big?12:6)));
    g.lineTo(Math.sin(t)*R,-Math.cos(t)*R);g.stroke();}
  g.fillStyle='#ef4444';g.beginPath();
  g.moveTo(0,-R+1);g.lineTo(-7,-R+16);g.lineTo(7,-R+16);g.closePath();g.fill();
  g.restore();
  g.fillStyle='#e6edf7';g.font='800 12px Consolas';g.textAlign='center';
  g.fillText('N',S2/2,S2/2-R+30);
}

// ── 축소 지도 ──
// 2026-08-26  제목줄을 눌러 접었다 펼 수 있다. 기본은 접힘.
//   210 x 210 캔버스가 우측 사격 패널을 통째로 가려서 필요할 때만 편다.
function toggleMini(){
  const m=$('#mini'); m.classList.toggle('fold');
  $('#mmt').textContent = m.classList.contains('fold') ? '펴기' : '접기';
  if(!m.classList.contains('fold')) sizeMini();
}
function sizeMini(){const c=$('#mmc'),r=c.getBoundingClientRect();
  if(!r.width) return;
  c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;drawMini();}
function drawMini(){
  const c=$('#mmc'); if(!c.width||!TR)return;
  if($('#mini').classList.contains('fold')) return;   // 접혀 있으면 그리지 않는다
  const g=c.getContext('2d'),P=Math.min(c.width,c.height);
  const X=v=>v/SPAN*P, Z=v=>P-(v/SPAN*P);
  g.clearRect(0,0,c.width,c.height);
  if(!drawMini._bg){          // 지형 배경은 한 번만 그린다
    const n=TR.n, off=document.createElement('canvas'); off.width=off.height=n;
    const og=off.getContext('2d'), im=og.createImageData(n,n);
    const rng=(TR.hi-TR.lo)||1, wl=waterLevel();
    for(let j=0;j<n;j++)for(let i=0;i<n;i++){
      const k=idx(i,j),h=TR.H[k],u=(h-TR.lo)/rng,q=(j*n+i)*4;
      let r,gg,b;
      if(h<wl+0.6){r=42;gg=92;b=124;}
      else if(u<0.4){r=64;gg=94;b=54;}
      else if(u<0.7){r=112;gg=116;b=68;}
      else {r=150;gg=140;b=118;}
      im.data[q]=r;im.data[q+1]=gg;im.data[q+2]=b;im.data[q+3]=255;}
    og.putImageData(im,0,0); drawMini._bg=off;
  }
  g.imageSmoothingEnabled=true;
  g.drawImage(drawMini._bg,0,0,P,P);
  const D=devicePixelRatio, f=(S&&S.fire)||{}, tm=f.tm||{}, dv=(S&&S.drive)||{};
  if(dv.path&&dv.path.length>1){g.strokeStyle='#f5a524';g.lineWidth=1.8*D;
    g.beginPath();dv.path.forEach((p,i)=>i?g.lineTo(X(p[0]),Z(p[1])):g.moveTo(X(p[0]),Z(p[1])));
    g.stroke();}
  const tr=f.trail||{};
  [['my','rgba(75,156,245,.55)'],['enemy','rgba(239,68,68,.55)']].forEach(([k,c2])=>{
    const q=tr[k]||[]; if(q.length<2)return; g.strokeStyle=c2;g.lineWidth=1.2*D;
    g.beginPath();q.forEach((p,i)=>i?g.lineTo(X(p[0]),Z(p[1])):g.moveTo(X(p[0]),Z(p[1])));
    g.stroke();});
  if(tm.my&&f.envelope){g.setLineDash([4*D,4*D]);g.strokeStyle='rgba(56,189,248,.5)';
    g.lineWidth=1*D;g.beginPath();g.arc(X(tm.my[0]),Z(tm.my[2]),f.envelope[1]/SPAN*P,0,7);
    g.stroke();g.setLineDash([]);}
  const dot=(x,z,c2,r)=>{g.fillStyle=c2;g.beginPath();g.arc(X(x),Z(z),r*D,0,7);g.fill();
    g.strokeStyle='rgba(0,0,0,.6)';g.lineWidth=1*D;g.stroke();};
  if(dv.dest) dot(dv.dest[0],dv.dest[1],'#f5a524',3);
  if(tm.enemy)dot(tm.enemy[0],tm.enemy[2],'#ef4444',4);
  if(tm.my){dot(tm.my[0],tm.my[2],'#4b9cf5',4);
    // 차체 방위(가는 선) + 포신 방위(굵은 선)
    const bx=(tm.body_x||0)*Math.PI/180, tx=(tm.turret_x||0)*Math.PI/180;
    const ray=(ang,len,col,w)=>{g.strokeStyle=col;g.lineWidth=w*D;g.beginPath();
      g.moveTo(X(tm.my[0]),Z(tm.my[2]));
      g.lineTo(X(tm.my[0])+Math.sin(ang)*len*D,Z(tm.my[2])-Math.cos(ang)*len*D);g.stroke();};
    ray(bx,11,'rgba(142,197,255,.55)',1.2);
    ray(tx,18,'#9fe8ff',2.0);}
}
</script></body></html>
"""


# Display-only cockpit; classic source remains intact.
_CLASSIC_HTML = _HTML
_HTML = _HTML.replace("</body>", '<script>\nconst FINAL_OBSTACLES=[{"kind": "Tree", "x": 96.70482635498047, "y": 14.916854858398438, "z": 130.12631225585938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 106.38688659667969, "y": 14.868049621582031, "z": 122.05781555175781, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 94.807861328125, "y": 14.372745513916016, "z": 117.55658721923828, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 157.02603149414062, "y": 10.351617813110352, "z": 113.45384979248047, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 152.18130493164062, "y": 10.913677215576172, "z": 106.24524688720703, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 140.72610473632812, "y": 11.413089752197266, "z": 102.11046600341797, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 97.67256164550781, "y": 13.834335327148438, "z": 110.49729919433594, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 92.89462280273438, "y": 13.318243026733398, "z": 103.2445297241211, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 90.8392333984375, "y": 13.045394897460938, "z": 92.15980529785156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "House", "x": 141.73350524902344, "y": 15.29325008392334, "z": 177.53546142578125, "rotation": {"x": 0.014566343277692795, "y": -0.007475273683667183, "z": -0.0063259173184633255, "w": 0.9998459815979004}}, {"kind": "House", "x": 143.5216064453125, "y": 16.63852310180664, "z": 158.21337890625, "rotation": {"x": 0.005687370430678129, "y": -0.01101476326584816, "z": -0.009708312340080738, "w": 0.999876081943512}}, {"kind": "House", "x": 156.1811065673828, "y": 14.7683744430542, "z": 166.32330322265625, "rotation": {"x": -0.0077208117581903934, "y": 0.002448252635076642, "z": -0.007342829369008541, "w": 0.9999402761459351}}, {"kind": "Tree", "x": 134.88113403320312, "y": 9.465188026428223, "z": 100.85075378417969, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 84.07483673095703, "y": 13.07322883605957, "z": 89.23763275146484, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 82.02613067626953, "y": 13.064298629760742, "z": 80.66999053955078, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 79.65726470947266, "y": 13.302350997924805, "z": 69.17756652832031, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 77.32071685791016, "y": 11.877519607543945, "z": 58.377044677734375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 70.86592102050781, "y": 11.393400192260742, "z": 50.00878143310547, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 66.26700592041016, "y": 11.324689865112305, "z": 44.49330520629883, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 128.14219665527344, "y": 11.699453353881836, "z": 98.75043487548828, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 117.1995620727539, "y": 12.107637405395508, "z": 90.36480712890625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 109.14920806884766, "y": 12.625303268432617, "z": 76.57752990722656, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 105.72832489013672, "y": 12.928003311157227, "z": 63.2777099609375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 107.27690124511719, "y": 12.84298324584961, "z": 50.47983169555664, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 102.46004486083984, "y": 12.888559341430664, "z": 40.66581726074219, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 94.81741333007812, "y": 11.87217903137207, "z": 30.241153717041016, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 153.99069213867188, "y": 10.66592788696289, "z": 140.9176483154297, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 170.10177612304688, "y": 11.40304183959961, "z": 162.73818969726562, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 161.614013671875, "y": 15.956621170043945, "z": 181.2543487548828, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 184.49038696289062, "y": 10.319921493530273, "z": 147.398193359375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 191.84246826171875, "y": 9.580789566040039, "z": 126.64569854736328, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 164.5305938720703, "y": 10.237037658691406, "z": 108.60320281982422, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 172.42315673828125, "y": 10.096416473388672, "z": 106.8546371459961, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 188.2216339111328, "y": 9.93703842163086, "z": 108.80563354492188, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 197.8970184326172, "y": 10.320829391479492, "z": 125.52484893798828, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 49.90999221801758, "y": 14.401786804199219, "z": 61.980438232421875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 56.116275787353516, "y": 13.92686653137207, "z": 67.00292205810547, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 61.03843688964844, "y": 12.634088516235352, "z": 84.12574768066406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 61.053157806396484, "y": 11.965597152709961, "z": 97.43379211425781, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 75.84140014648438, "y": 13.456808090209961, "z": 113.89202117919922, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 77.29843139648438, "y": 16.351011276245117, "z": 134.40890502929688, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 70.40619659423828, "y": 23.715421676635742, "z": 176.00958251953125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 57.58247756958008, "y": 19.934221267700195, "z": 170.5890655517578, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 63.738121032714844, "y": 12.291723251342773, "z": 111.69442749023438, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 76.33989715576172, "y": 15.270870208740234, "z": 126.90557098388672, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 58.413177490234375, "y": 14.649959564208984, "z": 139.82188415527344, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 54.03324508666992, "y": 13.36212158203125, "z": 80.5915298461914, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 47.86014938354492, "y": 14.569732666015625, "z": 63.830299377441406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 49.326690673828125, "y": 14.467550277709961, "z": 57.897361755371094, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 60.79076385498047, "y": 13.732686996459961, "z": 62.13726806640625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 64.01112365722656, "y": 13.018495559692383, "z": 76.24678039550781, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 63.95523452758789, "y": 12.209287643432617, "z": 91.31392669677734, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 69.43115234375, "y": 12.240119934082031, "z": 100.94148254394531, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 77.2727279663086, "y": 13.528701782226562, "z": 113.42279815673828, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 84.61434173583984, "y": 15.721378326416016, "z": 130.0625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 70.79969787597656, "y": 23.61100959777832, "z": 175.3215789794922, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 74.6633529663086, "y": 23.83072853088379, "z": 175.77462768554688, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 82.38079833984375, "y": 23.569293975830078, "z": 174.77947998046875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 64.68707275390625, "y": 14.177875518798828, "z": 215.2760772705078, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 69.41914367675781, "y": 14.099405288696289, "z": 210.93736267089844, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 74.89867401123047, "y": 15.598986625671387, "z": 206.41012573242188, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 75.67180633544922, "y": 24.018543243408203, "z": 185.74874877929688, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 76.42725372314453, "y": 23.414960861206055, "z": 172.4781951904297, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 60.3661003112793, "y": 13.354263305664062, "z": 73.27600860595703, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 50.13200759887695, "y": 14.256736755371094, "z": 53.2945671081543, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 99.1400146484375, "y": 12.473550796508789, "z": 30.082744598388672, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 120.91566467285156, "y": 11.810928344726562, "z": 29.247804641723633, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 137.27284240722656, "y": 11.343000411987305, "z": 29.82886505126953, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 154.23660278320312, "y": 10.769487380981445, "z": 31.6667423248291, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 119.15436553955078, "y": 12.105558395385742, "z": 39.04806900024414, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 119.35882568359375, "y": 12.31890869140625, "z": 53.66444778442383, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 121.12324523925781, "y": 12.273645401000977, "z": 65.89472961425781, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 125.36579132080078, "y": 11.930530548095703, "z": 84.01130676269531, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 139.99061584472656, "y": 11.417762756347656, "z": 94.5255126953125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 149.72882080078125, "y": 11.186670303344727, "z": 97.53658294677734, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 167.4971466064453, "y": 10.420015335083008, "z": 103.85929870605469, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 180.4774932861328, "y": 10.009599685668945, "z": 107.11067199707031, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 189.1918487548828, "y": 10.222064971923828, "z": 100.81107330322266, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 163.00523376464844, "y": 11.50222396850586, "z": 92.75679016113281, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 148.48965454101562, "y": 11.185928344726562, "z": 83.27058410644531, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 137.6851043701172, "y": 11.426204681396484, "z": 75.29814910888672, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 129.56936645507812, "y": 11.850906372070312, "z": 68.96809387207031, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 126.03363800048828, "y": 12.0352783203125, "z": 59.5358772277832, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 128.42906188964844, "y": 11.803295135498047, "z": 44.828765869140625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 133.68690490722656, "y": 11.575231552124023, "z": 51.59650421142578, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 139.27894592285156, "y": 11.2440185546875, "z": 53.79604721069336, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 140.01553344726562, "y": 11.197978973388672, "z": 58.44195556640625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 151.5017547607422, "y": 10.952535629272461, "z": 74.44734954833984, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 166.432373046875, "y": 12.25674819946289, "z": 84.37826538085938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 178.32931518554688, "y": 10.682939529418945, "z": 93.38005828857422, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 182.771484375, "y": 10.329654693603516, "z": 94.76921844482422, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 187.16709899902344, "y": 10.207677841186523, "z": 95.06692504882812, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 177.79263305664062, "y": 11.490324020385742, "z": 80.42635345458984, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 153.25686645507812, "y": 10.750247955322266, "z": 70.49702453613281, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 149.36512756347656, "y": 10.630376815795898, "z": 62.65446090698242, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 138.34860229492188, "y": 11.283855438232422, "z": 49.19342803955078, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 137.70822143554688, "y": 11.323883056640625, "z": 47.21226119995117, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 139.7280731201172, "y": 11.197633743286133, "z": 44.17915725708008, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 148.7218475341797, "y": 10.708736419677734, "z": 45.44282150268555, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 156.91543579101562, "y": 10.301877975463867, "z": 50.421836853027344, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 162.2880859375, "y": 10.377832412719727, "z": 55.40986633300781, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 165.85977172851562, "y": 10.971611022949219, "z": 64.03815460205078, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 165.39328002929688, "y": 11.82911491394043, "z": 74.6102066040039, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 163.89674377441406, "y": 12.177867889404297, "z": 81.03279876708984, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 172.4454345703125, "y": 11.95671272277832, "z": 74.7156753540039, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 186.03555297851562, "y": 11.454185485839844, "z": 75.49395751953125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 188.8632049560547, "y": 10.23985481262207, "z": 85.53462982177734, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 199.7943878173828, "y": 10.328649520874023, "z": 98.2307357788086, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 206.03445434570312, "y": 10.146688461303711, "z": 113.26025390625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 207.615478515625, "y": 10.431756973266602, "z": 124.39408874511719, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 205.62252807617188, "y": 10.327360153198242, "z": 119.8450698852539, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 197.9816131591797, "y": 9.851644515991211, "z": 116.27669525146484, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 196.37051391601562, "y": 9.760320663452148, "z": 116.34916687011719, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 171.68980407714844, "y": 16.398937225341797, "z": 187.18838500976562, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 178.80209350585938, "y": 15.00092887878418, "z": 170.75257873535156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 200.04554748535156, "y": 13.476396560668945, "z": 164.59243774414062, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 201.0501251220703, "y": 13.322547912597656, "z": 144.32058715820312, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 192.8326416015625, "y": 13.835247039794922, "z": 174.99789428710938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 191.05148315429688, "y": 14.247957229614258, "z": 163.33514404296875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 191.1605682373047, "y": 14.210777282714844, "z": 162.63119506835938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 119.93669128417969, "y": 18.001419067382812, "z": 187.13284301757812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 119.25408935546875, "y": 15.007287979125977, "z": 216.38739013671875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 121.34030151367188, "y": 15.76991081237793, "z": 207.70477294921875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 124.87942504882812, "y": 17.586015701293945, "z": 182.9417266845703, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 128.0423583984375, "y": 16.9360408782959, "z": 200.57843017578125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 85.57643127441406, "y": 23.259044647216797, "z": 174.0067901611328, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 82.91146850585938, "y": 21.70252227783203, "z": 191.78125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 65.64106750488281, "y": 17.07637596130371, "z": 153.50479125976562, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 59.671077728271484, "y": 13.918449401855469, "z": 131.5335693359375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 68.03518676757812, "y": 15.577194213867188, "z": 148.39830017089844, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 60.86294174194336, "y": 13.279959678649902, "z": 136.7318878173828, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 60.173004150390625, "y": 12.200461387634277, "z": 127.96517181396484, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 64.20877838134766, "y": 13.076577186584473, "z": 130.64833068847656, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 59.69926452636719, "y": 14.620426177978516, "z": 151.36868286132812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 61.17749786376953, "y": 21.781963348388672, "z": 179.42787170410156, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 61.08834457397461, "y": 14.764650344848633, "z": 207.9913787841797, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 56.8271369934082, "y": 17.379886627197266, "z": 200.77035522460938, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 71.02880096435547, "y": 17.033313751220703, "z": 233.95240783691406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 68.18367767333984, "y": 16.974321365356445, "z": 232.50225830078125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 210.79257202148438, "y": 10.295304298400879, "z": 147.53817749023438, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 181.62362670898438, "y": 9.076615333557129, "z": 93.3126220703125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 152.28228759765625, "y": 9.410418510437012, "z": 135.57659912109375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 152.9503173828125, "y": 8.81125545501709, "z": 132.515625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 80.90708923339844, "y": 13.561041831970215, "z": 122.4966049194336, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 70.25791931152344, "y": 11.006941795349121, "z": 104.9202880859375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 60.391990661621094, "y": 11.247136116027832, "z": 84.9493408203125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 72.82534790039062, "y": 11.415751457214355, "z": 109.08840942382812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 63.39377212524414, "y": 12.030959129333496, "z": 70.04934692382812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 153.0274200439453, "y": 9.92611026763916, "z": 92.81622314453125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 132.05966186523438, "y": 10.2897367477417, "z": 91.25273895263672, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 170.06985473632812, "y": 9.685805320739746, "z": 95.88941192626953, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 154.938232421875, "y": 10.039313316345215, "z": 78.54103088378906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 138.15634155273438, "y": 10.011549949645996, "z": 70.62613677978516, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 105.83695983886719, "y": 11.23622989654541, "z": 35.117828369140625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 122.73181915283203, "y": 10.60210132598877, "z": 38.438053131103516, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 137.6067657470703, "y": 9.954344749450684, "z": 38.213966369628906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 190.69796752929688, "y": 10.895649909973145, "z": 148.6371307373047, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 189.22789001464844, "y": 11.873305320739746, "z": 156.3870391845703, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 184.4264373779297, "y": 12.783387184143066, "z": 166.04368591308594, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 181.76959228515625, "y": 10.257462501525879, "z": 192.88868713378906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 171.03160095214844, "y": 10.558627128601074, "z": 214.7111358642578, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 161.66769409179688, "y": 14.942258834838867, "z": 185.8172149658203, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 157.968505859375, "y": 12.7966947555542, "z": 208.972412109375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 162.96499633789062, "y": 11.722668647766113, "z": 213.4644317626953, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 165.3807830810547, "y": 15.054210662841797, "z": 191.1511993408203, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 167.79302978515625, "y": 15.477027893066406, "z": 181.28781127929688, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 181.26795959472656, "y": 11.75932788848877, "z": 189.17726135253906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 159.38180541992188, "y": 13.514666557312012, "z": 205.2670440673828, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 164.4561767578125, "y": 9.079495429992676, "z": 104.76642608642578, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 140.8058624267578, "y": 9.785359382629395, "z": 54.69185256958008, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 147.57748413085938, "y": 9.349446296691895, "z": 57.39439010620117, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 143.31149291992188, "y": 9.61608600616455, "z": 56.419090270996094, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 156.8645782470703, "y": 9.055184364318848, "z": 63.983917236328125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 158.16748046875, "y": 9.677277565002441, "z": 71.55247497558594, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 53.83198547363281, "y": 13.389341354370117, "z": 128.4658966064453, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 54.542720794677734, "y": 14.684209823608398, "z": 141.27532958984375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 63.04765319824219, "y": 16.244001388549805, "z": 223.63555908203125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 56.4687614440918, "y": 21.776670455932617, "z": 179.54232788085938, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 171.2926788330078, "y": 11.756636619567871, "z": 227.69944763183594, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 163.8911895751953, "y": 12.196171760559082, "z": 225.9965057373047, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 252.60862731933594, "y": 13.511550903320312, "z": 128.62734985351562, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 247.0039825439453, "y": 12.144109725952148, "z": 143.51608276367188, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 246.94000244140625, "y": 13.419105529785156, "z": 162.55625915527344, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 246.1246337890625, "y": 11.96282958984375, "z": 178.90965270996094, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 241.74179077148438, "y": 11.733806610107422, "z": 208.51736450195312, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 242.03555297851562, "y": 12.249271392822266, "z": 212.2994842529297, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 241.85580444335938, "y": 11.452932357788086, "z": 206.23716735839844, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 243.6733856201172, "y": 10.558130264282227, "z": 197.24676513671875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 256.01470947265625, "y": 13.47755241394043, "z": 118.8557357788086, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 265.2669372558594, "y": 12.58364486694336, "z": 109.00320434570312, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 268.948974609375, "y": 12.045989990234375, "z": 99.42890167236328, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 273.12017822265625, "y": 13.154088973999023, "z": 81.58382415771484, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 271.213134765625, "y": 12.395586013793945, "z": 93.2691879272461, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 271.9546203613281, "y": 13.02655029296875, "z": 71.82147979736328, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 269.8798828125, "y": 13.073434829711914, "z": 62.11323547363281, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 266.64459228515625, "y": 12.699773788452148, "z": 55.17182922363281, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 207.4891815185547, "y": 10.326080322265625, "z": 104.05941009521484, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 205.5640411376953, "y": 10.235811233520508, "z": 90.37025451660156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 204.6187744140625, "y": 10.814088821411133, "z": 78.29296875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 202.43923950195312, "y": 13.832235336303711, "z": 67.67933654785156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 203.83921813964844, "y": 16.182395935058594, "z": 57.439292907714844, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 190.90179443359375, "y": 15.268440246582031, "z": 54.0035514831543, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 183.75067138671875, "y": 14.048919677734375, "z": 50.12630081176758, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 192.89215087890625, "y": 14.253395080566406, "z": 65.53073120117188, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 174.2010955810547, "y": 12.208520889282227, "z": 64.63621520996094, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 177.65318298339844, "y": 12.419652938842773, "z": 67.5023193359375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 196.85342407226562, "y": 10.141304016113281, "z": 88.58651733398438, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 203.33889770507812, "y": 17.455114364624023, "z": 50.98720169067383, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 195.56727600097656, "y": 19.04623031616211, "z": 43.42543029785156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 183.72946166992188, "y": 16.819459915161133, "z": 38.265602111816406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 166.09185791015625, "y": 10.601417541503906, "z": 38.151527404785156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 148.9171905517578, "y": 10.769773483276367, "z": 41.38238525390625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 170.7390899658203, "y": 11.671838760375977, "z": 48.921173095703125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 164.532470703125, "y": 10.562009811401367, "z": 51.5460090637207, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 173.78616333007812, "y": 12.216087341308594, "z": 61.95887756347656, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 188.74281311035156, "y": 11.382879257202148, "z": 75.85993194580078, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 204.0988006591797, "y": 13.026317596435547, "z": 143.82179260253906, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 200.5149688720703, "y": 13.843988418579102, "z": 159.7112579345703, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 200.8321533203125, "y": 13.443801879882812, "z": 145.5970001220703, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 206.64308166503906, "y": 10.368114471435547, "z": 122.4270248413086, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Human", "x": 239.97598266601562, "y": 2.2025420665740967, "z": 94.0415267944336, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Human", "x": 129.00582885742188, "y": 9.674386024475098, "z": 37.303733825683594, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "House", "x": 142.99449157714844, "y": 14.964720726013184, "z": 142.95640563964844, "rotation": {"x": -0.020104052498936653, "y": 0.012707259505987167, "z": -0.016070052981376648, "w": 0.9995879530906677}}, {"kind": "House", "x": 155.07894897460938, "y": 11.723335266113281, "z": 150.54383850097656, "rotation": {"x": 0.003866030601784587, "y": 0.028888100758194923, "z": -0.007507674396038055, "w": 0.999547004699707}}, {"kind": "Tree", "x": 154.28668212890625, "y": 10.599748611450195, "z": 38.646820068359375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 154.04818725585938, "y": 10.595613479614258, "z": 38.937255859375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.56735229492188, "y": 10.79301643371582, "z": 36.38561248779297, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.902587890625, "y": 10.768770217895508, "z": 37.019107818603516, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.902587890625, "y": 10.768770217895508, "z": 37.019107818603516, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.902587890625, "y": 10.768770217895508, "z": 37.019107818603516, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.902587890625, "y": 10.768770217895508, "z": 37.019107818603516, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.902587890625, "y": 10.768770217895508, "z": 37.019107818603516, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 152.252685546875, "y": 10.506988525390625, "z": 48.026763916015625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 151.86517333984375, "y": 10.503795623779297, "z": 49.59719467163086, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 148.4046630859375, "y": 10.655315399169922, "z": 53.90335464477539, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 148.45724487304688, "y": 10.652027130126953, "z": 54.39470672607422, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 147.21124267578125, "y": 10.729907989501953, "z": 55.955421447753906, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.43722534179688, "y": 10.551231384277344, "z": 51.91978073120117, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.43722534179688, "y": 10.551231384277344, "z": 51.91978073120117, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.43722534179688, "y": 10.551231384277344, "z": 51.91978073120117, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 150.43722534179688, "y": 10.551231384277344, "z": 51.91978073120117, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 80.37828826904297, "y": 14.528493881225586, "z": 119.47293853759766, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 68.57698822021484, "y": 12.283597946166992, "z": 94.14366912841797, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 78.20703125, "y": 12.886100769042969, "z": 102.14190673828125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 86.04215240478516, "y": 14.285737991333008, "z": 115.2065658569336, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 67.66709899902344, "y": 12.727975845336914, "z": 81.10892486572266, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 54.670867919921875, "y": 14.070587158203125, "z": 58.5551872253418, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 46.32527160644531, "y": 13.882427215576172, "z": 114.98106384277344, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 44.48670196533203, "y": 14.28818130493164, "z": 104.12507629394531, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 43.12513732910156, "y": 14.853286743164062, "z": 97.91029357910156, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 46.951805114746094, "y": 13.525007247924805, "z": 96.48098754882812, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 49.62005615234375, "y": 12.84829330444336, "z": 108.98323822021484, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 40.92987060546875, "y": 16.165546417236328, "z": 86.92083740234375, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 40.50973892211914, "y": 16.707687377929688, "z": 81.51673126220703, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 38.239418029785156, "y": 17.766429901123047, "z": 75.84994506835938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 35.193687438964844, "y": 18.384885787963867, "z": 67.65391540527344, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 28.733354568481445, "y": 20.213483810424805, "z": 55.76759719848633, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 28.98383903503418, "y": 19.989870071411133, "z": 60.50270462036133, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 32.19771957397461, "y": 19.170562744140625, "z": 57.467750549316406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 28.91642189025879, "y": 20.65253257751465, "z": 72.8895034790039, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 32.644020080566406, "y": 18.77068519592285, "z": 95.33191680908203, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 36.454811096191406, "y": 17.498489379882812, "z": 111.5117416381836, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 38.02508544921875, "y": 17.423439025878906, "z": 124.7520980834961, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 42.421852111816406, "y": 15.82205581665039, "z": 125.50798034667969, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 47.4017333984375, "y": 14.293027877807617, "z": 127.32001495361328, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 41.97740173339844, "y": 19.31088638305664, "z": 145.87774658203125, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 42.25447463989258, "y": 18.92971420288086, "z": 159.0692138671875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 46.0224609375, "y": 18.122962951660156, "z": 162.14866638183594, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 57.72721862792969, "y": 15.526094436645508, "z": 147.93724060058594, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 62.73774337768555, "y": 15.56568717956543, "z": 143.1965789794922, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 38.54279327392578, "y": 20.342111587524414, "z": 145.41197204589844, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 38.937171936035156, "y": 15.694259643554688, "z": 172.91481018066406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 41.21628952026367, "y": 17.354389190673828, "z": 180.65516662597656, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 49.24258041381836, "y": 18.18227767944336, "z": 170.06771850585938, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 49.341278076171875, "y": 18.063478469848633, "z": 166.16432189941406, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 53.44834899902344, "y": 12.191646575927734, "z": 105.17525482177734, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 51.35676193237305, "y": 17.814105987548828, "z": 161.6571807861328, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 53.75463104248047, "y": 16.25397300720215, "z": 151.24636840820312, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tank", "x": 96.85845947265625, "y": 14.194231033325195, "z": 147.2486572265625, "rotation": {"x": -0.09765312075614929, "y": 0.7039141654968262, "z": -0.04182393476366997, "w": 0.7022959589958191}}, {"kind": "Tank", "x": 95.42987823486328, "y": 15.753525733947754, "z": 154.16806030273438, "rotation": {"x": -0.06679417937994003, "y": 0.7022325396537781, "z": -0.07841891795396805, "w": 0.7044562101364136}}, {"kind": "Tree", "x": 90.65467071533203, "y": 23.397510528564453, "z": 178.44508361816406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 79.4809799194336, "y": 22.896587371826172, "z": 188.94358825683594, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tent", "x": 72.53744506835938, "y": 12.382383346557617, "z": 289.14752197265625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 51.972145080566406, "y": 18.43059730529785, "z": 202.3113555908203, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 43.64801025390625, "y": 18.59097671508789, "z": 200.26034545898438, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 37.08545684814453, "y": 19.312942504882812, "z": 196.39173889160156, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 48.042484283447266, "y": 18.4222469329834, "z": 212.53184509277344, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 48.25239944458008, "y": 18.617599487304688, "z": 222.17906188964844, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 48.51878356933594, "y": 16.728107452392578, "z": 230.0331268310547, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 52.50572204589844, "y": 17.267786026000977, "z": 225.34942626953125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 53.981441497802734, "y": 17.179235458374023, "z": 216.2646484375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 53.84228515625, "y": 17.222383499145508, "z": 216.1360321044922, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 31.510053634643555, "y": 19.47675132751465, "z": 218.19264221191406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 32.853843688964844, "y": 17.096324920654297, "z": 231.89483642578125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 68.27520751953125, "y": 17.245100021362305, "z": 225.30593872070312, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 75.18667602539062, "y": 16.362445831298828, "z": 221.20462036132812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 81.72858428955078, "y": 14.855043411254883, "z": 215.02203369140625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 84.12066650390625, "y": 14.65778923034668, "z": 211.00282287597656, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 89.9200668334961, "y": 21.92202377319336, "z": 192.53598022460938, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 92.59068298339844, "y": 23.142520904541016, "z": 189.4481658935547, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 111.26929473876953, "y": 18.776731491088867, "z": 190.09963989257812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 111.61279296875, "y": 17.245182037353516, "z": 203.36415100097656, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 112.33277893066406, "y": 16.541528701782227, "z": 210.23318481445312, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 107.30835723876953, "y": 16.145416259765625, "z": 220.15460205078125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 119.13796997070312, "y": 16.115955352783203, "z": 207.52516174316406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 108.3805923461914, "y": 16.596406936645508, "z": 210.36740112304688, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 49.556396484375, "y": 19.192665100097656, "z": 196.6924285888672, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 40.63437271118164, "y": 19.27775764465332, "z": 195.4336700439453, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 41.17171096801758, "y": 19.345245361328125, "z": 191.89382934570312, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 54.33884048461914, "y": 20.034934997558594, "z": 195.21144104003906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 61.49363708496094, "y": 21.972219467163086, "z": 193.52076721191406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 81.7378158569336, "y": 15.84211540222168, "z": 205.23941040039062, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 109.16944122314453, "y": 16.83050537109375, "z": 207.9093017578125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 117.8305435180664, "y": 16.73611068725586, "z": 203.80189514160156, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 59.34109878540039, "y": 15.440956115722656, "z": 214.72027587890625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 55.57044982910156, "y": 16.64601707458496, "z": 213.96343994140625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 60.79765319824219, "y": 22.727628707885742, "z": 177.84239196777344, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 62.519622802734375, "y": 21.410320281982422, "z": 171.67237854003906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 55.103515625, "y": 21.60203742980957, "z": 184.09423828125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Rock", "x": 9.200470924377441, "y": 9.886741638183594, "z": 157.50460815429688, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Rock", "x": 5.0531463623046875, "y": 9.886741638183594, "z": 165.795166015625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 128.50242614746094, "y": 17.838905334472656, "z": 190.82894897460938, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tree", "x": 130.38729858398438, "y": 14.660032272338867, "z": 208.32041931152344, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tank", "x": 258.1126708984375, "y": 14.350055694580078, "z": 144.8376922607422, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tank", "x": 258.4700927734375, "y": 15.13542652130127, "z": 150.18695068359375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tank", "x": 258.77215576171875, "y": 16.022930145263672, "z": 157.73037719726562, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": 0.7071068286895752}}, {"kind": "Tank", "x": 260.2694091796875, "y": 14.51253890991211, "z": 138.61387634277344, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tank", "x": 143.48155212402344, "y": 10.699457168579102, "z": 132.57806396484375, "rotation": {"x": 0.14768612384796143, "y": 0.7007120251655579, "z": 0.01821603812277317, "w": -0.697753369808197}}, {"kind": "Tent", "x": 279.92169189453125, "y": 19.549602508544922, "z": 145.85548400878906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 171.79537963867188, "y": 12.288695335388184, "z": 205.25149536132812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 174.7890625, "y": 14.987491607666016, "z": 189.89694213867188, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 178.52777099609375, "y": 15.660051345825195, "z": 183.4609375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 178.2097930908203, "y": 13.707559585571289, "z": 166.32974243164062, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 178.9376983642578, "y": 11.266018867492676, "z": 158.07858276367188, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 204.6738739013672, "y": 13.156578063964844, "z": 153.79690551757812, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 204.68856811523438, "y": 12.177937507629395, "z": 136.89593505859375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 190.93251037597656, "y": 10.506630897521973, "z": 137.7700653076172, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 195.25949096679688, "y": 9.877211570739746, "z": 111.60740661621094, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 168.3380889892578, "y": 11.954724311828613, "z": 225.0699005126953, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 160.12680053710938, "y": 12.376399040222168, "z": 224.2899169921875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 161.2822723388672, "y": 12.412150382995605, "z": 220.95919799804688, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 43.24761962890625, "y": 17.772600173950195, "z": 141.25787353515625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 61.20012283325195, "y": 12.991432189941406, "z": 121.97402954101562, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 67.36002349853516, "y": 14.028938293457031, "z": 124.63394927978516, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 46.05021667480469, "y": 19.355270385742188, "z": 186.6727294921875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 66.27267456054688, "y": 17.321706771850586, "z": 204.40249633789062, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 35.311729431152344, "y": 19.486358642578125, "z": 219.4080047607422, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 49.501365661621094, "y": 13.311178207397461, "z": 119.19677734375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 79.56376647949219, "y": 13.161100387573242, "z": 106.84822082519531, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 74.81024169921875, "y": 12.655648231506348, "z": 93.555908203125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 72.20719909667969, "y": 12.627278327941895, "z": 88.08573150634766, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 50.659847259521484, "y": 12.842466354370117, "z": 93.05683898925781, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 43.79290008544922, "y": 15.565694808959961, "z": 74.5716552734375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 66.7807846069336, "y": 13.406038284301758, "z": 63.459171295166016, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 73.02570343017578, "y": 13.150436401367188, "z": 73.41565704345703, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 56.821224212646484, "y": 12.611313819885254, "z": 50.520931243896484, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 63.67735290527344, "y": 12.660323143005371, "z": 56.05145263671875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 36.288246154785156, "y": 17.548545837402344, "z": 59.17578887939453, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 183.45484924316406, "y": 13.787317276000977, "z": 58.39989471435547, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 195.9298858642578, "y": 10.77502727508545, "z": 79.31427001953125, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 117.66674041748047, "y": 12.33516788482666, "z": 76.84111022949219, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 140.30479431152344, "y": 11.334696769714355, "z": 81.83815002441406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 152.71832275390625, "y": 11.384724617004395, "z": 87.24911499023438, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 174.9532470703125, "y": 11.317570686340332, "z": 89.7114486694336, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 128.0861358642578, "y": 11.698241233825684, "z": 34.12527084350586, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 39.17399215698242, "y": 18.612289428710938, "z": 138.0909881591797, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 38.2042350769043, "y": 20.225778579711914, "z": 154.7968292236328, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 246.6746826171875, "y": 13.16357421875, "z": 154.90231323242188, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Tree", "x": 248.2395477294922, "y": 12.24484920501709, "z": 139.10960388183594, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 246.1468963623047, "y": 10.879922866821289, "z": 225.24903869628906, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 247.062744140625, "y": 11.752031326293945, "z": 216.09226989746094, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 236.14210510253906, "y": 8.93699836730957, "z": 247.7978515625, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 227.37571716308594, "y": 8.43691635131836, "z": 244.2158660888672, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 21.766260147094727, "y": 14.976007461547852, "z": 106.55165100097656, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 29.70309066772461, "y": 16.83571434020996, "z": 102.3974609375, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 24.735984802246094, "y": 16.08159828186035, "z": 123.25563049316406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 21.221418380737305, "y": 17.674293518066406, "z": 214.74378967285156, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 16.72467613220215, "y": 17.173397064208984, "z": 213.5135498046875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 142.5615997314453, "y": 9.454399108886719, "z": 115.02391052246094, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 163.61892700195312, "y": 13.411378860473633, "z": 201.37322998046875, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 158.2216033935547, "y": 10.729633331298828, "z": 220.88034057617188, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}, {"kind": "Rock", "x": 25.342191696166992, "y": 16.104917526245117, "z": 113.56636047363281, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Rock", "x": 15.367907524108887, "y": 9.988401412963867, "z": 164.00643920898438, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 5.597458839416504, "y": 11.818506240844727, "z": 132.82711791992188, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tree", "x": 22.473262786865234, "y": 19.2967529296875, "z": 208.640869140625, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Rock", "x": 18.52288818359375, "y": 13.694856643676758, "z": 121.21538543701172, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tent", "x": 78.89331817626953, "y": 19.960552215576172, "z": 153.87322998046875, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"kind": "Tank", "x": 142.13079833984375, "y": 10.27641773223877, "z": 127.32789611816406, "rotation": {"x": 0.0, "y": 0.7071068286895752, "z": 0.0, "w": -0.7071068286895752}}];\n// All changes below affect visualization only, never controller inputs.\nconst visualEffects=[];\nconst visualSeen=new Map();\nlet visualLastTime=null,visualInitialized=false;\nfunction disposeVisual(object){object.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose())});if(object.parent)object.parent.remove(object);}\nconst originalDecodeTerrain=decodeTerrain;\ndecodeTerrain=function(data){const t=originalDecodeTerrain(data);if(!t)return t;const raw=t.H,smoothed=new Float32Array(raw.length),n=t.n,weights=[1,2,1];\n  for(let j=0;j<n;j++)for(let i=0;i<n;i++){let sum=0;for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++)sum+=raw[Math.max(0,Math.min(n-1,j+y))*n+Math.max(0,Math.min(n-1,i+x))]*weights[y+1]*weights[x+1];smoothed[j*n+i]=raw[j*n+i]+Math.max(-.12,Math.min(.12,sum/16-raw[j*n+i]));}t.H=smoothed;return t;};\nconst originalPaintTerrain=paintTerrain;\npaintTerrain=function(){originalPaintTerrain();if(!RN||!TR)return;const col=RN.geo.attributes.color,n=TR.n,raw=col.array.slice();\n  if(SURF===\'terrain\'){for(let j=0;j<n;j++)for(let i=0;i<n;i++){const dst=(j*n+i)*3;for(let c=0;c<3;c++){let sum=0,total=0;for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){const w=(x===0?2:1)*(y===0?2:1);sum+=raw[(Math.max(0,Math.min(n-1,j+y))*n+Math.max(0,Math.min(n-1,i+x)))*3+c]*w;total+=w}col.array[dst+c]=sum/total}}col.needsUpdate=true;}\n};\nconst originalAddTrees=addTrees;\naddTrees=function(list){const land=list.filter(o=>hAt(o.x,o.z)>waterLevel()+.15);originalAddTrees(land);};\nbuildTent=function(){const g=new THREE.Group(),cloth=mat(0x514733,.98,0),stripe=mat(0xd4d1bd,.97,0),pole=mat(0xaaa993,.7,.2);cloth.side=stripe.side=THREE.DoubleSide;\n  const profile=[[-2.1,.1],[-1.8,2.2],[-1.1,2.9],[1.1,2.9],[1.8,2.2],[2.1,.1]],bounds=[-3,-1.1,1.1,3];\n  for(let s=0;s<3;s++)for(let i=0;i<profile.length-1;i++){const a=profile[i],b=profile[i+1],z0=bounds[s],z1=bounds[s+1],geo=new THREE.BufferGeometry();geo.setAttribute(\'position\',new THREE.Float32BufferAttribute([a[0],a[1],z0,b[0],b[1],z0,b[0],b[1],z1,a[0],a[1],z1],3));geo.setIndex([0,1,2,0,2,3]);geo.computeVertexNormals();const m=new THREE.Mesh(geo,s===1?stripe:cloth);m.castShadow=m.receiveShadow=true;g.add(m);}\n  for(const z of [-3,3]){const shape=new THREE.Shape();shape.moveTo(-2.1,.1);profile.slice(1).forEach(p=>shape.lineTo(...p));shape.lineTo(.55,.1);shape.lineTo(.55,1.5);shape.lineTo(-.55,1.5);shape.lineTo(-.55,.1);shape.closePath();const end=new THREE.Mesh(new THREE.ShapeGeometry(shape),cloth);end.position.z=z;g.add(end);}\n  for(const z of [-3,0,3])for(const side of [-1,1]){const m=new THREE.Mesh(new THREE.CylinderGeometry(.04,.04,2.25,8),pole);m.position.set(side*1.86,1.1,z);m.rotation.z=side*.12;g.add(m);}return g;};\nconst originalBuildMap=buildMapObstacles;\nbuildMapObstacles=function(){originalBuildMap();if(!RN)return;const lookup=new Map(ST.obstacles.map(o=>[o.kind+\':\'+o.x+\':\'+o.z,o]));\n  (RN.mapObjs||[]).forEach(o=>{const src=lookup.get(o.kind+\':\'+o.x+\':\'+o.z);if(o.kind===\'Rock\'&&o.obj.isMesh){o.obj.geometry.dispose();o.obj.geometry=new THREE.DodecahedronGeometry(1.5,0);o.obj.scale.set(1.3,.75,1.0);o.obj.material.color.setHex(0x66635a);o.obj.position.y+=.65;}if(src&&src.rotation){const q=src.rotation;o.obj.quaternion.set(-q.x,-q.y,q.z,q.w).normalize();}});};\nfunction camouflage(){const canvas=document.createElement(\'canvas\');canvas.width=canvas.height=512;const c=canvas.getContext(\'2d\');c.fillStyle=\'#465038\';c.fillRect(0,0,512,512);\n  let seed=921;const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296};\n  [\'#92794b\',\'#202820\',\'#a18a57\'].forEach((color,index)=>{c.fillStyle=color;for(let k=0;k<12;k++){const x=rand()*512,y=rand()*512;c.beginPath();c.moveTo(x,y);for(let a=0;a<8;a++){const angle=a*Math.PI/4,r=25+rand()*65;c.lineTo(x+Math.cos(angle)*r,y+Math.sin(angle)*r)}c.closePath();c.fill();}});\n  const texture=new THREE.CanvasTexture(canvas);texture.encoding=THREE.sRGBEncoding;texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.anisotropy=Math.min(8,RN.rd.capabilities.getMaxAnisotropy());return texture;}\nfunction detailTank(tank,enemy){const steel=new THREE.MeshStandardMaterial({color:0x272b26,roughness:.75,metalness:.55});const armor=tank.userData.mats[0],turret=tank.userData.turret;\n  function mesh(parent,geo,mat,x,y,z,rx=0,ry=0,rz=0){const o=new THREE.Mesh(geo,mat);o.position.set(x,y,z);o.rotation.set(rx,ry,rz);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}\n  // Side skirts, panel seams, track shoes, front towing loops and headlights.\n  for(const side of [-1,1]){\n    for(let i=0;i<7;i++){mesh(tank,new THREE.BoxGeometry(.13,.82,.77),armor,side*1.96,1.14,-2.53+i*.83);mesh(tank,new THREE.BoxGeometry(.16,.035,.72),steel,side*2.04,1.52,-2.53+i*.83);}\n    for(let i=0;i<25;i++){const z=-3+i*.25;for(const y of [.18,1.28])mesh(tank,new THREE.BoxGeometry(.72,.075,.19),steel,side*1.8,y,z);}\n    mesh(tank,new THREE.TorusGeometry(.13,.035,6,12),steel,side*1.1,.9,3.04,Math.PI/2);\n    mesh(tank,new THREE.BoxGeometry(.24,.16,.12),new THREE.MeshStandardMaterial({color:0xc2b984,roughness:.25,metalness:.2}),side*1.23,1.22,3.12);\n    // Wedge armor cheeks, optics and rear storage rack.\n    mesh(turret,new THREE.BoxGeometry(.42,.58,1.6),armor,side*1.2,.47,.48,0,side*.14);\n    for(let k=0;k<4;k++)mesh(turret,new THREE.CylinderGeometry(.085,.085,.34,10),steel,side*1.48,.45,-.5+k*.23,Math.PI/2,0,side*.5);\n    mesh(turret,new THREE.BoxGeometry(.06,.55,1.2),steel,side*1.1,.5,-1.55);\n  }\n  for(let i=0;i<9;i++)mesh(tank,new THREE.BoxGeometry(1.8,.025,.07),steel,0,1.91,-1.35-i*.15);\n  mesh(turret,new THREE.BoxGeometry(.4,.43,.4),armor,.72,1.11,.4);\n  mesh(turret,new THREE.BoxGeometry(.26,.16,.03),new THREE.MeshStandardMaterial({color:0x293f47,metalness:.7,roughness:.12}),.72,1.15,.615);\n  const mg=mesh(turret,new THREE.BoxGeometry(.15,.19,.64),steel,-.53,1.48,.12);mesh(mg,new THREE.CylinderGeometry(.035,.035,.75,10),steel,0,0,.6,Math.PI/2);\n  steel.userData={emis:0,inten:0};tank.userData.mats.push(steel);tank.traverse(o=>{if(o.isMesh)o.receiveShadow=true});\n  if(!enemy){const texture=camouflage();[tank.userData.mats[0],tank.userData.mats[1]].forEach(m=>{m.color.setHex(0xffffff);m.map=texture;m.roughness=.88;m.metalness=.15;m.needsUpdate=true;});}\n}\nfunction refineScene(){detailTank(RN.me,false);detailTank(RN.foe,true);RN.water.material.color.setHex(0x123e43);RN.water.material.roughness=.5;RN.water.material.metalness=.12;RN.terrain.material.metalness=0;}\nfunction shellModel(){const g=new THREE.Group(),metal=new THREE.MeshStandardMaterial({color:0xc5a665,metalness:.35,roughness:.35}),tip=new THREE.MeshStandardMaterial({color:0xd1d0c9,metalness:.4,roughness:.25});\n  const body=new THREE.Mesh(new THREE.CylinderGeometry(.09,.09,.5,12),metal);body.rotation.x=Math.PI/2;g.add(body);\n  const nose=new THREE.Mesh(new THREE.ConeGeometry(.09,.26,12),tip);nose.rotation.x=Math.PI/2;nose.position.z=.38;g.add(nose);return g;}\nfunction impactVisual(position,hit,maxDuration=950){if(!RN||maxDuration<=0)return;const g=new THREE.Group();g.position.copy(position);const sparks=new THREE.BufferGeometry(),pts=[];for(let i=0;i<24;i++)pts.push(Math.sin(i*2.4)*.4,.15+((i*7)%13)/13,Math.cos(i*2.4)*.4);sparks.setAttribute(\'position\',new THREE.Float32BufferAttribute(pts,3));\n  const spark=new THREE.Points(sparks,new THREE.PointsMaterial({color:hit?0xffaa48:0xb7a68a,size:.09,transparent:true,opacity:.9,depthWrite:false}));g.add(spark);\n  const puff=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTex(),color:hit?0x514335:0x655f52,transparent:true,opacity:.36,depthWrite:false,depthTest:true}));puff.scale.set(1.2,1.2,1);g.add(puff);\n  const flash=new THREE.PointLight(0xff9d44,hit?1.4:.35,7,2);g.add(flash);RN.fxG.add(g);visualEffects.push({type:\'impact\',g,spark,puff,flash,born:performance.now(),duration:Math.min(maxDuration,hit?950:650)});}\nconst previousHitFlash=hitFlash;\nhitFlash=function(tank,base,mine){previousHitFlash(tank,base,mine);const now=performance.now();if(!tank.userData.lastImpact||now-tank.userData.lastImpact>250){tank.userData.lastImpact=now;impactVisual(base.clone().add(new THREE.Vector3(0,2,0)),true)}};\nconst previousStepHit=stepHit;\nstepHit=function(dt){previousStepHit(dt);if(RN)[RN.me,RN.foe].forEach(t=>{if(t)t.userData.mats.forEach(m=>m.emissiveIntensity=Math.min(.3,m.emissiveIntensity))});};\nfunction recentShot(sh,simTime){return sh&&Number.isFinite(sh.t)&&Number.isFinite(simTime)&&simTime>=sh.t&&simTime-sh.t<5;}\nfunction displayShot(sh,enemy,simTime){if(!RN||!sh.fire||!sh.imp||!recentShot(sh,simTime))return;\n  const g=new THREE.Group(),a=V(sh.fire[0],sh.fire[1],2.7),b=V(sh.imp[0],sh.imp[1],.25),points=[],lift=Math.min(20,a.distanceTo(b)*.12);\n  for(let k=0;k<=30;k++){const u=k/30;points.push(a.clone().lerp(b,u).add(new THREE.Vector3(0,Math.sin(Math.PI*u)*lift,0)))}\n  const arc=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:enemy?0xca5f54:0xbaa86b,transparent:true,opacity:.28}));g.add(arc);\n  const mark=new THREE.Mesh(new THREE.RingGeometry(.16,.24,24),new THREE.MeshBasicMaterial({color:enemy?0xbd4c41:0x998d65,transparent:true,opacity:.6,depthWrite:false,side:THREE.DoubleSide}));mark.rotation.x=-Math.PI/2;mark.position.copy(b);g.add(mark);\n  const projectile=shellModel();g.add(projectile);RN.fxG.add(g);const age=(simTime-sh.t)*1000;\n  const flight=Math.min(4000,Math.max(150,(sh.tof||.9)*1000));\n  const replay=!sh.pending&&age>=flight;\n  visualEffects.push({type:\'shot\',key:(enemy?\'e\':\'a\')+sh.id,g,arc,mark,projectile,a,b,lift,born:performance.now()-age,duration:5000,flight:replay?Math.min(900,Math.max(0,5000-age)):flight,flightStart:replay?performance.now():performance.now()-age,hit:sh.hit,enemy,confirmed:!sh.pending,impacted:false});}\nconst originalUpdate3D=update3D;\nupdate3D=function(){if(!S||!S.fire)return originalUpdate3D();const f=S.fire,shots=f.shots,foe=f.foe,fs=foe&&foe.shots,t=f.tm&&f.tm.t;\n  // A bounding-box size is not an enemy identification. Keep uncertain objects neutral.\n  const known=S.drive&&S.drive.known;\n  if(known)S.drive.known=known.map(o=>o.g?{...o,g:false,t:\'nature\'}:o);\n  f.shots=[];if(foe)foe.shots=[];try{originalUpdate3D()}finally{f.shots=shots;if(foe)foe.shots=fs;if(known)S.drive.known=known;}\n  if(Number.isFinite(t)&&visualLastTime!==null&&t<visualLastTime){visualEffects.splice(0).forEach(e=>disposeVisual(e.g));visualSeen.clear();visualInitialized=false;}\n  if(Number.isFinite(t))visualLastTime=t;\n  [[f.pending_shot?[f.pending_shot,...(shots||[])]:shots,false],[fs,true]].forEach(([list,enemy])=>(list||[]).forEach(sh=>{const key=(enemy?\'e\':\'a\')+sh.id;if(!visualSeen.has(key)){visualSeen.set(key,t);if(visualInitialized)displayShot(sh,enemy,t)}else if(!sh.pending){const effect=visualEffects.find(e=>e.key===key);if(effect&&!effect.confirmed){effect.confirmed=true;effect.impacted=true;effect.hit=sh.hit;effect.b.copy(V(sh.imp[0],sh.imp[1],.25));effect.arc.geometry.setFromPoints([effect.a,effect.b]);impactVisual(effect.b,sh.hit,Math.max(0,5000-(performance.now()-effect.born)));}}}));\n  if(Number.isFinite(t))visualInitialized=true;\n  for(const [key,when] of visualSeen)if(Number.isFinite(t)&&t-when>60)visualSeen.delete(key);\n};\nstepFX=function(){if(!RN)return;const now=performance.now();for(let i=visualEffects.length-1;i>=0;i--){const e=visualEffects[i],age=now-e.born;if(age>=e.duration){disposeVisual(e.g);visualEffects.splice(i,1);continue;}\n  if(e.type===\'shot\'){const u=Math.min(1,Math.max(0,(now-e.flightStart)/Math.max(1,e.flight)));e.projectile.visible=u<1;e.projectile.position.copy(e.a).lerp(e.b,u).add(new THREE.Vector3(0,Math.sin(Math.PI*u)*e.lift,0));const tangent=e.b.clone().sub(e.a);tangent.y+=Math.PI*e.lift*Math.cos(Math.PI*u);e.projectile.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),tangent.normalize());e.arc.visible=L[e.enemy?\'foeshots\':\'shots\'];e.mark.visible=e.confirmed&&u>=1&&e.arc.visible;e.arc.material.opacity=.28*Math.min(1,(5000-age)/700);if(u>=1&&!e.impacted&&e.confirmed){e.impacted=true;impactVisual(e.b,e.hit,5000-age);}}\n  else{const u=age/e.duration;e.spark.scale.setScalar(1+u*3);e.spark.material.opacity=Math.max(0,1-u*2);e.puff.position.y=u*1.2;e.puff.scale.setScalar(1.2+u*1.6);e.puff.material.opacity=.36*(1-u);e.flash.intensity=Math.max(0,1.4*(1-u*6));}}\n};\n// Scope shows physical shells and brief impacts, not historical annotations.\nfunction scopeAnnotations(){return visualEffects.filter(e=>e.type===\'shot\').flatMap(e=>[e.arc,e.mark]);}\nfunction sizeProjectiles(camera,height){for(const e of visualEffects)if(e.type===\'shot\'){const distance=camera.position.distanceTo(e.projectile.position);e.projectile.scale.setScalar(Math.min(8,Math.max(1.5,distance*Math.tan(camera.fov*Math.PI/360)*14/Math.max(height,1)/.18)));}}\n</script>\n<style>\n:root{--acc:#d0e78b;--glass:rgba(18,24,23,.92);--line:rgba(205,223,196,.22);--dim:#abb7ac;--fg:#edf0e6}\n#top,#left,#right,#mini,#comp,#leg{display:none}\nbody.inspect #right{display:block;top:65px;bottom:65px;max-height:none}\nbody.settings #left{display:block;top:65px;bottom:245px;max-height:none}\n#cockpit-bar{position:fixed;z-index:20;top:12px;left:14px;right:14px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}\n.cockpit-control{font:12px \'Malgun Gothic\',sans-serif;color:#d8dfd4;background:rgba(17,23,22,.91);border:1px solid #465343;border-radius:5px;padding:9px 13px;cursor:pointer;text-decoration:none}\n.cockpit-control[aria-pressed=true]{color:#d5f192;border-color:#aabd79;background:#293222}\n#cockpit-bar strong{letter-spacing:2px;padding:0 12px 0 2px;text-shadow:0 1px 5px #000}\n#cockpit-tools{margin-left:auto;display:flex;gap:6px}\n#cockpit-bearing{position:fixed;top:68px;left:50%;transform:translateX(-50%);color:#edf4df;background:#151d19bb;border-radius:5px;padding:7px 24px;z-index:12;letter-spacing:2px;pointer-events:none}\n#cockpit-status{position:fixed;bottom:15px;left:16px;right:16px;z-index:12;display:flex;justify-content:space-between;gap:12px;pointer-events:none;font-size:12px;text-shadow:0 1px 4px #000}\n#cockpit-status span{background:#151d19d9;padding:7px 11px;border-radius:5px}\n#scope-inset{position:fixed;left:16px;bottom:66px;width:260px;height:178px;border:1px solid #92a279;border-radius:6px;z-index:11;pointer-events:none;overflow:hidden;box-shadow:0 8px 30px #0007}\n#scope-inset .caption{position:absolute;top:0;left:0;right:0;padding:6px 10px;background:#121915c9;color:#e3eacb;display:flex;justify-content:space-between;font-size:12px;z-index:2}\n.reticle{position:absolute;inset:0;background:radial-gradient(ellipse at center,transparent 36%,#0008 69%,#000 92%)}\n.reticle:before{content:\'\';position:absolute;left:12%;right:12%;top:50%;height:1px;background:#d6e4bba0}\n.reticle:after{content:\'\';position:absolute;top:20%;bottom:15%;left:50%;width:1px;background:#d6e4bba0}\n.reticle i{position:absolute;left:42%;top:50%;width:16%;height:24%;border-top:1px solid #d6e4bb;background:repeating-linear-gradient(90deg,transparent 0 19%,#d6e4bb80 19% 20%) top/100% 6px no-repeat}\n#full-reticle{position:fixed;inset:0;z-index:8;pointer-events:none;display:none}\nbody.scope-mode #full-reticle{display:block}body.scope-mode #scope-inset{visibility:hidden}\n#cockpit-notice{position:fixed;top:110px;left:50%;transform:translateX(-50%);z-index:15;background:#1a211dde;padding:9px 15px;border:1px solid #667655;border-radius:6px;max-width:85%;text-align:center}\n@media(max-width:700px){#cockpit-bar strong{display:none}#cockpit-tools{margin-left:0}#cockpit-bar{gap:4px}.cockpit-control{padding:7px 9px}#cockpit-bearing{top:100px}#scope-inset{width:190px;height:130px}#cockpit-status{font-size:10px}#cockpit-status span:last-child{display:none}#cockpit-notice{top:142px}}\n</style>\n<div id="cockpit-bar"><strong>3D VIEW</strong><button class="cockpit-control" data-camera="chase" aria-pressed="true">추적 시점</button><button class="cockpit-control" data-camera="scope" aria-pressed="false">조준경</button><button class="cockpit-control" data-camera="orbit" aria-pressed="false">자유 시점</button><div id="cockpit-tools"><button class="cockpit-control" id="emg_stop" aria-pressed="false">긴급제동</button><button class="cockpit-control" id="cockpit-settings" aria-pressed="false">레이어</button><button class="cockpit-control" id="cockpit-details" aria-pressed="false">상태</button><a class="cockpit-control" href="?style=classic">기존 화면</a></div></div>\n<div id="cockpit-bearing">방위 데이터 대기</div>\n<div id="scope-inset"><div class="caption"><span>조준경 · 포신 방향</span><span>4.0×</span></div><div class="reticle"><i></i></div></div>\n<div id="full-reticle" class="reticle"><i></i></div>\n<div id="cockpit-notice" role="status">시뮬레이터 데이터 대기 중</div>\n<div id="cockpit-status"><span id="cockpit-readout">연결 대기</span><span>── 실제\u3000┄ 계획 경로 · 탄체 확대 / 착탄 기록 재생</span></div>\n<script>\n(()=>{\n  let mode=\'chase\',ready=false,lastHUD=0,lastTime=null,lastAdvance=performance.now();\n  let optic,uncertainty,ghost,previewTag;\n  const el=id=>document.getElementById(id);\n  const finite=v=>typeof v===\'number\'&&Number.isFinite(v);\n  const fmt=(v,u=\'\')=>finite(v)?v.toFixed(1)+u:\'—\';\n  function selectCamera(next){mode=next;document.body.classList.toggle(\'scope-mode\',next===\'scope\');\n    document.querySelectorAll(\'[data-camera]\').forEach(b=>b.setAttribute(\'aria-pressed\',String(b.dataset.camera===next)));\n    if(RN&&next===\'orbit\'){RN.cam3.tgt.copy(RN.me.position);RN.cam3.dis=75;RN.cam3.pit=.55;follow=false;}}\n  document.querySelectorAll(\'[data-camera]\').forEach(b=>b.onclick=()=>selectCamera(b.dataset.camera));\n  [[\'cockpit-settings\',\'settings\'],[\'cockpit-details\',\'inspect\']].forEach(([id,cls])=>el(id).onclick=()=>{const yes=document.body.classList.toggle(cls);el(id).setAttribute(\'aria-pressed\',String(yes))}); \n el(\'emg_stop\').onclick=()=>{const btn=el(\'emg_stop\'),next=btn.getAttribute(\'aria-pressed\')!==\'true\';btn.setAttribute(\'aria-pressed\',String(next));btn.innerText=next?\'긴급제동 해제\':\'긴급제동\';fetch(\'/get_emg_stop\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify({emg_stop:next?\'True\':\'False\'})})}; \n  const originalBoot=boot;\n  boot=function(){VS=1;if(ST)ST.obstacles=FINAL_OBSTACLES;originalBoot();if(!RN||ready)return;ready=true;setup();};\n  function setup(){\n    VS=1;el(\'vs\').value=\'1\';el(\'v-vs\').textContent=\'1.0×\';paintTerrain();\n    Object.assign(L,{rings:false,ray:false,los:false,trail:false,known:false,det:false,aim:false});\n    document.querySelectorAll(\'[data-l]\').forEach(e=>e.checked=L[e.dataset.l]);applyLayers();\n    RN.sc.fog.color.setHex(0x4f5d5d);RN.sc.fog.density=.0015;RN.rd.toneMappingExposure=.65;\n    RN.sc.children.filter(o=>o.isHemisphereLight).forEach(o=>{o.color.setHex(0x99a8ac);o.groundColor.setHex(0x343629);o.intensity=.7});\n    RN.sun.color.setHex(0xd7d0b7);RN.sun.intensity=.65;\n    const oldSky=RN.sc.children.find(o=>o.isMesh&&o.geometry.type===\'SphereGeometry\'&&o.geometry.parameters.radius===1600);\n    if(oldSky){const p=oldSky.geometry.attributes.position,c=oldSky.geometry.attributes.color;\n      const horizon=new THREE.Color(0x667778),zenith=new THREE.Color(0x304b5a),bottom=new THREE.Color(0x323d36),v=new THREE.Color();\n      for(let i=0;i<p.count;i++){const t=Math.max(0,Math.min(1,p.getY(i)/1600));v.copy(p.getY(i)<0?bottom:horizon).lerp(zenith,t);c.setXYZ(i,v.r,v.g,v.b)}c.needsUpdate=true;}\n    [RN.me,RN.foe].forEach((tank,index)=>{const colors=index?[0x9c2520,0xb6372c,0x542724,0x77241f,0x222723]:[0x555e42,0x65704c,0x424b3c,0x58613e,0x242922];\n      tank.userData.mats.forEach((m,i)=>{m.color.setHex(colors[i]);m.emissive.setHex(0);m.emissiveIntensity=0;m.userData.emis=0;m.userData.inten=0});tank.userData.ring.visible=false;});\n    refineScene();\n    RN.routeL.material.dispose();RN.routeL.material=new THREE.LineDashedMaterial({color:0xd2ed87,dashSize:2.5,gapSize:1.8,transparent:true,opacity:.9});\n    optic=new THREE.PerspectiveCamera(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(46/2))/4)*180/Math.PI,1,.1,4000);\n    uncertainty=new THREE.Mesh(new THREE.CircleGeometry(1,64),new THREE.MeshBasicMaterial({color:0xef987f,transparent:true,opacity:.15,side:THREE.DoubleSide,depthWrite:false}));uncertainty.rotation.x=-Math.PI/2;RN.sc.add(uncertainty);\n    ghost=buildTank({hull:0xd2ed98,turret:0xd2ed98,track:0xd2ed98,metal:0xd2ed98,accent:0xd2ed98,ring:0xd2ed98});\n    ghost.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.transparent=true;o.material.opacity=.19;o.material.depthWrite=false;o.castShadow=false}});ghost.userData.ring.visible=false;ghost.userData.flash.visible=false;RN.sc.add(ghost);\n    const canvas=tagCanvas(\'계획 경로 미리보기\',\'#d2ed98\');previewTag=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),transparent:true,depthTest:false}));previewTag.scale.set(canvas.width*.032,canvas.height*.032,1);RN.sc.add(previewTag);\n    const render=RN.rd.render.bind(RN.rd);\n    RN.rd.render=function(scene,camera){if(scene!==RN.sc){render(scene,camera);return}renderCockpit(render,camera)};\n    RN.rd.domElement.addEventListener(\'pointerdown\',()=>{if(mode!==\'orbit\')selectCamera(\'orbit\')});\n    if(S)update3D();\n  }\n  function renderCockpit(render,camera){\n    const f=S&&S.fire,tm=f&&f.tm,my=tm&&tm.my,hasMy=Array.isArray(my)&&my.length>=3&&my.every(finite);\n    const hasFoe=!!(tm&&Array.isArray(tm.enemy)&&tm.enemy.length>=3&&tm.enemy.every(finite));\n    RN.me.visible=hasMy;RN.foe.visible=hasFoe;RN.meTag.visible=hasMy&&L.label&&mode===\'orbit\';RN.foeTag.visible=hasFoe&&L.label;\n    uncertainty.visible=false;ghost.visible=previewTag.visible=false;\n    if(hasMy){\n      if(RN.routeL.geometry.getAttribute(\'position\'))RN.routeL.computeLineDistances();\n      const heading=(tm.body_x||0)*Math.PI/180,dir=new THREE.Vector3(Math.sin(heading),0,-Math.cos(heading)),base=RN.me.position;\n      if(mode===\'chase\'){const pos=base.clone().addScaledVector(dir,-11);pos.y=Math.max(base.y+6.5,hAt(pos.x+150,150-pos.z)*VS+2);\n        camera.position.copy(pos);camera.fov=57;camera.updateProjectionMatrix();camera.lookAt(base.clone().addScaledVector(dir,35).add(new THREE.Vector3(0,3.5,0)));}\n      else if(mode===\'orbit\'){camera.fov=46;camera.updateProjectionMatrix();}\n      RN.sc.updateMatrixWorld(true);\n      optic.position.copy(RN.me.userData.gun.localToWorld(new THREE.Vector3(0,.35,6)));\n      const gunDir=RN.me.userData.gun.localToWorld(new THREE.Vector3(0,.35,7)).sub(optic.position).normalize();\n      optic.lookAt(optic.position.clone().addScaledVector(gunDir,100));\n      const err=f.track&&f.track.pred_err;\n      if(hasFoe&&finite(err)&&err>0){uncertainty.position.copy(V(tm.enemy[0],tm.enemy[2],.45));uncertainty.scale.set(err,err,1);uncertainty.visible=true;}\n      // Spatial path preview only: the source has no timestamped planned pose.\n      const path=(S.drive&&S.drive.path||[]).filter(p=>Array.isArray(p)&&p.length>=2&&p.every(finite));\n      if(L.route&&path.length>1){let nearest=0,best=Infinity;path.forEach((p,i)=>{const d=Math.hypot(p[0]-my[0],p[1]-my[2]);if(d<best){best=d;nearest=i}});\n        let j=nearest;while(j<path.length-1&&Math.hypot(path[j][0]-my[0],path[j][1]-my[2])<12)j++;\n        const p=path[j],q=path[Math.min(j+1,path.length-1)];ghost.position.copy(V(p[0],p[1],0));ghost.rotation.y=j<path.length-1?yawOf(brg(p,q)):RN.me.rotation.y;\n        ghost.visible=true;previewTag.position.copy(ghost.position).add(new THREE.Vector3(0,5,0));previewTag.visible=L.label;}\n    }\n    const rd=RN.rd,w=innerWidth,h=innerHeight;\n    rd.setScissorTest(false);rd.setViewport(0,0,w,h);sizeProjectiles(mode===\'scope\'?optic:camera,h);\n    if(mode===\'scope\'&&hasMy){optic.aspect=w/h;optic.updateProjectionMatrix();const objects=[RN.aimSp,RN.meTag,RN.foeTag,RN.routeL,RN.losL,uncertainty,ghost,previewTag,...scopeAnnotations()],visibility=objects.map(o=>o.visible);objects.forEach(o=>o.visible=false);render(RN.sc,optic);objects.forEach((o,i)=>o.visible=visibility[i])}else render(RN.sc,camera);\n    el(\'scope-inset\').style.visibility=hasMy&&mode!==\'scope\'?\'visible\':\'hidden\';\n    if(hasMy&&mode!==\'scope\'){\n      const r=el(\'scope-inset\').getBoundingClientRect();optic.aspect=r.width/r.height;optic.updateProjectionMatrix();sizeProjectiles(optic,r.height);\n      const hidden=[RN.meTag,RN.foeTag,RN.aimSp,RN.routeL,RN.losL,uncertainty,ghost,previewTag,...scopeAnnotations()],prev=hidden.map(o=>o.visible);hidden.forEach(o=>o.visible=false);\n      rd.setViewport(r.left,h-r.bottom,r.width,r.height);rd.setScissor(r.left,h-r.bottom,r.width,r.height);rd.setScissorTest(true);render(RN.sc,optic);\n      hidden.forEach((o,i)=>o.visible=prev[i]);rd.setScissorTest(false);rd.setViewport(0,0,w,h);\n    }\n    if(performance.now()-lastHUD>250){lastHUD=performance.now();if(tm&&tm.t!==lastTime){lastTime=tm.t;lastAdvance=lastHUD}\n      const stale=hasMy&&lastHUD-lastAdvance>5000;\n      el(\'cockpit-notice\').hidden=hasMy&&!stale;el(\'cockpit-notice\').textContent=hasMy?\'데이터 갱신 대기 · 일시정지 또는 연결 상태 확인\':\'시뮬레이터 데이터 대기 중\';\n      el(\'cockpit-bearing\').textContent=hasMy?\'N\u3000·\u3000포탑 \'+fmt(tm.turret_x,\'°\')+\'\u3000·\u3000차체 \'+fmt(tm.body_x,\'°\'):\'방위 데이터 대기\';\n      el(\'cockpit-readout\').textContent=hasMy?\'시뮬레이션 \'+fmt(tm.t,\'s\')+\'\u3000/\u3000거리 \'+(hasFoe?fmt(Math.hypot(tm.enemy[0]-my[0],tm.enemy[2]-my[2]),\'m\'):\'—\')+\'\u3000/\u3000예측오차 \'+fmt(f.track&&f.track.pred_err,\'m\'):\'지형 미리보기 · 실시간 전차 상태 대기\';}\n  }\n  setTimeout(()=>{if(!ready)el(\'cockpit-notice\').textContent=\'3D 로딩 대기 · 서버 / 지형 데이터 / 인터넷 연결을 확인해 주세요\'},12000);\n})();\n</script>\n' + "</body>")

# Read-only UI adapters. Controller state and commands are never changed.
_base_collect_fire = _collect_fire
def _collect_fire():
    out = _base_collect_fire()
    pending = _g(_g(_REF['fm'], 'log'), 'pending')
    if isinstance(pending, dict):
        out['pending_shot'] = {
            'id': pending.get('id'), 't': pending.get('t'),
            'fire': _xz(pending.get('fire_pos')),
            'imp': _xz(pending.get('aim_point')),
            'tof': pending.get('tof'), 'pending': True,
        }
    return out

_base_attach_viz = attach_viz_taek
def attach_viz_taek(app, fm=None, drive=None, detect=None):
    bp_result = _base_attach_viz(app, fm=fm, drive=drive, detect=detect)
    if app.extensions.get('cockpit_console'):
        return bp_result
    import logging
    logging.getLogger('werkzeug').setLevel(logging.INFO)
    status = {'last': 0., 'received': time.monotonic(), 'info': 0, 'action': 0}
    app.extensions['cockpit_console'] = status
    print('[아군 모니터] PID=%s · /info 와 /get_action 수신 시 1초 간격 상태 출력' % os.getpid(), flush=True)
    @app.after_request
    def cockpit_request_status(response):
        if request.path not in ('/info', '/get_action'):
            return response
        try:
            key = 'info' if request.path == '/info' else 'action'
            with _LOCK:
                status[key] += 1
                now = time.monotonic()
                status['received'] = now
                if now-status['last'] < 1.:
                    return response
                status['last'] = now
                counts = (status['info'], status['action'])
            tm = _g(fm, 'tm')
            print('[아군] t=%s 위치=%s 차체=%s° 포탑=%s° 상태=%s 수신 info=%s action=%s HTTP=%s' % (
                _num(_g(tm, 't')), _g(tm, 'my'), _num(_g(tm, 'body_x')),
                _num(_g(tm, 'turret_x')), _g(_g(fm, 'fc'), 'state', '대기'),
                counts[0], counts[1], response.status_code), flush=True)
        except Exception as exc:
            app.logger.warning('[아군 모니터] 상태 출력 실패: %s', type(exc).__name__)
        return response
    def watch_connection():
        while True:
            time.sleep(5)
            with _LOCK:
                quiet = time.monotonic()-status['received']
            if quiet >= 5:
                print('[아군 연결 대기] %.0f초간 /info · /get_action 수신 없음 — 시뮬레이터 아군 주소/실행 상태 확인' % quiet, flush=True)
    if not app.testing:
        threading.Thread(target=watch_connection, daemon=True).start()
    return bp_result
