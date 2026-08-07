# -*- coding: utf-8 -*-
"""
tank_server.py - Tank Challenge EndPoint + A* 경로 추종

구조
    /init            : 시뮬레이터 설정 (trackingMode=True 로 API 제어 활성화)
    /update_obstacle : 장애물 수신 -> A* 그리드 갱신
    /set_destination : 목적지 수신 -> 경로 재계산
    /info            : 전차 상태 저장 (차체 방위각은 여기에만 있음)
    /get_action      : 경로 추종 제어기 -> W/A/S/D 명령 반환
    /status          : 브라우저로 상태 확인 (디버그용)

실행
    python tank_server.py
    시뮬레이터 Setting > Request Port 5000 > Save > Run

주의
    - astar_planner_v2.py 가 같은 폴더에 있어야 함
    - A/D 방향과 방위각 기준은 실측 검증 필요 (아래 TURN_LEFT/RIGHT, bearing 함수)
"""
from flask import Flask, request, jsonify
import math, threading, time, json

import numpy as np

from astar_planner_v2 import AStarPlanner, ObstacleRect

app = Flask(__name__)

# ── 튜닝 파라미터 ────────────────────────────────────────
MAP_MIN, MAP_MAX = 0.0, 300.0
CELL_SIZE        = 2.0     # 격자 크기(m). 작을수록 정밀하지만 느림
OBSTACLE_MARGIN  = 3.0     # 전차 반경 + 여유

ANGLE_SPIN       = 30.0    # 이 각도 이상 틀어지면 제자리 회전
ANGLE_OK         = 8.0     # 이 이하면 조향 없이 직진
WP_REACH         = 6.0     # waypoint 도달 판정 반경(m)
GOAL_REACH       = 4.0     # 최종 목적지 도달 반경(m)

SPEED_FAR        = 1.0     # 직선 구간 속도 weight
SPEED_TURN       = 0.4     # 조향 중 속도 weight

TURN_LEFT, TURN_RIGHT = "A", "D"   # 반대로 돌면 이 두 값을 바꿀 것

SCAN_CELL        = 2.0     # 지형 스캔 격자 크기(m)
PATROL_POINTS    = [(40., 40.), (260., 40.), (260., 150.), (40., 150.),
                    (40., 260.), (260., 260.), (150., 150.)]
# ────────────────────────────────────────────────────────

lock = threading.Lock()
planner = AStarPlanner(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX,
                       cell_size=CELL_SIZE, obstacle_margin=OBSTACLE_MARGIN)

# ── 지형 스캐너 ─────────────────────────────────────────
SCAN_N = int((MAP_MAX - MAP_MIN) / SCAN_CELL)
SCAN = {
    "sum":    np.zeros((SCAN_N, SCAN_N)),   # 셀별 높이 합
    "cnt":    np.zeros((SCAN_N, SCAN_N)),   # 셀별 포인트 수
    "frames": 0,                            # LiDAR 프레임 수
    "points": 0,                            # 누적 포인트 수
    "last_origin": None,                    # 직전 lidarOrigin (갱신 확인용)
    "stale":  0,                            # origin 이 그대로인 연속 횟수
    "centroid": None,                       # 최근 점군의 무게중심 (x, z)
    "hashes": set(),                        # 점군 내용 해시 (동일 여부 판정)
}


# 전차가 지나간 자리의 지면 높이를 직접 샘플링한다.
# (LiDAR 점군이 갱신되지 않는 문제의 우회책)
TRACK = {
    "sum":  np.zeros((SCAN_N, SCAN_N)),
    "cnt":  np.zeros((SCAN_N, SCAN_N)),
    "n":    0,
    "last": None,
}


def track_add(pos):
    """playerPos = (x, y, z). y 가 그 지점의 지면 높이."""
    x, y, z = pos.get("x"), pos.get("y"), pos.get("z")
    if x is None or y is None:
        return
    ix = int((x - MAP_MIN) / SCAN_CELL)
    iz = int((z - MAP_MIN) / SCAN_CELL)
    if not (0 <= ix < SCAN_N and 0 <= iz < SCAN_N):
        return
    TRACK["sum"][iz, ix] += y
    TRACK["cnt"][iz, ix] += 1
    TRACK["n"] += 1
    TRACK["last"] = (round(x, 1), round(y, 2), round(z, 1))


def track_heightmap():
    with np.errstate(invalid="ignore"):
        h = TRACK["sum"] / TRACK["cnt"]
    h[TRACK["cnt"] == 0] = np.nan
    return h


def interpolate(h, max_r=12):
    """
    관측된 셀로부터 IDW(역거리가중) 보간하여 빈 칸을 채운다.
    max_r: 참조할 최대 반경(셀). 너무 멀면 채우지 않는다.
    """
    known = ~np.isnan(h)
    if known.sum() < 3:
        return h.copy()
    kz, kx = np.nonzero(known)
    kv = h[known]
    out = h.copy()
    uz, ux = np.nonzero(~known)
    for iz, ix in zip(uz, ux):
        d2 = (kx - ix) ** 2 + (kz - iz) ** 2
        m = d2 <= max_r * max_r
        if not m.any():
            continue
        w = 1.0 / np.maximum(d2[m], 0.25)
        out[iz, ix] = float(np.sum(kv[m] * w) / np.sum(w))
    return out


def scan_add(points, origin):
    """LiDAR 점군을 격자에 누적. origin 변화로 갱신 여부도 확인한다."""
    o = (round(origin.get("x", 0), 2), round(origin.get("z", 0), 2)) if origin else None
    if o is not None and o == SCAN["last_origin"]:
        SCAN["stale"] += 1
    else:
        SCAN["stale"] = 0
    SCAN["last_origin"] = o

    # 점군 '내용'이 실제로 바뀌는지 해시로 확인한다.
    # lidarOrigin 필드만 낡은 것인지, 좌표 배열 전체가 고정인지 구분하기 위함.
    import hashlib
    sig = hashlib.md5(
        "".join(f"{p.get('distance', 0):.3f}" for p in points[:200]).encode()
    ).hexdigest()[:12]
    SCAN["hashes"].add(sig)

    n = 0
    sx = sz = 0.0
    for p in points:
        if not p.get("isDetected"):
            continue
        q = p.get("position") or {}
        x, y, z = q.get("x"), q.get("y"), q.get("z")
        if x is None:
            continue
        ix = int((x - MAP_MIN) / SCAN_CELL)
        iz = int((z - MAP_MIN) / SCAN_CELL)
        if 0 <= ix < SCAN_N and 0 <= iz < SCAN_N:
            SCAN["sum"][iz, ix] += y
            SCAN["cnt"][iz, ix] += 1
            n += 1
            sx += x
            sz += z
    SCAN["frames"] += 1
    SCAN["points"] += n
    if n:
        SCAN["centroid"] = (sx / n, sz / n)


def scan_heightmap():
    """평균 높이 격자 반환 (미관측 셀은 NaN)"""
    with np.errstate(invalid="ignore"):
        h = SCAN["sum"] / SCAN["cnt"]
    h[SCAN["cnt"] == 0] = np.nan
    return h


S = {
    "pos":      None,       # (x, z)
    "heading":  None,       # 차체 방위각(도)
    "health":   None,
    "goal":     None,       # (x, z)
    "path":     [],         # [(x, z), ...]
    "wp_idx":   0,
    "need_replan": False,
    "last_action": {},
    "log":      [],         # 최근 이벤트
    "patrol":   [],         # 순찰 대기 지점
}


def note(msg):
    ts = time.strftime("%H:%M:%S")
    S["log"].insert(0, f"[{ts}] {msg}")
    del S["log"][30:]
    print(f"  {msg}")


# ══════════════════════════════════════════════════════
# 각도 유틸
#   Unity 기준: yaw 0 = +Z 방향, 시계방향(위에서 볼 때)으로 증가
#   따라서 목표 방위각 = atan2(dx, dz)
# ══════════════════════════════════════════════════════
def bearing_to(cur, tgt):
    dx = tgt[0] - cur[0]
    dz = tgt[1] - cur[1]
    return math.degrees(math.atan2(dx, dz)) % 360.0


def angle_diff(target_deg, current_deg):
    """-180 ~ +180 로 정규화. 양수면 오른쪽(시계방향)으로 틀어야 함"""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ══════════════════════════════════════════════════════
# 경로 재계산
# ══════════════════════════════════════════════════════
def replan():
    if S["pos"] is None or S["goal"] is None:
        return
    t = time.time()
    path = planner.find_path(S["pos"], S["goal"], smooth=True)
    el = time.time() - t
    S["path"] = path
    S["wp_idx"] = 0
    S["need_replan"] = False
    if path:
        note(f"경로 생성 {len(path)}개 waypoint ({el:.3f}s) -> {S['goal']}")
    else:
        note(f"경로 실패: {S['pos']} -> {S['goal']}")


# ══════════════════════════════════════════════════════
# 엔드포인트
# ══════════════════════════════════════════════════════
@app.route("/init", methods=["GET", "POST"])
def init():
    with lock:
        S.update(path=[], wp_idx=0, goal=None, need_replan=False, patrol=[])
        note("=== 에피소드 초기화 ===")
    return jsonify({
        "startMode": "start",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True,       # API 로 전차를 제어하려면 반드시 True
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,   # 자체 수집하므로 파일 저장은 불필요
        "lux": 30000,
        "destoryObstaclesOnHit": False,   # 오타 아님. 시뮬레이터 스펙 그대로
    })


@app.route("/start", methods=["GET", "POST"])
def start():
    return jsonify({"control": ""})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    data = request.get_json(silent=True) or {}
    items = data.get("obstacles", [])
    rects = []
    for it in items:
        try:
            if "x_min" in it:
                rects.append(ObstacleRect.from_min_max(
                    float(it["x_min"]), float(it["x_max"]),
                    float(it["z_min"]), float(it["z_max"])))
            else:
                # 다른 형식 대비 (center + size)
                rects.append(ObstacleRect(
                    float(it.get("x", it.get("centerX", 0))),
                    float(it.get("z", it.get("centerZ", 0))),
                    float(it.get("sizeX", it.get("width", 4))),
                    float(it.get("sizeZ", it.get("depth", 4)))))
        except Exception as e:
            note(f"장애물 파싱 실패: {e} / {json.dumps(it)[:120]}")
    with lock:
        # 빈 목록이 나중에 도착해 기존 장애물을 지우는 것을 방지
        if rects or not planner._obstacles:
            planner.set_obstacles(rects)
            S["need_replan"] = True
        note(f"장애물 {len(rects)}개 수신")
    return jsonify({"status": "success"})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    data = request.get_json(silent=True) or {}
    raw = data.get("destination", "")
    try:
        x, y, z = map(float, str(raw).split(","))
    except Exception:
        return jsonify({"status": "ERROR", "message": "bad format"}), 400
    with lock:
        S["goal"] = (x, z)
        replan()
    return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})


@app.route("/info", methods=["POST"])
def info():
    d = request.get_json(force=True, silent=True) or {}
    p = d.get("playerPos") or {}
    with lock:
        if p:
            S["pos"] = (p.get("x", 0.0), p.get("z", 0.0))
        # 차체 방위각(yaw). /get_action 에는 없고 여기에만 있다
        if "playerBodyX" in d:
            S["heading"] = float(d["playerBodyX"])
        S["health"] = d.get("playerHealth")

        if p:
            track_add(p)                      # 지면 높이 직접 샘플링

        pts = d.get("lidarPoints") or []
        if pts:
            scan_add(pts, d.get("lidarOrigin") or {})
    return jsonify({"status": "success", "control": ""})


@app.route("/get_action", methods=["POST"])
def get_action():
    d = request.get_json(force=True, silent=True) or {}
    pos = d.get("position") or {}

    with lock:
        if pos:
            S["pos"] = (pos.get("x", 0.0), pos.get("z", 0.0))
        cur = S["pos"]
        hdg = S["heading"]

        if S["need_replan"]:
            replan()

        act = drive(cur, hdg)
        S["last_action"] = act
    return jsonify(act)


def drive(cur, hdg):
    """경로 추종 제어기 - 현재 위치/방위에서 다음 명령을 결정"""
    stop = {"moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False}

    if cur is None or hdg is None or not S["path"]:
        return stop

    # 도달한 waypoint 는 건너뛴다
    while S["wp_idx"] < len(S["path"]) - 1 and dist(cur, S["path"][S["wp_idx"]]) < WP_REACH:
        S["wp_idx"] += 1

    wp = S["path"][S["wp_idx"]]

    # 최종 목적지 도달
    if S["wp_idx"] == len(S["path"]) - 1 and dist(cur, wp) < GOAL_REACH:
        if S["path"]:
            note(f"목적지 도달 {cur}")
            S["path"] = []
            # 순찰 중이면 다음 지점으로 계속
            if S.get("patrol"):
                S["goal"] = S["patrol"].pop(0)
                note(f"다음 순찰 지점 {S['goal']} (남은 {len(S['patrol'])}개)")
                replan()
                return stop
        return stop

    err = angle_diff(bearing_to(cur, wp), hdg)
    turn = TURN_RIGHT if err > 0 else TURN_LEFT
    mag = abs(err)

    if mag > ANGLE_SPIN:
        # 크게 틀어짐 -> 제자리 회전
        move_cmd, move_w = "STOP", 1.0
        turn_w = 1.0
    elif mag > ANGLE_OK:
        # 약간 틀어짐 -> 감속하며 조향
        move_cmd, move_w = "W", SPEED_TURN
        turn_w = min(1.0, mag / ANGLE_SPIN)
    else:
        move_cmd, move_w = "W", SPEED_FAR
        turn, turn_w = "", 0.0

    return {"moveWS": {"command": move_cmd, "weight": round(move_w, 2)},
            "moveAD": {"command": turn, "weight": round(turn_w, 2)},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False}


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    d = request.get_json(silent=True) or {}
    note(f"착탄 ({d.get('x')}, {d.get('z')}) hit={d.get('hit')}")
    return jsonify({"status": "OK"})


@app.route("/collision", methods=["POST"])
def collision():
    d = request.get_json(silent=True) or {}
    with lock:
        S["need_replan"] = True
    note(f"충돌 {d.get('objectName')} -> 재계산 예약")
    return jsonify({"status": "success"})


@app.route("/detect", methods=["POST"])
def detect():
    return jsonify([])


@app.route("/stereo_image", methods=["POST"])
def stereo_image():
    return jsonify({"result": "success"})


# ── 디버그용 상태 화면 ──────────────────────────────────
@app.route("/status")
def status():
    with lock:
        wp = S["path"][S["wp_idx"]] if S["path"] else None
        err = None
        if S["pos"] and S["heading"] is not None and wp:
            err = round(angle_diff(bearing_to(S["pos"], wp), S["heading"]), 1)
        body = f"""
        <html><head><meta charset="utf-8"><meta http-equiv="refresh" content="1">
        <style>body{{background:#0f1419;color:#e6edf3;font-family:monospace;padding:24px}}
        td{{padding:4px 14px;border-bottom:1px solid #21262d}} b{{color:#58a6ff}}</style></head><body>
        <h2>Tank Server</h2><table>
        <tr><td><b>위치</b></td><td>{S['pos']}</td></tr>
        <tr><td><b>방위각</b></td><td>{S['heading']}</td></tr>
        <tr><td><b>HP</b></td><td>{S['health']}</td></tr>
        <tr><td><b>목적지</b></td><td>{S['goal']}</td></tr>
        <tr><td><b>경로</b></td><td>{len(S['path'])}개 (현재 {S['wp_idx']}번)</td></tr>
        <tr><td><b>다음 waypoint</b></td><td>{wp}</td></tr>
        <tr><td><b>방위 오차</b></td><td>{err}</td></tr>
        <tr><td><b>명령</b></td><td>{json.dumps(S['last_action'], ensure_ascii=False)}</td></tr>
        </table><h3>로그</h3><pre>{"<br>".join(S['log'])}</pre></body></html>"""
    return body


# ── 수동 조작 (브라우저에서 호출) ────────────────────────
@app.route("/")
def root():
    return """<html><head><meta charset="utf-8">
    <style>body{background:#0f1419;color:#e6edf3;font-family:monospace;padding:28px}
    a{color:#58a6ff;display:block;margin:8px 0;font-size:15px}</style></head><body>
    <h2>Tank Server</h2>
    <a href="/status">/status &mdash; 현재 상태 보기</a>
    <a href="/goal?x=200&z=250">/goal?x=200&z=250 &mdash; 목적지 지정</a>
    <a href="/stop">/stop &mdash; 정지</a>
    </body></html>"""


@app.route("/goal")
def set_goal_manual():
    """브라우저에서 목적지 지정:  /goal?x=200&z=250"""
    try:
        x = float(request.args["x"])
        z = float(request.args["z"])
    except Exception:
        return "사용법: /goal?x=200&z=250", 400
    with lock:
        S["goal"] = (x, z)
        replan()
        n = len(S["path"])
    msg = f"목적지 ({x}, {z}) 설정 · 경로 {n}개" if n else f"목적지 ({x}, {z}) · 경로 생성 실패"
    return f'<meta charset="utf-8">{msg} <a href="/status">상태 보기</a>'


@app.route("/stop")
def stop_manual():
    with lock:
        S["goal"] = None
        S["path"] = []
        note("수동 정지")
    return '<meta charset="utf-8">정지 <a href="/status">상태 보기</a>'


def zigzag(lines=8, margin=25.0):
    """맵을 가로로 훑는 지그재그 경로 생성"""
    lo, hi = MAP_MIN + margin, MAP_MAX - margin
    pts = []
    for i in range(lines):
        z = lo + (hi - lo) * i / max(1, lines - 1)
        pts += [(lo, z), (hi, z)] if i % 2 == 0 else [(hi, z), (lo, z)]
    return pts


@app.route("/patrol")
def patrol():
    """
    맵 전체를 훑는 순찰 경로를 자동 실행 (지형 스캔용)
      /patrol            기본 7개 지점
      /patrol?lines=10   가로 10줄 지그재그 (촘촘한 스캔)
    """
    lines = request.args.get("lines", type=int)
    with lock:
        S["patrol"] = zigzag(lines) if lines else list(PATROL_POINTS)
        S["goal"] = S["patrol"].pop(0)
        replan()
        note(f"순찰 시작 - 남은 지점 {len(S['patrol'])}개")
    return '<meta charset="utf-8">순찰 시작 <a href="/scan">스캔 현황</a>'


@app.route("/scan")
def scan_status():
    h = scan_heightmap()
    seen = int(np.sum(~np.isnan(h)))
    total = SCAN_N * SCAN_N
    vmin = float(np.nanmin(h)) if seen else 0.0
    vmax = float(np.nanmax(h)) if seen else 0.0
    stale = SCAN["stale"]
    warn = ("<p style='color:#f85149'>LiDAR origin 이 "
            f"{stale}회 연속 동일 - 갱신되지 않는 중일 수 있음</p>") if stale > 20 else ""
    th = track_heightmap()
    tseen = int(np.sum(~np.isnan(th)))
    tvmin = float(np.nanmin(th)) if tseen else 0.0
    tvmax = float(np.nanmax(th)) if tseen else 0.0

    # ── LiDAR 갱신 판정 ──────────────────────────────────
    # lidarOrigin 필드만 낡은 것인지, 점군 좌표 전체가 고정인지 구분한다.
    #   1) 점군 내용 해시가 1가지뿐  -> 배열 자체가 고정
    #   2) 점군 무게중심이 전차를 따라오지 않음 -> 초기 스캔 반복 전송
    cent_gap = "-"
    verdict = "판정 불가 (데이터 부족)"
    vcolor = "#8b949e"
    if SCAN.get("centroid") and S.get("pos"):
        g = ((SCAN["centroid"][0] - S["pos"][0]) ** 2
             + (SCAN["centroid"][1] - S["pos"][1]) ** 2) ** 0.5
        cent_gap = f"{g:.1f} m"
        if len(SCAN.get("hashes", ())) <= 1 and SCAN["frames"] > 30:
            verdict = "점군 고정 - 좌표 배열 전체가 갱신되지 않음"
            vcolor = "#f85149"
        elif g > 60:
            verdict = "점군이 전차를 따라오지 않음 - 초기 스캔 반복 전송"
            vcolor = "#f85149"
        elif g < 30:
            verdict = "정상 - 점군이 전차를 따라옴"
            vcolor = "#3fb950"
        else:
            verdict = "애매 - 더 이동해 보십시오"
            vcolor = "#e3b341"

    return f"""<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
    <style>body{{background:#0f1419;color:#e6edf3;font-family:monospace;padding:26px}}
    td{{padding:5px 16px;border-bottom:1px solid #21262d}} b{{color:#58a6ff}}
    a{{color:#58a6ff}} h3{{color:#8b949e;margin-top:26px}}</style></head><body>
    <h2>지형 스캔 현황</h2>{warn}

    <h3>주행 궤적 샘플링 (주 수집원)</h3><table>
    <tr><td><b>샘플 수</b></td><td>{TRACK['n']:,}</td></tr>
    <tr><td><b>커버 셀</b></td><td>{tseen:,} / {total:,} ({tseen/total*100:.1f}%)</td></tr>
    <tr><td><b>고도 범위</b></td><td>{tvmin:.2f} ~ {tvmax:.2f} m</td></tr>
    <tr><td><b>마지막 지점</b></td><td>{TRACK['last']}</td></tr>
    </table>

    <h3>LiDAR 갱신 판정</h3><table>
    <tr><td><b>점군 내용 종류</b></td><td>{len(SCAN['hashes'])} 가지{
        ' - 모든 프레임이 동일' if len(SCAN['hashes'])<=1 else ' - 내용이 바뀌고 있음'}</td></tr>
    <tr><td><b>점군 무게중심</b></td><td>{'-' if not SCAN.get('centroid') else
        f"({SCAN['centroid'][0]:.1f}, {SCAN['centroid'][1]:.1f})"}</td></tr>
    <tr><td><b>전차 현재 위치</b></td><td>{'-' if not S['pos'] else
        f"({S['pos'][0]:.1f}, {S['pos'][1]:.1f})"}</td></tr>
    <tr><td><b>중심-전차 거리</b></td><td>{cent_gap}</td></tr>
    <tr><td><b>판정</b></td><td style="color:{vcolor}"><b>{verdict}</b></td></tr>
    </table>

    <h3>LiDAR 점군 (원시 수치)</h3><table>
    <tr><td><b>프레임</b></td><td>{SCAN['frames']}</td></tr>
    <tr><td><b>누적 포인트</b></td><td>{SCAN['points']:,}</td></tr>
    <tr><td><b>커버 셀</b></td><td>{seen:,} ({seen/total*100:.1f}%)</td></tr>
    <tr><td><b>고도 범위</b></td><td>{vmin:.2f} ~ {vmax:.2f} m</td></tr>
    <tr><td><b>origin 정체</b></td><td>{stale}회 연속</td></tr>
    </table>

    <p><a href="/patrol?lines=8">지그재그 순찰(8줄)</a> ·
       <a href="/patrol?lines=12">촘촘히(12줄)</a> ·
       <a href="/scan_save">heightmap 저장</a> ·
       <a href="/status">전차 상태</a></p></body></html>"""


@app.route("/scan_save")
def scan_save():
    th = track_heightmap()
    lh = scan_heightmap()
    filled = interpolate(th)

    np.save("heightmap_track.npy", th)       # 관측 원본 (희소)
    np.save("heightmap_filled.npy", filled)  # IDW 보간본 (조밀)
    np.save("heightmap_lidar.npy", lh)       # LiDAR 원본 (참고용)
    np.save("track_count.npy", TRACK["cnt"])

    a = int(np.sum(~np.isnan(th)))
    b = int(np.sum(~np.isnan(filled)))
    note(f"heightmap 저장 - 관측 {a}셀 → 보간 후 {b}셀")
    return (f'<meta charset="utf-8">저장 완료 · 격자 {SCAN_N}x{SCAN_N}<br>'
            f'heightmap_track.npy (관측 {a}셀)<br>'
            f'heightmap_filled.npy (보간 {b}셀)<br>'
            f'heightmap_lidar.npy (LiDAR 참고)<br>'
            f'<a href="/scan">돌아가기</a>')


# ── 정의되지 않은 엔드포인트도 기록 (시뮬레이터가 뭘 부르는지 확인용) ──
@app.route("/<path:unknown>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def unknown_endpoint(unknown):
    note(f"[미정의] {request.method} /{unknown}")
    return jsonify({})


if __name__ == "__main__":
    print("=" * 52)
    print("  Tank Server")
    print("    상태 :  http://localhost:5000/status")
    print("    이동 :  http://localhost:5000/goal?x=200&z=250")
    print("    정지 :  http://localhost:5000/stop")
    print("    순찰 :  http://localhost:5000/patrol?lines=8   (지형 스캔)")
    print("    스캔 :  http://localhost:5000/scan")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False)
