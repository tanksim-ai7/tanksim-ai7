"""
mission_log.py — 아군 전차 작전 상황 로그 (대시보드 '작전 로그' 패널용)

주행/사격/인식 코드가 이 모듈의 함수를 불러 상황을 남기고,
ally-controller.py 의 /mission_log 가 대시보드에 그대로 내려준다.

같은 함수가 단계마다 다른 문구로 쓰이므로 두 가지를 관리한다.

  phase  (어느 단계인가)
      rescue  구출 지점으로 가는 중          (ALLY_DEST_LIST[0])
      combat  교전 지점으로 이동 + 적 전차 교전 (SEQ_FLAG == 'second')
      base    아군 기지로 복귀               (SEQ_FLAG == 'third')

  flag   (같은 문구가 프레임마다 반복 출력되는 것을 막는다)
      탐지/도착/경로 탐색은 매 프레임 호출되므로, 한 번 남기면
      다음 계기(목적지 변경, 후퇴 완료 등)가 오기 전까지 다시 남기지 않는다.

이 모듈은 로그만 쌓는다. 제어 로직에는 아무 영향도 주지 않는다.
"""

import functools
import threading
import time
from collections import deque

PHASE_RESCUE = "rescue"
PHASE_COMBAT = "combat"
PHASE_BASE = "base"

# 이 단계에서는 이동 관련 로그(목적지/경로 탐색/이동 시작/도착)를 남기지 않는다.
# combat 은 구출 지점 도착 직후의 짧은 교전 위치 이동이라 시나리오 로그에서 뺐다.
# 남기고 싶으면 비워 두면 된다.
QUIET_NAV_PHASES = {PHASE_COMBAT}

_DEST_LABEL = {
    PHASE_RESCUE: "구출 지점",
    PHASE_COMBAT: "교전 지점",
    PHASE_BASE: "아군 기지",
}

MAX_EVENTS = 200


class MissionLog:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    # ── 상태 ────────────────────────────────────────────
    def reset(self):
        """새 에피소드 시작. 로그와 단계, 플래그를 모두 비운다."""
        with self._lock:
            self._events = deque(maxlen=MAX_EVENTS)
            self._seq = 0
            self._epoch = 0 if not hasattr(self, "_epoch") else self._epoch + 1
            self._flags = set()
            self._pflags = set()      # 단계(phase) 안에서 한 번만: 목적지가 바뀌어도 유지
            self._last_shot = 0
            self._my_hp = None
            self._my_hits = 0
            self._phase = PHASE_RESCUE
            self._sim_t = None
            self._t0 = time.monotonic()

    def set_phase(self, phase):
        with self._lock:
            if phase != self._phase:
                self._pflags.clear()
            self._phase = phase

    @property
    def phase(self):
        return self._phase

    def set_time(self, sim_t):
        """/info 의 시뮬레이션 시간. 없으면 서버 경과 시간을 쓴다."""
        if sim_t is None:
            return
        try:
            self._sim_t = float(sim_t)
        except (TypeError, ValueError):
            pass

    # ── 내부 ────────────────────────────────────────────
    def _now(self):
        return self._sim_t if self._sim_t is not None else time.monotonic() - self._t0

    def _add(self, text, level="info"):
        self._seq += 1
        self._events.append({"id": self._seq, "t": round(self._now(), 1),
                             "text": text, "level": level})

    def _emit(self, text, level="info", nav=False):
        with self._lock:
            if nav and self._phase in QUIET_NAV_PHASES:
                return
            self._add(text, level)

    def _once(self, flag, text, level="info", nav=False, persist=False):
        """
        flag 가 서 있지 않을 때만 남기고 flag 를 세운다.
        persist  True 면 목적지가 바뀌어도 유지되고 단계(phase)가 바뀔 때만 풀린다.
        """
        with self._lock:
            flags = self._pflags if persist else self._flags
            if flag in flags:
                return False
            flags.add(flag)
            if nav and self._phase in QUIET_NAV_PHASES:
                return False
            self._add(text, level)
            return True

    def _clear(self, *flags):
        for f in flags:
            self._flags.discard(f)

    # ── 이동 (pid_controller) ───────────────────────────
    def dest_set(self):
        """1·14. 새 목적지. 이전 목적지에서 쌓인 플래그를 모두 비운다."""
        with self._lock:
            # 기지 복귀 목적지는 단계당 한 번만 알린다. 같은 목적지가 다시 들어와도
            # (동시에 들어온 /info 등) 로그와 진행 플래그를 되돌리지 않는다.
            if self._phase == PHASE_BASE:
                if "dest" in self._pflags:
                    return
                self._pflags.add("dest")
            self._flags.clear()
            if self._phase == PHASE_COMBAT:
                # 구출 지점 도착 뒤 왜 움직이는지 한 번만 알린다.
                # 이후 교전 지점이 계속 바뀌어도(순찰) 다시 남기지 않는다.
                if "move" not in self._pflags:
                    self._pflags.add("move")
                    self._add("교전 지점으로 이동 (적 전차 요격 위치)")
                return
            if self._phase in QUIET_NAV_PHASES:
                return
            if self._phase == PHASE_BASE:
                self._add("아군 기지로 목적지 변경")
            else:
                self._add("목적지 설정됨 (%s)" % _DEST_LABEL[self._phase])

    def path_search(self, replan=False, auto=False):
        """
        2·7·15. 경로 탐색 시작.

        replan  후퇴 완료·장애물 변경 등 '다시 찾는' 이유가 있을 때 True.
        auto    get_action 이 경로가 빈 것을 보고 매 tick 부르는 호출.
                이 구간에서 처음 하는 탐색일 때만 남기고, 나머지는 조용히 넘긴다.

        문구는 이 구간(목적지 하나)에서 이미 탐색을 남겼는지로 정한다.
        첫 탐색이면 replan 이어도 '경로 탐색중...' 이다.
        """
        with self._lock:
            if self._phase in QUIET_NAV_PHASES:
                return
            searched = "searched" in self._flags
            # auto(매 tick 의 경로 갱신)와 replan 이 아닌 일반 호출(목적지 설정)은
            # 이 구간의 첫 탐색일 때만 남긴다.
            if searched and (auto or not replan):
                return
            # 이미 탐색 중이면(실패 후 재시도 포함) 남기지 않는다.
            # (여기서 'moving' 을 건드리면 이동 시작이 매 tick 다시 나온다.)
            if "searching" in self._flags:
                return
            self._flags.update(("searching", "searched"))
            self._clear("moving")
            self._add("경로 재탐색..." if (replan and searched) else "경로 탐색중...")

    def path_found(self, ok):
        """
        3·8·16. 경로 탐색 완료. 탐색 시작을 남긴 뒤의 성공만 남긴다.

        실패는 화면에 남기지 않는다. 실패하면 '탐색 중' 상태를 그대로 두어,
        이후 재시도가 성공했을 때 '경로 탐색 완료'가 이어서 나오게 한다.
        """
        if not ok:
            return
        with self._lock:
            searching = "searching" in self._flags
            self._clear("searching")
            if self._phase in QUIET_NAV_PHASES:
                return
            if searching:
                self._add("경로 탐색 완료")

    def move_start(self):
        """4·9·17. 경로를 따라 실제로 움직이기 시작. 탐색 뒤 한 번만."""
        self._once("moving", "경로 이동 시작", nav=True)

    def arrived(self):
        """10·18. 목적지 도착. 목적지마다 한 번."""
        if self._phase == PHASE_COMBAT:
            self._once("wait", "교전 지점 도착 · 적 전차 대기중...", "info", persist=True)
            return
        text = "목적지 도착 (%s)" % _DEST_LABEL[self._phase]
        self._once("arrived", text, "ok", nav=True)

    # ── 위협 (pid_controller, 인식) ─────────────────────
    def tank_detected(self, matched):
        """
        5·11. 스테레오로 잡은 Tank1 탐지 한 건.

        matched  탐지 좌표가 맵에 등록된 오브젝트와 겹치는가.

        rescue/base : 맵 좌표와 맞는(matched) 적군 탱크만 '발견'으로 본다.
        combat      : 맵에 없는(unmatched) Tank1 = 움직이는 적 전차다.
                      먼 거리에서 좌표가 요동쳐 생기는 unmatched 도 섞일 수 있어
                      combat 단계에서만 인정한다.
        """
        if self._phase == PHASE_COMBAT:
            if not matched:
                self._once("enemy_found", "적 전차 발견", "warn", persist=True)
        elif matched:
            self._once("threat", "적군 탱크 발견", "warn")

    def retreat(self):
        """6. 후퇴 시작."""
        with self._lock:
            self._clear("moving")
        self._once("retreat", "후퇴", "warn")

    def retreat_done(self):
        """후퇴 종료. 다음 탐지·후퇴를 다시 남길 수 있게 플래그를 푼다."""
        with self._lock:
            self._clear("threat", "retreat", "moving")

    # ── 교전 (ally-controller) ──────────────────────────
    def shots(self, fired, dist=None):
        """
        12. 포격. fired 는 FireModule 의 누적 발사 수. 새로 늘어난 만큼만 남긴다.
        첫 발에서 '적 전차 발견'(아직 없으면)과 '교전중...'을 먼저 남긴다.
        """
        if self._phase != PHASE_COMBAT:
            return
        with self._lock:
            if fired < self._last_shot:      # 사격 모듈이 초기화됨
                self._last_shot = fired
            new = list(range(self._last_shot + 1, fired + 1))
            self._last_shot = max(self._last_shot, fired)
        if not new:
            return
        self._once("enemy_found", "적 전차 발견", "warn", persist=True)
        self._once("engaging", "적 전차와 교전중...", "warn", persist=True)
        for n in new:
            d = " (거리 %.0f m)" % dist if (dist is not None and n == new[-1]) else ""
            self._emit("포격 #%d%s" % (n, d), "warn")

    def enemy_hit(self, count, total=2):
        """13. 적 전차 명중 (count/total)."""
        self._emit("적 전차 명중 (%d/%d)" % (count, total), "ok")

    def ally_hp(self, hp):
        """
        아군 피격. /info 의 playerHealth 가 줄어드는 순간을 한 번의 피격으로 본다
        (대시보드 체력 표시와 같은 기준). 값이 다시 늘어나면 새 에피소드로 보고 센 횟수를 비운다.
        """
        if hp is None:
            return
        with self._lock:
            prev = getattr(self, "_my_hp", None)
            self._my_hp = hp
            if prev is None:
                return
            if hp > prev:
                self._my_hits = 0
                return
            if hp < prev:
                self._my_hits = getattr(self, "_my_hits", 0) + 1
                self._add("아군 피격 (누적 %d회)" % self._my_hits, "warn")

    def shot_missed(self):
        """포탄이 적 전차에 맞지 않았다."""
        self._emit("빗나감")

    def enemy_destroyed(self):
        """13. 적 전차 격파."""
        self._once("destroyed", "적 전차 격파 확인", "ok")

    # ── 조회 ────────────────────────────────────────────
    def snapshot(self, after=0):
        with self._lock:
            return {"epoch": self._epoch, "phase": self._phase,
                    "events": [e for e in self._events if e["id"] > after]}



def _guard(fn):
    """
    로그 기록이 실패해도 호출한 제어 코드(주행/사격)가 끊기지 않게 한다.
    이 모듈은 화면 표시용이라, 여기서 예외가 나도 조용히 넘기고 콘솔에만 남긴다.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print("[mission_log] 기록 실패(무시): %s: %s" % (type(exc).__name__, exc))
            return None
    return wrapper


# snapshot 은 /mission_log 응답을 만드는 조회용이라 제외한다.
for _name in ("reset", "set_phase", "set_time", "dest_set", "path_search", "path_found",
              "move_start", "arrived", "tank_detected", "retreat", "retreat_done",
              "shots", "enemy_hit", "ally_hp", "shot_missed", "enemy_destroyed"):
    setattr(MissionLog, _name, _guard(getattr(MissionLog, _name)))

mission_log = MissionLog()
