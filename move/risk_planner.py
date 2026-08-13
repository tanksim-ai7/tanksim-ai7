# -*- coding: utf-8 -*-
"""
risk_planner.py - DStarPlanner 에 위험 비용 레이어를 얹는다

원본을 수정하지 않고 상속으로 확장한다.
기존 코드에서 DStarPlanner(...) 를 RiskDStarPlanner(...) 로 바꾸기만 하면 된다.

이식 지점이 명확한 이유
    원본 movement_cost 가 이미
        base_cost + proximity_cost
    형태이고, proximity_cost 는 셀별 가산 비용 딕셔너리의 두 셀 평균이다.
    위험 비용도 정확히 같은 자리에 한 항 더 붙이면 된다.

휴리스틱
    손대지 않는다. octile 거리는 위험 비용을 더해도 여전히 과소추정이므로
    admissible 이 유지되고 최적성이 깨지지 않는다.
    (반대로 휴리스틱에 위험도를 반영하면 과대추정이 되어 최적해를 놓친다)

정적 / 동적 구분
    set_risk_layers()  정적 레이어. 시작 시 1회. 전체 재탐색.
    update_risk_cells() 동적 레이어. 바뀐 셀 주변만 갱신.
                        D* Lite 를 쓰는 의미가 여기에 있다.
"""
import math
import numpy as np
import threading

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from move.dstar_lite_planner_cost import DStarPlanner, INF


class RiskDStarPlanner(DStarPlanner):

    def __init__(self, *args,
                 slope_weight=3.0,
                 exposure_weight=6.0,
                 threat_weight=40.0,
                 cell_size=1.0,
                 **kw):
        """
        slope_weight     경사 저항 가중치. 이동 저항 성격
        exposure_weight  사전 피탐도 가중치. 능선/개활지 회피 강도
        threat_weight    관측된 적 위협. 다른 항을 압도해야 한다
        cell_size        격자 한 칸의 실제 크기 [m]

        기준: base_cost 가 1.0 (직진) 이므로
              가중치 6.0 은 "피탐도 1.0 인 셀 1칸 = 평지 7칸" 을 뜻한다.

        cell_size 를 왜 두는가
            원본은 300x300 격자에 셀 1 m 를 가정한다. 그러나 비용이
            셀마다 다르면 D* Lite 의 확장 노드 수가 급증해
            300x300 에서 계획에 20초 이상 걸린다.
            고도맵이 2 m 격자이고 전차 폭이 3.6 m 이므로
            150x150 (셀 2 m) 로 충분하고, 노드가 1/4 이라 훨씬 빠르다.
        """
        super().__init__(*args, **kw)
        self.cell_size = float(cell_size)
        self.slope_weight = float(slope_weight)
        self.exposure_weight = float(exposure_weight)
        self.threat_weight = float(threat_weight)

        self.slope_cost = None        # (H, W) float, 0~1
        self.exposure = None          # (H, W) float, 0~1
        self.threat = None            # (H, W) float, 0~1  동적
        self.terrain_blocked = set()  # 경사 한계 초과 셀

    # ── 좌표 변환 (셀 크기 반영) ──────────────────────────
    def world_to_grid(self, position, clamp=False):
        if position is None:
            return None
        if self.cell_size == 1.0:
            return super().world_to_grid(position, clamp)
        x = int(round(float(position[0]) / self.cell_size))
        z = int(round(float(position[1]) / self.cell_size))
        if clamp:
            x = min(max(x, 0), self.width - 1)
            z = min(max(z, 0), self.height - 1)
        node = (x, z)
        if not self.in_bounds(node):
            raise ValueError(f"좌표 {position} 가 Grid 범위를 벗어났습니다.")
        return node

    def grid_to_world(self, node):
        return (float(node[0]) * self.cell_size,
                float(node[1]) * self.cell_size)

    # ── 셀 비용 조회 ──────────────────────────────────────
    def get_risk(self, node):
        x, z = node
        c = 0.0
        if self.slope_cost is not None:
            c += self.slope_weight * self.slope_cost[z, x]
        if self.exposure is not None:
            c += self.exposure_weight * self.exposure[z, x]
        if self.threat is not None:
            c += self.threat_weight * self.threat[z, x]
        return c

    # ── 원본 훅 두 개만 재정의 ────────────────────────────
    def is_free(self, node):
        return super().is_free(node) and node not in self.terrain_blocked

    def movement_cost(self, a, b):
        base = super().movement_cost(a, b)
        if base >= INF:
            return INF
        return base + 0.5 * (self.get_risk(a) + self.get_risk(b))

    # ── 정적 레이어 주입 ──────────────────────────────────
    def set_risk_layers(self, npz_path="move/risk_layers.npz",
                        use_blocked=True, reset=True):
        d = np.load(npz_path)
        sc = d["slope_cost"].astype(np.float64)
        ex = d["exposure"].astype(np.float64)
        bl_arr = d["blocked"]

        # 저장된 레이어는 300x300 이다. 격자가 더 성기면 다운샘플한다.
        if sc.shape != (self.height, self.width):
            f = sc.shape[0] // self.height
            if f >= 1 and sc.shape[0] == self.height * f:
                sc = sc.reshape(self.height, f, self.width, f).mean(axis=(1, 3))
                ex = ex.reshape(self.height, f, self.width, f).mean(axis=(1, 3))
                # 통행 불가는 하나라도 막히면 막힌 것으로 본다 (보수적)
                bl_arr = bl_arr.reshape(self.height, f,
                                        self.width, f).any(axis=(1, 3))

        self.slope_cost = sc
        self.exposure = ex

        if self.slope_cost.shape != (self.height, self.width):
            raise ValueError(
                f"레이어 크기 {self.slope_cost.shape} 가 플래너 격자 "
                f"({self.height}, {self.width}) 와 다릅니다. "
                "risk_map.PLANNER_CELL 확인 필요")

        if use_blocked:
            zs, xs = np.where(bl_arr)
            self.terrain_blocked = set(zip(xs.tolist(), zs.tolist()))

        if reset:
            self.refresh_costmap()
        return {
            "blocked": len(self.terrain_blocked),
            "mean_slope_cost": float(self.slope_cost.mean()),
            "mean_exposure": float(self.exposure.mean()),
        }

    # ── 동적 레이어 (관측 위협) ───────────────────────────
    def ensure_threat(self):
        if self.threat is None:
            self.threat = np.zeros((self.height, self.width), dtype=np.float64)

    def add_threat(self, wx, wz, sigma_m=25.0, peak=1.0, radius_m=None):
        """
        적을 관측했다. 중심이 강하고 주변으로 번지는 봉우리를 더한다.

        시간이 지나면 위험이 사라지는 것이 아니라 번진다.
        적이 t 초 전 이 자리에 있었다면 지금은 반경 v*t 안 어디든 있다.
        따라서 decay_threat() 에서 sigma 를 키우고 peak 를 낮춘다.

        반환: 값이 바뀐 셀 목록 (update_risk_cells 에 넘긴다)
        """
        self.ensure_threat()
        if radius_m is None:
            radius_m = 3.0 * sigma_m
        cx = int(round(wx / self.cell_size))
        cz = int(round(wz / self.cell_size))
        r = int(round(radius_m / self.cell_size))
        sigma_m = sigma_m / self.cell_size
        x0, x1 = max(0, cx - r), min(self.width - 1, cx + r)
        z0, z1 = max(0, cz - r), min(self.height - 1, cz + r)

        xs = np.arange(x0, x1 + 1)
        zs = np.arange(z0, z1 + 1)
        dx = (xs - cx)[None, :]
        dz = (zs - cz)[:, None]
        g = peak * np.exp(-(dx * dx + dz * dz) / (2.0 * sigma_m ** 2))

        before = self.threat[z0:z1 + 1, x0:x1 + 1].copy()
        self.threat[z0:z1 + 1, x0:x1 + 1] = np.maximum(before, g)
        changed = np.where(self.threat[z0:z1 + 1, x0:x1 + 1] != before)
        return [(int(x0 + a), int(z0 + b))
                for b, a in zip(changed[0], changed[1])]

    def decay_threat(self, dt, tau=6.9):
        """
        재장전 시간(6.9 s)을 시정수로 감쇠시킨다.
        적이 한 발 쏘면 다음 발까지 6.9 s 이므로
        그 안에 노출 구간을 벗어나야 한다는 전술적 의미가 붙는다.
        """
        if self.threat is None:
            return []
        old = self.threat.copy()
        self.threat *= math.exp(-dt / tau)
        self.threat[self.threat < 0.01] = 0.0
        ch = np.where(old != self.threat)
        return [(int(x), int(z)) for z, x in zip(ch[0], ch[1])]

    # ── 증분 갱신 ─────────────────────────────────────────
    def update_risk_cells(self, cells):
        """
        바뀐 셀과 그 이웃만 다시 계산한다.
        refresh_costmap() 은 내부에서 탐색 상태를 통째로 버리므로
        실시간 갱신에 쓰면 D* Lite 를 쓰는 의미가 사라진다.
        """
        if not cells:
            return 0
        affected = set()
        for c in cells:
            if not self.in_bounds(c):
                continue
            affected.add(c)
            affected.update(self.get_neighbors(c))
        for n in affected:
            self.update_vertex(n)
        return len(affected)

    # ── 진단 ──────────────────────────────────────────────
    def path_risk_profile(self, path=None):
        """경로가 실제로 위험을 피해 갔는지 확인한다"""
        p = path if path is not None else self.get_path()
        if not p:
            return None
        vals = []
        for wp in p:
            n = self.world_to_grid(wp, clamp=True)
            x, z = n
            vals.append((
                float(self.slope_cost[z, x]) if self.slope_cost is not None else 0.0,
                float(self.exposure[z, x]) if self.exposure is not None else 0.0,
                float(self.threat[z, x]) if self.threat is not None else 0.0,
            ))
        a = np.array(vals)
        return {"n": len(p),
                "slope_mean": float(a[:, 0].mean()),
                "exposure_mean": float(a[:, 1].mean()),
                "exposure_max": float(a[:, 1].max()),
                "threat_max": float(a[:, 2].max())}

    def plot(
            self,
            path=None,
            show_grid=True,
            title="D* Lite",
            save_path=None,
            show=False,
        ):
            """
            서버 호출:
                planner.plot(path=current_path, show_grid=True, title="...")
            """
            active_path = self.last_path if path is None else path
    
            fig, ax = plt.subplots(figsize=(8, 8))
    
            height_matrix = np.zeros((self.width, self.height))
            for ix in range(self.width):
                for iz in range(self.height):
                    height_matrix[ix, iz] = self.y.get((ix, iz), 0.0)
            height_matrix = height_matrix.T
    
            # 2. 플롯 설정 및 그리드 히트맵 드로잉
            im = ax.imshow(
                height_matrix, 
                cmap='terrain', 
                origin='lower',
                extent=[0, self.width, 0, self.height],
                zorder=1
            )

            for x,z in self.terrain_blocked:
                rect_grid = plt.Rectangle(
                    ((x+0.5) - 1.0 * 0.5, (z+0.5) - 1.0 * 0.5), # 사각형의 시작점(좌측 하단)
                    1.0, 1.0,
                    facecolor='magenta', 
                    edgecolor='none', 
                    alpha=0.3, # 다른 표시와 겹쳐도 다 보이도록 투명도 설정
                    zorder=2
                )
                ax.add_patch(rect_grid)
            ax.plot([], [], color='magenta', alpha=0.3, label="Blocked Terrain", linestyle='-', linewidth=5)
    
            for obs in self.obstacle_rectangles:
                patch = Rectangle(
                    (
                        obs.x_min - self.obstacle_margin,
                        obs.z_min - self.obstacle_margin,
                    ),
                    (obs.x_max - obs.x_min) + 2 * self.obstacle_margin,
                    (obs.z_max - obs.z_min) + 2 * self.obstacle_margin,
                    alpha=0.5,
                )
                ax.add_patch(patch)
    
            if active_path:
                px = [point[0] for point in active_path]
                pz = [point[1] for point in active_path]
    
                ax.plot(px, pz, marker="o", markersize=2, label="D* Lite Path")
                ax.scatter(px[0], pz[0], s=80, marker="o", label="Start")
                ax.scatter(px[-1], pz[-1], s=120, marker="*", label="Goal")
    
            else:
                ax.scatter(
                    self.start[0],
                    self.start[1],
                    s=80,
                    marker="o",
                    label="Start",
                )
                ax.scatter(
                    self.goal[0],
                    self.goal[1],
                    s=120,
                    marker="*",
                    label="Goal",
                )
    
            
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Height / Altitude (meters)", rotation=275, labelpad=15)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_title("Path Map")
    
            if show_grid:
                ax.grid(True, alpha=0.3)
    
            ax.legend()
            fig.tight_layout()
    
            if save_path:
                output = Path(save_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                print("grid 저장 완료")
    
            if show:
                plt.show()
            else:
                plt.close(fig)
    
            return fig, ax

    # --------------------------------------------------
    # 논블로킹 렌더링
    # --------------------------------------------------

    def plot_async(self, path=None, show_grid=True, title="D* Lite", save_path=None):
        # plot()(matplotlib 저장)이 응답을 막지 않도록 
        # 별도 스레드로 돌리되, 그리는 데 필요한 값만 미리 
        # 복사(스냅샷)해서 넘겨 다른 요청과 충돌 없이 안전하게 렌더링하기 위함.
        active_path = list(self.last_path if path is None else path)
        terrain_blocked_snapshot = set(getattr(self, "terrain_blocked", set()))
        obstacle_rectangles_snapshot = list(self.obstacle_rectangles)
        start_snapshot = self.start
        goal_snapshot = self.goal
        obstacle_margin_snapshot = self.obstacle_margin

        thread = threading.Thread(
            target=self._render_and_save,
            args=(
                active_path,
                terrain_blocked_snapshot,
                obstacle_rectangles_snapshot,
                start_snapshot,
                goal_snapshot,
                obstacle_margin_snapshot,
                show_grid,
                title,
                save_path,
            ),
            daemon=True,
        )
        thread.start()

    def _render_and_save(self, active_path, terrain_blocked, obstacle_rectangles,
                          start, goal, obstacle_margin, show_grid, title, save_path):
        """
        plot_async()가 뜬 스냅샷으로 실제 렌더링을 수행한다.
        pyplot을 쓰지 않고 Figure를 직접 만들어서, 다른 스레드가 동시에
        plot()/plot_async()를 불러도 서로 간섭하지 않는다.
        """
        fig = Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        height_matrix = np.zeros((self.width, self.height))
        for ix in range(self.width):
            for iz in range(self.height):
                height_matrix[ix, iz] = self.y.get((ix, iz), 0.0)
        height_matrix = height_matrix.T

        im = ax.imshow(
            height_matrix,
            cmap='terrain',
            origin='lower',
            extent=[0, self.width, 0, self.height],
            zorder=1,
        )

        for x, z in terrain_blocked:
            rect_grid = Rectangle(
                ((x + 0.5) - 0.5, (z + 0.5) - 0.5),
                1.0, 1.0,
                facecolor='magenta',
                edgecolor='none',
                alpha=0.3,
                zorder=2,
            )
            ax.add_patch(rect_grid)
        ax.plot([], [], color='magenta', alpha=0.3, label="Blocked Terrain", linestyle='-', linewidth=5)

        for obs in obstacle_rectangles:
            patch = Rectangle(
                (obs.x_min - obstacle_margin, obs.z_min - obstacle_margin),
                (obs.x_max - obs.x_min) + 2 * obstacle_margin,
                (obs.z_max - obs.z_min) + 2 * obstacle_margin,
                alpha=0.5,
            )
            ax.add_patch(patch)

        if active_path:
            px = [point[0] for point in active_path]
            pz = [point[1] for point in active_path]
            ax.plot(px, pz, marker="o", markersize=2, label="D* Lite Path")
            ax.scatter(px[0], pz[0], s=80, marker="o", label="Start")
            ax.scatter(px[-1], pz[-1], s=120, marker="*", label="Goal")
        else:
            ax.scatter(start[0], start[1], s=80, marker="o", label="Start")
            ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height / Altitude (meters)", rotation=275, labelpad=15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(title)

        if show_grid:
            ax.grid(True, alpha=0.3)

        ax.legend()
        fig.tight_layout()

        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.print_figure(output, dpi=150, bbox_inches="tight")
            print("grid 저장 완료 (백그라운드)")