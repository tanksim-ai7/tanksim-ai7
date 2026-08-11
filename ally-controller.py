from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.dstar_lite_planner_cost import ObstacleRect as DStarLiteObstacleRect
import numpy as np
import math
import matplotlib
from move.navigation_controller import NavigationController as nav
from move.pid_controller import (
    update_info_speed,
    read_player_speed_kmh,
    read_player_body_yaw_deg,
    get_control_dt,
    check_destination_change,
    update_arrival_state,
    calculate_target_speed_kmh,
    find_upcoming_corner,
    calculate_corner_speed_limit,
    calculate_steering_command,
    apply_alignment_speed_limit,
    make_longitudinal_command,
    make_stop_command,
    reset_all_controller_state,
)
matplotlib.use("Agg")

app = Flask(__name__)
model = YOLO('yolov8n.pt')
print(model.names)

nav = NavigationController()

path_flag = False # 경로 최초 탐색 여부
path = []
path_idx = 0 # path를 위한 idx
all_info = None # info 정보
dest = None # 목적지

path_planner = DStarLitePlanner() # 초기화할 때 고도정보도 같이 넣어준다.

# def update_dstar_obstacles_from_payload(payload: dict):
#     obs_list = []
#     for item in payload.get("obstacles", []):
#         obs = DStarLiteObstacleRect.from_min_max(
#             x_min=item["x_min"],
#             x_max=item["x_max"],
#             z_min=item["z_min"],
#             z_max=item["z_max"],
#         )
#         obs_list.append(obs)
#     path_planner.set_obstacles(obs_list)

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    # print(results[0].boxes.data)
    target_classes = {0: "tank",1: "rock", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return jsonify({"result": "error", "message": str(e)}), 500

    return jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    global all_info
    all_info = data

    # /info의 현재 위치를 읽는다.
    player_pos = data.get(
        "playerPos",
        {}
    )

    # x/z가 실제로 있을 때만 navigation 위치와
    # PID 속도추정 상태를 갱신한다.
    if (
        player_pos.get("x") is not None
        and player_pos.get("z") is not None
    ):
        current_position = [
            float(player_pos["x"]),
            float(player_pos["z"]),
        ]

        # NavigationController가 현재 위치를 기억하도록 한다.
        nav.set_current_position(
            current_position
        )

        # PIDController 쪽 속도 상태 갱신.
        #
        # explicit speed가 있으면 그것을 우선 사용하고,
        # 없으면 position / dt로 계산한다.
        update_info_speed(
            data,
            current_position,
        )

    return jsonify({"status": "success", "control": ""})

@app.route('/get_action', methods=['POST'])
def get_action():
    """
    trackingMode에서 실제 전차 이동 command를 반환한다.

    전체 처리 흐름
    --------------
    1. 현재 위치 갱신
    2. 목적지 존재 여부 확인
    3. 현재 속도 확인
    4. 차체 yaw 확인
    5. 목적지 변경 여부 확인
    6. PID dt 계산
    7. 현재 위치 기준 D* Lite 경로 재계산
    8. 목적지 도착/최종 제동 확인
    9. Look-ahead 조향 계산
    10. 앞쪽 코너 탐색
    11. 목적지 제동속도 + 코너 제한속도 결정
    12. 속도 PID로 W/S 계산
    13. 차체 정렬 상태에 따라 W 제한
    14. A/D 조향을 command에 결합
    """
    data = request.get_json(force=True)

    position = data.get("position", {})
    previous_position = (
        nav.get_current_position()
    )
    turret = data.get("turret", {})

    pos_x = float(
        position.get(
            "x",
            previous_position[0]
            if previous_position is not None
            else 0.0,
        )
    )
    pos_y = position.get("y", 0)
    pos_z = float(
        position.get(
            "z",
            previous_position[1]
            if previous_position is not None
            else 0.0,
        )
    )

    current_position = [
        pos_x,
        pos_z,
    ]

    nav.set_current_position(
        current_position
    )
    # 목적지 확인
    destination = nav.get_destination()
    # 목적지가 아직 설정되지 않았으면 움직이지 않는다.
    if destination is None:
        return jsonify(
            make_stop_command()
        )

    # --------------------------------------------------------
    # 현재 속도 확인
    # --------------------------------------------------------

    current_speed_kmh = (
        read_player_speed_kmh()
    )

    # /info가 아직 충분히 들어오지 않아
    # 속도를 계산하지 못했다면 안전하게 정지 명령.
    if current_speed_kmh is None:
        return jsonify(
            make_stop_command()
        )

    # --------------------------------------------------------
    # 차체 yaw 확인
    # --------------------------------------------------------

    body_yaw_deg = (
        read_player_body_yaw_deg(
            all_info or {}
        )
    )

    # yaw가 없으면 어느 방향으로 조향해야 할지 계산할 수 없다.
    if body_yaw_deg is None:
        return jsonify(
            make_stop_command()
        )

    # --------------------------------------------------------
    # 목적지 변경 감지
    # --------------------------------------------------------

    # 목적지가 이전 제어의 목적지와 다르면
    # PID와 도착/제동 상태를 새 목적지 기준으로 초기화한다.
    check_destination_change(
        destination
    )

    # --------------------------------------------------------
    # PID 제어주기 dt
    # --------------------------------------------------------

    now, dt = get_control_dt()

    # --------------------------------------------------------
    # 현재 위치 기준 경로 재계산
    # --------------------------------------------------------

    try:
        current_path = nav.replan(
            render=False
        )

    except ValueError:
        # 경로계산에 필요한 위치/목적지 상태가 잘못된 경우
        # 이동 명령을 내리지 않는다.
        return jsonify(
            make_stop_command()
        )

    # D* Lite가 이동 가능한 경로를 찾지 못한 경우.
    if not current_path:
        return jsonify(
            make_stop_command()
        )

    # --------------------------------------------------------
    # 목적지까지 실제 직선거리
    # --------------------------------------------------------

    goal_dx = (
        float(destination[0])
        - pos_x
    )

    goal_dz = (
        float(destination[1])
        - pos_z
    )

    distance_to_goal = math.hypot(
        goal_dx,
        goal_dz,
    )

    # --------------------------------------------------------
    # 목적지 도착 / 최종 S 제동
    # --------------------------------------------------------

    arrival_command = (
        update_arrival_state(
            distance_to_goal=distance_to_goal,
            current_speed_kmh=current_speed_kmh,
            now=now,
        )
    )

    # None이 아니라 command가 반환되면
    # 이미 목적지 도착 제어 상태다.
    if arrival_command is not None:
        return jsonify(
            arrival_command
        )

    # --------------------------------------------------------
    # D* Lite Look-ahead 기반 조향
    # --------------------------------------------------------

    steering_command, steering_info = (
        calculate_steering_command(
            current_position=current_position,
            body_yaw_deg=body_yaw_deg,
            path=current_path,
            current_speed_kmh=current_speed_kmh,
            dt=dt,
        )
    )

    # 경로는 있으나 조향 target을 계산하지 못한 경우.
    if steering_info is None:
        return jsonify(
            make_stop_command()
        )

    # --------------------------------------------------------
    # 앞쪽 코너 탐색 및 코너 제한속도
    # --------------------------------------------------------

    corner = find_upcoming_corner(
        path=current_path,
        current_position=current_position,
    )

    corner_speed_limit_kmh = (
        calculate_corner_speed_limit(
            corner
        )
    )

    # --------------------------------------------------------
    # 목적지 제동거리 기반 목표속도
    # --------------------------------------------------------

    destination_speed_limit_kmh = (
        calculate_target_speed_kmh(
            distance_to_goal
        )
    )

    # 두 제한 중 더 낮은 속도를 실제 목표속도로 사용한다.
    #
    # 예:
    # 목적지 기준 60 km/h 가능
    # 코너 기준 18 km/h
    # -> 실제 목표속도 18 km/h
    target_speed_kmh = min(
        destination_speed_limit_kmh,
        corner_speed_limit_kmh,
    )

    # --------------------------------------------------------
    # 속도 PID
    # --------------------------------------------------------

    # PID error = 목표속도 - 현재속도.
    speed_error_kmh = (
        target_speed_kmh
        - current_speed_kmh
    )

    # + output -> W
    # - output -> S
    pid_output = speed_pid.update(
        speed_error_kmh,
        dt,
    )

    command = make_longitudinal_command(
        pid_output
    )

    # --------------------------------------------------------
    # 방향이 크게 틀어진 상태에서 W 가속 제한
    # --------------------------------------------------------

    command = apply_alignment_speed_limit(
        command=command,
        heading_error_deg=steering_info[
            "heading_error_deg"
        ],
        current_speed_kmh=current_speed_kmh,
    )

    # --------------------------------------------------------
    # A/D 조향 결합
    # --------------------------------------------------------

    command["moveAD"] = (
        steering_command
    )

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    # print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    # print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    # if path_idx >= len(path):
    #     return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})

    # # print("🔁 Sent Combined Action:", command)
    # return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})
    return jsonify(command)
@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))

        # global dest
        # dest = (x, y, z)

        result = nav.apply_destination(x,y,z,render=True)

        # print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        # return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    try:
        result = (
            nav.update_dstar_obstacles_from_payload(
                payload=data,
                replan=True,
                render=True,
            )
        )
        return jsonify({
            "status": "success",
            "message": "Obstacle data received",
            "changed_cell_count": result[
                "changed_cell_count"
            ],
            "path_length": result[
                "path_length"
            ],
            "obstacle_count": result[
                "obstacle_count"
            ],
            "replanned": result[
                "replanned"
            ],
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400

    # update_dstar_obstacles_from_payload(data)

    # if path_flag:
    #     global path, path_idx
    #     path = [] # path 초기화
    #     path_idx = 0 # path idx 초기화
    #     path = path_planner.find_path((all_info['playerPos']['x'], all_info['playerPos']['z']), (dest[0], dest[2]))
    #     path_planner.plot(path, save_path='terrain_map')

    # # print("🪨 Obstacle Data:", data)
    # return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": False,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }
    # 기존 목적지를 보관한다.
    #
    # reset_navigation()은 dest를 None으로 만들기 때문에
    # 필요하면 초기화 후 다시 적용할 수 있도록 먼저 저장한다.
    previous_destination = (
        nav.get_destination()
    )

    # PID, dt, 도착 latch, 최종제동, 속도추정 상태 초기화.
    reset_all_controller_state()

    # Navigation 목적지/경로/현재위치 초기화.
    nav.reset_navigation(
        clear_position=True
    )

    # 시뮬레이터 Blue 시작 위치를 navigation 현재 위치로 등록.
    nav.set_current_position([
        float(config["blStartX"]),
        float(config["blStartZ"]),
    ])

    # /init 전에 목적지가 이미 설정되어 있었다면
    # 동일 목적지로 경로를 다시 만든다.
    if previous_destination is not None:
        try:
            nav.apply_destination(
                previous_destination[0],
                0.0,
                previous_destination[1],
                render=True,
            )

        except Exception as e:
            print(
                "초기 목적지 재설정 실패:",
                e,
            )
    # global path, path_idx, path_flag

    # path_planner.set_risk_layers()
    # path = path_planner.find_path((60, 27.23), (dest[0], dest[2]))
    # path_planner.plot(path, save_path='terrain_map')

    # path_idx = 0
    # path_flag = True
    
    # print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
