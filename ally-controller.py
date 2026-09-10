from flask import Flask, request, jsonify
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController
from fire.fire_module import FireModule
import detect.LibraryFile.TankSim as ts
import detect.LibraryFile.TankSim_kijun as tskijun
import detect.LibraryFile.TankSim_injee as tsinjee
import matplotlib
import requests
import threading
import datetime

# 서버 환경에서 matplotlib GUI 창을 열지 않고 map 이미지만 저장한다.
matplotlib.use("Agg")

# Flask 서버 객체.
app = Flask(__name__)

SEQ_FLAG = 'first'
ALLY_DEST_LIST = [(77.0, 281.20)]
ALLY_DEST_IDX = 0

# YOLO 객체 인식 모델.
model = YOLO(ts.MODEL_PATH)
print(model.names)

# 가장 최근 /info JSON snapshot.
# 원본 FireModule.get_turret_command()에 my_vel/body_rate를 넘길 때 사용한다.
all_info = None

# 원본 사격 모듈. fire_module.py 자체는 수정하지 않는다.
fm = FireModule()

# 위험 비용 확장 D* Lite planner.
# 서버가 planner 객체를 하나만 생성하고 PID controller에 주입한다.
path_planner = DStarLitePlanner()

# D* Lite 경로 추종 + 속도/조향 PID controller.
drive_controller = TankDriveController(path_planner)

# 적 전차 타격 횟수
enemy_hit_count = 0
pre_hit_time = None

def send_to_5100(target_data, name):
    try:
        requests.post("http://127.0.0.1:5100/"+name, json=target_data, timeout=1)
    except requests.exceptions.RequestException:
        pass

@app.route('/detect', methods=['POST'])
def detect():
    """인식팀 객체 탐지 모듈을 호출한다."""
    # tsinjee.detect()

    # 기존 통합 서버의 반환 형식을 그대로 유지한다.
    filtered_results = tsinjee.detect()
    return filtered_results


# tskijun.DETECTED_OBJECTS_INFO에 담기는 class_name 중, D* Lite 맵에
# '적 전차' 장애물로 패딩해야 하는 것들. 다른 클래스(Human/Rock/Tree 등)도
# 필요해지면 여기에 obj_type 매핑만 추가하면 된다.
#
# 각 클래스별 실제 바운딩 박스 크기(pad_object 폴백 시 쓸 half-extent)는
# pid_controller.py의 TankDriveController._OBJECT_HALF_EXTENTS_M에서
# 관리한다 (update_obstacles_type() 매칭 실패 시 그쪽에서 계산).
ENEMY_TANK_CLASSES = {"Tank1", "Tank2"}


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    """
    인식팀 stereo image 모듈을 호출하고, 새로 확보된 오브젝트 world 좌표를
    D* Lite planner에 반영한다.

    처리 방식: 먼저 update_obstacles_type()으로 Unity /update_obstacle가
    이미 등록해둔 고정 오브젝트(예: 고정 배치된 Tank1 모형)와 좌표가
    겹치는지 확인해서 타입만 재분류하고, 겹치는 게 없는 탐지(예: 실시간
    으로 움직이는 적 전차)만 pad_object() 기반으로 새 장애물을 등록한다.
    (drive_controller.handle_objects_detected() 안에서 이 두 단계를
    자동으로 처리한다.)
    """
    tskijun.stereo_image()

    # tskijun.stereo_image()가 채워둔 최신 탐지 결과.
    # [(x, y, z, class_name), ...] 형태.
    detections = [
        (x, y, z, class_name)
        for x, y, z, class_name in tskijun.DETECTED_OBJECTS_INFO
        if class_name in ENEMY_TANK_CLASSES
    ]

    if detections:
        try:
            drive_controller.handle_objects_detected(detections)
        except Exception as exc:
            # 탐지/회피 쪽 예외로 인식 파이프라인 응답 자체가 죽지 않게
            # 방어한다. 원인은 콘솔에 남긴다.
            print(f"[/stereo_image] handle_objects_detected 처리 실패: {exc}")

    return ts.jsonify({"result": "success"})


@app.route('/info', methods=['POST'])
def info():
    """
    동일한 /info JSON 한 개를 주행 모듈과 사격 모듈에 전달한다.

    request.get_json()을 여러 번 호출하지 않고 같은 snapshot을 공유한다.
    """
    global all_info

    # 이번 /info 요청의 단일 JSON snapshot.
    data = request.get_json(force=True)

    # /get_action 사격 운동 보정에 사용할 최신 telemetry.
    all_info = data

    # 주행 모듈의 위치/속도/yaw 상태 갱신.
    response, status = drive_controller.handle_info(data)

    # 원본 FireModule의 player/enemy/turret/target tracker 상태 갱신.
    fm.on_info(data)

    # 기존 인식팀 /info 처리.
    tskijun.info()

    global ALLY_DEST_IDX, ALLY_DEST_LIST, enemy_hit_count, SEQ_FLAG
    if enemy_hit_count < 2:
        if ALLY_DEST_IDX <= len(ALLY_DEST_LIST):
            if ALLY_DEST_IDX == 0:
                dest = {
                    "destination": f"{ALLY_DEST_LIST[ALLY_DEST_IDX][0]}, {data['playerPos']['y']}, {ALLY_DEST_LIST[ALLY_DEST_IDX][1]}"
                }
                ALLY_DEST_IDX += 1
                drive_controller.handle_set_destination(dest)
            elif ALLY_DEST_LIST[ALLY_DEST_IDX-1][0]-1 <= data['playerPos']['x'] <= ALLY_DEST_LIST[ALLY_DEST_IDX-1][0]+1 and\
                 ALLY_DEST_LIST[ALLY_DEST_IDX-1][1]-1 <= data['playerPos']['z'] <= ALLY_DEST_LIST[ALLY_DEST_IDX-1][1]+1:
                ALLY_DEST_LIST.append(path_planner.get_random_destination(data))
                dest = {
                    "destination": f"{ALLY_DEST_LIST[ALLY_DEST_IDX][0]}, {data['playerPos']['y']}, {ALLY_DEST_LIST[ALLY_DEST_IDX][1]}"
                }

                threading.Thread(target=send_to_5100, args=({'next_dest': ALLY_DEST_LIST[ALLY_DEST_IDX]}, 'get_next_dest'), daemon=True).start()

                ALLY_DEST_IDX += 1

                # 여기서 5100에 좌표를 넘겨줘야 함
                drive_controller.handle_set_destination(dest)
                if ALLY_DEST_IDX == 2:
                    SEQ_FLAG = 'second'
    elif enemy_hit_count == 2:
        ALLY_DEST_LIST.append((280.0, 170.0))
        dest = {
            "destination": "280.0, 0, 170.0"
        }
        drive_controller.handle_set_destination(dest)
        enemy_hit_count += 1

    threading.Thread(target=send_to_5100, args=(data, 'info'), daemon=True).start()

    return jsonify(response), status


@app.route('/get_action', methods=['POST'])
def get_action():
    """
    주행 명령과 사격 명령을 독립 계산한 뒤 하나로 합친다.

    TankDriveController 소유:
        moveWS, moveAD

    FireModule 소유:
        turretQE, turretRF, fire
    """
    # 이번 /get_action JSON snapshot.
    data = request.get_json(force=True)

    # D* Lite + PID 차체 이동/조향 명령.
    rst_cmd = drive_controller.get_action(data)

    global SEQ_FLAG
    if SEQ_FLAG == 'second':
        fire_inputs = (
            drive_controller.get_fire_control_inputs(
                rst_cmd
            )
        )
        turret_cmd = fm.get_turret_command(
            my_vel=fire_inputs["my_vel"],
            body_rate_dps=fire_inputs["body_rate_dps"],
            hull_settled=fire_inputs["hull_settled"],
        )

        rst_cmd["turretQE"] = turret_cmd["turretQE"]
        rst_cmd["turretRF"] = turret_cmd["turretRF"]
        rst_cmd["fire"] = turret_cmd["fire"]

    return jsonify(rst_cmd)


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """착탄 결과를 원본 FireModule의 보정/로그 모듈에 전달한다."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    # FireModule 내부 ShotLog/BiasEstimator에 착탄 결과를 전달한다.
    fm.on_impact(data)

    global enemy_hit_count, pre_hit_time, SEQ_FLAG
    if data.get('hit') == 'enemy':
        if pre_hit_time == None:
            enemy_hit_count += 1
        elif abs((datetime.datetime.now() - pre_hit_time).total_seconds()) > 1: # 여기서 두번 호출되는 거를 걸러준다.
            enemy_hit_count += 1

        pre_hit_time = datetime.datetime.now()

    # 여기서 enemy_hit_count가 2일 때 5100포트로 액션 넘겨줘야한다.
    if enemy_hit_count == 2:
        SEQ_FLAG = 'third'
        threading.Thread(target=send_to_5100, args=({}, 'go_third_step'), daemon=True).start()

    print(
        f"💥 Bullet Impact at X={data.get('x')}, "
        f"Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}"
    )
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route('/set_destination', methods=['POST'])
def set_destination():
    """목적지 설정을 주행 controller 모듈에 전달한다."""
    response, status = drive_controller.handle_set_destination(request.get_json())
    return jsonify(response), status


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """장애물 정보를 주행 controller -> RiskDStarPlanner에 전달한다."""
    response, status = drive_controller.handle_update_obstacles(request.get_json())

    threading.Thread(target=send_to_5100, args=(request.get_json(), 'update_obstacle'), daemon=True).start()

    return jsonify(response), status


@app.route('/collision', methods=['POST'])
def collision():
    """simulator collision 이벤트를 로그로 출력한다."""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    # 충돌 object 이름.
    object_name = data.get('objectName')

    # 충돌 위치 dictionary.
    position = data.get('position', {})

    # 충돌 위치 X/Y/Z 좌표.
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")
    return jsonify({'status': 'success', 'message': 'Collision data received'})


@app.route('/init', methods=['GET'])
def init():
    """episode 설정을 반환하고 TankDriveController 상태를 초기화한다."""
    # 기존 통합 서버의 simulator config를 유지한다.
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 111,
        "rdStartY": 15,
        "rdStartZ": 172,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True,
    }

    # D* Lite/PID 내부 상태의 시작 위치 [x, z]를 simulator와 일치시킨다.
    drive_controller.initialize(start_position=(60.0, 27.23))

    threading.Thread(target=send_to_5100, args=(config, 'init'), daemon=True).start()

    return jsonify(config)


@app.route('/start', methods=['GET'])
def start():
    return jsonify({"control": ""})


if __name__ == '__main__':
    # 기존 refactored 서버와 동일하게 병렬 Flask 요청을 허용한다.
    app.run(host='0.0.0.0', port=5000, threaded=True)