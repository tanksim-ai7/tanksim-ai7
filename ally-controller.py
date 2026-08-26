from flask import Flask, request, jsonify
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController
from fire.fire_module import FireModule
import detect.LibraryFile.TankSim as ts
import detect.LibraryFile.TankSim_kijun as tskijun
import detect.LibraryFile.TankSim_injee as tsinjee
import matplotlib

# 서버 환경에서 matplotlib GUI 창을 열지 않고 map 이미지만 저장한다.
matplotlib.use("Agg")

# Flask 서버 객체.
app = Flask(__name__)


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
ENEMY_TANK_CLASSES = {"Tank1"}

# 적 전차의 x/z 절반 크기[m]. compute_stereo_for_pair()가 world_pos를
# 점(point) 하나로만 계산해주기 때문에, pad_object()가 받는 bounding
# box(x_min/x_max/z_min/z_max)를 만들기 위해 이 정도 크기의 여유를 둔다.
#
# 실측 하단 몸체 크기(x, y, z) = (3.303, 1.131, 6.339) 기준.
# y는 높이(수직)축이라 2D 평면(x-z) 위에서 도는 D* Lite 충돌판정엔
# 안 쓴다 -- get_bb_corners(cx, cz, 3.303, 6.339, angle) 호출에서도
# 동일하게 x, z만 쓰는 것과 같은 이유.
# 포탑 크기(2.681, 3.094, 2.822)는 x/z 모두 몸체보다 작아 몸체 풋프린트
# 안에 들어오므로 몸체 크기만으로 충분하다.
#
# set_obstacles()가 obj_type='enemy_tank' 기준으로 이미 큰 안전 반경을
# 더 얹어주므로, 여기서는 실제 차체 크기만 잡으면 된다.
ENEMY_TANK_BODY_X_M = 3.303
ENEMY_TANK_BODY_Z_M = 6.339
ENEMY_TANK_HALF_WIDTH_M = ENEMY_TANK_BODY_X_M / 2   # 1.6515
ENEMY_TANK_HALF_LENGTH_M = ENEMY_TANK_BODY_Z_M / 2  # 3.1695


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    """
    인식팀 stereo image 모듈을 호출하고, 새로 확보된 적 전차 world 좌표를
    D* Lite planner의 회피/후퇴 로직(pad_object -> handle_object_detected)에
    반영한다.
    """
    tskijun.stereo_image()

    # tskijun.stereo_image()가 채워둔 최신 탐지 결과.
    # [(x, y, z, class_name), ...] 형태.
    for x, y, z, class_name in tskijun.DETECTED_OBJECTS_INFO:
        if class_name not in ENEMY_TANK_CLASSES:
            continue

        try:
            drive_controller.handle_object_detected(
                x - ENEMY_TANK_HALF_WIDTH_M,
                x + ENEMY_TANK_HALF_WIDTH_M,
                z - ENEMY_TANK_HALF_LENGTH_M,
                z + ENEMY_TANK_HALF_LENGTH_M,
                obj_type='enemy_tank',
            )
        except Exception as exc:
            # 탐지/회피 쪽 예외로 인식 파이프라인 응답 자체가 죽지 않게
            # 방어한다. 원인은 콘솔에 남긴다.
            print(f"[/stereo_image] handle_object_detected 처리 실패: {exc}")

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

    # PID Controller가 관리하는 실제 속도/yaw 상태와
    # /info의 실제 차체 yaw 변화량으로 FireModule 협업 입력을 만든다.
    #
    # 서버는 playerSpeed/playerBodyX/moveAD를 다시 계산하지 않고
    # 주행팀이 만든 값을 사격팀에 그대로 전달한다.
    fire_inputs = (
        drive_controller.get_fire_control_inputs(
            rst_cmd
        )
    )

    # 원본 FireModule API는 수정하지 않고 그대로 사용한다.
    #
    # my_vel:
    #     PID Controller가 /info 실제 속도/yaw로 만든 속도 벡터 [m/s].
    #
    # body_rate_dps:
    #     연속된 /info의 playerBodyX 변화량으로 계산한 실제 차체 각속도 [deg/s].
    #
    # hull_settled:
    #     PID Controller가 실제 속도와 실제 차체 회전으로 판단한 정지 여부.
    turret_cmd = fm.get_turret_command(
        my_vel=fire_inputs["my_vel"],
        body_rate_dps=fire_inputs["body_rate_dps"],
        hull_settled=fire_inputs["hull_settled"],
    )

    # 이동/조향 명령은 PID controller 결과를 유지하고,
    # 포탑/사격 key만 FireModule 결과로 병합한다.
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
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,
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

    return jsonify(config)


@app.route('/start', methods=['GET'])
def start():
    return jsonify({"control": ""})


if __name__ == '__main__':
    # 기존 refactored 서버와 동일하게 병렬 Flask 요청을 허용한다.
    app.run(host='0.0.0.0', port=5000, threaded=True)