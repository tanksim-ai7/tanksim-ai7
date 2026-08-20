from flask import Flask, request, jsonify
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController
import matplotlib
from fire_module import FireModule
import detect.LibraryFile.TankSim as ts
import detect.LibraryFile.TankSim_kijun as tskijun
import detect.LibraryFile.TankSim_injee as tsinjee
import math

matplotlib.use("Agg")

app = Flask(__name__)
model = YOLO('yolov8n.pt')
print(model.names)

all_info = None
fm = FireModule()
path_planner = DStarLitePlanner()
drive_controller = TankDriveController(path_planner)

@app.route('/detect', methods=['POST'])
def detect():
    tsinjee.detect()
    
    filtered_results = []
    return ts.jsonify(filtered_results)

@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    tskijun.stereo_image()
    
    return ts.jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    global all_info
    all_info = request.get_json(force=True)
    response, status = drive_controller.handle_info(request.get_json(force=True))
    fm.on_info(request.get_json(force=True))
    tskijun.info()
    return jsonify(response), status

@app.route('/get_action', methods=['POST'])
def get_action():
    rst_cmd = drive_controller.get_action(request.get_json(force=True))

    # TODO 
    body_rate = abs(rst_cmd["moveAD"]["weight"]*all_info.get("playerBodyX", 0))
    my_speed = all_info.get("playerSpeed", 0)

    angle_deg = all_info.get("playerBodyX", 0)
    angle_rad = math.radians(angle_deg)

    vx = my_speed * math.sin(angle_rad)
    vz = my_speed * math.cos(angle_rad)

    my_vel = (vx, 0.0, vz)

    """
        my_vel        경로팀 이동에 따른 자기 속도 벡터 [m/s]
        body_rate_dps 경로팀 선회에 따른 차체 각속도 [deg/s]
        hull_settled  차체 정지 여부. None 이면 속도로 자동 판정
    """
    turret_cmd = fm.get_turret_command(
        my_vel=my_vel,        # 경로팀이 내는 이동에 따른 속도
        body_rate_dps=body_rate,     # 경로팀이 내는 차체 선회 각속도
        hull_settled=(my_speed < 0.3 and body_rate < 1e-6),
    )
    rst_cmd["turretQE"] = turret_cmd["turretQE"]
    rst_cmd["turretRF"] = turret_cmd["turretRF"]
    rst_cmd["fire"] = turret_cmd["fire"]

    return jsonify(rst_cmd)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    fm.on_impact(data)
    
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    response, status = drive_controller.handle_set_destination(request.get_json())
    return jsonify(response), status

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    response, status = drive_controller.handle_update_obstacles(request.get_json())
    return jsonify(response), status

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
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }
    drive_controller.initialize(start_position=(60, 27.23))

    # print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    # threaded=True: 병렬 처리 - 여러 요청이 동시에 들어와도 하나가 끝날 때까지 기다리지 않고 동시에 처리