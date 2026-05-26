# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import time
import os

from lingzu_motor import LingZuMotorController, SPITransport

app = FastAPI(title="RS04 Control API")

# ---- static dir (use Path consistently) ----
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 共享 SPI 传输层，多电机按 motor_id 区分
_shared_transport = SPITransport(targets=((0, 0), (1, 0)), max_speed_hz=1000000)

MOTOR_IDS = [
    0x11, 0x12, 0x13,
    0x21, 0x22, 0x23,
    0x31, 0x32, 0x33,
    0x41, 0x42, 0x43,
]

motors: dict[int, LingZuMotorController] = {
    mid: LingZuMotorController(motor_id=mid, transport=_shared_transport)
    for mid in MOTOR_IDS
}


def _refresh_all_states():
    # A板代表：0x11；B板代表：0x31
    motors[0x11].refresh_board_snapshot()
    motors[0x31].refresh_board_snapshot()


def _get_motor(motor_id: int):
    if motor_id not in motors:
        raise HTTPException(status_code=400, detail=f"motor_id {motor_id} not in {list(motors.keys())}")
    return motors[motor_id]


# --------------------------
# Basic APIs（均支持 motor_id / motor_ids）
# --------------------------
@app.get("/api")
def api_root():
    return {"ok": True, "msg": "RS04 API running", "motor_ids": MOTOR_IDS}


class MotorIdsBody(BaseModel):
    motor_id: int | None = None
    motor_ids: list[int] | None = None

class ZeroCmd(BaseModel):
    motor_id: int | None = None
    motor_ids: list[int] | None = None
    stop_first: bool = True
    set_motion_mode: bool = True

@app.post("/api/enable")
def api_enable(body: MotorIdsBody | None = None):
    ids = _resolve_motor_ids(body)
    for mid in ids:
        _get_motor(mid).enable()
    return {"ok": True, "motor_ids": ids}


@app.post("/api/stop")
def api_stop(clear_error: bool = False, body: MotorIdsBody | None = None):
    ids = _resolve_motor_ids(body)
    for mid in ids:
        _get_motor(mid).stop(clear_error=clear_error)
    return {"ok": True, "motor_ids": ids}


def _resolve_motor_ids(body: MotorIdsBody | None) -> list[int]:
    if body is None:
        return list(motors.keys())
    if body.motor_ids is not None and len(body.motor_ids) > 0:
        return body.motor_ids
    if body.motor_id is not None:
        return [body.motor_id]
    return list(motors.keys())



@app.post("/api/zero")
def api_zero(cmd: ZeroCmd | None = None):
    ids = _resolve_motor_ids(cmd)

    stop_first = True if cmd is None else cmd.stop_first
    set_motion_mode = True if cmd is None else cmd.set_motion_mode

    # 为了避免 PP 模式下标零被屏蔽，先停，再切回运控模式(0)
    if stop_first:
        for mid in ids:
            _get_motor(mid).stop(clear_error=False)
        time.sleep(0.05)

    if set_motion_mode:
        for mid in ids:
            _get_motor(mid).set_mode(0)   # 运控模式
        time.sleep(0.05)

    for mid in ids:
        _get_motor(mid).set_zero()
        time.sleep(0.02)

    _refresh_all_states()

    return {
        "ok": True,
        "action": "set_zero",
        "motor_ids": ids,
        "count": len(ids),
    }


@app.get("/api/state")
def api_state(motor_id: int | None = None):
    _refresh_all_states()
    if motor_id is not None:
        return _get_motor(motor_id).get_state_dict()
    return {hex(mid): _get_motor(mid).get_state_dict() for mid in motors}

class SpeedExactCmd(BaseModel):
    motor_id: int = 0x11
    speed: float = 3.0
    accel: float = 2.0
    current_limit: float = 2.0


class MotionCmd(BaseModel):
    motor_id: int = 0x11
    position: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    kp: float = 4.0
    kd: float = 4.0
    enable_first: bool = True
    stop_first: bool = False

@app.post("/api/rs04/speed_mode_run_exact")
def rs04_speed_mode_run_exact(cmd: SpeedExactCmd):
    m = _get_motor(cmd.motor_id)
    m.set_mode(2)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7018, cmd.current_limit)
    time.sleep(0.05)
    m.set_param_f32(0x7022, cmd.accel)
    time.sleep(0.05)
    m.set_param_f32(0x700A, cmd.speed)
    time.sleep(0.05)
    return {
        "ok": True,
        "mode": "speed_exact",
        "motor_id": cmd.motor_id,
        "speed": cmd.speed,
        "accel": cmd.accel,
        "current_limit": cmd.current_limit,
    }


@app.post("/api/rs04/motion_mode_run")
def rs04_motion_mode_run(cmd: MotionCmd):
    m = _get_motor(cmd.motor_id)

    if cmd.stop_first:
        m.stop(clear_error=False)
        time.sleep(0.05)

    # 运控模式 = 0
    m.set_mode(0)
    time.sleep(0.03)

    if cmd.enable_first:
        m.enable()
        time.sleep(0.03)

    m.motion_control(
        torque=cmd.torque,
        position=cmd.position,
        speed=cmd.speed,
        kp=cmd.kp,
        kd=cmd.kd,
    )

    return {
        "ok": True,
        "mode": "motion",
        "motor_id": cmd.motor_id,
        "position": cmd.position,
        "speed": cmd.speed,
        "torque": cmd.torque,
        "kp": cmd.kp,
        "kd": cmd.kd,
    }




class MotionBatchItem(BaseModel):
    motor_id: int
    position: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    kp: float = 4.0
    kd: float = 4.0

class MotionBatchCmd(BaseModel):
    items: list[MotionBatchItem]
    enable_first: bool = True
    stop_first: bool = False

@app.post("/api/rs04/motion_mode_run_batch")
def rs04_motion_mode_run_batch(cmd: MotionBatchCmd):
    if cmd.stop_first:
        for item in cmd.items:
            _get_motor(item.motor_id).stop(clear_error=False)
        time.sleep(0.05)

    for item in cmd.items:
        _get_motor(item.motor_id).set_mode(0)
    time.sleep(0.03)

    if cmd.enable_first:
        for item in cmd.items:
            _get_motor(item.motor_id).enable()
        time.sleep(0.03)

    for item in cmd.items:
        _get_motor(item.motor_id).motion_control(
            torque=item.torque,
            position=item.position,
            speed=item.speed,
            kp=item.kp,
            kd=item.kd,
        )

    return {"ok": True, "mode": "motion_batch", "count": len(cmd.items)}

@app.post("/api/rs04/motion_batch_fast")
def rs04_motion_batch_fast(cmd: MotionBatchCmd):
    """
    高频策略控制专用接口。

    这个接口只发送 motion_control，不重复 set_mode，不重复 enable，不 sleep。

    使用前提：
    1. 电机已经进入运控模式；
    2. 电机已经 enable；
    3. 已经完成默认站立；
    4. ROS2 策略循环只负责连续下发目标角。
    """
    for item in cmd.items:
        _get_motor(item.motor_id).motion_control(
            torque=item.torque,
            position=item.position,
            speed=item.speed,
            kp=item.kp,
            kd=item.kd,
        )

    return {
        "ok": True,
        "mode": "motion_batch_fast",
        "count": len(cmd.items),
    }
# --------------------------
# Raw Hex (12 bytes) - debug helper
# --------------------------
class RawHexCmd(BaseModel):
    hex12: str

@app.post("/api/raw_hex")
def raw_hex(cmd: RawHexCmd):
    parts = cmd.hex12.strip().split()
    if len(parts) != 12:
        return {"ok": False, "error": "Need exactly 12 bytes"}
    vals = bytes(int(x, 16) for x in parts)
    ext_id = int.from_bytes(vals[:4], "big")
    data = vals[4:12]
    target_id = ext_id & 0xFF
    m = motors.get(target_id)
    if m is None:
        m = next(iter(motors.values()))
    m._send(ext_id, data)
    return {"ok": True}


# --------------------------
# RS04 mode-based APIs（body 中带 motor_id）
# --------------------------
class SpeedCmd(BaseModel):
    motor_id: int = 0x11
    speed: float = 2.0
    accel: float = 2.0


class PosCmd(BaseModel):
    motor_id: int = 0x11
    position: float = 0.0
    speed: float = 2.0
    accel: float = 1.0


class CurrentCmd(BaseModel):
    motor_id: int = 0x11
    iq: float = 0.0


@app.post("/api/rs04/speed_mode_run")
def rs04_speed_mode_run(cmd: SpeedCmd):
    m = _get_motor(cmd.motor_id)
    m.set_param_u8(0x7005, 2)
    time.sleep(0.08)
    m.enable()
    time.sleep(0.08)
    m.set_param_f32(0x7018, 2.0)
    time.sleep(0.08)
    m.set_param_f32(0x7022, cmd.accel)
    time.sleep(0.08)
    m.set_param_f32(0x700A, cmd.speed)
    time.sleep(0.08)
    return {
        "ok": True,
        "mode": "speed",
        "motor_id": cmd.motor_id,
        "speed": cmd.speed,
        "accel": cmd.accel,
        "current_limit": 2.0,
    }


@app.post("/api/rs04/pp_mode_run")
def rs04_pp_mode_run(cmd: PosCmd):
    m = _get_motor(cmd.motor_id)
    m.set_mode(1)
    time.sleep(0.05)
    m.set_param_f32(0x7024, cmd.speed)
    time.sleep(0.05)
    m.set_param_f32(0x7025, cmd.accel)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7016, cmd.position)
    return {
        "ok": True,
        "mode": "pp",
        "motor_id": cmd.motor_id,
        "position": cmd.position,
        "speed": cmd.speed,
        "accel": cmd.accel,
    }


@app.post("/api/rs04/current_mode_run")
def rs04_current_mode_run(cmd: CurrentCmd):
    m = _get_motor(cmd.motor_id)
    m.set_mode(3)
    time.sleep(0.05)
    m.enable()
    time.sleep(0.05)
    m.set_param_f32(0x7006, cmd.iq)
    return {"ok": True, "mode": "current", "motor_id": cmd.motor_id, "iq": cmd.iq}
    
    
    
    
    
