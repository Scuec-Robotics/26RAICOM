#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 分段巡线 - 增强版 + 红点触发 + ORB图案识别 + 优化直角检测 + 远层保线 + 爬楼梯模式 + 蓝色启停区 + 窄道检测 + 跳跃 + 中转平台

功能：
  - 当第一层（最远层）无法识别黑线时自动减速
  - 尽量保持第一层能识别到黑线，提前感知弯道
  - 优化90°直角检测灵敏度（多条件综合判断 + 近距离障碍物黑线过滤）
  - 第一次丢线触发窄道模式，后续丢线正常摆头搜索+记忆转弯
  - 红点检测 → 循迹补偿 → 左转 → 后退 → ORB识别动作 → 右转 → 检测黑线 → 恢复循迹
  - 爬楼梯模式：窄道完成后才允许触发（楼梯ROI高阈值宽度突变 → 盲目前进 → 立即边转弯边直行 → 直接衔接循迹）
  - 蓝色启停区检测 → 停止循迹 → 直走 → 左转 → 前进10cm → 坐下(红点处理完成后才启用)
  - 窄道检测 → 自动执行写死路径 → 转弯87%时检测黑线(仅一次) → 检测到黑线后补偿转弯
  - 跳跃功能：窄道前跳一次，红点后跳一次（红点处理后延迟2.5秒开启检测）
  - 中转平台检测：楼梯完成后启用，第1个平台转弯，第2个平台停止(加强深度+色块双重检测)
  - 相机预热：启动时趴着预热40帧，再站立
  - 【集成】楼梯后识别grasp → 左转100° → 3D视觉抓取第1个黄色物块(grasp_3d管线)
    → 循迹 → 第1个直角弯后2.5s减速停止 → 中转: 放下物块A + 抓第2个黄色物块B
"""

import sys
import time
import argparse
import os
import random
import glob
import cv2
import numpy as np
from enum import Enum
from collections import deque
import pyrealsense2 as rs
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.vui.vui_client import VuiClient

# ==================== 【集成块·A】3D视觉抓取管线 (移植自 d1_arm/arm_control/grasp_3d.py) ====================
# 完整复制 grasp_3d.py 的抓取逻辑 (检测/手眼标定/IK/中转), 无缝嵌入本文件。
# 抓取用夹爪相机(锁定序列号335222075495), 与巡线深度相机互不冲突。
# 与原文件的差异仅为以下【集成】适配:
#   1) __init__ 新增 sport_client/phase/enable_mjpeg 参数 (复用巡线运动客户端/分阶段/关MJPEG)
#   2) _init_stand 支持注入运动客户端, 不重建DDS通道
#   3) phase='first' 抓完第1个物块确认容量(1/1)后提前返回; phase='second' 起始夹持直接中转
#   4) 第二阶段允许持物前移搜索 (原逻辑抓后一律不前移)
#   5) 集成模式不启动MJPEG流; 第二阶段起始不张开夹爪
import threading
import traceback


def now_s():
    """Monotonic seconds for motion timing; robot wall clocks jump after boot."""
    return time.monotonic()


sys.path.insert(0, os.path.expanduser('~/go2_zong_project/d1_arm/arm_control'))
from d1_udp_client import D1UDPClient
from d1_ik import D1Kinematics


# ===================================================================
# 配置常量
# ===================================================================
CALIB_DIR = os.path.expanduser('~/go2_zong_project/d1_arm/place_zone')
IW, IH = 640, 480

# ---- 检测 ----
Y_LOW  = np.array([18, 80, 80])
Y_HIGH = np.array([35, 255, 255])

# ---- 抓取参数 ----
APPROACH_Z_OFFSET = 0.05   # 预接近高度5cm, 避免撞物体
GRASP_DOWN_EXTRA  = 0.0775  # 2026-08-10: 0.0715→0.0775 (用户: 目标抓取点再向下0.6cm)
GRASP_FWD_EXTRA    = -0.009  # 2026-08-09: 原基础(0位)往后0.9cm (用户: 0.9)
GRASP_B_DOWN_EXTRA = 0.015  # 2026-08-07: 抓B额外下探1.5cm (用户, 只影响phase=second)
MIN_GRASP_DEPTH   = 0.05   # 最小抓取距离(m)
# v18: TCP_OFFSET/DESKTOP_Z_COMP 已被新标定替代, 保留Z_ABS_MIN安全限位
# TCP_OFFSET        = 0.08
# DESKTOP_Z_COMP    = 0.08
Z_ABS_MIN         = -0.20  # Z轴绝对硬限位-2cm, 避免撞桌面

# ---- D1官方硬限位 ----
JOINT_HARD_LIMITS = [
    (-60,  60),   # J0/angle0 底座旋转 (v19: 全关节 IK, 放开锁定)
    (-90,  90),   # J1/angle1 大臂俯仰
    (-90,  90),   # J2/angle2 肘部俯仰
    (-135, 135),  # J3/angle3 腕部俯仰
    (-90,  90),   # J4/angle4 腕部旋转（官方±90°）
    (-135, 135),  # J5/angle5 末端法兰旋转
]

# ---- 运动参数 ----
SPEED_FACTOR      = 0.30   # v16: 调试阶段30%低速, 方向对了再提速
HEARTBEAT_INTERVAL = 0.10  # 心跳间隔(s)

# ---- 伺服参数 (v16: 解耦式, 先底座再大臂小臂) ----
SERVO_GAIN_X      = 0.018  # 底座对准增益，稍微减小防超调
ALIGN_CX_TARGET   = 320    # 底座对准目标: 画面中心 X (v20)
ALIGN_PX_TOL      = 40     # 对准容差(px)，太严容易振荡
SERVO_GAIN_Y      = 0.015  # 上下方向增益
SERVO_GAIN_D      = 25.0   # 深度方向增益
SERVO_MAX_DELTA   = 3.0    # 单步最多转3度
ALIGN_TOLERANCE   = 30     # X方向对准容差(像素)
ALIGN_STABLE_REQ    = 15     # 原来的2改成15
STABLE_PX_TOLERANCE = 150    # 稳定帧内像素跳变容差(±px), 底座转相机也转
# v18: angle4=夹爪, 由open_gripper/close_gripper独立控制
# WRIST_ANGLE         = -15

# ===================================================================
# 权威标定 (2026-08-01, 用户实测) — 以这套为准!!
#   handeye: d1_handeye_session_final_18 (HORAUD, RMS 11.10mm/0.672°)
#   tcp:     d1_tcp_pivot_session (RMS 4.64mm)
#   flange  = Empty_Link6 (URDF: d1_550_description.urdf)
# ===================================================================
# 相机序列号 (标定相机, 多相机时必须锁定)
CAM_SERIAL = "335222075495"  # 新夹爪相机 D435I (旧 334622072209 已换下)
# 相机内参 (新相机 335222075495 出厂值; 旧标定 605.0/328.8 已不适用, 未重标定)
CAM_FX, CAM_FY = 606.5, 606.764
CAM_CX, CAM_CY = 313.235, 251.302

# 手眼: p_flange = T_FLANGE_CAMERA @ p_cam
T_FLANGE_CAMERA = np.array([
    [ 0.01773454528500995, -0.5164469373449353,  0.8561355306320194,  0.025971185967540923],
    [-0.9995297333366298,   0.012266262655669182, 0.028104287510411947, 0.038190425906970644],
    [-0.02501595649859686, -0.8562313353921931,  -0.5159865329763578,  0.06538213746806978 ],
    [ 0.0,                  0.0,                  0.0,                 1.0                 ],
])

# 关节反馈修正: q_real = JOINT_SIGNS * angle_fb + JOINT_OFFSETS
JOINT_SIGNS   = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
JOINT_OFFSETS = [0.0, 4.8652025166, -2.0432974036, 0.4414531022, 3.5961901477, 0.0]

# 抓取 TCP (effective_grasp_tcp, 相对 flange)
TCP_OFFSET_FLANGE = np.array([0.100701, -0.003506, 0.013825])  # 2026-08-11: TCP 往右移 0.17cm (相机右=flange -Y, 方向反了改符号)
# ---- 中转功能 (v39): 夹着物块A看到远处物块B -> 把A放到B右下30°、7cm处 ----
TRANSFER_DIST      = 0.051  # 2026-08-08: 0.09→0.051 (用户: 放B右边5.1cm)
TRANSFER_ANGLE_DEG = 30     # 放置方向: 左边30° (2026-08-04: 右下→左边)
TRANSFER_LIFT_Z    = 0.05   # 中转: 放置点上方高度
PLACE_SIDE_Y       = 0.22   # 红点后放置点: 狗侧向距离(m) (2026-08-02 用户: 左右22cm)
PLACE_HEIGHT_Z     = 0.05   # 红点后放置点: 高度(m) (用户: 高度5cm)
PLACE_LIFT_Z       = 0.05   # 红点后放置: 下探前抬升(m)
CAP_Z_MAX          = 0.15   # v39: 容量判据 — 近处(~0.1m)黄色 = 爪上物块 (1/1)
CAP_AREA_MIN       = 10000  # v39: 容量/夹持判据 — 面积超大(近距) = 爪上物块
CAP_HELD_Z_MAX     = 0.25   # 2026-08-11: 前方检查的"爪上物块"深度阈值 (比容量判据0.15宽松,
                            # 覆盖近距漏判区 0.15~0.25m; 此距离物块不可抓, 排除安全)
CAP_ANGLE6_THRESH  = 0.0    # 2026-08-09晚: 夹爪角度判据已废弃 (用户: 不要) — 参数保留无引用
EXCLUDE_PLACED_DIST = 0.03  # 2026-08-11: 7cm→3cm (用户: 最大干扰半径22.5mm; 圆心已锚定实测A落点, 3cm=干扰+余量, B距A~10cm更安全)
                            # (2026-08-07: 用户: 抓不到重新识别时去掉A放置坐标点, 防再抓回A)
WS_WALK_STEP_M      = 0.08   # 2026-08-10: 0.12→0.08 每次搜索8cm (用户); 搜索前移每步距离
WS_WALK_STEP_A     = 0.10   # 2026-08-11: 0.12→0.10 (用户: 抓A每步搜索10cm); 中转抓B仍用上面8cm
WS_WALK_SPEED_M_S   = 0.25   # 挪动速度 (m/s, 2026-08-04: 0.30→0.25) — 中转前移
GRASP_SEARCH_SPEED_M_S = 0.50  # 2026-08-11: 0.30→0.50 (用户: 每步往前搜索速度 0.5); 10cm/步 ≈0.2s/步
WS_WALK_MAX_STEPS   = 10     # 2026-08-11: 6→10 (用户: 最多走10步, 安全范围内停下); 安全门<24cm停
GRASP_FWD_SAFE_DIST  = 0.24   # 2026-08-11: 抓取前移安全距离(m) (用户) — 循迹相机前方深度<此值 → 原地停+angle0搜索
SAFE_SWEEP_RANGE     = 40     # 2026-08-11: 安全扫描 angle0 左右最大搜索度数 (5°/步, 每步停3s, 用户)
SAFE_SWEEP_FRAMES    = 3      # 2026-08-11: 安全扫描连续看到黄色物块帧数 → 停扫抓取 (用户)
SAFE_SWEEP_COOLDOWN  = 10.0   # 2026-08-11: 扫描失败冷却(s), 避免连续重扫
SAFE_SWEEP_MAX_ATTEMPTS = 2   # 2026-08-11: 扫描失败放弃次数 (防撞, 不前移)
# v39.13: 搜索时不再前移 — 只有锁定目标且超出工作空间才向前探索 (WS_WALK_STEP_M)

# 抓取规划偏移 (基座系, 正=前/左/上)
GRASP_PLAN_OFFSET_X = 0.024   # 往前补 2.4cm (用户实测再前0.6cm, 原1.8cm)
GRASP_PLAN_OFFSET_Y = -0.0371  # 2026-08-15: -0.0291→-0.0371 (用户: 再向右 0.8cm, +Y=左 -Y=右)
GRASP_PLAN_OFFSET_Z = 0.070   # 往上补 7.0cm (2026-08-06: 8.5→7.0, 下1.5cm 用户)
# 夹爪
GRIPPER_OPEN  = 60.0   # 张开再加大 (用户要求)
GRIPPER_CLOSE = -10.0

def fix_feedback_angles(angles_fb):
    """反馈(舵机)角度 → 真实关节角 (sign + offset)"""
    if isinstance(angles_fb, dict):
        return [JOINT_SIGNS[i] * angles_fb.get(f'angle{i}', 0) + JOINT_OFFSETS[i]
                for i in range(6)]
    return [JOINT_SIGNS[i] * float(angles_fb[i]) + JOINT_OFFSETS[i]
            for i in range(6)]

def inv_fix_angles(q_real):
    """真实关节角 → 命令(舵机)角度: fb = (q_real - offset) / sign"""
    return {f'angle{i}': (q_real[i] - JOINT_OFFSETS[i]) / JOINT_SIGNS[i]
            for i in range(6)}


# ===================================================================
# 物体检测器 (保持不变)
# ===================================================================
class ObjectDetector:
    """物体检测: 支持 HSV 颜色阈值 或 YOLOv8 模型"""

    def __init__(self, mode='hsv', model_path=None, conf=0.4):
        self.mode = mode
        self.model = None
        if mode == 'yolo' and model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf = conf
            print(f'[Detector] YOLO loaded: {model_path}')
        else:
            print('[Detector] HSV mode (yellow objects)')

    def detect(self, color_img):
        if self.mode == 'yolo' and self.model:
            return self._detect_yolo(color_img)
        return self._detect_hsv(color_img)

    def _detect_hsv(self, color):
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, Y_LOW, Y_HIGH)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(c)
            detections.append({
                'cx': cx, 'cy': cy,
                'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                'conf': 1.0, 'name': 'yellow_obj', 'area': area,
            })
        detections.sort(key=lambda d: d['area'], reverse=True)
        return detections

    def _detect_yolo(self, color):
        results = self.model(color, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(boxes.cls[i])
                detections.append({
                    'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': float(boxes.conf[i]),
                    'name': str(r.names.get(cls_id, f'cls_{cls_id}')),
                })
        return detections


# ===================================================================
# 3D 抓取管线 v15 — 底座对准 + 纯IK盲抓
# ===================================================================
class GraspPipeline3D:
    """完整的 3D 视觉抓取管线"""

    def __init__(self, mode='hsv', model_path=None, dry_run=False,
                 sport_client=None, phase='first', enable_mjpeg=True):
        """【集成】sport_client: 复用巡线运动客户端; phase: 'first'只抓第1个物块,
        'second'从夹持状态直接中转抓第2个; enable_mjpeg: 集成时关闭HTTP流"""
        self.mode = mode
        self.dry_run = dry_run
        self._sport_client = sport_client   # 【集成】复用巡线运动客户端 (不重建DDS)
        self.phase = phase                  # 【集成】'first'/'second'
        self._enable_mjpeg = enable_mjpeg   # 【集成】关闭MJPEG流
        self._phase2_search = True   # 2026-08-04: 中转阶段用 angle0 扫描搜索找B (原禁止前移搜索)
        self._angle0_sweep_done = False  # 2026-08-04: angle0 扫描只执行一次

        # ---- 手眼标定 (权威: HORAUD, flange=Empty_Link6) ----
        self.T_cam2gripper = T_FLANGE_CAMERA
        t = self.T_cam2gripper[:3, 3]
        print(f'[Calib] 权威标定: cam->flange offset '
              f'[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m (HORAUD RMS 11.1mm)')

        # ---- 相机内参 (权威标定值) ----
        self.mtx = np.array(
            [[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]],
            dtype=np.float32)
        self.dist = np.zeros(5, dtype=np.float32)
        print(f'[Calib] 相机内参: fx={CAM_FX:.1f} fy={CAM_FY:.1f} '
              f'cx={CAM_CX:.1f} cy={CAM_CY:.1f} (标定值)')

        # ---- 组件 ----
        self.detector = ObjectDetector(mode=mode, model_path=model_path)
        self.holding_block = False  # 中转功能: 爪上是否抓着物块
        if phase == 'second':
            self.holding_block = True  # 【集成】第二阶段: 起始即夹持物块A, 直接中转
        self.transfer_done = False  # 中转功能: 是否已完成一次中转 (中转后抓取即完成)
        self.placed_pos = None      # v39: 中转放下的物块位置 (基座系, 抓取时屏蔽)
        self._b_mem_angle0 = None   # 2026-08-11: 记忆第一次锁定B的 angle0 方向 (放A后对准)
        self._b_mem_base = None     # 2026-08-11: 记忆B的基座位置 (重试时重算A屏蔽)
        self._ws_walk_steps = 0     # v39.5: 超工作空间已向前挪动次数
        self._at_ready = False      # v39.12: 臂是否在ready姿态 (只在ready才搜索)
        self.arm = None
        self.pipe = None
        self.align = None
        self.depth_scale = None
        self.factory_intrinsics = None

        # 心跳计时
        self._last_cmd_time = 0.0
        self._heartbeat_seq = 0

        # MJPEG流
        self.stream_frame = None
        self.stream_lock = threading.Lock()
        self.has_display = bool(os.environ.get('DISPLAY', ''))

    # ==================================================================
    # 初始化
    # ==================================================================

    def init_arm(self):
        """连接机械臂: 桥健康检查→enable→等1.5s→home"""
        if self.dry_run:
            print('[Arm] DRY-RUN: skipping hardware init')
            return
        self.arm = D1UDPClient('192.168.123.100')
        # 2026-08-03: 桥健康检查 — 桥僵死/未起时明确报错, 不再闷头继续 (指令是发后即忘, 桥死了程序无感知)
        bridge_ok = False
        for attempt in range(3):
            try:
                resp = self.arm._send_recv(9)
                if resp is not None:
                    bridge_ok = True
                    ang = {k: v for k, v in resp.get("data", {}).items() if k.startswith("angle")}
                    print(f"[Arm] ✓ 桥健康检查通过 (角度: {ang})")
                    break
            except Exception as e:
                print(f"[Arm] ⚠️ 桥查询异常: {e}")
            print(f"[Arm] ⚠️ 桥无回应 ({attempt+1}/3) — 检查: ssh ubuntu@192.168.123.100 && sudo systemctl restart arm-udp-bridge")
            time.sleep(2)
        if not bridge_ok:
            print("[Arm] ❌❌ 桥健康检查失败: 9999 无回应 — 机械臂不可用, 终止程序")
            sys.exit(1)
        print('[Arm] Enabling motors...')
        self.arm.enable()
        time.sleep(1.5)
        print('[Arm] Homing...')
        self.arm.home()
        time.sleep(3.0)
        self._last_cmd_time = now_s()
        print('[Arm] Ready')

    def init_camera(self):
        """初始化 D435i"""
        if self.dry_run:
            print('[Camera] DRY-RUN: skipping camera init')
            return
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(CAM_SERIAL)  # 锁定标定相机 (多相机时防连错)
        # 2026-08-03: 双相机带宽不足(夹爪30fps+巡线30fps时夹爪无帧), 降到15fps实测正常
        cfg.enable_stream(rs.stream.color, IW, IH, rs.format.bgr8, 15)
        cfg.enable_stream(rs.stream.depth, IW, IH, rs.format.z16, 15)
        self.align = rs.align(rs.stream.color)
        # 2026-08-10 (评审点3): start 加重试 — 反复 start/stop 偶发 device busy,
        # 重试间先 USB 复位 (夹爪相机)
        prof = None
        for attempt in range(3):
            try:
                prof = self.pipe.start(cfg)
                break
            except RuntimeError as e:
                print(f'[Camera] ⚠️ start 失败 (a{attempt+1}/3): {e}')
                if attempt < 2:
                    usb_authorized_reset(CAM_SERIAL, '夹爪相机')
        if prof is None:
            raise RuntimeError('夹爪相机 start 3 次失败')

        cp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = cp.get_intrinsics()
        # 相机内参 (权威标定值)
        self.factory_intrinsics = np.array([
            [CAM_FX, 0, CAM_CX],
            [0, CAM_FY, CAM_CY],
            [0, 0, 1],
        ])
        self.depth_scale = \
            prof.get_device().first_depth_sensor().get_depth_scale()

        print('[Camera] Warming up...')
        for _ in range(30):
            try:
                if not self.pipe.wait_for_frames(timeout_ms=1000):
                    time.sleep(0.1)
                    continue
            except RuntimeError:
                time.sleep(0.1)
                break  # 2026-08-10 (评审点3): 预热尽力而为, 掉线交给 _get_frames 自恢复
        print(f'[Camera] Ready (fx={intr.fx:.1f} fy={intr.fy:.1f})')

    def start_mjpeg(self, port=8080):
        """MJPEG 流服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        sf = self.stream_frame
        sl = self.stream_lock

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/stream':
                    self.send_response(200)
                    self.send_header(
                        'Content-Type',
                        'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    while True:
                        with sl:
                            f = sf.copy() if sf is not None else None
                        if f is None:
                            time.sleep(0.1)
                            continue
                        _, jpg = cv2.imencode(
                            '.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                            + jpg.tobytes() + b'\r\n')
                        time.sleep(0.05)

        server = HTTPServer(('0.0.0.0', port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

    # ==================================================================
    # 核心坐标变换
    # ==================================================================

    def pixel_to_base(self, px, py, depth_img, arm_angles,
                      debug_log=False):
        """
        像素 → 基坐标系 (v19, 权威标定)

        链路: 像素→相机系(标定内参)→T_FLANGE_CAMERA→flange系→FK(flange, 修正关节角)→基座
        抓取目标: 物块基座系 - R_flange_base @ TCP_OFFSET_FLANGE + Y规划偏移
        """
        # [Step 1] 像素 → 相机系
        z_m = self._median_depth(px, py, depth_img)
        if z_m <= 0:
            return None

        fx, fy, cx, cy = CAM_FX, CAM_FY, CAM_CX, CAM_CY  # 权威标定内参

        p_cam = np.array([
            (px - cx) * z_m / fx,
            (py - cy) * z_m / fy,
            z_m,
        ])

        # [Step 2] 相机系 → flange 系 (权威手眼)
        p_flange = (T_FLANGE_CAMERA @ np.append(p_cam, 1.0))[:3]

        # [Step 3] FK: flange → 基座 (反馈角度先修正为真实关节角)
        if isinstance(arm_angles, dict):
            angle_list = fix_feedback_angles(arm_angles)
        else:
            angle_list = fix_feedback_angles(list(arm_angles))
        try:
            T_f2b, _, _ = D1Kinematics.forward_kinematics(
                angle_list, output='flange')
            p_base_flange = (T_f2b @ np.append(p_flange, 1.0))[:3]
        except Exception as e:
            print(f'    [3] FK ERROR: {e}, angles={angle_list}')
            return None

        # [Step 4] 抓取目标: TCP(夹爪中心)对准物块, flange 需移到物块 - TCP偏移
        p_grasp = p_base_flange - T_f2b[:3, :3] @ TCP_OFFSET_FLANGE
        p_grasp[0] += GRASP_PLAN_OFFSET_X  # X 方向补偿
        p_grasp[1] += GRASP_PLAN_OFFSET_Y  # Y 方向补偿
        p_grasp[2] += GRASP_PLAN_OFFSET_Z  # Z 方向补偿


        # Z 硬限位
        p_grasp[2] = max(p_grasp[2], Z_ABS_MIN)

        if debug_log:
            print(f'    [1] 像素({px},{py}) z={z_m:.3f}m → 相机 '
                  f'[{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}]')
            print(f'    [2] → flange [{p_flange[0]:.4f}, {p_flange[1]:.4f}, '
                  f'{p_flange[2]:.4f}]')
            print(f'    [3] FK(flange,修正角) → 基座 物块 '
                  f'[{p_base_flange[0]:.4f}, {p_base_flange[1]:.4f}, '
                  f'{p_base_flange[2]:.4f}]')
            print(f'    [4] 抓取目标(含TCP补偿) '
                  f'[{p_grasp[0]:.4f}, {p_grasp[1]:.4f}, {p_grasp[2]:.4f}]')

        return p_grasp

    def get_tcp_pose(self, angles_fb=None):
        """
        获取当前 TCP 在基座系中的坐标 (x, y, z)
        单位: 米
        """
        if angles_fb is None:
            if not self.dry_run:
                self.arm.query_angles()
                angles_fb = dict(self.arm.get_angles())
            else:
                return np.array([0.3, 0.0, 0.2])  # dry-run 默认值

        # 反馈角 → 真实关节角
        angle_list = fix_feedback_angles(angles_fb)

        # FK: flange 位姿
        T_f2b, _, _ = D1Kinematics.forward_kinematics(angle_list, output='flange')

        # flange → TCP
        p_flange = T_f2b[:3, 3]
        R_f2b = T_f2b[:3, :3]
        p_tcp = p_flange + R_f2b @ TCP_OFFSET_FLANGE

        return p_tcp

    @staticmethod
    def _median_depth(px, py, depth_img, patch=9):
        """深度图 patch 中值滤波 (mm → m)"""
        h, w = depth_img.shape[:2]
        x1 = max(0, int(px) - patch)
        x2 = min(w, int(px) + patch + 1)
        y1 = max(0, int(py) - patch)
        y2 = min(h, int(py) + patch + 1)
        region = depth_img[y1:y2, x1:x2]
        valid = region[region > 0]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid)) * 0.001

    def _mask_placed(self, dets, depth_img, angles):
        """
        v39: 屏蔽中转放下的物块位置 — 基座距离 placed_pos 在
        EXCLUDE_PLACED_DIST 内的检测剔除, 防止再抓回A.
        未中转过直接返回原检测.
        """
        if getattr(self, 'placed_pos', None) is None:
            return dets
        kept = []
        for d in dets:
            pb = self.pixel_to_base(d['cx'], d['cy'], depth_img, angles)
            if pb is None:
                continue
            if np.linalg.norm(pb - self.placed_pos) > EXCLUDE_PLACED_DIST:
                kept.append(d)
        return kept

    def _get_frames(self):
        """
        v39.14: 取帧带自恢复 — USB2 下偶发掉帧/超时抛异常会中断整个流程.
        失败重试3次, 仍失败则重建相机 pipeline 再试, 避免运行一半中断.
        """
        for round_ in range(2):
            for attempt in range(3):
                try:
                    frames = self.pipe.wait_for_frames()
                    aligned = self.align.process(frames)
                    color = np.asanyarray(
                        aligned.get_color_frame().get_data())
                    depth = np.asanyarray(
                        aligned.get_depth_frame().get_data())
                    return color, depth
                except Exception as e:
                    print(f'  [CAM] 取帧失败 (r{round_+1} '
                          f'a{attempt+1}/3): {e}')
                    time.sleep(1.0)
            print('  [CAM] 连续取帧失败, 重建相机 pipeline...')
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.init_camera()
        raise RuntimeError('Camera frames unavailable after recovery')

    def _get_frames_quick(self, timeout_ms=1000):
        """2026-08-12: 单次快速取帧 (超时上限, 不触发恢复阶梯) — 容量/前方/平台判定链用,
        相机慢帧时最多等 timeout_ms, 超时返回 (None, None) 走兜底链."""
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=timeout_ms)
            aligned = self.align.process(frames)
            color = np.asanyarray(aligned.get_color_frame().get_data())
            depth = np.asanyarray(aligned.get_depth_frame().get_data())
            return color, depth
        except Exception as e:
            print(f'  [CAP] ⚠️ 快速取帧超时/失败 ({timeout_ms}ms): {e}')
            return None, None

    def _check_front_yellow(self):
        """2026-08-11 (用户): 抓A容量0兜底 — 前方是否还有黄色物块.
        取1-2帧检测: 有任何黄色检测返回 True (前方还有 = 疑似抓空, 重试);
        全部无黄色 = 物块已在爪上, 判容量+1. 取帧失败保守返回 True (重试)."""
        for _ in range(2):
            try:
                color, depth = self._get_frames_quick(1000)  # 2026-08-12: 快速取帧1s, 慢帧不站30s
            except Exception as e:
                print(f'  [CAP] ⚠️ 前方检查取帧失败: {e}, 视为前方有黄色 (保守重试)')
                return True
            if color is None:
                print('  [CAP] ⚠️ 前方检查无帧, 视为前方有黄色 (保守重试)')
                return True
            dets = self.detector.detect(color)
            # 2026-08-11: 排除爪上物块 (近处/超大 = held), 只数远处黄色 —
            # 否则爪上的A部分入镜会被当成"前方还有黄色"反复重试
            for d in dets:
                w = d.get('x2', 0) - d.get('x1', 0)
                h = d.get('y2', 0) - d.get('y1', 0)
                area = d.get('area', 0) or (w * h)
                z = self._min_depth(d['cx'], d['cy'], depth)  # 2026-08-12: 最小深度防中值稀释
                if area >= CAP_AREA_MIN or (z is not None and 0 < z < CAP_HELD_Z_MAX):
                    continue  # 爪上物块 (近处<0.25m), 跳过
                return True  # 远处还有黄色
            time.sleep(0.2)
        return False

    def _check_front_only_masked(self):
        """2026-08-11 (用户): 抓B容量0兜底 — 前方除被屏蔽的A外是否无其它黄色物块?
        取1-2帧检测, 用 _mask_placed 过滤掉放下的A; 过滤后无任何检测
        → B不在平台上 (在爪上, 近距漏判) → 容量+1.
        取帧/角度查询失败保守返回 False (维持重试)."""
        for _ in range(2):
            try:
                color, depth = self._get_frames_quick(1000)  # 2026-08-12: 快速取帧1s
            except Exception as e:
                print(f'  [CAP] ⚠️ 平台检查取帧失败: {e}, 视为有其它物块 (保守重试)')
                return False
            if color is None or depth is None:
                print('  [CAP] ⚠️ 平台检查无帧, 视为有其它物块 (保守重试)')
                return False
            dets = self.detector.detect(color)
            try:
                self.arm.query_angles()
                ang = dict(self.arm.get_angles())
            except Exception as e:
                print(f'  [CAP] ⚠️ 平台检查角度查询失败: {e}, 视为有其它物块 (保守重试)')
                return False
            dets = self._mask_placed(dets, depth, ang) or []
            # 2026-08-11: 排除爪上物块 (近处/超大 = held), 只数远处黄色
            for d in dets:
                w = d.get('x2', 0) - d.get('x1', 0)
                h = d.get('y2', 0) - d.get('y1', 0)
                area = d.get('area', 0) or (w * h)
                z = self._min_depth(d['cx'], d['cy'], depth)  # 2026-08-12: 最小深度防中值稀释
                if area >= CAP_AREA_MIN or (z is not None and 0 < z < CAP_HELD_Z_MAX):
                    continue  # 爪上物块 (近处<0.25m), 跳过
                return False  # 远处还有黄色 (B还在平台上)
            time.sleep(0.2)
        return True

    def _min_depth(self, cx, cy, depth_img, r=12):
        """2026-08-12: 框中心 r×r 方块的最小深度(m) — 爪上物块排除用:
        爪上B部分入镜时中值被背景稀释读值偏大, 最小值能抓住物块本身的近距像素."""
        try:
            h, w = depth_img.shape[:2]
            y0, y1 = max(0, cy - r), min(h, cy + r)
            x0, x1 = max(0, cx - r), min(w, cx + r)
            patch = depth_img[y0:y1, x0:x1].astype(np.float32) * 0.001
            pos = patch[patch > 0]
            if pos.size:
                return float(pos.min())
        except Exception:
            pass
        return None

    def _platform_yellow_scan(self):
        """2026-08-11 (用户): 中转容量0平台扫描 — 取帧, 排除爪上物块 (近处<CAP_HELD_Z_MAX
        或超大面积), 返回 (按图像cx左→右排序的远处黄色检测列表, 对应深度帧).
        取帧/检测失败返回 None (调用方走原逻辑)."""
        try:
            color, depth = self._get_frames_quick(1000)  # 2026-08-12: 快速取帧1s
        except Exception as e:
            print(f'  [CAP] ⚠️ 平台扫描取帧失败: {e}')
            return None
        if color is None or depth is None:
            print('  [CAP] ⚠️ 平台扫描无帧, 走原逻辑')
            return None
        dets = self.detector.detect(color)
        far = []
        for d in dets:
            w = d.get('x2', 0) - d.get('x1', 0)
            h = d.get('y2', 0) - d.get('y1', 0)
            area = d.get('area', 0) or (w * h)
            z = self._min_depth(d['cx'], d['cy'], depth)  # 2026-08-12: 最小深度防中值稀释
            if area >= CAP_AREA_MIN or (z is not None and 0 < z < CAP_HELD_Z_MAX):
                continue  # 爪上物块 (近处<0.25m)
            far.append(d)
        far.sort(key=lambda d: d['cx'])  # 左→右
        return far, depth

    def _check_capacity(self):
        """
        v39: 容量检测 — ready 姿态下, 画面里近处(~0.1m)有黄色物块
        = 爪上确实抓着物块 (容量 1/1). 面积超大也视为近处 (深度近距
        可能读0低于D435最小深度).
        """
        # 2026-08-09晚: 夹爪角度标准已移除 (用户: 容量+1不要夹爪角度标准) — 只用相机近处黄色判据
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=1000)  # 2026-08-12: 每帧1s超时 (用户), 原默认5s/帧站30s
            aligned = self.align.process(frames)
            color2 = np.asanyarray(aligned.get_color_frame().get_data())
            depth2 = np.asanyarray(aligned.get_depth_frame().get_data())
            dets = self.detector.detect(color2)
            for d in dets:
                w = d.get('x2', 0) - d.get('x1', 0)
                h = d.get('y2', 0) - d.get('y1', 0)
                area = d.get('area', 0) or (w * h)
                z = self._median_depth(d['cx'], d['cy'], depth2)
                if 0 < z < CAP_Z_MAX or area >= CAP_AREA_MIN:
                    print(f'  [CAP] 近处黄色确认 (z={z:.2f}m '
                          f'area={area:.0f}px)')
                    return True
            print('  [CAP] 画面里没有近处(~0.1m)黄色, 爪上无物块')
            self._heartbeat()  # 2026-08-11: 裸wait期间保活
            return False
        except Exception as e:
            print(f'  [CAP] 检测异常: {e}')
            self._heartbeat()  # 2026-08-11: 保活
            return False

    # ==================================================================
    # 画面标注
    # ==================================================================

    def annotate(self, color, detection, step, info=''):
        img = color.copy()
        cx_s, cy_s = IW // 2, IH // 2
        cv2.line(img, (cx_s - 20, cy_s), (cx_s + 20, cy_s),
                 (150, 150, 150), 1)
        cv2.line(img, (cx_s, cy_s - 20), (cx_s, cy_s + 20),
                 (150, 150, 150), 1)
        cv2.rectangle(img, (0, 0), (IW, 40), (0, 0, 0), -1)
        cv2.putText(img, f'Step:{step} | {info}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
        if detection:
            d = detection
            cx, cy = d['cx'], d['cy']
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2)
            cv2.line(img, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
            cv2.line(img, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
            ex, ey = cx - IW // 2, cy - IH // 2
            cv2.putText(img, f'err=({ex:+d},{ey:+d}) px',
                        (10, IH - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
            cv2.rectangle(img,
                          (d.get('x1', cx - 10), d.get('y1', cy - 10)),
                          (d.get('x2', cx + 10), d.get('y2', cy + 10)),
                          (0, 255, 0), 1)
            label = f'{d.get("name","obj")} {d.get("conf",1.0):.2f}'
            cv2.putText(img, label, (cx - 30, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, 'SEARCHING...',
                        (IW // 2 - 90, IH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return img

    # ==================================================================
    # 主循环 v15 — 底座对准 + 纯IK盲抓
    # ==================================================================

    def run(self):
        print('=' * 60)
        print('  3D Grasp Pipeline v17')
        print(f'  Mode: {self.mode} | Dry-run: {self.dry_run} | '
              f'Speed: {SPEED_FACTOR*100:.0f}%')
        print('=' * 60)

        try:
            self._run_inner()
        except KeyboardInterrupt:
            print('\n[STOP] User interrupt')
        except Exception as e:
            print(f'\n[ERROR] {e}')
            traceback.print_exc()
            self._emergency_stop()
        finally:
            self._cleanup()

    def _run_inner(self):
        """
        v16 主循环 — 解耦伺服: 底座对准 → 臂联动靠近 → 精对准 → IK盲抓

        状态机:
          COARSE:   检测稳定15帧(跳变<±50px) → 锁定位置 → 进PRECISION或继续伺服
          PRECISION: 距离<0.25m且画面中心±60px → 算p_base → 工作空间→EXECUTE
          EXECUTE:   IK三段式盲抓 (预接近→二次定位→下探→夹→提)
          DONE:      完成
        """
        self.init_camera()
        self.init_arm()
        self._init_stand()  # 抓取前: 平稳站立锁定

        if not self.has_display and self._enable_mjpeg:
            self.start_mjpeg()
            print('[MJPEG] http://192.168.123.18:8080/stream')
        elif not self.has_display:
            print('[MJPEG] 已禁用 (集成模式, 不启动HTTP流)')
        else:
            cv2.namedWindow('Grasp 3D', cv2.WINDOW_NORMAL)

        # 张开夹爪 (【集成】第二阶段起始夹持物块A, 不张爪防掉落)
        if not self.dry_run and self.phase != 'second':
            self.arm.open_gripper()
            time.sleep(0.5)

        # 就绪姿态 (v16: 臂前伸, 不挡相机, 腕部微低头)
        ready_pose = {
            'angle0': 0, 'angle1': -2, 'angle2': 42,
            'angle3': 0, 'angle4': -60.0, 'angle5': 0,
        }  # v39.6: angle4 -35→-50; 08-09: -50→-53.5; 2026-08-10: -53.5→-60 (用户)
        self._ready_pose = ready_pose
        self._zero_pose = {'angle0': 0, 'angle1': -90, 'angle2': 90,
                           'angle3': 0, 'angle4': 0, 'angle5': 0}  # 2026-08-04: 用户设置的零位 (桥home_all一致)
        # v18: 固定靠近姿态已删除, 15帧后直接IK解算全链路运动
        if not self.dry_run:
            self._safe_move(ready_pose, 'Ready pose')
            time.sleep(1.5)
        self._at_ready = True  # v39.12: 已到ready, 允许搜索

        ws_limits = self._compute_workspace()
        self._ws_limits_cache = ws_limits

        state = 'COARSE'
        stable_count = 0
        lost_count = 0
        grasp_count = 0
        self._grasp_a_attempts = 0  # 2026-08-06: 抓A尝试次数 (每次run重置, 同抓B)
        p_base = None
        step = 0

        while step < 200:
            step += 1
            # 2026-08-11: 180s 总超时已删除 (用户: 不需要, 6次尝试兜底足够)
            # 2026-08-11 (用户): 抓B容量0重试预算 8s — 超时强制容量+1 开始转
            if (getattr(self, '_capb_retry_deadline', None) is not None
                    and now_s() > self._capb_retry_deadline):
                print('  [CAP] ⏱ 8s 重试超时未确认, 强制容量判定 1/1, 直接左转衔接')
                self.holding_block = True
                self._at_ready = True
                self._capb_retry_deadline = None
                state = 'DONE'
                time.sleep(0.3)
                continue
            # 2026-08-11: 抓A同样 8s 预算 — 超时强制+1 (用户: 不要长等待)
            if (getattr(self, '_capa_retry_deadline', None) is not None
                    and now_s() > self._capa_retry_deadline):
                print('  [CAP] ⏱ 8s 重试超时未确认, 强制容量判定 1/1, 继续后续流程')
                self.holding_block = True
                self._at_ready = True
                self._capa_retry_deadline = None
                state = 'DONE'
                time.sleep(0.3)
                continue
            self._heartbeat()

            if self.dry_run:
                # 合成帧: 中央黄色方块 (dry-run 端到端验证)
                color = np.zeros((IH, IW, 3), np.uint8)
                cv2.rectangle(color, (260, 180), (380, 300), (0, 200, 255), -1)
                depth = np.zeros((IH, IW), np.uint16)
                depth[180:300, 260:380] = 300
            else:
                color, depth = self._get_frames()  # v39.14: 取帧带自恢复

            detections = self.detector.detect(color)

            if not self.dry_run:
                # v39.14: 角度查询防护 — 桥失联/无数据时跳过本帧, 不中断
                try:
                    self.arm.query_angles()
                    got = self.arm.get_angles()
                except Exception as e:
                    print(f'  [ARM] 角度查询异常, 跳过本帧: {e}')
                    time.sleep(0.3)
                    continue
                if not got:
                    print('  [ARM] 角度查询无数据, 跳过本帧')
                    time.sleep(0.3)
                    continue
                angles = dict(got)
            else:
                angles = ready_pose.copy()
            if not angles:
                angles = ready_pose.copy()

            # v39: 屏蔽中转放下的物块位置, 防止再抓回A
            detections = self._mask_placed(detections, depth, angles) or []

            # v39.1/v39.4: 夹持时排除"爪上物块" (面积超大 或 深度<0.15m),
            #              其余取面积最大 = 更远处的B. 不再用0.3m深度门槛 —
            #              B稍近(0.25~0.3m)或扫描时深度读值不稳也能锁定,
            #              不会一直搜索
            if self.holding_block:
                far_dets = []
                for d in detections:
                    w = d.get('x2', 0) - d.get('x1', 0)
                    h = d.get('y2', 0) - d.get('y1', 0)
                    area = d.get('area', 0) or (w * h)
                    z = self._median_depth(d['cx'], d['cy'], depth)
                    held = area >= CAP_AREA_MIN or 0 < z < CAP_Z_MAX
                    if not held:
                        far_dets.append(d)
                detections = far_dets

            target = detections[0] if detections else None
            # 2026-08-11 (用户): 目标可见 → 8s预算清零, 按6次保底抓; 8s只计"无目标"时间
            if target is not None:
                self._capb_retry_deadline = None
                self._capa_retry_deadline = None

            info = f'{state} | lost={lost_count}'
            display = self.annotate(color, target, step, info)
            if self.has_display:
                cv2.imshow('Grasp 3D', display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                with self.stream_lock:
                    self.stream_frame = display

            if target is None:
                if stable_count > 0:
                    print('  [COARSE] 目标丢失，重置稳定计数')
                stable_count = 0
                lost_count += 1
                # v39.15: 空手搜索时看不到目标 -> 狗缓慢前移8cm/步探索找物块 (2026-08-10: 12→8cm)
                #         (不再FAIL退出); 抓到物块后一律不向前; 步数封顶
                if (not self.dry_run
                        and self._at_ready
                        and (not self.holding_block or self._phase2_search)  # 【集成】第二阶段持物搜索
                        and self._ws_walk_steps < WS_WALK_MAX_STEPS
                        and lost_count % 4 == 0):
                    if self.transfer_done:
                        # 2026-08-04: 抓完B后不扫描 — 失败重试直接EXECUTE (B坐标已锁定), 不用angle0搜索
                        continue
                    # ===== 抓A搜索 (2026-08-11 用户): 累计前移≥50cm 或 危险深度<24cm → angle0扫描±40°(5°/步,3s/步,连续3帧) =====
                    if self.phase == 'first':
                        fwd_d = self._get_forward_depth()  # 每次迭代取一次前方深度
                        # 扫描触发: 累计前移≥50cm 未见物块 (用户) 或 前方深度<24cm (临近危险距离)
                        if not getattr(self, '_a_swept_after_step', False):
                            danger = fwd_d is not None and fwd_d < GRASP_FWD_SAFE_DIST
                            if getattr(self, '_a_search_dist', 0.0) >= 0.50 or danger:
                                print(f'  [搜索A] 🔄 累计前移{getattr(self, "_a_search_dist", 0.0)*100:.0f}cm'
                                      f'{" (危险深度)" if danger else ""}未见物块, angle0 扫描 ±40° (5°/步, 3s/步)...')
                                found = self._search_angle0_sweep(sweep_range=SAFE_SWEEP_RANGE, frames_req=SAFE_SWEEP_FRAMES)
                                self._a_swept_after_step = True
                                if found:
                                    print('  [搜索A] ✅ 扫描找到黄色物块(连续3帧), 停止搜索, 保持姿态进入抓取')
                                else:
                                    print('  [搜索A] ⚠️ 扫描未找到, 冷却后继续前移搜索')
                                    self._a_sweep_cooldown_until = now_s() + SAFE_SWEEP_COOLDOWN
                                time.sleep(0.3)
                                continue
                        # 扫描失败冷却中 → 不前移不重扫
                        if now_s() < getattr(self, '_a_sweep_cooldown_until', 0.0):
                            time.sleep(0.3)
                            continue
                        # 危险深度 (<24cm) → 不前移, 冷却后重扫
                        if fwd_d is not None and fwd_d < GRASP_FWD_SAFE_DIST:
                            print(f'  [安全] 前方深度 {fwd_d*100:.0f}cm < 24cm, 原地不动 (冷却后重扫)')
                            self._a_swept_after_step = False
                            time.sleep(0.3)
                            continue
                        self._ws_walk_steps += 1
                        self._a_search_dist = getattr(self, '_a_search_dist', 0.0) + WS_WALK_STEP_A
                        self._a_swept_after_step = False  # 走完待扫描
                        self._step_forward(WS_WALK_STEP_A, speed=GRASP_SEARCH_SPEED_M_S)  # 2026-08-11: 抓A 10cm/步 @0.5
                        time.sleep(0.3)
                        continue
                    # ===== 中转搜索 (phase=second): 安全门 + 8cm/步 =====
                    if self._safety_forward_gate():
                        time.sleep(0.3)
                        continue
                    self._ws_walk_steps += 1
                    self._step_forward(WS_WALK_STEP_M, speed=GRASP_SEARCH_SPEED_M_S)  # 2026-08-10: 搜索前移速度 0.3 (防前摔)
                time.sleep(0.3)
                continue

            lost_count = 0
            cx, cy = target['cx'], target['cy']
            z_cam = self._median_depth(cx, cy, depth)
            err_x = cx - IW // 2
            err_y = cy - IH // 2

            if z_cam <= MIN_GRASP_DEPTH:
                time.sleep(0.1)
                continue

            # ============================================================
            # COARSE: 解耦伺服 + 15帧稳定锁定 + 盲接近
            # ============================================================
            if state == 'COARSE':
                # 跳变检测：X/Y误差和上帧差超过容差直接重置
                if hasattr(self, '_coarse_last_cx'):
                    jump_x = abs(cx - self._coarse_last_cx)
                    jump_y = abs(cy - self._coarse_last_cy)
                    if jump_x > STABLE_PX_TOLERANCE or jump_y > STABLE_PX_TOLERANCE:
                        if stable_count > 0:
                            print(f'  [COARSE] 目标跳变({jump_x}px,{jump_y}px)，重置稳定计数')
                        stable_count = 0

                self._coarse_last_cx = cx
                self._coarse_last_cy = cy

                # v21: 第一段识别 angle0 不动, 纯稳定计数 (靠近后第二阶段再伺服)
                stable_count += 1

                # 逐帧打印当前稳定进度
                print(f'  [COARSE] 识别到目标，稳定帧：{stable_count:2d}/15 | 误差X={cx - ALIGN_CX_TARGET:+d} 距离={z_cam:.3f}m')

                # 没到15帧继续等待，不动臂
                if stable_count < ALIGN_STABLE_REQ:
                    time.sleep(0.2)
                    continue

                # ===== 15帧到了，直接拍最新帧算坐标，不移动臂 =====
                print('\n' + '='*60)
                print('  ✅ 15帧锁定，直接计算坐标开始抓取')
                print('='*60 + '\n')

                # 15帧到了直接用当前帧算，不重新拍
                p_base = self.pixel_to_base(cx, cy, depth, angles, debug_log=True)
                if p_base is None:
                    print('  [COARSE] 坐标计算失败，重试')
                    stable_count = 0
                    time.sleep(0.2)
                    continue

                in_ws = self._in_workspace(p_base, ws_limits)
                print(f'  [COARSE] p_base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) 工作空间={in_ws}')
                if not in_ws:
                    # v39.5/v39.13: 锁定目标且超出工作空间 -> 狗缓慢向前挪
                    # 一步再识别; 只在ready姿态、且爪上无物块时才挪
                    if (self._ws_walk_steps < WS_WALK_MAX_STEPS
                            and self._at_ready
                            and (not self.holding_block or self._phase2_search)):  # 【集成】第二阶段持物前移
                        # 2026-08-11: 前移前安全门 (用户)
                        if self._safety_forward_gate():
                            pass  # 安全扫描/冷却中: 本次不前移
                        else:
                            self._ws_walk_steps += 1
                            print(f'  [WARN] 超出工作空间'
                                  f' ({self._ws_walk_steps}/{WS_WALK_MAX_STEPS}),'
                                  f' 狗向前挪一点再识别')
                            if not self.dry_run:
                                self._step_forward(dist=WS_WALK_STEP_A if self.phase == 'first' else WS_WALK_STEP_M, speed=GRASP_SEARCH_SPEED_M_S)  # 2026-08-11: 抓A 12cm/中转 8cm
                    else:
                        print(f'  [WARN] 已挪 {WS_WALK_MAX_STEPS} 次仍超出'
                              f' 工作空间, 放弃挪动')
                    stable_count = 0
                    time.sleep(0.3)
                    continue

                # 中转检查: 爪上抓着物块A, 又看到新物块B -> 先把A放到B右边
                if self.holding_block:
                    print('  [TRANSFER] 检测到新物块, 先中转手上物块...')
                    # 2026-08-11 (用户): 记忆第一次锁定B的方向/位置 (放A后对准B用)
                    self._b_mem_angle0 = angles.get('angle0', 0.0)
                    self._b_mem_base = np.array(p_base, dtype=float)
                    # v39.14: 中转异常不中断 — 保持夹持等待重试
                    try:
                        ok_t = self._transfer_place(p_base)
                    except Exception as e:
                        print(f'  [TRANSFER] 中转异常: {e}, 保持夹持等待')
                        ok_t = False
                    if not ok_t:
                        # 中转失败 (工作空间/IK): 保持夹持, 等待/重试, 不去抓B
                        print('  [TRANSFER] 中转失败, 保持夹持等待 (不抓新物块)')
                        stable_count = 0
                        time.sleep(1.0)
                        continue
                    self.holding_block = False
                    self.transfer_done = True  # 中转完成, 抓B即收尾
                    self._transfer_search_start = now_s()  # 2026-08-04: 抓B搜索总超时起点
                    self._grasp_b_attempts = 0  # 2026-08-04: 抓B尝试次数
                    # v39: 中转后回 ready
                    if not self.dry_run:
                        self._safe_move(self._ready_pose, 'Return Ready')
                        time.sleep(1.5)
                    self._at_ready = True  # v39.12: 在ready才搜索
                    self._ws_walk_steps = 0  # v39.7: 新抓取循环, 挪动预算重置
                    # 2026-08-11 (用户): A从始至终不动 — 放A后检测A实际落点, 锚定屏蔽中心 (须在转B方向前)
                    if not self.dry_run:
                        try:
                            color_a, depth_a = self._get_frames()
                            if color_a is not None and depth_a is not None:
                                dets_a = self.detector.detect(color_a)
                                best = None; best_d = 1e9
                                for d in dets_a:
                                    w = d.get('x2', 0) - d.get('x1', 0)
                                    h = d.get('y2', 0) - d.get('y1', 0)
                                    area = d.get('area', 0) or (w * h)
                                    z = self._median_depth(d['cx'], d['cy'], depth_a)
                                    if area >= CAP_AREA_MIN or 0 < z < CAP_HELD_Z_MAX:
                                        continue  # 爪上物块
                                    pb = self.pixel_to_base(d['cx'], d['cy'], depth_a, angles)
                                    if pb is not None:
                                        dist = float(np.linalg.norm(pb - self.placed_pos))
                                        if dist < best_d:
                                            best_d = dist; best = pb
                                if best is not None and best_d < 0.15:
                                    self.placed_pos = best
                                    print(f'  [TRANSFER] 🎯 A实际落点锚定: {np.round(best,3)} (距计算位{best_d*100:.1f}cm)')
                                else:
                                    print(f'  [TRANSFER] ⚠️ 未锚定到A实际落点 (最近{best_d*100:.1f}cm), 用计算位')
                        except Exception as e:
                            print(f'  [TRANSFER] ⚠️ A锚定检测异常: {e}, 用计算位')
                    # 2026-08-11 (用户): 放A后 angle0 转回记忆的B方向 (相机对准B, 重试优先看到B)
                    if not self.dry_run and self._b_mem_angle0 is not None:
                        self._safe_move({'angle0': self._b_mem_angle0}, 'Face B mem')
                        time.sleep(1.0)
                    # 2026-08-07: 放A后直接用锁定B坐标抓 (用户: 抓B用第一次锁定的坐标)
                    print('  [TRANSFER] 已放下并回 ready, 直接用锁定B坐标抓...')
                    state = 'EXECUTE'
                    stable_count = 0
                    time.sleep(0.5)
                    continue

                # 2026-08-11 (用户修正): 抓B重试只更新B方向记忆 — A不动, 屏蔽锚定放A后的实际落点, 不跟随B
                if self.transfer_done and self.phase == 'second':
                    self._b_mem_angle0 = angles.get('angle0', 0.0)
                    self._b_mem_base = np.array(p_base, dtype=float)
                # 直接进抓取，用当前初始位置当种子，IK自己算预接近/下探
                state = 'EXECUTE'
                continue

            # ============================================================
            # PRECISION: 确认工作空间 → EXECUTE
            # ============================================================
            elif state == 'PRECISION':
                p_base = self.pixel_to_base(cx, cy, depth, angles,
                                            debug_log=self.dry_run)
                if p_base is None:
                    time.sleep(0.2)
                    continue

                in_ws = self._in_workspace(p_base, ws_limits)
                print(f'  S{step} [PRECISION]: p_base='
                      f'({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) ws={in_ws}')
                if in_ws:
                    state = 'EXECUTE'
                    print('  -> EXECUTE')
                    continue
                else:
                    print('  [WARN] Outside workspace, back to COARSE')
                    state = 'COARSE'
                    stable_count = 0
                    time.sleep(0.3)
                    continue

            # ============================================================
            # EXECUTE: IK盲抓
            # ============================================================
            elif state == 'EXECUTE':
                print(f'\n  *** IK GRASP @ '
                      f'base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) ***')
                self._at_ready = False  # v39.12: 抓取过程中不搜索
                # 从当前初始位置出发, IK自动解预接近→下探→夹取→提起
                # v39.14: 抓取过程异常不中断 — 视为失败, 回ready重试
                try:
                    success = self._execute_grasp(
                        p_base, angles, color, depth, skip_approach=False)
                except Exception as e:
                    print(f'  [FAIL] 抓取过程异常: {e}')
                    success = False
                    if not self.dry_run:
                        self._safe_move(self._ready_pose, 'Return Ready')
                        time.sleep(1.5)
                    self._at_ready = True
                if success:
                    grasp_count += 1
                    print(f'  Grasp #{grasp_count} OK!')
                    if self.transfer_done:
                        # 中转后抓取 -> 目标物块B, 容量确认 (2026-08-04)
                        print('  [TRANSFER] 目标物块已抓起, 回 ready 容量确认...')
                        if not self.dry_run:
                            self._safe_move(self._ready_pose, 'Return Ready')
                            time.sleep(1.5)
                            self._at_ready = True
                            # 2026-08-08: 连续 5 帧确认容量 1/1 再走 (用户)
                            cap_ok = True
                            for _cf in range(5):
                                if not self._check_capacity():
                                    cap_ok = False
                                    break
                                time.sleep(0.05)
                            if cap_ok:
                                self.holding_block = True
                                print('  [CAP] 中转容量 1/1 (B连续5帧确认), 直接左转')
                                state = 'DONE'
                            else:
                                # ===== 2026-08-12 (用户): 容量确认失败 → 用记忆位置直接重抓, 跳过平台扫描 =====
                                print('  [CAP] 中转容量未确认, 用记忆位置重抓B...')
                                self.holding_block = False
                                self._grasp_b_attempts += 1  # 2026-08-04: 重试计数
                                if self._grasp_b_attempts > Z0_GRASP_B_MEM_MAX_ATTEMPTS:
                                    print(f'  [中转搜索] 记忆重抓 {self._grasp_b_attempts} 次达上限, 放弃抓B')
                                    break
                                if getattr(self, '_b_mem_base', None) is not None:
                                    p_base = np.array(self._b_mem_base, dtype=float)
                                    print(f'  [CAP] 🎯 记忆位置重抓: base={np.round(p_base,3)}')
                                    if getattr(self, '_b_mem_angle0', None) is not None and not self.dry_run:
                                        self._safe_move({'angle0': self._b_mem_angle0}, 'Face B mem')
                                    state = 'EXECUTE'
                                    stable_count = 0
                                    time.sleep(0.3)
                                    continue
                                print('  [CAP] ⚠️ 无记忆位置, 回COARSE重新定位')
                                state = 'COARSE'
                                stable_count = 0
                                time.sleep(0.3)
                                continue
                        else:
                            self.holding_block = True
                            state = 'DONE'
                    else:
                        # 第一次抓 -> 回 ready, 实测近处~0.1m黄色确认容量1/1
                        print('  [TRANSFER] 已抓第1个物块, 回 ready 姿态...')
                        if not self.dry_run:
                            self._safe_move(self._ready_pose, 'Return Ready')
                            time.sleep(1.5)
                            self._at_ready = True  # v39.12: 在ready才搜索
                            if self._check_capacity():
                                self.holding_block = True
                                print('  [CAP] 容量 1/1 (近处~0.1m黄色确认)')
                            else:
                                print('  [CAP] 近处无黄色, 疑似抓空, 检查前方是否还有黄色物块...')
                                # 2026-08-11 (用户): 容量0 但前方已无黄色物块 → 物块在爪上, 判定容量+1, 继续后续流程
                                if not self._check_front_yellow():
                                    print('  [CAP] ✅ 前方无黄色物块 (物块已在爪上), 容量判定 1/1, 继续后续流程')
                                    self.holding_block = True
                                else:
                                    # 2026-08-11 (用户): 8s 重试预算 — 超时强制容量+1, 不再6次×2轮长等
                                    if getattr(self, '_capa_retry_deadline', None) is not None and now_s() > self._capa_retry_deadline:
                                        print('  [CAP] ⏱ 8s 重试未确认, 强制容量判定 1/1, 继续后续流程')
                                        self.holding_block = True
                                        self._at_ready = True
                                        self._capa_retry_deadline = None
                                        time.sleep(0.3)
                                        continue
                                    if getattr(self, '_capa_retry_deadline', None) is None:
                                        self._capa_retry_deadline = now_s() + 8.0
                                        print('  [CAP] ⏱ 8s 内未确认则自动容量+1 继续')
                                    print('  [CAP] 前方仍有黄色物块, 疑似抓空, 重新抓取')
                                    self.holding_block = False
                                    if self.phase == 'first':  # 2026-08-06: 抓A疑似抓空也计数 (已是COARSE重新定位)
                                        self._grasp_a_attempts += 1
                                        if self._grasp_a_attempts >= Z0_GRASP_A_MAX_ATTEMPTS:
                                            print(f'  [抓取1] 抓A尝试 {self._grasp_a_attempts} 次达上限, 放弃抓A')
                                            break
                                    state = 'COARSE'
                                    stable_count = 0
                                    time.sleep(0.3)
                                    continue
                        else:
                            self.holding_block = True
                            self._at_ready = True  # v39.12: dry-run也视为就绪
                        self._ws_walk_steps = 0  # v39.7: 新抓取循环, 挪动预算重置
                        self._a_search_dist = 0.0  # 2026-08-11: 抓A累计前移距离重置
                        # 【集成】第一阶段: 抓完第1个物块并确认容量(1/1)后直接返回, 由巡线继续走
                        if self.phase == 'first':
                            print('  [PHASE1] 已抓第1个物块并确认容量(1/1), 返回巡线流程')
                            break
                        state = 'COARSE'
                        stable_count = 0
                        time.sleep(0.5)
                        continue
                else:
                    print('  [FAIL] IK failed, 回 ready 再重试')
                    if not self.dry_run:
                        self._safe_move(self._ready_pose, 'Return Ready')
                        time.sleep(1.5)
                    self._at_ready = True  # v39.12: 回ready才允许搜索
                    if self.phase == 'first':
                        self._grasp_a_attempts += 1  # 2026-08-06: 抓A重试计数 (用户要求: 失败重新定位再抓)
                        if self._grasp_a_attempts >= Z0_GRASP_A_MAX_ATTEMPTS:
                            print(f'  [抓取1] 抓A尝试 {self._grasp_a_attempts} 次达上限, 放弃抓A')
                            break
                    if self.phase == 'second':
                        self._grasp_b_attempts += 1  # 2026-08-04: 中转重试计数
                        if self._grasp_b_attempts >= Z0_GRASP_B_MAX_ATTEMPTS:
                            print(f'  [中转搜索] 抓B尝试 {self._grasp_b_attempts} 次达上限, 放弃抓B')
                            break
                    # 2026-08-06: 抓A/B每次重试都回COARSE重新定位 (用户指定)
                    state = 'COARSE'
                    stable_count = 0
                    time.sleep(0.3)
                    continue

            elif state == 'DONE':
                break

        # 2026-08-11 (用户): 重试到上限时平台只可能 0/1 个黄色 (B抓住了未确认/掉出)
        # → 强制容量+1 继续, 不空爪退出
        if self.transfer_done and not self.holding_block:
            print('  [CAP] ✅ 重试上限到达, 平台已无B (0/1个黄色), 强制容量判定 1/1, 继续运行')
            self.holding_block = True
        if self.has_display:
            cv2.destroyAllWindows()
        print(f'\n{"="*60}')
        print(f'  Grasped: {grasp_count} objects')
        print(f'{"="*60}')

    # ==================================================================
    # 解耦视觉伺服 v16 — 先底座对准, 再大臂小臂联动靠近
    # ==================================================================

    def _visual_servo_step(self, cx, cy, depth_cam, angles):
        """
        解耦伺服 v16: Step1底座对准X → Step2大臂小臂联动靠近
        angle4(夹爪)不参与伺服, 由open_gripper/close_gripper独立控制

        方向验证 (反了改符号):
        - 水平: 右偏(err_x>0)→底座右转(angle0+), 反了改 d0=-err_x*SERVO_GAIN_X
        - 深度: 物体远(err_d>0)→大臂下俯(angle1-), 反了改 d1=+err_d*SERVO_GAIN_D
        - 上下: 物体在下(err_y>0)→小臂抬(angle2-), 反了改 d2=-err_y*SERVO_GAIN_Y
        """
        err_x = cx - IW // 2
        err_y = cy - IH // 2
        err_d = depth_cam - 0.18
        moves = {}

        # Step 1: 底座对准X (只转angle0)
        if abs(err_x) > 30:
            j0 = angles.get('angle0', 0)
            d0 = err_x * SERVO_GAIN_X
            d0 = np.clip(d0, -SERVO_MAX_DELTA, SERVO_MAX_DELTA)
            target_a0 = float(np.clip(j0 + d0, -60, 60))
            moves['angle0'] = target_a0
            print(f'  [SrvX] ex={err_x:+d} d0={d0:+.2f} a0:{j0:.1f}->{target_a0:.1f}')

        # Step 2: 大臂小臂联动靠近 (X对准后才动)
        if abs(err_x) < 60 and (abs(err_d) > 0.02 or abs(err_y) > 30):
            j1 = angles.get('angle1', -20)
            j2 = angles.get('angle2', 40)
            d1 = -err_d * SERVO_GAIN_D
            d1 = np.clip(d1, -SERVO_MAX_DELTA, SERVO_MAX_DELTA)
            d2 = err_y * SERVO_GAIN_Y
            d2 = np.clip(d2, -SERVO_MAX_DELTA, SERVO_MAX_DELTA)
            target_a1 = float(np.clip(j1 + d1, -78, 55))
            target_a2 = float(np.clip(j2 + d2, -40, 80))
            moves['angle1'] = target_a1
            moves['angle2'] = target_a2
            print(f'  [SrvDY] ed={err_d:+.3f} ey={err_y:+d} d1={d1:+.2f} d2={d2:+.2f} '
                  f'a1:{j1:.1f}->{target_a1:.1f} a2:{j2:.1f}->{target_a2:.1f}')

        if self.dry_run:
            for k, v in moves.items():
                angles[k] = v
        else:
            self._safe_move(moves, 'Servo')

    # ==================================================================
    # 三段式抓取 (保持 v14 逻辑, 含二次定位)
    # ==================================================================

    def _init_stand(self):
        """抓取前: 平稳站立锁定 (Go2 运动控制, 防狗晃动)"""
        if self.dry_run:
            print('[GO2] DRY-RUN: skip stand')
            return
        # 【集成】复用巡线运动客户端 (不重建DDS通道, 避免冲突)
        if self._sport_client is not None:
            self._sport = self._sport_client
            try:
                self._sport.RecoveryStand()
                print('[GO2] 平稳站立锁定 (复用巡线运动客户端)')
            except Exception as e:
                print(f'[GO2] 站立初始化失败: {e}')
            return
        try:
            sys.path.insert(0, '/home/unitree/unitree_sdk2_python')
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            ChannelFactoryInitialize(0, "eth0")
            time.sleep(1)
            self._sport = SportClient()
            self._sport.SetTimeout(10.0)
            self._sport.Init()
            time.sleep(1)
            self._sport.RecoveryStand()
            print('[GO2] 平稳站立锁定 (抓取稳定)')
        except Exception as e:
            print(f'[GO2] 站立初始化失败: {e}')

    def _resume_motion(self):
        """抓完: 恢复运动 (解锁, 可走路)"""
        if getattr(self, '_sport', None):
            try:
                self._sport.Move(0, 0, 0)  # 零速激活运动控制
                print('[GO2] 已恢复运动 (可走路)')
            except Exception as e:
                print(f'[GO2] 恢复运动失败: {e}')

    def _search_angle0_sweep(self, sweep_range=None, frames_req=2):
        """2026-08-04: angle0 梯度搜索B — ±sweep_range° (5°/步, 先右, 每步停3s), 找到即停.
        第一层±15°(夹A前移搜索), 第二层±20°(放A后重搜); 全部未找到回0位
        2026-08-11: frames_req=连续看到目标帧数 (中转保持2, 安全扫描用3)"""
        if sweep_range is None: sweep_range = Z0_SWEEP_LAYER1_RANGE  # 常量在文件后部, 运行时解析
        if self.dry_run:
            return False
        if not self._at_ready:  # 2026-08-04: 只在 ready 姿态才扫描 (双保险)
            print('  [中转搜索] ⚠️ 臂不在ready姿态, 跳过扫描')
            return False
        print(f'  [中转搜索] 🔄 angle0 梯度扫描 ±{sweep_range}° (5°/步, 先右, 每步停3s) 找B...')
        sweep = list(range(5, sweep_range + 1, 5)) + list(range(-5, -sweep_range - 1, -5))
        for a0 in sweep:
            try:
                self.arm.move_joints({0: float(a0)})
            except Exception as e:
                print(f'  [中转搜索] angle0={a0} 移动失败: {e}')
                continue
            # 2026-08-04: 每步停3s期间持续检测; 连续2帧看到B(帧数上升) → 立即停, 保持当前姿态
            found_frames = 0
            pause_end = now_s() + 3.0
            while now_s() < pause_end:
                try:
                    color, depth = self._get_frames()
                except Exception:
                    time.sleep(0.2)
                    continue
                if color is None or depth is None:
                    found_frames = 0
                    time.sleep(0.2)
                    continue
                detections = self.detector.detect(color)
                try:
                    self.arm.query_angles()
                    ang = dict(self.arm.get_angles())
                except Exception:
                    ang = None
                dets = self._mask_placed(detections, depth, ang) or []
                far = []
                for d in dets:
                    w = d.get('x2', 0) - d.get('x1', 0)
                    h = d.get('y2', 0) - d.get('y1', 0)
                    area = d.get('area', 0) or (w * h)
                    z = self._median_depth(d['cx'], d['cy'], depth)
                    if not (area >= CAP_AREA_MIN or 0 < z < CAP_Z_MAX):
                        far.append(d)
                if far:
                    found_frames += 1
                    if found_frames >= frames_req:  # 帧数上升 → 停 (中转2帧 / 安全扫描3帧)
                        print(f'  [中转搜索] ✅ angle0={a0}° 看到目标(连续{found_frames}帧), 停止扫描保持姿态')
                        return True
                else:
                    found_frames = 0
                time.sleep(0.3)
            print(f'  [中转搜索] angle0={a0}°: 未见B')
        try:
            self.arm.move_joints({0: 0.0})
        except Exception:
            pass
        print('  [中转搜索] ⏱ angle0 扫描完成, 未找到B')
        self._sweep_cooldown_until = now_s() + Z0_SWEEP_COOLDOWN  # 2026-08-04: 失败冷却
        return False

    def _get_forward_depth(self):
        """2026-08-11: 循迹相机(244222070235)中心前方深度(m) — 抓取前移安全距离用 (用户).
        抓取期间巡线detector已停, 此处临时开短流(仅depth 6fps)读中心深度后关闭.
        失败返回 None — 安全检查降级跳过, 不阻断流程."""
        if self.dry_run:
            return None
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(LINE_CAM_SERIAL)  # 循迹相机 (锁序列号)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
            pipe.start(cfg)
            try:
                vals = []
                for _ in range(10):
                    self._heartbeat()  # 2026-08-11: 短流读取期间保活
                    frames = pipe.wait_for_frames()
                    d = frames.get_depth_frame()
                    if d is None:
                        continue
                    scale = d.get_units()  # 深度比例 (米/单位)
                    cx, cy, r = 320, 240, 12  # 画面中心 12px 方块
                    patch = np.asanyarray(d.get_data())[cy-r:cy+r, cx-r:cx+r]
                    patch = patch[patch > 0] * scale
                    if patch.size:
                        vals.append(float(np.median(patch)))
                if vals:
                    return float(np.median(vals))
                print('  [安全] ⚠️ 循迹相机深度帧全无效')
                return None
            finally:
                pipe.stop()
        except Exception as e:
            print(f'  [安全] ⚠️ 循迹相机短流失败: {e} — 安全检查跳过')
            return None

    def _safety_forward_gate(self):
        """2026-08-11: 抓取前移安全门 (用户) —
        循迹相机前方深度 < GRASP_FWD_SAFE_DIST(24cm) → 原地停下, angle0 扫描搜索
        (±40°, 5°/步, 每步停3s, 连续3帧看到黄色物块即停 → 保持姿态回主循环直接抓取).
        返回 True = 本次不前移 (已扫描/冷却中/放弃); False = 深度正常, 可正常前移."""
        if self.dry_run or not self._at_ready:
            return False
        if getattr(self, '_safe_sweep_gave_up', False):
            return True
        if now_s() < getattr(self, '_safe_sweep_cooldown_until', 0.0):
            return True  # 扫描失败冷却中: 不前移不重扫
        fwd_d = self._get_forward_depth()
        if fwd_d is None or fwd_d >= GRASP_FWD_SAFE_DIST:
            return False
        print(f'  [安全] 🛑 前方深度 {fwd_d*100:.0f}cm < {GRASP_FWD_SAFE_DIST*100:.0f}cm, 原地停下 angle0 扫描搜索...')
        if getattr(self, '_sport', None) is not None:
            try:
                self._sport.StopMove()
            except Exception:
                pass
        found = self._search_angle0_sweep(sweep_range=SAFE_SWEEP_RANGE, frames_req=SAFE_SWEEP_FRAMES)
        if found:
            print('  [安全] ✅ 扫描找到黄色物块, 停止搜索, 保持姿态进入抓取')
            self._safe_sweep_cooldown_until = 0.0
            self._safe_sweep_attempts = 0
        else:
            self._safe_sweep_attempts = getattr(self, '_safe_sweep_attempts', 0) + 1
            self._safe_sweep_cooldown_until = now_s() + SAFE_SWEEP_COOLDOWN
            if self._safe_sweep_attempts >= SAFE_SWEEP_MAX_ATTEMPTS:
                self._safe_sweep_gave_up = True
                print(f'  [安全] ⚠️ 扫描 {self._safe_sweep_attempts} 次未找到, 放弃搜索 (防撞不前移)')
        return True

    def _step_forward(self, dist=None, speed=None):
        """
        v39.5/v39.10: 狗缓慢向前挪动一小段.
        搜索前移 8cm/步 (2026-08-10: 12cm→8cm 用户), 速度 0.3 (GRASP_SEARCH_SPEED_M_S, 0.4 易前摔降回).
        平稳站立锁定期间调用, 挪完保持站立, 由主循环重新识别.
        2026-08-04: speed=None 用 WS_WALK_SPEED_M_S (中转0.25); 抓取搜索传 0.30.
        """
        if dist is None:
            dist = WS_WALK_STEP_M
        if speed is None:
            speed = WS_WALK_SPEED_M_S
        if getattr(self, '_sport', None) is None:
            print('  [GO2] 运动客户端未初始化, 跳过挪动')
            return
        try:
            self._sport.Move(speed, 0, 0)
            time.sleep(dist / speed)
            self._sport.Move(0, 0, 0)
            print(f'  [GO2] 向前挪动 {dist*100:.0f}cm'
                  f' ({self._ws_walk_steps}/{WS_WALK_MAX_STEPS})')
            time.sleep(0.5)  # 等狗站稳再重新识别
        except Exception as e:
            print(f'  [GO2] 挪动失败: {e}')

    def _transfer_place(self, p_block):
        """
        中转: 把爪上物块放到新物块右边30°、9cm处 (2026-08-05晚: 左边→右边)
        夹持状态下移动, 到位后张开放下
        """
        # 2026-08-03 (z0 嵌入): 中转放置方向 右下 → 左下镜像 (放A在B的方向反了)
        # 2026-08-04: 再往下 6.5cm+2cm=8.5cm (用户调参)
        dx = -(TRANSFER_DIST * np.sin(np.radians(TRANSFER_ANGLE_DEG)) + 0.015) + 0.005  # 2026-08-09: 再往里0.5cm (用户: 放置A往近走2.5cm, 原3cm)
        dy = -TRANSFER_DIST * np.cos(np.radians(TRANSFER_ANGLE_DEG)) - 0.05  # 2026-08-06: 再往右5cm (用户)
        place_pt = p_block + np.array([dx, dy, -0.055])  # 2026-08-07: -0.03→-0.055 (再低2.5cm 用户)
        print(f'  [TRANSFER] 放置手上物块到 {np.round(place_pt, 3)} '
              f'(新物块右边{TRANSFER_ANGLE_DEG}° {TRANSFER_DIST*100:.0f}cm)')

        if self.dry_run:
            print(f'[DRY-RUN] Would transfer to: {np.round(place_pt, 3)}')
            return True

        # 工作空间检查
        if not self._in_workspace(place_pt, self._ws_limits_cache):
            print('  [TRANSFER] 放置点超出工作空间, 跳过中转')
            return False

        self.arm.query_angles()
        ang = dict(self.arm.get_angles())
        seed_real = fix_feedback_angles(ang) if ang else [0.0]*6

        # 1. 放置点上方 (夹持移动, 不张爪)
        hover = place_pt + np.array([0.0, 0.0, TRANSFER_LIFT_Z])
        ik_h = D1Kinematics.solve_grasp_ik(hover, seed_angles=seed_real,
                                           output='flange')
        if not ik_h['success']:
            print('  [TRANSFER] hover IK 失败')
            return False
        cmd_h = inv_fix_angles([ik_h['grasp'][f'angle{i}'] for i in range(6)])
        self._safe_move(cmd_h, 'Transfer Hover')
        time.sleep(0.5)

        # 2. 下降到位
        ik_p = D1Kinematics.solve_grasp_ik(place_pt, seed_angles=seed_real,
                                           output='flange')
        if not ik_p['success']:
            print('  [TRANSFER] place IK 失败')
            return False
        cmd_p = inv_fix_angles([ik_p['grasp'][f'angle{i}'] for i in range(6)])
        self._safe_move(cmd_p, 'Transfer Place')
        time.sleep(0.5)

        # 3. 张开放下
        print('  [TRANSFER] 张开夹爪放下物块...')
        self.arm.open_gripper()
        time.sleep(0.8)

        # 4. 提起
        self._safe_move(cmd_h, 'Transfer Lift')
        time.sleep(0.5)
        print('  [TRANSFER] 完成, 手上物块已放到新物块右下')
        self.placed_pos = place_pt  # v39: 屏蔽该位置, 防止再抓回A
        return True

    def _execute_grasp(self, p_base, seed_angles, color=None, depth=None,
                        skip_approach=False):
        """
        三段式抓取 v17: [盲接近(可选)] → 二次定位 → 两步下探 → 夹紧 → 提起

        skip_approach=True: 跳过Phase1预接近 (COARSE锁定后已盲接近到目标上方5cm)
        """
        if self.dry_run:
            print(f'[DRY-RUN] Would grasp at: {p_base} (Z={p_base[2]:.3f}m)')
            return True

        print(f'  当前TCP: {np.round(self.get_tcp_pose(seed_angles), 4)}')
        print(f'  目标TCP: {np.round(p_base, 4)}')

        # ---- flange IK 三段式 (权威标定: 修正关节角 + flange 目标) ----
        seed_angles = seed_angles.copy()
        seed_real = fix_feedback_angles(seed_angles)
        grasp_pt = p_base.copy()
        grasp_pt[0] += -0.0005  # 2026-08-10: TCP抓取点 X = 原基础-0.05cm (用户: 往前0.8cm后再往近0.85cm, 0.008-0.0085)
        result = D1Kinematics.solve_grasp_ik(
            grasp_pt, seed_angles=seed_real,
            approach_z=APPROACH_Z_OFFSET, down_extra=GRASP_DOWN_EXTRA + (GRASP_B_DOWN_EXTRA if self.phase == 'second' else 0.0),
            output='flange', angle4_bias=-6.0 if self.phase in ('first', 'second') else 0.0)  # 2026-08-06: A/B 抓取 angle4 偏-6° TCP不变
        if not result['success']:
            print(f'  [FAIL] Grasp IK 不可达: {p_base}, 停止')
            return False
        # IK 结果(真实关节角) → 命令(舵机)角
        for k in ('approach', 'grasp', 'lift'):
            result[k] = inv_fix_angles(
                [result[k][f'angle{i}'] for i in range(6)])
        print(f'  IK TARGET: grasp@{np.round(p_base, 3)} Z={p_base[2]:.3f}m '
              f'-> { {k: round(result["grasp"][k], 1) for k in result["grasp"]} }')

        # ---- Phase 1: 预接近 (张开夹爪 + 目标上方5cm) ----
        if not skip_approach:
            print('  [1/4] Approach...')
            self.arm.open_gripper()
            time.sleep(0.5)
            # 2026-08-08: 先只动 angle0 转到目标朝向, 到位后再动其他舵机 (用户)
            self._safe_move({'angle0': result['approach']['angle0']}, 'Approach a0')
            time.sleep(1.0)
            rest_angles = {f'angle{i}': result['approach'][f'angle{i}'] for i in range(1, 6)}
            self._safe_move(rest_angles, 'Approach rest')
            time.sleep(1.5)
        else:
            print('  [1/4] Approach... (skipped, 已在COARSE完成盲接近)')

        # ---- Phase 1.5: 二次定位 + angle0 对准伺服 ----
        if color is not None and depth is not None:
            print('  [2/4] Second look (angle0 align)...')
            # 用变量记录目标角度，不用反馈（反馈有延迟导致振荡）
            self.arm.query_angles()
            angles_now = dict(self.arm.get_angles())
            target_a0 = float(angles_now.get('angle0', 0)) if angles_now else 0.0

            stable_count = 0
            for _try in range(15):
                color2, depth2 = self._get_frames()  # v39.14
                dets = self.detector.detect(color2)
                if not dets:
                    time.sleep(0.2)
                    continue
                # v39.7: 每帧现查角度 + 屏蔽刚放下的A — 第二次抓取时A与B只
                #        隔7cm, 面积相近, 不屏蔽会对准A/来回切
                if not self.dry_run:
                    self.arm.query_angles()
                    fb_now = dict(self.arm.get_angles())
                else:
                    fb_now = angles_now
                dets = self._mask_placed(dets, depth2, fb_now) or []
                if not dets:
                    time.sleep(0.2)
                    continue
                dx = dets[0]['cx'] - ALIGN_CX_TARGET

                # 连续3帧在容差内才算对准
                if abs(dx) <= ALIGN_PX_TOL:
                    stable_count += 1
                    if stable_count >= 3:
                        break
                    time.sleep(0.1)
                    continue

                stable_count = 0
                d_a0 = float(np.clip(SERVO_GAIN_X * dx, -3.0, 3.0))
                target_a0 += d_a0
                target_a0 = float(np.clip(target_a0, -60.0, 60.0))
                self.arm.move_joint(0, target_a0)
                print(f'    [ALIGN] a0 target={target_a0:.1f}° (dx={dx:+d}px)')
                time.sleep(0.5)  # 等到位再拍下一帧

            # 对准后重新检测计算
            color2, depth2 = self._get_frames()  # v39.14
            dets = self.detector.detect(color2)
            if dets:
                self.arm.query_angles()
                new_angles = dict(self.arm.get_angles())
                if new_angles:
                    # v39: 屏蔽刚放下的物块位置, 防止二次定位对准A
                    dets = self._mask_placed(dets, depth2, new_angles) or []
                    if dets:
                        p_base2 = self.pixel_to_base(
                            dets[0]['cx'], dets[0]['cy'],
                            depth2, new_angles)
                    if dets and p_base2 is not None:
                        result2 = D1Kinematics.solve_grasp_ik(
                            p_base2, seed_angles=fix_feedback_angles(new_angles),
                            approach_z=0.02, down_extra=GRASP_DOWN_EXTRA + (GRASP_B_DOWN_EXTRA if self.phase == 'second' else 0.0),
                            output='flange', angle4_bias=-6.0 if self.phase in ('first', 'second') else 0.0)  # 2026-08-06: A/B 抓取 angle4 偏-6° TCP不变
                        if result2['success']:
                            for k in ('approach', 'grasp', 'lift'):
                                result2[k] = inv_fix_angles(
                                    [result2[k][f'angle{i}'] for i in range(6)])
                            if self._validate_joints(result2['grasp'],
                                                     'Grasp(v2)'):
                                result = result2
                                print(f'    Corrected: base='
                                      f'({p_base2[0]:.3f},{p_base2[1]:.3f},'
                                      f'{p_base2[2]:.3f})')

        # 抓取后读取实际TCP坐标 (调试: 用偏差数据精确调偏移)
        self.arm.query_angles()
        actual_angles = dict(self.arm.get_angles())
        if actual_angles:
            actual_tcp = self.get_tcp_pose(actual_angles)
            print(f'  实际TCP: {np.round(actual_tcp, 4)}')
            print(f'  偏差(mm): {np.round((actual_tcp - p_base) * 1000, 1)}')


        # ---- Phase 2: 两步下探抓取 ----
        print('  [3/4] Grasp (2-step)...')

        # Step 2a: 先下探到物体上方2cm
        p_mid = p_base.copy()
        p_mid[2] = p_base[2] + 0.02
        result_mid = D1Kinematics.solve_grasp_ik(
            p_mid, seed_angles=fix_feedback_angles(result['grasp']),
            approach_z=0.0, down_extra=0.0, output='flange', angle4_bias=-6.0 if self.phase in ('first', 'second') else 0.0)  # 2026-08-06: A/B 抓取 angle4 偏-6° TCP不变
        if result_mid['success']:
            result_mid['grasp'] = inv_fix_angles(
                [result_mid['grasp'][f'angle{i}'] for i in range(6)])
            print('    [3a] Pre-grasp hover (+2cm)...')
            self._safe_move_a0_first(result_mid['grasp'], 'PreGrasp')
            time.sleep(0.3)
        else:
            print('    [3a] Pre-grasp skipped (IK fail)')

        # Step 2b: 最终下探
        print('    [3b] Final descent...')
        self._safe_move_a0_first(result['grasp'], 'Grasp')
        time.sleep(1.0)
        time.sleep(0.2)

        # 夹紧 — 夹爪是独立执行器, 用专用接口
        print('    [3c] Closing gripper...')
        self.arm.close_gripper()
        time.sleep(0.8)

        # ---- Phase 3: 提起 ----
        print('  [4/4] Lift...')
        self._safe_move_a0_first(result['lift'], 'Lift')
        time.sleep(1.5)

        # ---- 抓取完毕: 回初始位置 (真实零位 Home), 夹爪保持闭紧 ----
        print('  [5/4] 抓取完毕, 回初始位置 (夹爪保持闭紧)...')
        home_cmd = inv_fix_angles([0.0] * 6)  # 真实零位 -> 命令(舵机)角
        self._safe_move_a0_first(home_cmd, 'Return Home')
        time.sleep(1.5)
        print('  ✅ 抓取完成: 已回初始位置, 夹爪闭紧')

        return True

    # ==================================================================
    # 心跳
    # ==================================================================

    def _heartbeat(self):
        if self.dry_run or self.arm is None:
            return
        now = now_s()
        if now - self._last_cmd_time > HEARTBEAT_INTERVAL:
            try:
                self.arm.query_angles()
                self._last_cmd_time = now
            except Exception as e:
                print(f'[Heartbeat] WARNING: query failed ({e})')

    # ==================================================================
    # 关节安全检查
    # ==================================================================

    def _validate_joints(self, joints_dict, label=''):
        for i in range(6):
            key = f'angle{i}'
            val = joints_dict.get(key, 0)
            lo, hi = JOINT_HARD_LIMITS[i]
            if not (lo <= val <= hi):
                print(f'  [REJECT] {label}: {key}={val:.1f} 超出限位 '
                      f'[{lo}, {hi}]!')
                return False
        return True

    def _safe_move_a0_first(self, angles_dict, label='Move'):
        """2026-08-09: 先只动 angle0 转到目标朝向, 到位后再动其余舵机 (用户)"""
        if self.dry_run:
            print(f'[DRY-RUN] {label}: {angles_dict}')
            return
        if not self._validate_joints(angles_dict, label):
            return
        a0 = angles_dict.get('angle0')
        if a0 is not None:
            self._safe_move({'angle0': a0}, f'{label} a0')
            time.sleep(0.8)
        rest = {k: v for k, v in angles_dict.items() if k != 'angle0'}
        if rest:
            self._safe_move(rest, f'{label} rest')
            time.sleep(0.8)

    def _safe_move(self, angles_dict, label='Move'):
        if self.dry_run:
            print(f'[DRY-RUN] {label}: {angles_dict}')
            return

        if not self._validate_joints(angles_dict, label):
            return

        self.arm.query_angles()
        current = dict(self.arm.get_angles())
        if current:
            max_delta = 0
            for i in range(6):
                k = f'angle{i}'
                if k in angles_dict and k in current:
                    max_delta = max(max_delta,
                                    abs(angles_dict[k] - current[k]))
            # 2026-08-10 (用户): 大角度跳变分段插值防舵机突变 (原仅1个中点且受SPEED_FACTOR闸)
            if max_delta > 15:
                steps = 5 if max_delta > 60 else (3 if max_delta > 30 else 2)
                for s in range(1, steps):
                    wp = {}
                    for k, v in angles_dict.items():
                        wp[k] = current.get(k, v) + (v - current.get(k, v)) * (s / steps)
                    print(f'  [Slow] {label} waypoint {s}/{steps} (delta={max_delta:.0f}deg)')
                    self.arm.move_joints(wp)
                    time.sleep(0.6)

        self.arm.move_joints(angles_dict)
        self._last_cmd_time = now_s()
        time.sleep(0.3)

    # ==================================================================
    # 紧急停止 & 清理
    # ==================================================================

    def _emergency_stop(self):
        print('[EMERGENCY] Stopping...')
        try:
            if self.arm:
                self.arm.open_gripper()
                time.sleep(0.3)
                self.arm.disable()
        except Exception:
            pass

    def _cleanup(self):
        _t0 = now_s()
        self._resume_motion()  # 抓完: 恢复运动 (可走路)
        print(f'  [Cleanup] 恢复运动: {now_s()-_t0:.1f}s')
        try:
            if self.pipe:
                self.pipe.stop()
        except Exception:
            pass
        print(f'  [Cleanup] 夹爪相机stop: {now_s()-_t0:.1f}s')
        try:
            if self.has_display:
                cv2.destroyAllWindows()
        except Exception:
            pass
        print(f'[Cleanup] Done (总 {now_s()-_t0:.1f}s)')

    # ==================================================================
    # 工具方法
    # ==================================================================

    def _compute_workspace(self):
        """v15: 用径向距离(r=sqrt(x²+y²)) - 底座可转, 只需检查距离和Z.
        放宽预检, 最终由IK决定是否可达."""
        r_vals, z_vals = [], []
        # 全关节范围采样 (a1/a2 ±90, 覆盖下探姿态; 桌面抓取目标 z 可为负)
        for a1 in range(-90, 91, 10):
            for a2 in range(-90, 91, 10):
                _, _, t = D1Kinematics.forward_kinematics(
                    [0, a1, a2, 0, 0, 0], output='flange')
                tv = t.ravel()
                r = np.sqrt(tv[0]**2 + tv[1]**2)
                r_vals.append(r)
                z_vals.append(tv[2])
        r_max_fk = max(r_vals)
        r_max = max(r_max_fk, 0.55)
        z_min = max(min(z_vals), Z_ABS_MIN)  # FK 可达下限与安全硬限位取严
        z_max = max(z_vals)
        print(f'[WS] limits: r_max={r_max:.2f}m z=[{z_min:.2f},{z_max:.2f}]')
        return r_max, z_min, z_max

    def _in_workspace(self, p_base, limits):
        r_max, z_min, z_max = limits
        r = np.sqrt(p_base[0]**2 + p_base[1]**2)
        r_ok = r <= r_max
        z_ok = z_min <= p_base[2] <= z_max
        if not r_ok:
            print(f'  [WS] R out: r={r:.3f}m > r_max={r_max:.3f}m')
        if not z_ok:
            print(f'  [WS] Z out: {p_base[2]:.3f} not in [{z_min:.2f}, {z_max:.2f}]')
        return r_ok and z_ok

    # v39.11: _search_scan (angle0/angle1 摆动搜索) 已删除 —
    #          搜索统一为狗缓慢前移把物块带入视野


# ===================================================================
# 主入口
# ===================================================================
# ==================== 【集成块·A】结束 ====================


# ==================== 命令行参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('interface', nargs='?', default='eth0')
parser.add_argument('--duration', type=float, default=0, help='运行时间(秒)')
parser.add_argument('--show', action='store_true', default=True, help='显示画面(需GUI)')
args = parser.parse_args()

INTERFACE = args.interface
SHOW_GUI = args.show

if SHOW_GUI:
    if 'DISPLAY' not in os.environ:
        print("[警告] 未检测到图形界面(DISPLAY)，自动关闭显示")
        SHOW_GUI = False
    else:
        try:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test")
        except:
            print("[警告] 无法创建图形窗口，自动关闭显示")
            SHOW_GUI = False

if not SHOW_GUI:
    print("[信息] 以无头模式运行 (无GUI)")


# ==================== 窄道检测配置 ====================
class NarrowConfig:
    """窄道检测和固定路径配置 - 每个转弯独立参数"""
    NARROW_WALL_DIST = 0.5
    NARROW_DETECT_FRAMES = 10
    MIN_NARROW_WIDTH = 0.3
    MAX_NARROW_WIDTH = 1.0

    # ====== 直行速度 ====== (2026-08-03: 正常直行0.3 用户)
    STRAIGHT_1_SPEED = 0.35  # 2026-08-14: 0.32→0.35 (用户: 窄道前进速度0.35)
    STRAIGHT_1B_SPEED = 0.35
    STRAIGHT_2_SPEED = 0.35
    STRAIGHT_2B_SPEED = 0.35
    STRAIGHT_3_SPEED = 0.35

    TURN_FWD_SPEED = 0.242

    # ====== 2026-08-06: 转弯后直行开头固定横向平移 ======
    SHIFT_AFTER_TURN2_VY = -0.3   # TURN_2 后 STRAIGHT_2 开头右移 (m/s, 负=右)
    SHIFT_AFTER_TURN2_TIME = 0.4  # 2026-08-14: 0.2→0.4 (用户)

    # ====== 各段直行帧数 ======
    STRAIGHT_1_FRAMES = 220
    STRAIGHT_2_FRAMES = 200
    STRAIGHT_3_FRAMES = 200 + 33  # 2026-08-11: +4帧≈5cm (用户: 窄道第五个距离+5cm, @0.4m/s 30fps)

    # ====== 各转弯帧数 ======
    TURN_1_FRAMES = 51
    TURN_2_FRAMES = 54  # 2026-08-03: 52→53→54 (还差一点)
    TURN_3_FRAMES = 54
    TURN_4_FRAMES = 59  # 2026-08-03: 58→59→60 (还差一点)
    TURN_5_FRAMES = 35  # 2026-08-04: 35→34 (少转一帧)

    # ====== 各转弯角速度 ======
    TURN_1_YAW = 0.9
    TURN_2_YAW = 0.9
    TURN_3_YAW = 0.9
    TURN_4_YAW = 0.9
    TURN_5_YAW = 0.9

    PAUSE_FRAMES = 45

    # ============================================================
    # 转弯5（最后转弯）看到黑线后的补偿参数
    # ============================================================
    FINAL_TURN_COMPENSATE_ENABLED = True
    FINAL_TURN_COMPENSATE_ANGLE = 8.0
    FINAL_TURN_COMPENSATE_YAW = 0.6
    FINAL_TURN_COMPENSATE_VX = 0.05
    FINAL_TURN_EXTRA_FORWARD = 0.0
    FINAL_TURN_EXTRA_FORWARD_SPEED = 0.15
    FINAL_TURN_CHECK_START_RATIO = 0.87

    # ============================================================
    # 转弯1 (TURN_1) 前的直行 STRAIGHT_1 - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_1_WALL_SLOW = 0.60
    STRAIGHT_1_WALL_STOP = 0.30
    STRAIGHT_1_TURN_TRIGGER = 0.30  # 2026-08-03: 直行到底 (原0.5太早转)

    # ============================================================
    # 转弯2 (TURN_2) 前的直行 STRAIGHT_1B - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_1B_WALL_SLOW = 0.60
    STRAIGHT_1B_WALL_STOP = 0.34  # 2026-08-03: 直行到底 (原0.42太早停)
    STRAIGHT_1B_TURN_TRIGGER = None

    # ============================================================
    # 转弯3 (TURN_3) 前的直行 STRAIGHT_2 (有侧移) - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_2_WALL_SLOW = 0.60
    STRAIGHT_2_WALL_STOP = 0.34  # 2026-08-03: 直行到底 (原0.38太早停)
    STRAIGHT_2_TURN_TRIGGER = 0.28  # 2026-08-03: 直行到底 (原0.40太早转)

    # ===========================================================
    # 转弯4 (TURN_4) 前的直行 STRAIGHT_2B - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_2B_WALL_SLOW = 0.74
    STRAIGHT_2B_WALL_STOP = 0.38  # 2026-08-03: 直行到底 (原0.38太早停)
    STRAIGHT_2B_TURN_TRIGGER = 0.28  # 2026-08-03: 直行到底 (原0.40太早转)

    # ============================================================
    # 转弯5 (TURN_5) 前的直行 STRAIGHT_3 (有侧移) - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_3_WALL_SLOW = 0.70
    STRAIGHT_3_WALL_STOP = 0.24
    STRAIGHT_3_TURN_TRIGGER = 0.28  # 2026-08-03: 直行到底 (原0.40太早转)

    # ====== 通用参数 ======
    WALL_CONFIRM_FRAMES = 5
    MIN_FRAMES_BEFORE_CHECK = 30

    # ====== 2026-08-03: 第一段直行左右墙距横向补偿 (防右偏) ======
    NARROW_SIDE_COMP_GAIN = 0.30   # 横向补偿增益 (m/s 每米墙距差)
    NARROW_SIDE_COMP_CAP = 0.12    # 横向补偿上限 (m/s)
    NARROW_LEFT_SHIFT_VY = 0.30    # 第一段直行开头左平移速度 (m/s, 正=左) (2026-08-06: 0.27→0.30, +0.03 用户)
    NARROW_LEFT_SHIFT_TIME = 0.4    # 2026-08-12: 0.55→0.4 (用户: 这个时间都改成0.4s); 第一/第五段共用



# ==================== ORB特征点识别配置 ====================
SIMILARITY_THRESHOLD = 13
MIN_GAP = 7
MAX_ACTIONS = 3
COOLDOWN_TIME = 2.0
PRINT_THRESHOLD = 10
CONFIRM_FRAMES = 3

CHESSBOARD_SIZE = (3, 3)  # 2026-08-03: 4x4棋盘格 = 3x3内角点 (若实际是4x4内角点改 (4,4))

# ===================================================================
# z0 嵌入参数区 (2026-08-03: 自 go2_z0.py 楼梯后流程嵌入)
# ===================================================================
Z0_GRASP_PROBE_THRESHOLD  = 6    # 匹配点数底线 (棋盘格检测下不用)
Z0_GRASP_STOP_THRESHOLD   = 20   # 识别点数阈值 (棋盘格检测下不用)
Z0_GRASP_STOP_HIGH        = 40   # 高置信点数 → 立即确认 (棋盘格检测直接命中此档)
Z0_GRASP_PROBE_FRAMES     = 2    # 迹象连续帧数才确认 (棋盘格检测直接高置信)
Z0_CHESS_INNER_CORNERS = (3, 3)  # 4×4 棋盘格 → 3×3 内角点
Z0_CHESS_CORNER_COUNT  = 100     # 检测到返回的点数(必过 HIGH=40 → 立即确认)
Z0_GRASP_TURN_ANGLE       = 112.2  # 2026-08-12: 109→112.2 (用户: grasp左转多3.2度)
Z0_BACKUP_SPEED           = 0.3  # 2026-08-12: 0.4→0.3 (用户: grasp后退速度减少; 0.3×2s=0.6m)
Z0_BACKUP_TIME            = 1.25  # 2026-08-15: 2.0→1.25 (用户: 时长1.25s; 0.3×1.25≈0.375m)
Z0_PLACE_DETECT_TIMEOUT   = 7.0  # 放置区识别超时(s), 超时随机选一个放置区
Z0_GRASP_FWD_TIME = 3.24  # 2026-08-12: 3.44→3.34→3.24 (用户: 识别到grasp再少走0.1s, 0.25×3.24≈0.81m)
Z0_GRASP_FWD_VX   = 0.25  # 前进速度(m/s)
Z0_GRASP_FWD_VY   = 0.15  # 右平移速度(m/s, 正值, 代码取负=向右) (2026-08-03: 0.20→0.15 慢一点)
Z0_GRASP_FWD_WZ   = 0.30  # 左转弯角速度(rad/s, 正值=左转)
Z0_TRANSFER_APPROACH_TIME = 11.0  # (2026-08-05 起不再作兜底, 仅旧GRASP2_APPROACH路径引用)
Z0_GRASP2_RECOG_ENABLE_DELAY = 2.0  # 中转阶段: 右转60°完成后再等N秒开始识别grasp (和第一次一样, 运动中识别)
Z0_TRANSFER_FWD_DURATION = 3.5   # (2026-08-05 起仅旧路径引用, 主逻辑改路程制)
Z0_TRANSFER_FWD_DECEL = 1.0      # (2026-08-05 起仅旧路径引用)
Z0_TRANSFER_FWD_DIST = 1.10  # 2026-08-10: 1.13→1.10 (用户: 中转前距离1.1m)
Z0_TRANSFER_FWD_DECEL_DIST = 0.3  # 2026-08-05: 最后0.3m线性减速 (原最后1s减速)
Z0_GRASP_ENABLE_DELAY     = 7.5   # 启动后延迟识别(s): 先循迹走一段再开grasp识别
Z0_SWEEP_LAYER1_RANGE = 15  # 中转第一层: 夹A前移搜索 angle0 ±15° (2026-08-04)
Z0_SWEEP_LAYER2_RANGE = 20  # 中转第二层: 放A后抓B失败 angle0 ±20° (2026-08-04)
Z0_SWEEP_COOLDOWN = 8.0      # 扫描失败冷却(s): 避免连续重扫 (2026-08-04)
Z0_TRANSFER_SEARCH_TIMEOUT = 180.0  # 2026-08-05晚: 20→180s, 让抓B重试跑满6次才放弃左转 (Z0_GRASP_B_MAX_ATTEMPTS=6)
Z0_GRASP_B_MAX_ATTEMPTS = 6     # 中转抓B最多尝试次数: 失败重新抓, 达上限放弃 (2026-08-04)
Z0_GRASP_B_MEM_MAX_ATTEMPTS = 3  # 2026-08-12 (用户): 容量确认失败后记忆位置直接重抓, 最多3次
Z0_GRASP_A_MAX_ATTEMPTS = 6     # 抓A最多尝试次数: 失败重新定位再抓, 达上限右转衔接 (2026-08-06, 用户要求同抓B)
Z0_GRASP_A_ROUNDS_MAX = 2     # 2026-08-11: 抓A整轮重试上限 (每轮含Z0_GRASP_A_MAX_ATTEMPTS次); 达上限按已夹持继续中转


# ==================== 红点检测配置 ====================
RED_HSV_LOWER1 = np.array([0, 50, 50])   # 2026-08-03: 适配闪光(过曝泛白)
RED_HSV_UPPER1 = np.array([10, 255, 255])
RED_HSV_LOWER2 = np.array([160, 50, 50]) # 2026-08-03: 适配闪光(过曝泛白)
RED_HSV_UPPER2 = np.array([180, 255, 255])
RED_MIN_RADIUS = 4   # 2026-08-03: 适配闪光(远处小光点)
RED_MAX_RADIUS = 120
RED_ROI_TOP = 350
RED_ROI_BOTTOM = 470


# ==================== Go2 步态模式 ====================
class GaitMode(Enum):
    FREE_WALK = "free_walk"
    CLASSIC_WALK = "classic_walk"
    TROT_RUN = "trot_run"
    STATIC_WALK = "static_walk"
    ECONOMIC_GAIT = "economic_gait"
    FREE_BOUND = "free_bound"
    FREE_JUMP = "free_jump"
    LEGACY_IDLE = "legacy_idle"
    LEGACY_TROT = "legacy_trot"
    LEGACY_TROT_RUN = "legacy_trot_run"
    LEGACY_STAIR_UP = "legacy_stair_up"
    LEGACY_STAIR_DOWN = "legacy_stair_down"

STAIRS_GAIT = GaitMode.LEGACY_STAIR_UP   # 爬楼梯步态: 官方STAIR_UP (SwitchGait 3, 2026-08-03 换回)
NORMAL_GAIT = GaitMode.FREE_WALK         # 普通巡线步态
USE_LEGACY_GAIT_API = False

def switch_gait(sport_client, gait_mode, enter=True):
    try:
        if USE_LEGACY_GAIT_API:
            legacy_map = {
                GaitMode.LEGACY_IDLE: 0, GaitMode.LEGACY_TROT: 1,
                GaitMode.LEGACY_TROT_RUN: 2, GaitMode.LEGACY_STAIR_UP: 3,
                GaitMode.LEGACY_STAIR_DOWN: 4,
            }
            if gait_mode in legacy_map:
                if enter:
                    sport_client.SwitchGait(legacy_map[gait_mode])
                    print(f"[步态] 🚶 切换到: {gait_mode.value}")
            else:
                print(f"[步态] ⚠️ 未知旧版步态: {gait_mode}")
        else:
            if gait_mode == GaitMode.FREE_WALK:
                if enter: sport_client.FreeWalk(); print("[步态] 🆓 FreeWalk")
            elif gait_mode == GaitMode.CLASSIC_WALK:
                sport_client.ClassicWalk(enter)
                print(f"[步态] {'🚶 进入' if enter else '⬅ 退出'} ClassicWalk")
            elif gait_mode == GaitMode.TROT_RUN:
                if enter: sport_client.TrotRun(); print("[步态] 🏃 TrotRun")
            elif gait_mode == GaitMode.STATIC_WALK:
                if enter: sport_client.StaticWalk(); print("[步态] 🐢 StaticWalk")
            elif gait_mode == GaitMode.ECONOMIC_GAIT:
                if enter: sport_client.EconomicGait(); print("[步态] 🔋 EconomicGait")
            elif gait_mode == GaitMode.FREE_BOUND:
                sport_client.FreeBound(enter)
                print(f"[步态] {'🦘 进入' if enter else '⬅ 退出'} FreeBound")
            elif gait_mode == GaitMode.FREE_JUMP:
                sport_client.FreeJump(enter)
                print(f"[步态] {'🤸 进入' if enter else '⬅ 退出'} FreeJump")
            else:
                print(f"[步态] ⚠️ 未知步态: {gait_mode}")
        return True
    except Exception as e:
        print(f"[步态] ❌ 切换失败: {e}")
        return False


# ==================== 标识定义 ====================
class SignID(Enum):
    ELECTRIC_SHOCK = 1
    OXIDIZER = 2
    RADIATION = 3
    GRASP = 4
    PLATFORM_A = 5   # 1号放置平台 (d1arm/place_zone/pattern_A)
    PLATFORM_B = 6   # 2号放置平台 (d1arm/place_zone/pattern_B)

SIGN_NAMES = {SignID.ELECTRIC_SHOCK: "当心触电", SignID.OXIDIZER: "当心强氧化物", SignID.RADIATION: "当心辐射", SignID.GRASP: "抓取位置", SignID.PLATFORM_A: "1号放置平台", SignID.PLATFORM_B: "2号放置平台"}
SIGN_FILES = {SignID.ELECTRIC_SHOCK: "electric_shock.jpg", SignID.OXIDIZER: "oxidizer.jpg", SignID.RADIATION: "radiation.jpg"}  # 2026-08-04: grasp图像已移除 (棋盘格替代)


# ==================== 状态常量 ====================
TRACKING = "tracking"
CORNER_APPROACH = "corner_approach"
CORNER_TURN = "corner_turn"
LOST_MEMORY = "lost_memory"
LOST_SEARCH = "lost_search"
LOST_STOP = "lost_stop"
RED_APPROACH = "red_approach"
PLACE_AFTER_RED = "place_after_red"  # 红点后循迹3.3s停车 → IK放置物块 (2026-08-02)
TURN_BACK = "turn_back"
STAIRS = "stairs"
BLUE_STOP = "blue_stop"
BLUE_GO_STRAIGHT = "blue_go_straight"
BLUE_TURN_LEFT = "blue_turn_left"
BLUE_FINAL_APPROACH = "blue_final_approach"
BLUE_SIT_DOWN = "blue_sit_down"
NARROW_APPROACH = "narrow_approach"
NARROW_EXECUTING = "narrow_executing"
JUMP = "jump"
POST_JUMP_ALIGN = "post_jump_align"
PLATFORM1_TURN = "platform1_turn"
PLATFORM1_FORWARD = "platform1_forward"
PLATFORM2_STOP = "platform2_stop"
GRASP_FORWARD = "grasp_forward"
GRASP_TURN = "grasp_turn"
GRASP_DONE = "grasp_done"
# 【集成】3D视觉抓取流程状态 (grasp_3d 管线分阶段集成)
GRASP_ARM_1 = "grasp_arm_1"        # 抓第一个黄色物块 (管线 phase='first')
GRASP2_TURN_RIGHT = "grasp2_turn_right" # 抓A后右转60° (抓上→识别grasp抓B; 5次失败→衔接循迹)
GRASP2_DETECT = "grasp2_detect"         # 右转60°后巡线相机识别grasp图案 (超时15s进抓取)
GRASP2_APPROACH = "grasp2_approach" # 抓A后循迹7s减速停止进行中转 (2026-08-02)
GRASP_ARM_2 = "grasp_arm_2"        # 中转: 放下物块A + 抓第二个黄色物块B
GRASP2_DONE = "grasp2_done"        # 全部完成, 停车等待

STAIRS_PHASE_FORWARD = 0
STAIRS_PHASE_TURN = 1
STAIRS_PHASE_SHIFT = 2


# ==================== 窄道路径状态机 ====================
class NarrowPathFSM:
    def __init__(self, tracker=None):
        self.tracker = tracker
        self.reset()

    def reset(self):
        self.frame_cnt = 0
        self.finished = False
        self.state = "STRAIGHT_1"
        self.state_enter_time = now_s()  # 2026-08-04: 状态进入时刻 (STRAIGHT_1 左平移改时间制)
        self.wall_confirm_cnt = 0
        self.is_slowing = False
        self.trigger_confirm_cnt = 0
        self.line_detected = False
        nc = NarrowConfig
        self.state_machine = {
            "STRAIGHT_1": self._act_straight_1(
                nc.STRAIGHT_1_FRAMES, "PAUSE_1",
                speed=nc.STRAIGHT_1_SPEED,
                wall_slow=nc.STRAIGHT_1_WALL_SLOW,
                wall_stop=nc.STRAIGHT_1_WALL_STOP,
                trigger_dist=nc.STRAIGHT_1_TURN_TRIGGER
            ),
            "PAUSE_1": self._act_pause("TURN_1"),
            "TURN_1": self._act_turn(nc.TURN_1_FRAMES, nc.TURN_1_YAW, "STRAIGHT_1B"),
            "STRAIGHT_1B": self._act_straight_fast(
                nc.STRAIGHT_1_FRAMES, "PAUSE_1B",
                speed=nc.STRAIGHT_1B_SPEED,
                wall_slow=nc.STRAIGHT_1B_WALL_SLOW,
                wall_stop=nc.STRAIGHT_1B_WALL_STOP,
                trigger_dist=nc.STRAIGHT_1B_TURN_TRIGGER
            ),
            "PAUSE_1B": self._act_pause("TURN_2"),
            "TURN_2": self._act_turn(nc.TURN_2_FRAMES, nc.TURN_2_YAW, "STRAIGHT_2"),
            "STRAIGHT_2": self._act_straight_side(
                nc.STRAIGHT_2_FRAMES, "PAUSE_2",
                side_distance=0.03, direction='right',
                speed=nc.STRAIGHT_2_SPEED,
                wall_slow=nc.STRAIGHT_2_WALL_SLOW,
                wall_stop=nc.STRAIGHT_2_WALL_STOP,
                trigger_dist=nc.STRAIGHT_2_TURN_TRIGGER,
                shift_vy=nc.SHIFT_AFTER_TURN2_VY, shift_time=nc.SHIFT_AFTER_TURN2_TIME  # 2026-08-06: 第二弯后开头右移
            ),
            "PAUSE_2": self._act_pause("TURN_3"),
            "TURN_3": self._act_turn(nc.TURN_3_FRAMES, -nc.TURN_3_YAW, "STRAIGHT_2B"),
            "STRAIGHT_2B": self._act_straight_fast(
                nc.STRAIGHT_2_FRAMES, "PAUSE_2B",
                speed=nc.STRAIGHT_2B_SPEED,
                wall_slow=nc.STRAIGHT_2B_WALL_SLOW,
                wall_stop=nc.STRAIGHT_2B_WALL_STOP,
                trigger_dist=nc.STRAIGHT_2B_TURN_TRIGGER
            ),
            "PAUSE_2B": self._act_pause("TURN_4"),
            "TURN_4": self._act_turn(nc.TURN_4_FRAMES, -nc.TURN_4_YAW, "STRAIGHT_3"),
            "STRAIGHT_3": self._act_straight_side(
                nc.STRAIGHT_3_FRAMES, "PAUSE_3",
                side_distance=0.03, direction='left',
                speed=nc.STRAIGHT_3_SPEED,
                wall_slow=nc.STRAIGHT_3_WALL_SLOW,
                wall_stop=nc.STRAIGHT_3_WALL_STOP,
                trigger_dist=nc.STRAIGHT_3_TURN_TRIGGER,
                shift_vy=0.0, shift_time=0.0  # 2026-08-12 (用户): 窄道第三个左平移(第五段开头)不要了
            ),
            "PAUSE_3": self._act_pause("TURN_5"),
            "TURN_5": self._act_turn_final(nc.TURN_5_FRAMES, nc.TURN_5_YAW),
            "COMPENSATE_TURN": self._act_compensate_turn(),
            "EXTRA_FORWARD": self._act_extra_forward(),
        }

    def _act_straight_1(self, total, next_state, speed=None, wall_slow=None, wall_stop=None, trigger_dist=None):
        if speed is None: speed = NarrowConfig.STRAIGHT_1_SPEED
        if wall_slow is None: wall_slow = NarrowConfig.STRAIGHT_1_WALL_SLOW
        if wall_stop is None: wall_stop = NarrowConfig.STRAIGHT_1_WALL_STOP
        if trigger_dist is None: trigger_dist = NarrowConfig.STRAIGHT_1_TURN_TRIGGER
        def act(front_dist):
            nc = NarrowConfig
            if self.frame_cnt >= total:
                print(f"\n[窄道-兜底] {total}帧 → {next_state}")
                self._switch(next_state)
                return 0, 0, 0
            if front_dist < wall_stop:
                self.wall_confirm_cnt += 1; self.is_slowing = True
                if self.wall_confirm_cnt >= nc.WALL_CONFIRM_FRAMES:
                    print(f"\n[窄道-碰墙] {front_dist:.3f}m (stop={wall_stop}m) → {next_state}")
                    self._switch(next_state)
                    return 0, 0, 0
            elif front_dist < wall_slow:
                self.wall_confirm_cnt += 1; self.is_slowing = True
            else:
                self.wall_confirm_cnt = max(0, self.wall_confirm_cnt - 1); self.is_slowing = False
            if trigger_dist is not None and self.frame_cnt >= nc.MIN_FRAMES_BEFORE_CHECK:
                if front_dist <= trigger_dist:
                    self.trigger_confirm_cnt += 1
                    if self.trigger_confirm_cnt >= 3:
                        print(f"\n[窄道-距离触发] {front_dist:.2f}m (trigger={trigger_dist}m, 帧:{self.frame_cnt}/{total}) → {next_state}")
                        self._switch(next_state)
                        return 0, 0, 0
                else:
                    self.trigger_confirm_cnt = max(0, self.trigger_confirm_cnt - 1)
                if front_dist <= trigger_dist: self.is_slowing = True
            self.frame_cnt += 1
            # 2026-08-04: 第一段直行开头左平移 0.26m/s × 0.5s (时间制), 之后按左右墙距差横向补偿 (防右偏); 2026-08-12 恢复 (用户: 删的是第三个, 不是第一个)
            vy = 0.0
            if now_s() - self.state_enter_time < nc.NARROW_LEFT_SHIFT_TIME:
                vy = nc.NARROW_LEFT_SHIFT_VY
            else:
                bal = getattr(self, 'side_balance', 0.0) or 0.0
                if bal != 0.0:
                    vy = max(-nc.NARROW_SIDE_COMP_CAP, min(nc.NARROW_SIDE_COMP_CAP, bal * nc.NARROW_SIDE_COMP_GAIN))
            if self.is_slowing: return speed, vy, 0  # 2026-08-03: 碰墙不减速 (照抄旧版)
            return speed, vy, 0
        return act

    def _act_straight_fast(self, total, next_state, speed=None, wall_slow=None, wall_stop=None, trigger_dist=None):
        if speed is None: speed = NarrowConfig.STRAIGHT_1B_SPEED
        if wall_slow is None: wall_slow = NarrowConfig.STRAIGHT_1B_WALL_SLOW
        if wall_stop is None: wall_stop = NarrowConfig.STRAIGHT_1B_WALL_STOP
        def act(front_dist):
            nc = NarrowConfig
            if trigger_dist is not None and self.frame_cnt >= nc.MIN_FRAMES_BEFORE_CHECK:
                if front_dist <= trigger_dist:
                    self.trigger_confirm_cnt += 1
                    if self.trigger_confirm_cnt >= 3:
                        print(f"\n[窄道-距离触发] {front_dist:.2f}m (trigger={trigger_dist}m, 帧:{self.frame_cnt}/{total}) → {next_state}")
                        self._switch(next_state)
                        return 0, 0, 0
                else: self.trigger_confirm_cnt = max(0, self.trigger_confirm_cnt - 1)
            if front_dist < wall_stop:
                self.wall_confirm_cnt += 1; self.is_slowing = True
                if self.wall_confirm_cnt >= nc.WALL_CONFIRM_FRAMES:
                    print(f"\n[窄道-碰墙] {front_dist:.3f}m (stop={wall_stop}m) → {next_state}")
                    self._switch(next_state)
                    return 0, 0, 0
            elif front_dist < wall_slow:
                self.wall_confirm_cnt += 1; self.is_slowing = True
            else:
                self.wall_confirm_cnt = max(0, self.wall_confirm_cnt - 1); self.is_slowing = False
            if self.frame_cnt >= total:
                print(f"\n[窄道-兜底] {total}帧 → {next_state}")
                self._switch(next_state)
                return 0, 0, 0
            self.frame_cnt += 1
            if self.is_slowing: return speed, 0, 0  # 2026-08-03: 碰墙不减速 (照抄旧版)
            return speed, 0, 0
        return act

    def _act_straight_side(self, total, next_state, side_distance=0.0, direction='right',
                           speed=None, wall_slow=None, wall_stop=None, trigger_dist=None,
                           shift_vy=0.0, shift_time=0.0):
        if speed is None: speed = NarrowConfig.STRAIGHT_2_SPEED
        if wall_slow is None: wall_slow = NarrowConfig.STRAIGHT_2_WALL_SLOW
        if wall_stop is None: wall_stop = NarrowConfig.STRAIGHT_2_WALL_STOP
        total_side_frames = total
        max_side_speed = 2.0 * side_distance / (total_side_frames * 0.01) if total_side_frames > 0 else 0
        half_frames = total_side_frames // 2
        def act(front_dist):
            nc = NarrowConfig
            if trigger_dist is not None and self.frame_cnt >= nc.MIN_FRAMES_BEFORE_CHECK:
                if front_dist <= trigger_dist:
                    self.trigger_confirm_cnt += 1
                    if self.trigger_confirm_cnt >= 3:
                        print(f"\n[窄道-距离触发] {front_dist:.2f}m (trigger={trigger_dist}m, 帧:{self.frame_cnt}/{total}) → {next_state}")
                        self._switch(next_state)
                        return 0, 0, 0
                else: self.trigger_confirm_cnt = max(0, self.trigger_confirm_cnt - 1)
            if front_dist < wall_stop:
                self.wall_confirm_cnt += 1; self.is_slowing = True
                if self.wall_confirm_cnt >= nc.WALL_CONFIRM_FRAMES:
                    print(f"\n[窄道-碰墙] {front_dist:.3f}m (stop={wall_stop}m) → {next_state}")
                    self._switch(next_state)
                    return 0, 0, 0
            elif front_dist < wall_slow:
                self.wall_confirm_cnt += 1; self.is_slowing = True
            else:
                self.wall_confirm_cnt = max(0, self.wall_confirm_cnt - 1); self.is_slowing = False
            if self.frame_cnt >= total:
                print(f"\n[窄道-兜底] {total}帧 → {next_state}")
                self._switch(next_state)
                return 0, 0, 0
            vy = 0.0
            if shift_time > 0 and now_s() - self.state_enter_time < shift_time:
                vy = shift_vy  # 2026-08-06: 转弯后直行开头固定平移 (覆盖侧移斜坡)
            elif total_side_frames > 0 and self.frame_cnt < total_side_frames:
                if self.frame_cnt < half_frames:
                    progress = self.frame_cnt / max(1, half_frames)
                    vy = max_side_speed * progress
                else:
                    progress = (total_side_frames - self.frame_cnt) / max(1, total_side_frames - half_frames)
                    vy = max_side_speed * progress
                if direction == "right": vy = -vy  # 2026-08-03: 符号修正 (vy正=左移, 原映射反了)
            self.frame_cnt += 1
            if self.is_slowing: return speed, vy, 0  # 2026-08-03: 碰墙不减速 (照抄旧版)
            return speed, vy, 0
        return act

    def _act_pause(self, next_state):
        def act(front_dist):
            if self.frame_cnt >= NarrowConfig.PAUSE_FRAMES:
                print(f"[窄道-停顿] → {next_state}")
                self._switch(next_state)
                return 0, 0, 0
            self.frame_cnt += 1
            return 0, 0, 0
        return act

    def _act_turn(self, total, yaw, next_state):
        def act(front_dist):
            if self.frame_cnt >= total:
                print(f"[窄道-转弯] → {next_state}")
                self._switch(next_state)
                return 0, 0, 0
            self.frame_cnt += 1
            return NarrowConfig.TURN_FWD_SPEED, 0, yaw
        return act

    def _act_turn_final(self, total, yaw):
        nc = NarrowConfig
        check_start = int(total * nc.FINAL_TURN_CHECK_START_RATIO)
        def act(front_dist):
            if self.frame_cnt >= total:
                print(f"[窄道-最后转弯] 转弯完成({total}帧)，检测黑线...")
                if self.tracker is not None and not self.line_detected:
                    color, depth = self.tracker.detector.get_frames()
                    if color is not None:
                        display, mask, centers = self.tracker.detector.detect_layers(color, depth)
                        avg_offset = self.tracker.compute_weighted_offset(centers)
                        if avg_offset is not None:
                            print(f"[窄道-最后转弯] 🎯 转弯完成时检测到黑线! 偏移:{avg_offset:.1f}px")
                            self.line_detected = True

                if self.line_detected:
                    print("[窄道-最后转弯] ➡️ 检测到黑线, 直接循迹!")
                self.finished = True  # 2026-08-05晚: 第五转弯完直接循迹 (补偿转弯已移除)
                return 0, 0, 0

            self.frame_cnt += 1

            if self.tracker is not None and self.frame_cnt >= check_start and not self.line_detected:
                color, depth = self.tracker.detector.get_frames()
                if color is not None:
                    display, mask, centers = self.tracker.detector.detect_layers(color, depth)
                    avg_offset = self.tracker.compute_weighted_offset(centers)
                    if avg_offset is not None:
                        print(f"[窄道-最后转弯] 🎯 检测到黑线! 偏移:{avg_offset:.1f}px (帧:{self.frame_cnt}/{total})")
                        self.line_detected = True
                        # 2026-08-05晚: 第五个转弯看到黑线直接循迹 (用户指定, 不再转完/补偿)
                        print("[窄道-最后转弯] ➡️ 看到黑线, 直接循迹!")
                        self.finished = True
                        return 0, 0, 0

            return NarrowConfig.TURN_FWD_SPEED, 0, yaw
        return act

    def _act_compensate_turn(self):
        nc = NarrowConfig
        angle_rad = np.radians(nc.FINAL_TURN_COMPENSATE_ANGLE)
        compensate_frames = int(angle_rad / (nc.FINAL_TURN_COMPENSATE_YAW * 0.01))

        def act(front_dist):
            if self.frame_cnt >= compensate_frames:
                print(f"[窄道-补偿] ✅ 补偿转弯完成 ({compensate_frames}帧, {nc.FINAL_TURN_COMPENSATE_ANGLE}°)")

                if nc.FINAL_TURN_EXTRA_FORWARD > 0:
                    extra_frames = int(nc.FINAL_TURN_EXTRA_FORWARD / (nc.FINAL_TURN_EXTRA_FORWARD_SPEED * 0.01))
                    print(f"[窄道-补偿] ➡️ 额外前进 {nc.FINAL_TURN_EXTRA_FORWARD*100:.0f}cm ({extra_frames}帧)...")
                    self._switch("EXTRA_FORWARD")
                    return nc.FINAL_TURN_EXTRA_FORWARD_SPEED, 0, 0

                self.finished = True
                return 0, 0, 0

            self.frame_cnt += 1

            if self.frame_cnt % 100 == 0:
                progress = self.frame_cnt / compensate_frames * 100
                print(f"[窄道-补偿] 转弯中... {progress:.0f}% ({self.frame_cnt}/{compensate_frames}帧)")

            return nc.FINAL_TURN_COMPENSATE_VX, 0, nc.FINAL_TURN_COMPENSATE_YAW
        return act

    def _act_extra_forward(self):
        nc = NarrowConfig
        extra_frames = int(nc.FINAL_TURN_EXTRA_FORWARD / (nc.FINAL_TURN_EXTRA_FORWARD_SPEED * 0.01))

        def act(front_dist):
            if self.frame_cnt >= extra_frames:
                print(f"[窄道-补偿] ✅ 额外前进完成")
                self.finished = True
                return 0, 0, 0

            self.frame_cnt += 1
            return nc.FINAL_TURN_EXTRA_FORWARD_SPEED, 0, 0
        return act

    def _switch(self, s):
        self.state = s; self.frame_cnt = 0; self.wall_confirm_cnt = 0; self.is_slowing = False; self.trigger_confirm_cnt = 0
        self.state_enter_time = now_s()  # 2026-08-04: 状态进入时刻 (STRAIGHT_1 左平移改时间制)

    def get_cmd(self, front_dist):
        if self.finished: return 0, 0, 0
        return self.state_machine[self.state](front_dist)


# ==================== 窄道检测器 ====================
class NarrowDetector:
    def __init__(self):
        self.narrow_confirm_cnt = 0
        self.depth_history = deque(maxlen=5)

    def detect_narrow(self, depth_img):
        if depth_img is None: return False, 0.0
        nc = NarrowConfig
        h, w = depth_img.shape
        roi_top = int(h * 0.5); roi_bottom = int(h * 0.85)
        roi = depth_img[roi_top:roi_bottom, :]
        if roi.size == 0: return False, 0.0
        depth_m = roi.astype(np.float32) * 0.001
        left_third = depth_m[:, :w//3]; center_third = depth_m[:, w//3:2*w//3]; right_third = depth_m[:, 2*w//3:]
        left_valid = left_third[(left_third > 0.1) & (left_third < 5.0)]
        center_valid = center_third[(center_third > 0.1) & (center_third < 5.0)]
        right_valid = right_third[(right_third > 0.1) & (right_third < 5.0)]
        if len(left_valid) < 20 or len(right_valid) < 20 or len(center_valid) < 20: return False, 0.0
        left_dist = np.median(left_valid); center_dist = np.median(center_valid); right_dist = np.median(right_valid)
        estimated_width = left_dist + right_dist
        self.depth_history.append(estimated_width)
        smooth_width = np.mean(self.depth_history)
        is_narrow = (nc.MIN_NARROW_WIDTH <= smooth_width <= nc.MAX_NARROW_WIDTH and left_dist < nc.NARROW_WALL_DIST and right_dist < nc.NARROW_WALL_DIST and center_dist > left_dist * 0.8 and center_dist > right_dist * 0.8)
        if is_narrow: self.narrow_confirm_cnt += 1
        else: self.narrow_confirm_cnt = max(0, self.narrow_confirm_cnt - 1)
        return self.narrow_confirm_cnt >= nc.NARROW_DETECT_FRAMES, smooth_width


# ==================== 前方雷达 ====================
class FrontRadar:
    def __init__(self):
        self.history = deque(maxlen=5)

    def get_front_dist(self, depth_img, narrow_mode=False):
        if depth_img is None: return 8.0
        h, w = depth_img.shape
        cx, cy = w // 2, h // 2
        samples = []
        if narrow_mode:
            for dx in range(-40, 41, 4):
                for dy in range(-30, 31, 4):
                    x = cx + dx; y = cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        d = depth_img[y, x] * 0.001
                        if 0.05 < d < 8.0: samples.append(d)
        else:
            for dx in [-15, -8, 0, 8, 15]:
                for dy in [-8, -4, 0, 4, 8]:
                    x = cx + dx; y = cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        d = depth_img[y, x] * 0.001
                        if 0.1 < d < 8.0: samples.append(d)

        if narrow_mode and len(samples) < 10:
            y1, y2 = max(0, cy-30), min(h, cy+30)
            x1, x2 = max(0, cx-50), min(w, cx+50)
            center_region = depth_img[y1:y2, x1:x2]
            center_valid = center_region[(center_region > 50) & (center_region < 8000)]
            if len(center_valid) > 10:
                dist = float(np.median(center_valid)) * 0.001
                self.history.append(dist)
                return float(np.mean(self.history))

        if samples:
            samples.sort()
            dist = samples[len(samples) // 2]
            self.history.append(dist)
            return float(np.mean(self.history))
        return 8.0


# ==================== 灯光控制 ====================
def blink_front_lights(vui_client, times=3):
    print("💡 执行动作: 闪烁前灯三次...")
    vui_client.SetBrightness(0)
    time.sleep(0.1)
    for i in range(times):
        print(f"   闪烁 {i+1}/{times}")
        vui_client.SetBrightness(10)
        time.sleep(0.3)
        vui_client.SetBrightness(0)
        time.sleep(0.2)
    vui_client.SetBrightness(0)
    print("✅ 闪烁完成，灯光已关闭")


# ==================== 动作函数 ====================
def do_stretch(sport_client):
    print("🐕 执行动作: 伸懒腰...")
    sport_client.Stretch()
    time.sleep(2)
    print("✅ 伸懒腰完成")

def do_greet(sport_client):
    print("👋 执行动作: 打招呼...")
    sport_client.Hello()
    time.sleep(2)
    print("✅ 打招呼完成")


# ==================== ORB识别器 ====================
class ORBRecognizer:
    def __init__(self, template_folder="templates"):
        self.templates = {}; self.keypoints = {}; self.descriptors = {}; self.templates_bottom = {}; self.platform_descriptors = {}
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.load_templates(template_folder)
        self.load_platform_templates()
        self.action_count = 0; self.last_action_time = 0; self.confirm_counter = 0; self.pending_sign_id = None

    def load_templates(self, folder):
        loaded_count = 0
        for sign_id, filename in SIGN_FILES.items():
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[sign_id] = img
                    h, w = img.shape; roi_h = int(h * 2 / 3); start_y = h - roi_h
                    img_bottom = img[start_y:start_y+roi_h, 0:w]; self.templates_bottom[sign_id] = img_bottom
                    kp, des = self.orb.detectAndCompute(img_bottom, None)
                    if des is not None and len(kp) > 10:
                        self.keypoints[sign_id] = kp; self.descriptors[sign_id] = des
                        print(f"✓ 加载: {SIGN_NAMES[sign_id]} (特征点: {len(kp)})"); loaded_count += 1
                    else: print(f"✗ 特征点太少: {SIGN_NAMES[sign_id]}")
            else: print(f"✗ 参考照片不存在: {filename}")
        if loaded_count == 0: print("\n❌ 未找到任何有效参考照片！"); return False
        print(f"✅ 成功加载 {loaded_count} 张参考照片"); return True

    def get_detection_zone(self, frame_shape): h, w = frame_shape[:2]; return 0, 0, w, int(h * 2 / 3)
    def extract_bottom_region(self, img): h, w = img.shape; roi_h = int(h * 2 / 3); start_y = h - roi_h; return img[start_y:start_y+roi_h, 0:w]

    def load_platform_templates(self):
        """加载放置平台图案多视角模板 (d1arm/place_zone/pattern_A|B)"""
        base = os.path.expanduser('~/go2_zong_project/d1_arm/place_zone')
        for sign_id, folder in [(SignID.PLATFORM_A, 'pattern_A'), (SignID.PLATFORM_B, 'pattern_B')]:
            d = os.path.join(base, folder)
            if not os.path.isdir(d):
                print(f"✗ 平台模板文件夹不存在: {d}"); continue
            des_list = []; count = 0
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith('.jpg'): continue
                img = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None and len(kp) > 10:
                    des_list.append(des); count += 1
            if des_list:
                self.platform_descriptors[sign_id] = des_list
                print(f"✓ 平台模板: {SIGN_NAMES[sign_id]} ({count}张)")
            else:
                print(f"✗ 平台模板特征点太少: {SIGN_NAMES[sign_id]}")

    def match_platform(self, frame_gray):
        """识别放置平台: patternA(1号)/patternB(2号), 多视角模板每类取最大匹配点数"""
        if not self.platform_descriptors: return None
        kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)
        if des_frame is None or len(kp_frame) < 10: return None
        best = None
        for sign_id, des_list in self.platform_descriptors.items():
            class_best = 0
            for des_template in des_list:
                matches = self.bf.knnMatch(des_template, des_frame, k=2)
                good = [m for m, n in matches if len(matches[0]) == 2 and m.distance < 0.75 * n.distance]
                if len(good) > class_best: class_best = len(good)
            if best is None or class_best > best[1]:
                best = (sign_id, class_best)
        if best and best[1] >= SIMILARITY_THRESHOLD:
            return best
        return None

    def match(self, frame_gray, min_threshold=SIMILARITY_THRESHOLD, min_gap=MIN_GAP):
        if not self.templates: return None
        start_x, start_y, zone_w, zone_h = self.get_detection_zone(frame_gray.shape)
        roi = frame_gray[start_y:start_y+zone_h, start_x:start_x+zone_w]
        if roi.size == 0: return None
        roi_bottom = self.extract_bottom_region(roi)
        kp_frame, des_frame = self.orb.detectAndCompute(roi_bottom, None)
        if des_frame is None or len(kp_frame) < 10: return None
        results = []
        for sign_id, des_template in self.descriptors.items():
            matches = self.bf.knnMatch(des_template, des_frame, k=2)
            good_matches = [m for m, n in matches if len(matches[0]) == 2 and m.distance < 0.75 * n.distance]
            match_count = len(good_matches); results.append((sign_id, match_count))
            if match_count >= PRINT_THRESHOLD: print(f"   📊 {SIGN_NAMES[sign_id]}: 匹配点数={match_count}")
        for sign_id, match_count in results:
            if sign_id == SignID.RADIATION and match_count > 20:
                print(f"   🚫 辐射匹配点数={match_count} > 20，忽略识别"); return None
        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            best_id, best_count = results[0]
            second_count = results[1][1] if len(results) > 1 else 0
            gap = best_count - second_count
            if best_count >= PRINT_THRESHOLD:
                print(f"   🏆 最佳: {SIGN_NAMES[best_id]} (匹配点数={best_count}) 差距:{gap}")
            if best_count >= min_threshold and gap >= min_gap:
                print(f"   ✅ 确认匹配: {SIGN_NAMES[best_id]}")
                return (best_id, best_count)
            elif best_count >= PRINT_THRESHOLD:
                print(f"   ❌ 未确认 (需要>{SIMILARITY_THRESHOLD}点, 差距>{MIN_GAP})")
        return None


# ==================== 检测器 ====================
# 【修复】循迹相机序列号 (狗上另一个深度相机 D435IF; 夹爪相机 335222075495 是抓取用的,
#          由 grasp_3d 管线 enable_device 锁定)。无锁启动时 pyrealsense2 取枚举第一个,
#          顺序变化会抢到夹爪相机导致启动偏移大/循迹异常, 故锁定。
LINE_CAM_SERIAL = "244222070235"


def usb_unbind_reset(serial, label='相机'):
    """2026-08-11: USB unbind/rebind 硬复位 — authorized 复位无效时强制重新枚举.
    比 authorized 开关更强的设备级复位 (拔出式重枚举)."""
    import subprocess, re
    port = None
    try:
        for d in rs.context().query_devices():
            if d.get_info(rs.camera_info.serial_number) == serial:
                m = re.search(r'([\d.]+-[\d.]+):[\d.]+',
                              d.get_info(rs.camera_info.physical_port))
                if m:
                    port = m.group(1)
                break
    except Exception as e:
        print(f'  [{label}] unbind 定位失败: {e}')
    if not port:
        print(f'  [{label}] unbind 未定位到设备, 跳过')
        return False
    try:
        p = subprocess.run(['sudo', '-S', 'sh', '-c',
                            f'echo {port} > /sys/bus/usb/drivers/usb/unbind; '
                            f'sleep 2; echo {port} > /sys/bus/usb/drivers/usb/bind'],
                           input='123\n', text=True, capture_output=True, timeout=15)
        if p.returncode == 0:
            print(f'  [{label}] USB unbind/rebind 硬复位成功 ({port})')
            return True
        print(f'  [{label}] unbind 复位失败: {p.stderr.strip()[:120]}')
    except Exception as e:
        print(f'  [{label}] unbind 复位异常: {e}')
    return False


def usb_authorized_reset(serial, label='相机'):
    """2026-08-10: USB authorized 复位 (设备级, sudo 密码约定同 go2_check).
    掉线坏态 (errno=5) 重建 pipeline 救不回, 复位强制重新枚举.
    定位用 rs physical_port (本机 sysfs serial ≠ 相机SN, 不能按 sysfs 匹配)"""
    import subprocess, re
    port = None
    try:
        for d in rs.context().query_devices():
            if d.get_info(rs.camera_info.serial_number) == serial:
                m = re.search(r'([\d.]+-[\d.]+):[\d.]+',
                              d.get_info(rs.camera_info.physical_port))
                if m:
                    port = m.group(1)
                break
    except Exception as e:
        print(f'  [{label}] USB 复位定位失败: {e}')
    if not port:
        print(f'  [{label}] USB 复位未定位到设备, 跳过')
        return False
    try:
        p = subprocess.run(['sudo', '-S', 'sh', '-c',
                            f'echo 0 > /sys/bus/usb/devices/{port}/authorized; '
                            f'sleep 1; echo 1 > /sys/bus/usb/devices/{port}/authorized'],
                           input='123\n', text=True, capture_output=True, timeout=10)
        if p.returncode == 0:
            print(f'  [{label}] USB authorized 复位成功 ({port})')
            return True
        print(f'  [{label}] USB 复位失败: {p.stderr.strip()[:120]}')
    except Exception as e:
        print(f'  [{label}] USB 复位异常: {e}')
    return False


class LineDetector:
    def __init__(self, width=640, height=480, num_layers=10):
        self.width = width; self.height = height; self.num_layers = num_layers
        self.roi_top = 400; self.roi_bottom = height; self.roi_height = self.roi_bottom - self.roi_top
        self.roi_left = 130; self.roi_right = 510; self.roi_width = self.roi_right - self.roi_left
        self.stair_roi_top = max(0, self.roi_top - self.roi_height); self.stair_roi_bottom = self.roi_top
        self.stair_roi_height = self.stair_roi_bottom - self.stair_roi_top
        self.pipeline = rs.pipeline(); self.config = rs.config()
        self.config.enable_device(LINE_CAM_SERIAL)  # 【修复】锁定循迹相机, 防枚举顺序抢到夹爪相机
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.black_threshold = 50; self.min_contour_area_min = 30; self.min_contour_area_max = 150
        self.use_depth_filter = True
        self.depth_min = 0.2
        self.depth_max = 2.0
        self.black_threshold_high = 80
        self.kernel_close = np.ones((5,5), np.uint8); self.kernel_open = np.ones((3,3), np.uint8)
        self.layer_weights = np.linspace(0.5, 1.0, num_layers)
        self.red_detect_roi_top = 350; self.red_detect_roi_bottom = 470

    def start(self): self.profile = self.pipeline.start(self.config); return True
    def stop(self):
        # 2026-08-10: 防双异常 (掉线后 pipeline 未启动时 stop 抛错会掩盖真因)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass

    def _rebuild_pipeline(self):
        """2026-08-10: 循迹相机掉线自恢复 — 重建 pipeline (同抓取相机 v39.14 模式)"""
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = rs.pipeline(); self.config = rs.config()
        self.config.enable_device(LINE_CAM_SERIAL)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)

    def _usb_authorized_reset(self):
        """2026-08-10: 循迹相机 USB authorized 复位 (模块级 usb_authorized_reset 封装)"""
        usb_authorized_reset(LINE_CAM_SERIAL, '循迹相机')

    def _usb_unbind_reset(self):
        """2026-08-11: 循迹相机 unbind/rebind 硬复位 (模块级 usb_unbind_reset 封装)"""
        usb_unbind_reset(LINE_CAM_SERIAL, '循迹相机')

    def get_frames_quick(self, timeout_ms=800):
        """2026-08-11: 单次快速取帧 (不触发恢复阶梯) — 供转弯/衔接等关键时序 handler 使用,
        防止 30s 恢复阻塞吃掉转弯窗口/狗盲走. 失败返回 (None, None)."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            aligned = self.align.process(frames)
            color = aligned.get_color_frame(); depth = aligned.get_depth_frame()
            if not color or not depth: return None, None
            return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
        except Exception:
            return None, None

    def get_frames(self):
        """2026-08-10: 取帧带自恢复 (bak_0806_linedetect_recover 补回) —
        USB 偶发掉线 (errno=5/Device disconnected) 重试3次, 仍失败重建 pipeline,
        掉线不再中断整个流程"""
        for round_ in range(2):
            for attempt in range(3):
                try:
                    frames = self.pipeline.wait_for_frames()
                    aligned = self.align.process(frames)
                    color = aligned.get_color_frame(); depth = aligned.get_depth_frame()
                    if not color or not depth: return None, None
                    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
                except Exception as e:
                    print(f'  [循迹相机] 取帧失败 (r{round_+1} a{attempt+1}/3): {e}')
                    time.sleep(1.0)
            print('  [循迹相机] 连续取帧失败, 重建 pipeline...')
            try:
                self._rebuild_pipeline()
                continue  # 重建成功 → 下一轮在新 pipeline 上取帧
            except Exception as e2:
                print(f'  [循迹相机] 重建失败: {e2}')
            # 2026-08-11: 重建失败 → 硬件复位 + 等重枚举 + 立即重建(start) 再进下一轮
            # (原逻辑复位后直接 wait_for_frames, pipeline 未 start → 纯浪费一轮)
            if round_ == 0:
                self._usb_authorized_reset()  # 2026-08-10: 设备级授权复位
            else:
                self._usb_unbind_reset()      # 2026-08-11: 授权复位无效 → unbind/rebind 硬复位
            time.sleep(3.0)                   # 2026-08-11: 等设备重新枚举
            try:
                print('  [循迹相机] 复位后重建 pipeline...')
                self._rebuild_pipeline()
            except Exception as e3:
                print(f'  [循迹相机] 复位后重建失败: {e3}')
            time.sleep(2.0)
        # 2026-08-10 (评审点5): 恢复旧契约返回 None — 彻底失败不 raise,
        # 各调用点按"无帧"处理 (狗保持当前状态不猝死)
        print('  [循迹相机] ⚠️ 恢复失败, 返回无帧 — 流程按无帧继续')
        return None, None

    def _process_roi_mask(self, roi_color, roi_depth, roi_left, roi_top, threshold, num_layers_override=None):
        layers = num_layers_override if num_layers_override is not None else self.num_layers
        roi_h = roi_color.shape[0]
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, roi_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        if self.use_depth_filter and roi_depth is not None:
            depth_m = roi_depth * 0.001; valid = (depth_m > self.depth_min) & (depth_m < self.depth_max) & (depth_m > 0)
            depth_mask = np.zeros_like(depth_m, dtype=np.uint8); depth_mask[valid] = 255
            roi_mask = cv2.bitwise_and(roi_mask, depth_mask)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_close)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_open)
        layer_height = roi_h // layers
        if layer_height == 0: layer_height = 1
        layer_centers = []
        for i in range(layers):
            y_start = i * layer_height; y_end = (i+1) * layer_height if i < layers-1 else roi_h
            layer_mask = roi_mask[y_start:y_end, :]
            current_min_area = self.min_contour_area_min + (self.min_contour_area_max - self.min_contour_area_min) * i / (layers - 1) if layers > 1 else self.min_contour_area_min
            contours, _ = cv2.findContours(layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center_x_global = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > current_min_area:
                    M = cv2.moments(largest)
                    if M["m00"] > 0: cx_roi = int(M["m10"] / M["m00"]); center_x_global = cx_roi + roi_left
            layer_centers.append(center_x_global)
        return roi_mask, layer_centers

    def _process_roi_widths(self, roi_color, roi_depth, roi_left, roi_top, threshold, num_layers_override=None):
        layers = num_layers_override if num_layers_override is not None else self.num_layers
        roi_h = roi_color.shape[0]
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, roi_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        if self.use_depth_filter and roi_depth is not None:
            depth_m = roi_depth * 0.001; valid = (depth_m > self.depth_min) & (depth_m < self.depth_max) & (depth_m > 0)
            depth_mask = np.zeros_like(depth_m, dtype=np.uint8); depth_mask[valid] = 255
            roi_mask = cv2.bitwise_and(roi_mask, depth_mask)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_close)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_open)
        layer_height = roi_h // layers
        if layer_height == 0: layer_height = 1
        widths = []
        for i in range(layers):
            y_start = i * layer_height; y_end = (i+1) * layer_height if i < layers-1 else roi_h
            layer_mask = roi_mask[y_start:y_end, :]
            current_min_area = self.min_contour_area_min + (self.min_contour_area_max - self.min_contour_area_min) * i / (layers - 1) if layers > 1 else self.min_contour_area_min
            contours, _ = cv2.findContours(layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            width = 0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > current_min_area:
                    rect = cv2.boundingRect(largest); width = rect[2]
            widths.append(width)
        return widths

    def detect_layers(self, color_img, depth_img):
        display = color_img.copy() if SHOW_GUI else None
        h, w = color_img.shape[:2]
        roi_color = color_img[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        roi_depth = depth_img[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] if self.use_depth_filter else None
        roi_mask, layer_centers = self._process_roi_mask(roi_color, roi_depth, self.roi_left, self.roi_top, self.black_threshold)
        if SHOW_GUI:
            layer_height = self.roi_height // self.num_layers
            for i, cx in enumerate(layer_centers):
                if cx is not None:
                    y_global = self.roi_top + i * layer_height + layer_height//2
                    cv2.circle(display, (cx, y_global), 5, (0,255,0), -1)
            cv2.rectangle(display, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (255,255,0), 2)
            cv2.line(display, (w//2, self.roi_top), (w//2, self.roi_bottom), (255,255,255), 1)
            for i in range(1, self.num_layers):
                y_line = self.roi_top + i * layer_height; cv2.line(display, (self.roi_left, y_line), (self.roi_right, y_line), (255,255,0), 1)
            cv2.rectangle(display, (self.roi_left, self.stair_roi_top), (self.roi_right, self.stair_roi_bottom), (0,255,255), 1)
        return display, roi_mask, layer_centers

    def detect_stair_widths(self, color_img, depth_img):
        roi_color = color_img[self.stair_roi_top:self.stair_roi_bottom, self.roi_left:self.roi_right]
        roi_depth = depth_img[self.stair_roi_top:self.stair_roi_bottom, self.roi_left:self.roi_right] if self.use_depth_filter else None
        if roi_color.shape[0] == 0: return [0] * self.num_layers
        return self._process_roi_widths(roi_color, roi_depth, self.roi_left, self.stair_roi_top, self.black_threshold_high)

    def detect_stair_depth_block(self, depth_img):
        """2026-08-04: 深度相机黑色块 — 楼梯ROI内无效深度(0)或极近(<0.2m)像素占比.
        平地占比≈0; 楼梯到达=大块黑色块 → 占比突变"""
        if depth_img is None:
            return 0.0
        h, w = depth_img.shape
        y1 = self.stair_roi_top; y2 = min(self.stair_roi_bottom, h)
        x1 = self.roi_left; x2 = min(self.roi_right, w)
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        dark = (roi <= 0) | (roi < 200)  # 无效深度或极近 = 深度视图黑色块
        return float(dark.mean())

    def detect_stair_depth_stats(self, depth_img):
        """2026-08-08: 楼梯ROI深度统计 (方案C) — 返回 (黑块占比, 最小有效深度m).
        黑块=无效(0)或极近(<0.2m), 平地≈0, 楼梯到达=涨; min=有效深度最小值(米),
        平地≈0.5m+, 台阶顶面/立面接近时显著变小. 无有效像素 min 返回 None"""
        if depth_img is None:
            return 0.0, None
        h, w = depth_img.shape
        y1 = self.stair_roi_top; y2 = min(self.stair_roi_bottom, h)
        x1 = self.roi_left; x2 = min(self.roi_right, w)
        roi = depth_img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, None
        roi = roi.astype(np.float32)
        dark = (roi <= 0) | (roi < 200)
        valid = ~dark
        block = float(dark.mean())
        min_d = float(roi[valid].min() / 1000.0) if valid.any() else None
        return block, min_d

    def fit_line_heading(self, color_img):
        """ROI上方30%区域采集7个边界点，最小二乘法拟合黑线走向，返回斜率(dx/dy)"""
        top_h = max(1, int(self.roi_height * 0.3))
        roi = color_img[self.roi_top:self.roi_top + top_h, self.roi_left:self.roi_right]
        if roi.shape[0] == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)

        # 7个均匀采样行，每行取黑线中心点
        h = roi.shape[0]
        n_pts = 7
        points = []
        for i in range(n_pts):
            y = int(h * (i + 0.5) / n_pts)
            row = mask[y, :]
            black_pixels = np.where(row > 0)[0]
            if len(black_pixels) > 10:  # 至少10个黑像素才有效
                cx = np.median(black_pixels)  # 黑线中心
                gx = cx + self.roi_left
                gy = y + self.roi_top
                points.append((gx, gy))

        if len(points) < 3:
            return None

        # 最小二乘拟合：x = a*y + b
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        A = np.vstack([ys, np.ones_like(ys)]).T
        a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

        return {'slope': float(a), 'intercept': float(b), 'points': points}

    def get_first_layer_status(self, centers):
        if centers is None or len(centers) == 0: return False, None
        return centers[0] is not None, centers[0]

    def analyze_black_line_pattern(self, centers):
        top_layers = centers[:3]; mid_layers = centers[3:7]; bottom_layers = centers[7:]
        top_has = any(c is not None for c in top_layers); mid_has = any(c is not None for c in mid_layers); bottom_has = any(c is not None for c in bottom_layers)
        top_centers = [c for c in top_layers if c is not None]; bottom_centers = [c for c in bottom_layers if c is not None]
        top_center = np.mean(top_centers) if top_centers else None; bottom_center = np.mean(bottom_centers) if bottom_centers else None
        return {'top_has_line': top_has, 'middle_has_line': mid_has, 'bottom_has_line': bottom_has, 'top_center': top_center, 'bottom_center': bottom_center, 'is_broken': top_has and not mid_has and bottom_has, 'only_top': top_has and not mid_has and not bottom_has}

    def detect_wide_roi(self, color_img, depth_img):
        wide_top = int(self.height * 0.3); wide_bottom = self.height
        wide_left = int(self.width * 0.05); wide_right = int(self.width * 0.95)
        roi_color = color_img[wide_top:wide_bottom, wide_left:wide_right]
        roi_depth = depth_img[wide_top:wide_bottom, wide_left:wide_right] if depth_img is not None else None
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, roi_mask = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        if self.use_depth_filter and roi_depth is not None:
            depth_m = roi_depth.astype(np.float32) * 0.001; valid = (depth_m > self.depth_min) & (depth_m < self.depth_max) & (depth_m > 0)
            depth_mask = np.zeros_like(depth_m, dtype=np.uint8); depth_mask[valid] = 255; roi_mask = cv2.bitwise_and(roi_mask, depth_mask)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_close); roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_open)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return {"found": False, "offset_ratio": 0.0, "direction": 0.0}
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100: return {"found": False, "offset_ratio": 0.0, "direction": 0.0}
        M = cv2.moments(largest)
        if M["m00"] <= 0: return {"found": False, "offset_ratio": 0.0, "direction": 0.0}
        cx = M["m10"] / M["m00"]; roi_w = wide_right - wide_left
        offset_ratio = float(np.clip((cx - roi_w * 0.5) / (roi_w * 0.5), -1.0, 1.0))
        return {"found": True, "offset_ratio": offset_ratio, "direction": 1.0 if offset_ratio > 0 else -1.0}

    def detect_red_point(self, color_img):
        roi_red = color_img[self.red_detect_roi_top:self.red_detect_roi_bottom, :]
        if roi_red.size == 0: return False, 0, None
        hsv = cv2.cvtColor(roi_red, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
        mask2 = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel); red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea); area = cv2.contourArea(largest); radius = int(np.sqrt(area / np.pi))
            if RED_MIN_RADIUS < radius < RED_MAX_RADIUS:
                M = cv2.moments(largest)
                if M["m00"] > 0: cx_roi = int(M["m10"] / M["m00"]); cy_roi = int(M["m01"] / M["m00"]); return True, radius, (cx_roi, self.red_detect_roi_top + cy_roi)
        return False, 0, None

    def detect_blue_stop_area(self, color_img):
        h, w = color_img.shape[:2]; roi_top = int(h * 0.6); roi_bottom = h; roi = color_img[roi_top:roi_bottom, :]
        if roi.size == 0: return False, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 80, 80]); upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel); blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_pixels = cv2.countNonZero(blue_mask); roi_pixels = roi.shape[0] * roi.shape[1]
        blue_ratio = blue_pixels / roi_pixels if roi_pixels > 0 else 0
        if blue_ratio > 0.15: return True, blue_ratio
        return False, blue_ratio

    def check_cutoff(self, color_img, depth_img=None):
        """截断检测：窄ROI聚焦黑线，绝对比例判断"""
        h, w = color_img.shape[:2]
        # 收窄ROI，聚焦黑线区域（与jump_test一致）
        roi_y_top = h - 120
        roi_y_bottom = h - 40
        roi_x_center = w // 2
        roi_width_half = 40  # 收窄到80px宽
        roi_left = roi_x_center - roi_width_half
        roi_right = roi_x_center + roi_width_half

        roi = color_img[roi_y_top:roi_y_bottom, roi_left:roi_right]
        if SHOW_GUI:
            cv2.rectangle(color_img, (roi_left, roi_y_top), (roi_right, roi_y_bottom), (0, 0, 255), 2)
            cv2.line(color_img, (roi_x_center, roi_y_top), (roi_x_center, roi_y_bottom), (255, 0, 0), 1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        black_ratio = np.sum(black_mask > 0) / black_mask.size
        is_cutoff = black_ratio < 0.70
        return is_cutoff, black_ratio, False, 0.0


# ==================== 运动控制器 ====================
class Go2SegmentTracker:
    def __init__(self, interface="eth0", num_layers=10):
        ChannelFactoryInitialize(0, interface)
        self.sport = SportClient()
        self.sport.SetTimeout(5.0)
        self.sport.Init()
        print("[运动] 狗神降临...")
        self.sport.WaitLeaseApplied()

        print("[运动] 机器人保持趴卧状态，等待相机预热...")

        self.vui_client = VuiClient()
        self.vui_client.SetTimeout(1.0)
        self.vui_client.Init()
        self.vui_client.SetBrightness(0)

        self.detector = LineDetector(num_layers=num_layers)
        if not self.detector.start():
            sys.exit(1)

        # 保存原始ROI，记忆转弯时缩小一半，找到线后恢复
        self._orig_roi = {
            'top': self.detector.roi_top, 'bottom': self.detector.roi_bottom,
            'left': self.detector.roi_left, 'right': self.detector.roi_right,
            'height': self.detector.roi_height, 'width': self.detector.roi_width,
        }

        self.narrow_detector = NarrowDetector()
        self.narrow_fsm = NarrowPathFSM(tracker=self)
        self.front_radar = FrontRadar()
        self.narrow_enabled = True
        self.narrow_triggered = False
        self.stairs_allowed = False
        self.stairs_pending = 0.0          # 窄道完成后的楼梯触发时刻 (2026-08-03, 已弃用改距离门控)
        self.stairs_gate_dist_need = 0.4   # 2026-08-07: 门检测距离 (用户: 爬楼梯前循迹0.4m)
        self.stairs_gate_dist = 0.0        # 窄道后累计循迹距离
        self._stairs_gate_last_t = 0.0     # 距离积分上一帧时刻
        self.narrow_done_at = 0.0          # 窄道完成时刻 (2026-08-04: 1s后左平移)

        # ====== 基本运动参数 ======
        self.base_speed = 0.3
        self.max_rotation = 0.8
        self.max_vy = 0.2           # 平移纠偏最大侧移速度

        # 2026-08-10: 循迹速度回 0.5/0.38 (用户: 调大0.1后改回; 08-09 起的值)
        self.far_layer_normal_speed = 0.5
        self.far_layer_lost_speed = 0.38
        self.far_layer_lost_count = 0


        # ====== PID控制参数 ======
        self.kp = 0.0048
        self.ki = 0.0001
        self.kd = 0.0052
        self.integral = 0.0
        self.last_error = 0.0
        self.max_integral = 30.0
        self.last_wz = 0.0
        self.filter_alpha = 0.25
        self.dead_zone = 8

        # ====== 直角转弯参数 ======
        self.pre_turn_duration = 1.75  # 补偿直行 0.5m @ 0.30m/s
        self.pre_turn_speed = 0.25     # 慢速补偿直行
        self.corner_confirm_frames = 2
        self.corner_confirm_count = 0
        self.corner_direction = 0.0
        self.turn90_yaw = 0.9          # 固定90°转角速度 (0.9rad/s × 1.75s ≈ 90°)
        self.turn90_duration = 1.67    # 固定90°时长
        self.turn90_vx = 0.065          # 2026-08-10: 0.3→0.065 (用户: 恢复 GitHub 86c3f87 值, 直角转弯慢速)
        self.corner_max_time = 10.0
        self.corner_cooldown = 1.5

        # ====== 丢失恢复参数 ======
        self.transient_lost_frames = 4.5
        self.lost_memory_straight_dur = 1.75  # 补偿直行0.4m @ 0.24m/s
        self.lost_memory_straight_vx = 0.20
        self.lost_memory_turn_dur = 1.75      # 固定90° @ 0.9rad/s
        self.lost_memory_turn_yaw = 0.9
        self.lost_memory_turn_vx = 0.06  # 2026-08-10: 0.3→0.06 (用户: 恢复 GitHub 86c3f87 值, 丢线转弯慢速)
        self.lost_search_time = 13.0
        self.lost_search_vx = 0.15
        self.lost_search_yaw_base = 1.2
        self.lost_search_yaw_max = 1.4
        self.lost_search_yaw_ramp = 0.08

        self.first_lost_handled = False

        # ====== 红点接近参数 ======
        self.approach_speed = 0.47  # 2026-08-12: 0.45→0.47 (用户: 再大0.02, 距离不变8cm/步自动换算)
        self.red_search_right_yaw = 0.25  # 2026-08-11: 红点右转99°后丢线往右搜索角速度(rad/s) (用户)
        self.target_radius = 38
        self.approach_timeout = 12.0
        self.compensate_distance = 0.80# 2026-08-12: 0.79→0.80 (用户: 循迹补偿多1cm)
        self.compensate_speed = 0.27

        # ====== 转向序列参数 ======
        self.turn_step_angle = 25
        self.turn_step_time = 0.4   # 2026-08-12: 0.6→0.4 (用户: 红点转弯提速, 与speed同比例保证角度不变)
        self.turn_speed = 1.05      # 2026-08-12: 0.7→1.05 (1.5×, 每步0.42rad不变)
        self.red_turn_left_reduce_deg = 5.0   # 2026-08-08: 3→5 (用户: 左转再少2°, 共少5°)
        self.red_turn_right_reduce_deg = 2.0  # 2026-08-06: 红点右转少2°
        self.backup_speed = -0.4   # 2026-08-12: -0.23→-0.4 (用户)
        self.backup_step_time = 0.32  # 2026-08-12: 0.6→0.32s (用户)
        self.backup_steps = 3
        self.align_speed = 0.35
        self.align_timeout = 2.0

        # ====== 楼梯触发参数 ======
        self.width_trigger_ratio = 1.5  # 2026-08-10: 注释修正 — 实际值 1.5 (历史注释 2.0 过期; 1.5 楼梯实测正常, 勿改值)
        self.width_sample_layers = 3
        self.width_history_len = 30
        self.normal_widths_high = []
        self.stairs_trigger_enabled = True
        self.stairs_triggered_once = False

        # ====== 2026-08-08: 深度距离门控 (方案C) — 深度信号提前抬头/确认楼梯 ======
        self.stairs_depth_enabled = True    # 深度门控总开关
        self.stairs_depth_headup_gap = 0.15 # 最小有效深度比平地基线低≥0.15m → 提前抬头 (阈值待实测校准)
        self.stairs_depth_enter_m = 0.35    # 最小有效深度 <0.35m → 深度确认到楼梯, 0.4m门控后即进楼梯模式
        self.stairs_depth_block_enter = 0.30# 黑块占比 ≥0.35 → 同确认 (近距盲区/台阶立面)
        self.stairs_depth_block_max = 0.9   # 2026-08-08: 已废弃不参与判定 (全黑=近距盲区强信号, 不再掐死)
        self.stairs_gate_fallback_extra = 0.6 # 0.4m循迹后仍无深度确认, 再走0.6m保险兜底进 (防深度流故障卡死)
        self._stairs_min_depth_hist = deque(maxlen=20)  # 平地 min有效深度基线 (楼梯出现后冻结)
        self._stairs_depth_log_t = 0.0      # 深度门控日志节流时间戳

        self.stairs_forward_duration = 7.2  # 2026-08-08: 7.45→7.2s (用户恢复)
        self.stairs_fwd_left_turn_wz = 0.3    # 2026-08-06: 第一段直行开头左转(偏航)角速度 (rad/s)
        self.stairs_fwd_left_turn_time = 0.2   # 2026-08-08: 0.8→0.2s (用户; 0.3rad/s×0.2s≈3.4°)
        # 2026-08-08: 直行附加左偏航已删除 (用户: 去掉直行左偏航), 仅保留开头左转
        self.stairs_turn_left_px_thresh = 15   # 2026-08-06: 转弯中线偏左超过此px才跟随 (只看左边, 右边黑线忽略)
        self.stairs_turn_follow_gain = 0.002   # 2026-08-06: 左侧线跟随增益 (px -> rad/s)
        self.stairs_turn_omega_max = 1.5       # 2026-08-06: 跟随后角速度上限 (rad/s)
        self.stairs_head_up_pitch = 0.0  # 2026-08-08: 彻底关闭 — 实测 SportClient.Euler 与 Move 互斥 (发Euler→直行断+头不抬, 0.1/0.25 均如此; App摆姿态≠SDK Euler); 姿态方案废弃, 防蹭靠拆L1或ClassicWalk天然姿态; 0=关闭
        self.stairs_pitch = 0.0           # 当前生效俯仰 (直行开始置位, 转弯前/退出清零)
        self._stairs_pitch_sent = False   # 本次爬楼梯是否已发过后仰
        self._stairs_pitch_last = 0.0     # 最近一次重发后仰的时间戳
        self.stairs_fwd_left_shift_vy = 0.3    # 2026-08-08: 0.34→0.3 (用户): 0.5×0.3=15cm
        self.stairs_fwd_left_shift_time = 0.5  # 2026-08-08: 0.667→0.5s (用户: 恢复之前时长, 速度加大补足20cm)
        self.stairs_fwd_left_shift_delay = 0.5 # 2026-08-07: 先直行走0.5s再左平移 (用户: 不放开头)
        self.narrow_exit_shift_delay = 1.0     # 2026-08-04: 窄道完成后延迟(s)开始左平移 (替代楼梯识别门控) (0.8→1.0)
        self.stairs_turn_overlap = 0.0  # 2026-08-08: 归零 — stairs_turn_duration 直接等于实际转动时间
        self.stairs_forward_speed = 0.32  # 2026-08-08: 0.3→0.32 (用户: 恢复之前)
        self.stairs_forward_yaw = 0.18    # 直行时微左偏角速度
        self.stairs_turn_duration = 2.0  # 2026-08-08: 1.75→2.0s (用户: ω回0.785联动, 90°/2.0s)
        self.stairs_turn_vy0 = 0.6      # 2026-08-08: 恢复左移版 0.6 (用户)
        self.stairs_turn_omega = 0.785  # 2026-08-08: 0.943→0.785 (用户: 回到最开始的角速度, 90°/2.0s)
        self.stairs_turn_comp_time = 0.5  # 2026-08-08: 转完后前移补偿时长(s) (用户: 转完机器狗偏后; ≈0.32×0.5=16cm, ±0.1s≈3cm)
        self.stairs_turn_comp_vx = 0.32   # 补偿前移速度 (trot, 同直行速度)
        self._stairs_turn_comp = False    # 本次爬楼梯是否已做转弯补偿
        self._stairs_comp_t = 0.0         # 补偿开始时间戳
        self._stairs_mid_gait_switched = False  # 2026-08-08: 直行一半是否已切回循迹步态
        self.stairs_pre_shift_vy = 0.3    # 2026-08-05: 识别到楼梯前左平移速度 (正=左) (08-05晚: 0.2→0.3)
        self.stairs_pre_shift_time = 0.3  # 2026-08-05: 左平移时长(s) (08-05晚: 0.5→0.3, 0.3×0.3≈9cm)
        self.stairs_shift_vy = 0.25  # 2026-08-04: 与直行左平移速度交换
           # 侧移修正速度
        self.stairs_shift_duration = 0.2  # 侧移修正时长 (~0.05m)
        self.post_stairs_until = 0.0      # 楼梯后只看近处黑线的时间戳
        # 2026-08-09: 楼梯后衔接循迹期 — 无线时右找黑线, 找到立即循迹 (用户: 只有这一小部分右找, 其他照旧)
        self.post_stairs_right_search_until = 0.0    # 衔接期右找截止时间戳
        self.post_stairs_right_search_time = 2.5     # 衔接期右找时长(s)
        self.post_stairs_right_search_yaw = 1.0      # 右找角速度(rad/s, 应用时取负=右)
        self.post_stairs_right_search_vx = 0.15      # 右找时前进速度(m/s)
        self._slow_after_first_corner = False
        self._fast_after_memory = False

        self.stairs_phase = STAIRS_PHASE_FORWARD
        self.stairs_phase_start = 0.0

        # ====== 蓝色启停区参数 ======
        self.blue_stop_detected = False
        self.blue_confirm_frames = 3
        self.blue_confirm_count = 0
        self.blue_detection_enabled = False

        self.blue_go_straight_distance = 1.093   # 2026-08-11: 1.05→1.06 (用户); 2026-08-07: 1.26→1.20
        self.blue_go_straight_speed = 0.3
        self.blue_turn_angle = 109
        self.blue_turn_speed = 1
        self.blue_final_distance = 0.05
        self.blue_final_speed = 0.3

        # ====== 跳跃功能参数 ======
        self.jump_phase = 0
        self.jump_allowed = True
        self.last_corner_time = 0
        self.last_jump_time = 0
        self.red_complete_time = 0
        self.post_jump_align_timeout = 4.0
        self.post_jump_align_start = 0.0
        self.post_jump_search_direction = 3
        self.post_jump_search_speed = 0.8
        self.post_jump_align_threshold = 15

        self.jump_trigger_counter = 0
        self.jump_trigger_threshold = 10

        # ====== 中转平台检测参数（加强版） ======
        self.platform_detection_enabled = False
        self.platform_count = 0

        self.platform_confirm_frames = 6
        self.platform_confirm_count = 0
        self.platform_min_distance = 0.10
        self.platform_max_distance = 0.70
        self.platform_depth_history = deque(maxlen=5)

        self.platform_color_enabled = True
        self.platform_color_lower = np.array([0, 0, 0])
        self.platform_color_upper = np.array([80, 80, 80])
        self.platform_color_min_area = 200

        self.platform1_turn_duration = 1.5
        self.platform1_turn_yaw = 0.7
        self.platform1_turn_vx = 0.0
        self.platform1_forward_after = 0.3
        self.platform1_forward_speed = 0.2

        self.platform2_stop_duration = 2.0

        # ====== 楼梯后抓取流程参数 ======
        self.grasp_detect_enabled = False    # 爬楼梯完成后开启 grasp 图案识别
        self.grasp_enable_after = 0.0  # stairs+2s marker (2026-08-03)
        self.grasp_detect_start_time = 0.0  # 爬楼梯完成时刻 (grasp 兜底计时起点, 2026-08-03)
        self.grasp_fallback_timeout = 30.0  # 超时未识别到图案 → 跳过抓取开红点 (2026-08-03)
        self.grasp_processed = False         # 抓取流程是否已执行
        self.grasp_confirm_frames = 3
        self.grasp_confirm_count = 0
        self.grasp_pending_sign_id = None
        # 2026-08-03: 停车静止识别 (运动中低阈值探测→停车→满阈值确认)
        self.grasp_probe_threshold = 6     # 运动中迹象探测阈值 (匹配点)
        self.grasp_confirm_threshold = 13  # 静止确认阈值 (=SIMILARITY_THRESHOLD)
        self.grasp_probe_hits = 0          # 连续迹象命中数
        self.grasp_scan_until = 0.0        # 停车扫描截止时间戳
        self.grasp_scan_timeout = 2.5      # 停车扫描最长秒数
        self.grasp_scan_cooldown = 0.0     # 超时未确认后的冷却 (冷却内不再停车)
        self.grasp_last_check_time = 0.0
        self.grasp_compensate_time = 4.0        # 识别到棋盘格后保持循迹时长(s) (2026-08-03: 3.5s→4s)
        self.grasp_compensate_speed = 0.25      # 循迹补偿速度(m/s)
        self.grasp_turn_angle = Z0_GRASP_TURN_ANGLE  # 识别到grasp后左转角度 (2026-08-03 z0: 110°)
        self.grasp_turn_speed = 1.0             # 左转角速度(rad/s)
        self.grasp2_turn_right_angle = 65   # 抓A后右转角度(°) (2026-08-06: 60→65, +5° 用户)
        self.grasp2_turn_left_angle = 90    # 抓B后左转角度(°) (2026-08-02)
        self.grasp2_detect_timeout = 15.0   # 右转后识别grasp图案超时(s)
        self.grasp2_detect_frames = 3       # 识别grasp确认帧数
        self.grasp_retry_max = 1            # 抓A尝试次数 (2026-08-03: 只抓一次, 不再重试)
        self.grasp2_confirm_count = 0       # 右转后grasp图案确认计数
        self.grasp2_approach_duration = Z0_TRANSFER_APPROACH_TIME  # 中转窗口时长(s): 右转后循迹识别grasp, 窗口结束未识别到→跳过中转 (2026-08-03 z0: 15.7)
        self.grasp_target_phase = 'first'  # 左转完成后进哪个抓取: 'first'=抓A / 'second'=中转抓B (2026-08-03 z0)
        self.grasp2_approach_decel_time = 2.0 # 最后几秒线性减速到0
        self.grasp2_approach_speed = 0.3      # 循迹速度封顶(m/s)
        self.platform_zone = None            # grasp处识别放置平台: 'A'=1号(左转) 'B'=2号(右转)
        self.place_after_red_duration = 9.0  # 红点后循迹时长(s) 停车放置 (2026-08-04: 3.3→4.5→6.3→8.5→9.5→12→9)
        self.place_after_red_decel_time = 2.0 # 2026-08-04: 最后几秒线性减速到0.05, 不急停
        self.place_after_red_speed = 0.35    # 红点后循迹速度封顶(m/s) (2026-08-04: 0.3→0.35)
        self.place_after_red_dist = 2.655  # 2026-08-16: 2.630→2.655 (用户: 少4.5cm后加回2.5cm, 净少2cm)
        self.place_after_red_decel_dist = 0.5  # 2026-08-05: 最后0.5m线性减速到0.05, 不急停
        self.place_dist = 0.0              # 2026-08-05: 红点后累计循迹路程(m)
        self._place_last_t = 0.0           # 2026-08-05: 放置路程积分上一帧时刻
        self._place_lost_start = None      # 2026-08-12: 红点后丢线连续时长起点 (限时停车防乱走)
        self._place_failed = False         # 2026-08-05: 放置失败保持停止不前进
        self.post_place_speed_cap = False    # 2026-08-04: 放置完成后 0.3m/s 直到第二跳

        # 【集成】3D视觉抓取流程标志 (grasp_3d 管线)
        self.first_grasp_done = False    # 第一个物块抓取流程是否已执行 (之后才检测直角弯)
        self.first_grasp_held = False    # 第一个物块是否确认真抓到 (容量1/1)
        # 2026-08-11: 主类补 _zero_pose (原只在抓取管线定义, 收臂/蓝区收臂一直AttributeError)
        self._zero_pose = {'angle0': 0, 'angle1': -90, 'angle2': 90,
                           'angle3': 0, 'angle4': 0, 'angle5': 0}
        self._arm_ka_last = 0.0        # 2026-08-11: 主循环臂保活上次时间
        self._arm_ka_client = None     # 2026-08-11: 保活用常驻客户端 (免等待查询)
        self.second_phase_started = False  # 第二阶段(2.5s减速+中转抓B)是否已开始
        self.transfer_timer_start = 0.0    # 2026-08-03: 中转计时器起点(右转60°完成时)
        self.transfer_on_next_right = False  # 2026-08-04: 右转60°完成 → 循迹中只要右转停车中转
        self.transfer_right_time = 0.0     # 2026-08-04: 右转标志时刻, 之后循迹1.2m中转 (2026-08-05)
        self.transfer_dist = 0.0           # 2026-08-05: 右转标志后累计循迹路程(m)
        self._transfer_last_rt = 0.0       # 2026-08-05: 路程积分上一帧时刻
        self.transfer_right_wz_thresh = -0.03   # 2026-08-05: 只要机器狗开始右转即标志 (负=右)
        self.transfer_right_confirm_time = 0.3  # 2026-08-05: 右转持续确认时间(s), 防抖
        self.transfer_detect_delay = 6.0  # 2026-08-05晚: 右转60°完成后延迟(s)开启中转右转检测 (4.5→6, 防余动误触发)
        self._transfer_right_wz_since = None    # 2026-08-04: 右转持续计时起点
        self.transfer_left_shift_vy = 0.6    # 2026-08-11: 0.5→0.6 中转左平移速度 (用户)
        self.transfer_left_shift_time = 0.25   # 2026-08-12: 0.30→0.25s (用户: 平移15cm, 0.6×0.25=0.15m)
        self._prev_state = None          # 主循环上一帧状态 (用于检测第一个直角弯完成)
        self._corner_passed_after_grasp = False  # 直角弯后是否经历过丢线搜索

        self.state = TRACKING
        self.lost_count = 0
        self.last_valid_yaw = 0.0
        self.last_turn_direction = 1.0
        self.state_start_time = 0.0
        self._last_heading_error = 0.0
        self._last_avg_offset = None
        self.last_valid_offset_direction = 0.0
        self.last_valid_offset_magnitude = 0.0

        self.red_detect_enabled = False  # 中转完成(放A抓B)后才开启红点检测 (2026-08-02)
        self.red_processed = False
        self.red_detect_timeout = 20.0  # 2026-08-04: 红点检测超时(s), 超时未识别 → 跳过红点, 开启跳跃检测
        self.red_detect_start_time = 0.0  # 2026-08-04: 红点检测启用时刻
        self.red_center = None
        self.red_radius = None

        self.vy = 0.22  # 侧移速度覆盖（转弯甩尾 / 楼梯修正）

        print("[识别] 初始化ORB特征点识别器...")
        self.recognizer = ORBRecognizer("templates")
        # 2026-08-04: grasp 图像模板已彻底移除 (棋盘格识别替代; 红点识别排除grasp; 中转改定时)

        self.start_time = now_s()
        self.run_duration = args.duration
        self.frame_count = 0
        self.num_layers = num_layers

    # ==================== 状态转换 ====================
    def _transition_to(self, new_state):
        print(f"[状态] {self.state} -> {new_state}")
        old_state = self.state
        self.state = new_state
        self.state_start_time = now_s()

        if new_state == TRACKING:
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self.lost_count = 0; self.corner_confirm_count = 0

            # 从转弯状态恢复时还原ROI
            if old_state in (CORNER_APPROACH, CORNER_TURN):
                d = self.detector; o = self._orig_roi
                d.roi_top = o['top']; d.roi_bottom = o['bottom']
                d.roi_left = o['left']; d.roi_right = o['right']
                d.roi_height = o['height']; d.roi_width = o['width']

            # 窄道完成后启用楼梯
            if old_state == NARROW_EXECUTING:
                self.first_lost_handled = True
                self.stairs_allowed = True
                self.narrow_triggered = True

            # 2026-08-03: 楼梯完成后不再启用深度平台检测 (中转已改定时, 防 PLATFORM1_TURN/PLATFORM2_STOP 干扰红点识别)
            if old_state == STAIRS:
                self.platform_detection_enabled = False
                self.platform_count = 0
                # 退出爬楼梯，恢复普通步态
                # 2026-08-15: 兜底出口也切回 trot (正常出口在 _handle_stairs 转弯补偿后)
                switch_gait(self.sport, GaitMode.CLASSIC_WALK, False)
                print("[步态] 🪜 爬楼梯结束, 切回 trot")
        elif new_state in (CORNER_APPROACH, CORNER_TURN):
            # 缩小ROI一半，排除其他赛道线干扰
            d = self.detector
            cx = (d.roi_left + d.roi_right) // 2
            hw, hh = d.roi_width // 4, d.roi_height // 4
            d.roi_left = cx - hw; d.roi_right = cx + hw
            d.roi_top = d.roi_bottom - hh * 2
            d.roi_width = d.roi_right - d.roi_left
            d.roi_height = d.roi_bottom - d.roi_top
            print(f"[转弯] 🔍 ROI缩小: {d.roi_left}-{d.roi_right}, {d.roi_top}-{d.roi_bottom}")
        elif new_state == LOST_MEMORY:
            self._lm_phase = 0  # 从补偿直行开始
        elif new_state == RED_APPROACH: self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        elif new_state == STAIRS:
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self.stairs_phase = STAIRS_PHASE_FORWARD; self.stairs_phase_start = now_s()
            self._stairs_turn_comp = False  # 2026-08-08: 转弯补偿标志重置
            self._stairs_mid_gait_switched = False  # 2026-08-08: 直行中途切 trot 标志重置
            # 2026-08-07: 兜底恢复身体俯仰水平 (正常路径已在转弯前恢复)
            # 2026-08-08: 提前抬头方案 — 门控期已提前后仰则保留, 不复位压平
            if self.stairs_pitch != 0.0 and not self._stairs_pitch_sent:
                self._euler_fast(0.0, 0.0, 0.0)
                self.stairs_pitch = 0.0
                print("[楼梯] 俯仰恢复水平")
            self.stairs_triggered_once = True
            self.narrow_triggered = False
            # 2026-08-08: 进楼梯切 ClassicWalk, 楼梯结束切回 trot (用户)
            switch_gait(self.sport, GaitMode.CLASSIC_WALK, True)
            print("[步态] 🪜 进入爬楼梯: ClassicWalk")
            print("[步态] 🪜 进入爬楼梯: ClassicWalk (全程, 楼梯结束才切回 trot)")
        elif new_state == NARROW_APPROACH:
            self.narrow_triggered = True
            self.first_lost_handled = True
            self.sport.StopMove(); time.sleep(0.3)
            self.narrow_fsm.tracker = self; self.narrow_fsm.reset()
            print("[窄道] 🚀 准备执行窄道路径")
            time.sleep(0.2)
            self.state = NARROW_EXECUTING; self.state_start_time = now_s()
            print("[窄道] 开始执行写死路径...")
        elif new_state == POST_JUMP_ALIGN: self.post_jump_align_start = now_s()
        elif new_state == PLACE_AFTER_RED:
            self.place_dist = 0.0; self._place_last_t = 0.0; self._place_failed = False; self._place_lost_start = None  # 2026-08-05; 2026-08-12: 丢线计时重置
        elif new_state == JUMP:
            self.jump_trigger_counter = 0
        elif new_state == PLATFORM1_TURN:
            print(f"[平台1] 🔄 开始转弯")
        elif new_state == PLATFORM2_STOP:
            print(f"[平台2] ⏸️ 停止等待")

    def enable_blue_detection(self):
        self.blue_detection_enabled = True
        print("[蓝色] ✅ 蓝色启停区检测已启用（红点处理完成）")

    def compute_weighted_offset(self, centers):
        total_weight = 0.0; weighted_sum = 0.0
        image_center = self.detector.width // 2
        # 楼梯后2.5s / 转弯中 → 只看底层（近处黑线），忽略远处干扰线
        near_only = (now_s() < self.post_stairs_until or
                     self.state in (CORNER_APPROACH, CORNER_TURN))
        for i in range(self.num_layers):
            if near_only and i < 7:  # 上层0-6忽略，只用底层7-9
                continue
            weight = self.detector.layer_weights[i]; cx = centers[i]
            if cx is not None: weighted_sum += (cx - image_center) * weight; total_weight += weight
        if total_weight == 0: return None
        return weighted_sum / total_weight

    def check_corner(self, centers):
        """直角检测：上40%无黑线(物块过滤) + 底部20%有黑线 → 直角弯道"""
        if centers is None: return False
        far_valid = sum(1 for i in range(4) if centers[i] is not None)   # 上40%: 识别到黑物块就忽略
        mid_valid = sum(1 for i in range(4, 8) if centers[i] is not None) # 中40%
        near_valid = sum(1 for i in range(8, 10) if centers[i] is not None) # 底部20%: 左/右半区域黑线
        return (far_valid == 0) and (mid_valid <= 1) and (near_valid >= 2)

    def corner_detect_heading(self, centers):
        """底部20%黑线方向 → 记忆为左转/右转"""
        near_pts = []; image_center = self.detector.width // 2
        for i in range(8, 10):
            if centers[i] is not None: near_pts.append(centers[i] - image_center)
        return np.mean(near_pts) if near_pts else 0.0

    def get_dynamic_speed(self, avg_offset, first_layer_valid):
        if first_layer_valid: self.far_layer_lost_count = 0; base_vx = self.far_layer_normal_speed
        else:
            self.far_layer_lost_count += 1
            decel_factor = max(0.5, 1.0 - self.far_layer_lost_count * 0.1)
            base_vx = self.far_layer_lost_speed * decel_factor; base_vx = max(base_vx, 0.20)
        if avg_offset is not None:
            abs_off = abs(avg_offset)
            if abs_off > 40: base_vx *= 0.7
            elif abs_off > 25: base_vx *= 0.85
        # 看到弯道就减速，避免冲过头
        if self.corner_confirm_count > 0:
            base_vx = min(base_vx, 0.25)
        # 2026-08-04: 去掉定时减速; 放置完成后 0.3m/s 直到第二跳
        if self.post_place_speed_cap:
            base_vx = min(base_vx, 0.30)
        # 窄道前限速0.35，楼梯后恢复快速
        if not self.narrow_triggered and not self.stairs_triggered_once:
            base_vx = min(base_vx, 0.35)
        return max(base_vx, 0.15)

    def update_stairs_block_baseline(self, block_ratio):
        """2026-08-04: 深度黑色块基线 (滑动平均 20 帧)"""
        if not hasattr(self, 'stairs_block_history'):
            self.stairs_block_history = deque(maxlen=20)
        self.stairs_block_history.append(block_ratio)

    def is_stairs_block_triggered(self, block_ratio):
        """2026-08-04: 深度黑色块突变 → 爬楼梯; 当前占比 ≥ 基线+0.15 且 ≥ 0.25"""
        if not hasattr(self, 'stairs_block_history') or len(self.stairs_block_history) < 5:
            return False
        base = float(np.mean(self.stairs_block_history))
        return block_ratio >= base + 0.15 and block_ratio >= 0.25

    def update_normal_widths(self, widths_high):
        if self.stairs_triggered_once: return
        top_layers_widths = [w for w in widths_high[:self.width_sample_layers] if w > 0]
        if len(top_layers_widths) == 0: return
        avg_width = np.mean(top_layers_widths); self.normal_widths_high.append(avg_width)
        if len(self.normal_widths_high) > self.width_history_len: self.normal_widths_high.pop(0)

    def get_baseline_width_high(self):
        if len(self.normal_widths_high) == 0: return None
        return np.mean(self.normal_widths_high)

    def is_width_triggered(self, widths_high):
        if self.stairs_triggered_once: return False
        if not self.stairs_trigger_enabled: return False
        baseline = self.get_baseline_width_high()
        if baseline is None or baseline < 1.0: return False
        top_layers_widths = [w for w in widths_high[:self.width_sample_layers] if w > 0]
        if len(top_layers_widths) == 0: return False
        current_avg = np.mean(top_layers_widths)
        if current_avg > baseline * self.width_trigger_ratio:
            print(f"[楼梯触发] 楼梯ROI上侧宽度突变: cur={current_avg:.1f}px, base={baseline:.1f}px, ratio={current_avg/baseline:.2f}")
            return True
        return False

    def detect_platform(self, depth_img, color_img=None):
        if depth_img is None:
            return False, 8.0

        h, w = depth_img.shape

        roi_top = int(h * 0.35)
        roi_bottom = int(h * 0.80)
        roi_left = int(w * 0.2)
        roi_right = int(w * 0.8)

        roi = depth_img[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            return False, 8.0

        roi_m = roi.astype(np.float32) * 0.001
        valid_depths = roi_m[(roi_m > 0.1) & (roi_m < 5.0)]
        if len(valid_depths) < 30:
            return False, 8.0

        near_depths = np.sort(valid_depths)[:max(1, len(valid_depths)//3)]
        min_dist = np.median(near_depths)

        self.platform_depth_history.append(min_dist)
        smooth_dist = np.mean(self.platform_depth_history)

        depth_ok = (self.platform_min_distance <= smooth_dist <= self.platform_max_distance)

        color_ok = False
        combined = None
        if color_img is not None and self.platform_color_enabled:
            roi_color = color_img[roi_top:roi_bottom, roi_left:roi_right]
            if roi_color.size > 0:
                hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
                gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                _, dark_gray = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
                combined = cv2.bitwise_or(dark_mask, dark_gray)
                kernel = np.ones((5,5), np.uint8)
                combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

                dark_area = cv2.countNonZero(combined)
                color_ok = (dark_area > self.platform_color_min_area)

        is_platform = depth_ok and color_ok

        if depth_ok or color_ok:
            status = "✅" if is_platform else "⏳"
            dark_area_val = cv2.countNonZero(combined) if combined is not None else 0
            print(f"[平台检测] {status} 距离:{smooth_dist:.2f}m 深色面积:{dark_area_val} (深度:{'OK' if depth_ok else 'NG'} 色块:{'OK' if color_ok else 'NG'})")

        return is_platform, smooth_dist

    def _handle_platform1_turn(self, display):
        elapsed = now_s() - self.state_start_time
        if elapsed < self.platform1_turn_duration:
            return self.platform1_turn_vx, self.platform1_turn_yaw
        else:
            print("[平台1] ✅ 转弯完成，开始前进")
            self.sport.StopMove()
            time.sleep(0.2)
            self._transition_to(PLATFORM1_FORWARD)
            return 0.0, 0.0

    def _handle_platform1_forward(self, display):
        elapsed = now_s() - self.state_start_time
        expected_time = self.platform1_forward_after / self.platform1_forward_speed
        if elapsed < expected_time:
            return self.platform1_forward_speed, 0.0
        else:
            print("[平台1] ✅ 前进完成，恢复循迹")
            self.sport.StopMove()
            time.sleep(0.3)
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self._transition_to(TRACKING)
            return 0.0, 0.0

    def _handle_platform2_stop(self, display):
        elapsed = now_s() - self.state_start_time
        if elapsed < self.platform2_stop_duration:
            if int(elapsed * 2) % 2 == 0:
                print(f"[平台2] ⏸️ 等待中... {elapsed:.1f}s / {self.platform2_stop_duration}s")
            return 0.0, 0.0
        else:
            print("[平台2] ✅ 停止等待完成，恢复循迹")
            self.sport.StopMove()
            time.sleep(0.3)
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self._transition_to(TRACKING)
            return 0.0, 0.0

    def _narrow_side_balance(self, depth_img):
        """2026-08-03: 窄道横向补偿 — 左右墙距差 (左距-右距, 米).
        右偏时右墙近 → 差值正 → 向左补 (vy正=左移); 任一侧测不到墙 → 0 (不补偿)"""
        if depth_img is None:
            return 0.0
        h, w = depth_img.shape
        roi = depth_img[int(h * 0.5):int(h * 0.85), :]
        if roi.size == 0:
            return 0.0
        m = roi.astype(np.float32) * 0.001
        left = m[:, :w // 3]
        right = m[:, 2 * w // 3:]
        lv = left[(left > 0.1) & (left < 1.5)]
        rv = right[(right > 0.1) & (right < 1.5)]
        if len(lv) < 20 or len(rv) < 20:
            return 0.0
        return float(np.median(lv) - np.median(rv))

    def narrow_smooth_move(self, vx, vy, yaw):
        self.sport.Move(float(vx), float(vy), float(yaw))

    def _calculate_tracking_for_red(self, avg_offset):
        if avg_offset is not None:
            error = avg_offset; self.integral += error
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
            derivative = error - self.last_error
            wz_unfiltered = -np.clip(self.kp * error + self.ki * self.integral + self.kd * derivative, -self.max_rotation, self.max_rotation)
            wz = self.filter_alpha * wz_unfiltered + (1 - self.filter_alpha) * self.last_wz
            abs_off = abs(avg_offset)
            # 2026-08-11 (用户): 抓A右转60°后中转段循迹提到 0.5 (停车中转前)
            if getattr(self, 'transfer_on_next_right', False) and not getattr(self, 'second_phase_started', False):
                _bs = getattr(self, 'transfer_approach_speed', 0.5)
            else:
                _bs = self.base_speed
            if abs_off < 25: vx = _bs
            elif abs_off < 70: vx = _bs * 0.9
            else: vx = 0.20
            # 2026-08-07: 出窄道后循迹限速0.25 (用户), 进楼梯后解除 (左平移vy=0.3不受影响)
            if self.narrow_triggered and not self.stairs_triggered_once:
                vx = min(vx, 0.25)
            vx = max(vx, 0.15); self.last_error = error; self.last_wz = wz
        else: vx, wz = 0.0, 0.0
        return vx, wz

    def approach_red_point_with_compensate(self):
        print("[红点] 开始减速接近..."); self.sport.StopMove(); time.sleep(0.2)
        start_time = now_s(); radius_history = []; limit_reached = False
        compensate_time = self.compensate_distance / self.compensate_speed
        while True:
            if now_s() - start_time > self.approach_timeout:
                print("[红点] 接近超时"); self.sport.StopMove(); return False
            color, depth = self.detector.get_frames()
            if color is None: time.sleep(0.02); continue  # 2026-08-10 (评审点6): 防空转烧CPU
            display, mask, centers = self.detector.detect_layers(color, depth)
            avg_offset = self.compute_weighted_offset(centers)
            detected, radius, center = self.detector.detect_red_point(color)
            if not limit_reached:
                if not detected:
                    vx, wz = self._calculate_tracking_for_red(avg_offset)
                    self.sport.Move(vx, 0, wz); time.sleep(0.25); continue
                radius_history.append(radius)
                if len(radius_history) > 5: radius_history.pop(0)
                avg_radius = sum(radius_history) / len(radius_history)
                print(f"[红点] 半径:{radius}px 平均:{avg_radius:.1f}px")
                if avg_radius >= self.target_radius or (len(radius_history) >= 3 and abs(radius_history[-1] - radius_history[0]) < 3):
                    print(f"[红点] 到达视觉极限，补偿前进{self.compensate_distance*100:.0f}cm")
                    limit_reached = True; compensate_start_time = now_s()
                    self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; continue
                vx, wz = self._calculate_tracking_for_red(avg_offset)
                # 2026-08-12 (用户): 接近恒速 0.45m/s, 每步保持 8cm (0.08/0.45≈0.178s); wz 仍按偏移对准
                vx = self.approach_speed; self.sport.Move(vx, 0, wz); time.sleep(0.08 / self.approach_speed)
            else:
                elapsed = now_s() - compensate_start_time
                if elapsed >= compensate_time:
                    print(f"[红点] 补偿完成"); self.sport.StopMove(); return True
                vx, wz = self._calculate_tracking_for_red(avg_offset)
                vx = min(vx, self.compensate_speed); self.sport.Move(vx, 0, wz); time.sleep(0.05)

    def backup_before_turn(self):
        print(f"[转向] 后退{self.backup_steps}步...")
        for i in range(self.backup_steps):
            self.sport.Move(self.backup_speed, 0, 0); time.sleep(self.backup_step_time)
            self.sport.StopMove(); time.sleep(0.1)

    def align_to_line(self):
        print("[对准] 微调对准黑线..."); start_time = now_s()
        while now_s() - start_time < self.align_timeout:
            color, depth = self.detector.get_frames()
            if color is None: time.sleep(0.02); continue
            display, mask, centers = self.detector.detect_layers(color, depth)
            avg_offset = self.compute_weighted_offset(centers)
            if avg_offset is not None:
                if abs(avg_offset) < 10: break
                wz = -self.align_speed if avg_offset > 0 else self.align_speed
                self.sport.Move(0, 0, wz); time.sleep(0.05); self.sport.StopMove()
            else: self.sport.Move(0, 0, self.align_speed); time.sleep(0.1); self.sport.StopMove()
            time.sleep(0.02)
        self.sport.StopMove()

    def execute_action_by_id(self, sign_id, vui_client):
        self.recognizer.action_count += 1
        self.last_action_sign = sign_id  # 2026-08-14: 记录最后识别图案 (统一放置距离后仅记录备用, 无消费方)
        print(f"\n{'='*50}")
        print(f"匹配成功: {SIGN_NAMES[sign_id]} (第{self.recognizer.action_count}/{MAX_ACTIONS}次)")
        print(f"{'='*50}")
        if sign_id == SignID.ELECTRIC_SHOCK:
            do_stretch(self.sport)
            self.sport.Move(-0.5, 0, 0); time.sleep(0.3); self.sport.StopMove()  # 退0.15m
        elif sign_id == SignID.OXIDIZER:
            do_greet(self.sport)
            self.sport.Move(-0.25, 0, 0); time.sleep(0.12); self.sport.StopMove()  # 退0.03m
        elif sign_id == SignID.RADIATION: blink_front_lights(vui_client, times=3)
        elif sign_id == SignID.GRASP:
            print("[识别] grasp 图案由楼梯后抓取流程处理，红点动作跳过")

    def recognize_and_perform_action(self, color_img):
        print("\n[识别] 开始ORB识别...")
        if color_img is None:
            while True:
                color, depth = self.detector.get_frames()
                if color is not None: gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY); break
                time.sleep(0.1)
        else: gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        self.recognizer.confirm_counter = 0; self.recognizer.pending_sign_id = None; last_move_time = now_s()
        recog_start = now_s()
        while True:
            current_time = now_s()
            # 2026-08-04: 18s 未匹配到图案 → 放弃识别继续流程 (用户: 至少18s识别不到才右转)
            if current_time - recog_start >= 18.0:
                print("[识别] ⏱ 18s 未匹配到图案, 放弃识别, 继续右转接循迹")
                return
            if current_time - last_move_time >= 4.0:
                print("[识别] 4秒未识别, 前进0.3m/s×0.5s"); self.sport.Move(0.3, 0, 0); time.sleep(0.5); self.sport.StopMove(); last_move_time = current_time  # 2026-08-04: 0.3m/s×0.5s
            color, depth = self.detector.get_frames()
            if color is not None: gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            if current_time - self.recognizer.last_action_time < COOLDOWN_TIME: time.sleep(0.1); continue
            match = self.recognizer.match(gray)
            if match and match[0] == SignID.GRASP:
                match = None  # 2026-08-04: 红点识别不做 grasp 图案 (3552特征合并误报率高, 导致秒回右转)
            if match:
                sign_id, match_count = match
                print(f"[识别] 匹配: {SIGN_NAMES[sign_id]} ({match_count}点)")
                if self.recognizer.pending_sign_id == sign_id:
                    self.recognizer.confirm_counter += 1
                    if self.recognizer.confirm_counter >= CONFIRM_FRAMES:
                        self.execute_action_by_id(sign_id, self.vui_client)
                        self.recognizer.last_action_time = current_time
                        self.recognizer.confirm_counter = 0; self.recognizer.pending_sign_id = None; return
                else: self.recognizer.confirm_counter = 1; self.recognizer.pending_sign_id = sign_id
            time.sleep(0.05)

    def execute_turn_sequence(self):
        print("[转向] 开始转向序列...")
        steps_135 = 122 // self.turn_step_angle
        # 2026-08-06: 红点左转少3° (参数化: 每步sleep减 np.radians(deg)/(turn_speed*steps))
        _red_l = np.radians(self.red_turn_left_reduce_deg) / (self.turn_speed * steps_135)
        for i in range(steps_135): self.sport.Move(0, 0, self.turn_speed); time.sleep(self.turn_step_time - _red_l); self.sport.StopMove(); time.sleep(0.05)
        print("[转向] 左转完成"); time.sleep(0.3)
        self.backup_before_turn(); time.sleep(0.2)
        color, depth = self.detector.get_frames(); self.recognize_and_perform_action(color); time.sleep(0.5)
        steps_90 = 128 // self.turn_step_angle
        # 2026-08-07: 红点右转 99° (80+19 用户), 先转完再衔接循迹 (用户: 转完才看黑线)
        _r_step = np.radians(99.0) / (self.turn_speed * steps_90)
        for i in range(steps_90):
            self.sport.Move(0, 0, -self.turn_speed); time.sleep(_r_step); self.sport.StopMove(); time.sleep(0.05)
        print("[转向] 右转完成 (99°), 直接衔接循迹"); time.sleep(0.3)
        # 2026-08-12 (用户): 红点转弯后直接衔接循迹 — 不再对准/右搜 (PLACE_AFTER_RED 循迹自带丢线处理)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        print("[转向] 转向序列完成")

    def execute_jump(self):
        jump_num = self.jump_phase + 1
        print(f"\n[跳跃] ========== 开始执行第{jump_num}次跳跃 ==========")
        self.sport.StopMove()
        time.sleep(0.3)
        try:
            ret = self.sport.FrontJump()
            if ret == 0: print(f"[跳跃] ✓ FrontJump 第{jump_num}次执行成功")
            else: print(f"[跳跃] ✗ 返回码: {ret}")
        except Exception as e: print(f"[跳跃] ✗ 失败: {e}")
        time.sleep(1.0)   # 2026-08-02: 等跳跃落地站稳 (微调前提, 否则指令被运动控制忽略)
        # 2026-08-03: 跳1后不再右转10° (右偏导致窄道第一段直行跑偏); 跳2保留
        if jump_num == 2:
            yaw10_s = np.radians(10) / self.turn_speed
            self.sport.Move(0, 0, -self.turn_speed); time.sleep(yaw10_s); self.sport.StopMove()
            print(f"[跳跃] ========== 第{jump_num}次跳跃结束, 微调10°完成 ({yaw10_s:.2f}s) ==========\n")
        else:
            print(f"[跳跃] ========== 第{jump_num}次跳跃结束, 跳后不右转, 直接衔接循迹 ==========\n")
        if jump_num == 1:
            # 2026-08-05: 第一跳结束直接衔接循迹, 不做微调对准
            print("[跳跃] 🟢 第一跳结束, 直接衔接循迹")
        else:
            # 2026-08-03: 恢复微调对准 (仅闭环对准, 不搜索): 找到线后按偏移闭环转, 最长0.8s (2026-08-03: 1.5s→0.8s)
            self.sport.StopMove(); time.sleep(0.2)
            align_start = now_s()
            while now_s() - align_start < 0.8:
                color, depth = self.detector.get_frames()
                if color is None: time.sleep(0.02); continue  # 2026-08-10 (评审点6): 防空转
                _, _, centers = self.detector.detect_layers(color, depth)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is None: break
                if abs(avg_offset) < self.post_jump_align_threshold:
                    print(f"[跳跃] ✅ 微调对准完成 (偏移 {avg_offset:.1f}px)")
                    break
                wz = -0.3 if avg_offset > 0 else 0.3
                self.sport.Move(0, 0, wz); time.sleep(0.05); self.sport.StopMove(); time.sleep(0.02)
            self.sport.StopMove()
            print(f"[跳跃] 微调对准结束 (耗时 {now_s()-align_start:.2f}s)")
        self.last_jump_time = now_s()
        if self.jump_phase == 0:
            self.jump_allowed = False; self.jump_phase = 1
            print("[跳跃] ⚠️ 第一次跳跃完成，等待红点处理后开启第二次跳跃")
        elif self.jump_phase == 1:
            self.jump_allowed = False; self.jump_phase = 2
            print("[跳跃] ⚠️ 第二次跳跃完成，直行巡线等蓝区（不搜索不摆头）")
            self.sport.Move(0.0, -0.3, 0.0); time.sleep(0.6); self.sport.StopMove()  # 2026-08-05: 跳2完往右平移 0.3m/s × 0.6s (2026-08-04: 1.0s)
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self.state = TRACKING; self.state_start_time = now_s()
            return
        self.jump_trigger_counter = 0
        self.detector.roi_top = 390; self.detector.roi_left = 80; self.detector.roi_right = 560
        self.detector.roi_width = self.detector.roi_right - self.detector.roi_left
        self.detector.roi_height = self.detector.roi_bottom - self.detector.roi_top
        print("[跳跃] ROI已切换到循迹模式")
        # 2026-08-02: 跳完不摆头, 直接循迹 (丢线逻辑兜底, 消除4s停顿)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        self.state = TRACKING; self.state_start_time = now_s()
        return

    def post_jump_align(self):
        print("\n[摆头] ========== 开始摆头找线 =========="); self.sport.StopMove(); time.sleep(0.5)
        for retry in range(10):
            color, depth = self.detector.get_frames()
            if color is not None: break
            print(f"[摆头] 等待相机恢复... ({retry+1}/10)"); time.sleep(0.3)
        if color is None: print("[摆头] ❌ 相机无法恢复，跳过摆头"); return False
        search_direction = 1
        for attempt in range(8):
            color, depth = self.detector.get_frames()
            if color is not None:
                _, _, centers = self.detector.detect_layers(color, depth)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is not None:
                    print(f"[摆头] ✅ 找到黑线，偏移: {avg_offset:.1f}px")
                    if abs(avg_offset) < self.post_jump_align_threshold:
                        print("[摆头] ✅ 已在正中间"); self.sport.StopMove(); print("[摆头] ========== 摆头完成 ==========\n"); return True
                    print("[摆头] 微调对准中..."); align_start = now_s()
                    while now_s() - align_start < 0.8:
                        color, depth = self.detector.get_frames()
                        if color is None: time.sleep(0.02); continue  # 2026-08-10 (评审点6): 防空转
                        _, _, centers = self.detector.detect_layers(color, depth)
                        avg_offset = self.compute_weighted_offset(centers)
                        if avg_offset is None: break
                        if abs(avg_offset) < self.post_jump_align_threshold:
                            print("[摆头] ✅ 对准完成"); self.sport.StopMove(); time.sleep(0.2)
                            print("[摆头] ========== 摆头完成 ==========\n"); return True
                        wz = -0.3 if avg_offset > 0 else 0.3; self.sport.Move(0, 0, wz); time.sleep(0.05); self.sport.StopMove(); time.sleep(0.02)
                    self.sport.StopMove(); print("[摆头] ========== 摆头完成 ==========\n"); return True
            print(f"[摆头] 第 {attempt+1}/8 次搜索")
            self.sport.Move(0, 0, search_direction * 0.4); time.sleep(0.35); self.sport.StopMove(); time.sleep(0.1)
            if attempt % 2 == 1: search_direction *= -1
        print("[摆头] ❌ 超时未找到黑线！"); self.sport.StopMove(); print("[摆头] ========== 摆头失败 ==========\n"); return False

    def initial_align(self):
        print("\n[初始对准] ========== 启动前对准黑线 =========="); time.sleep(0.5)
        max_attempts = 8; search_direction = 1
        for attempt in range(max_attempts):
            color, depth = self.detector.get_frames()
            if color is not None:
                _, _, centers = self.detector.detect_layers(color, depth)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is not None:
                    print(f"[初始对准] 检测到黑线，偏移: {avg_offset:.1f}px")
                    if abs(avg_offset) < 15: print("[初始对准] ✅ 黑线已在正中间"); self.sport.StopMove(); print("[初始对准] ========== 对准完成 ==========\n"); return True
                    print("[初始对准] 微调中..."); align_start = now_s()
                    while now_s() - align_start < 0.8:
                        color, depth = self.detector.get_frames()
                        if color is None: time.sleep(0.02); continue  # 2026-08-10 (评审点6): 防空转
                        _, _, centers = self.detector.detect_layers(color, depth)
                        avg_offset = self.compute_weighted_offset(centers)
                        if avg_offset is None: break
                        if abs(avg_offset) < 15: print("[初始对准] ✅ 对准完成"); self.sport.StopMove(); time.sleep(0.3); print("[初始对准] ========== 对准完成 ==========\n"); return True
                        wz = -0.3 if avg_offset > 0 else 0.3; self.sport.Move(0, 0, wz); time.sleep(0.05); self.sport.StopMove(); time.sleep(0.02)
                    self.sport.StopMove(); print("[初始对准] ========== 对准完成 ==========\n"); return True
            print(f"[初始对准] 搜索中 {attempt+1}/{max_attempts}...")
            self.sport.Move(0, 0, search_direction * 0.4); time.sleep(0.35); self.sport.StopMove(); time.sleep(0.1)
            if attempt % 2 == 1: search_direction *= -1
        print("[初始对准] ⚠️ 未找到黑线，直接开始循迹"); self.sport.StopMove(); print("[初始对准] ========== 对准结束 ==========\n"); return False

    # ==================== 主循环 ====================
    def run(self):
        print("\n" + "=" * 50)
        print("Go2 循迹启动 - 完整版 (直角转弯优化 + 障碍物黑线过滤)")
        print("=" * 50)

        print("\n[预热阶段] ========================================")
        warmup_frames = 40
        for i in range(warmup_frames):
            color, depth = self.detector.get_frames()
            if color is not None and SHOW_GUI:
                display_preheat = color.copy()
                cv2.putText(display_preheat, f"预热中: {i+1}/{warmup_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("相机预热", display_preheat); cv2.waitKey(1)
            time.sleep(0.03)
        if SHOW_GUI: cv2.destroyWindow("相机预热")
        print("[预热] ✅ 相机预热完成\n")

        # 2026-08-04: 程序启动 → 机械臂显式回归0位 (上次强杀后臂可能停在非零位)
        try:
            arm0 = D1UDPClient('192.168.123.100')
            arm0.enable()
            time.sleep(1.0)
            arm0.home()
            time.sleep(3.0)
            print("[Arm] ✅ 启动归零完成 (0位)")
        except Exception as e:
            print(f"[Arm] ⚠️ 启动归零失败: {e}")


        print("[运动] 唤醒站立 (StandUp 慢速起立)...")
        # 2026-08-09: RecoveryStand 弹起太快(用户); Sit 在本机固件实测返回-1不可用; StandUp 单独起立最慢最稳 (stand_test.py 实测)
        # 2026-08-03: DDS 首请求间歇性发送失败, 不检查返回值会'假站立', 加重试
        def _stand_call(cmd_name, fn, retries=5):
            for attempt in range(retries):
                ret = fn()
                if ret == 0:
                    return True
                print(f"[运动] ⚠️ {cmd_name} 失败(码:{ret}), 重试 {attempt+1}/{retries}")
                time.sleep(1.0)
            return False

        if not _stand_call("StandUp", self.sport.StandUp):
            print("[运动] ❌ StandUp 失败(5次), 检查狗本体/DDS网络, 终止")
            sys.exit(1)
        time.sleep(2.0)  # 2026-08-09: 站立完成等稳定
        # 2026-08-03: StopMove 预热请求, 吸收信道首请求抖动 (initial_align 前)
        for attempt in range(3):
            ret = self.sport.StopMove()
            if ret == 0: break
            print(f"[运动] ⚠️ 预热请求失败(码:{ret}) {attempt+1}/3")
            time.sleep(0.5)
        self._ensure_level_standing()
        print("[运动] ✅ 站立完成\n")

        self.initial_align()
        print("[主循环] 开始循迹...")

        try:
            while True:
                if self.blue_stop_detected and self.state == BLUE_SIT_DOWN:
                    if now_s() - self.state_start_time > 3.0:
                        print("[退出] 到达终点，任务完成"); break

                # 2026-08-12: 臂动作阶段用快速取帧 (相机死时30s恢复会卡死转弯控制环 — 狗不动臂不收)
                if self.state in (GRASP_ARM_1, GRASP_ARM_2, GRASP2_TURN_RIGHT, GRASP2_DONE):
                    color, depth = self.detector.get_frames_quick()
                else:
                    color, depth = self.detector.get_frames()
                if color is None:
                    # 2026-08-04: 相机被挡/无帧时, 窄道/爬楼梯盲走段继续推进 (不原地停)
                    if self.state == NARROW_EXECUTING:
                        front_d = self.front_radar.get_front_dist(None, narrow_mode=(self.narrow_fsm.state == "STRAIGHT_1"))
                        vx, vy, yaw = self.narrow_fsm.get_cmd(front_d)
                        self.narrow_smooth_move(vx, vy, yaw)
                        time.sleep(0.01)
                        continue
                    if self.state == STAIRS:
                        vx, wz = self._handle_stairs(None)
                        self.sport.Move(vx, self.vy, wz)
                        self.vy = 0.0
                        time.sleep(0.01)
                        continue
                    time.sleep(0.02)  # 2026-08-10 (评审点6): 相机无帧防空转
                    continue
                self.frame_count += 1
                # 2026-08-11 (用户: 抓A后机械臂丢使能) — 主循环臂保活:
                # 抓A后到中转间~20s无臂指令, 控制器闲置超时会丢使能. 每1.5s免等待查询.
                if now_s() - self._arm_ka_last > 1.5:
                    self._arm_ka_last = now_s()
                    try:
                        if self._arm_ka_client is None:
                            self._arm_ka_client = D1UDPClient('192.168.123.100')
                        self._arm_ka_client._send(9)  # 发后即忘, 不阻塞
                    except Exception:
                        pass
                self._prev_state = self.state  # 【集成】记录本帧起始状态 (检测直角弯完成)

                if self.state == NARROW_EXECUTING:
                    narrow_mode = (self.narrow_fsm.state == "STRAIGHT_1")
                    front_d = self.front_radar.get_front_dist(depth, narrow_mode=narrow_mode)
                    self.narrow_fsm.side_balance = self._narrow_side_balance(depth)  # 2026-08-03: 左右墙距差
                    vx, vy, yaw = self.narrow_fsm.get_cmd(front_d)
                    self.narrow_smooth_move(vx, vy, yaw)
                    if self.narrow_fsm.finished:
                        print("[窄道] ✅ 窄道路径执行完成, 不停车直接衔接循迹")
                        self.state = TRACKING; self.state_start_time = now_s()
                        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
                        self.lost_count = 0; self.corner_confirm_count = 0
                        self._last_avg_offset = None; self._last_heading_error = 0.0
                        self.first_lost_handled = True
                        self.stairs_allowed = True; self.narrow_triggered = True
                        self.stairs_gate_dist = 0.0                 # 2026-08-07: 门检测=循迹0.4m
                        self._stairs_gate_last_t = now_s()
                        self.narrow_done_at = now_s()  # 2026-08-04: 记录窄道完成时刻
                        # 2026-08-11 (用户): 窄道出来丢线 → 右搜索, 看见黑线即循迹
                        try:
                            color_n, depth_n = self.detector.get_frames_quick()
                            seen = False
                            if color_n is not None:
                                _, _, centers = self.detector.detect_layers(color_n, depth_n)
                                avg_off = self.compute_weighted_offset(centers)
                                if avg_off is not None:
                                    seen = True
                                    print(f"[窄道] 🎯 出来即见黑线(偏移{avg_off:.0f}px), 直接循迹")
                            if not seen:
                                print("[窄道] 🔍 出来丢线, 往右搜索对准黑线...")
                                search_end = now_s() + 3.0
                                while now_s() < search_end:
                                    color_n, depth_n = self.detector.get_frames_quick()
                                    if color_n is not None:
                                        _, _, centers = self.detector.detect_layers(color_n, depth_n)
                                        avg_off = self.compute_weighted_offset(centers)
                                        if avg_off is not None:
                                            print(f"[窄道] ✅ 右搜找到黑线(偏移{avg_off:.0f}px), 停止搜索对准")
                                            self.sport.StopMove(); time.sleep(0.2)
                                            self.align_to_line()
                                            seen = True
                                            break
                                    self.sport.Move(0.0, 0.0, -self.red_search_right_yaw); time.sleep(0.15)
                                    self.sport.StopMove()
                                if not seen:
                                    print("[窄道] ⚠️ 右搜3s未找到黑线, 直接循迹 (丢线兜底)")
                        except Exception:
                            pass
                        print("[窄道] ✅ 窄道完成, 门检测=循迹0.4m后允许爬楼梯"); continue
                    if SHOW_GUI and color is not None:
                        display_n = color.copy()
                        cv2.putText(display_n, f"NARROW: {self.narrow_fsm.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Go2 Full", display_n); cv2.waitKey(1)
                    time.sleep(0.01); continue

                skip_states = [STAIRS, BLUE_STOP, BLUE_GO_STRAIGHT, BLUE_TURN_LEFT, BLUE_FINAL_APPROACH, BLUE_SIT_DOWN,
                              POST_JUMP_ALIGN, JUMP, NARROW_APPROACH, NARROW_EXECUTING, PLATFORM1_TURN, PLATFORM1_FORWARD, PLATFORM2_STOP,
                              GRASP_FORWARD, GRASP_TURN, GRASP_DONE,
                              GRASP_ARM_1, GRASP2_TURN_RIGHT, GRASP2_DETECT, GRASP2_APPROACH, GRASP_ARM_2, GRASP2_DONE]  # 【集成】3D抓取流程状态
                if self.state in skip_states:
                    display = color.copy() if SHOW_GUI else None
                    avg_offset = None; pattern = None
                    centers = [None] * self.num_layers; first_layer_valid = False
                else:
                    display, mask, centers = self.detector.detect_layers(color, depth)
                    avg_offset = self.compute_weighted_offset(centers)
                    pattern = self.detector.analyze_black_line_pattern(centers)
                    first_layer_valid, _ = self.detector.get_first_layer_status(centers)

                red_detected, red_radius, red_center = self.detector.detect_red_point(color)
                # 2026-08-06: 红点诊断打印已移除 (用户: 不用有这个)
                widths_high = self.detector.detect_stair_widths(color, depth)
                vx, wz = 0.0, 0.0

                # ====== 蓝区检测（最高优先级） ======
                if self.state == TRACKING and not self.blue_stop_detected and self.blue_detection_enabled:
                    blue_detected, blue_ratio = self.detector.detect_blue_stop_area(color)
                    if blue_detected:
                        self.blue_confirm_count += 1
                        if self.blue_confirm_count >= self.blue_confirm_frames:
                            print(f"[蓝色] ✅ 检测到蓝色启停区！占比: {blue_ratio:.1%}")
                            self.blue_stop_detected = True
                            self.sport.StopMove(); time.sleep(0.2)
                            self._transition_to(BLUE_STOP); continue
                    else: self.blue_confirm_count = max(0, self.blue_confirm_count - 1)

                # 2026-08-03: 窄道只由"第一次丢线"触发 (深度检测已关闭), 跳跃1完成后才开启
                # 2026-08-03: 楼梯衔接: 宽度突变检测为主; 窄道完成后超时(3s)兜底直接爬

                if self.state != STAIRS and not self.stairs_triggered_once and not self.blue_stop_detected and self.stairs_allowed:
                    if self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]:
                        # 2026-08-04: 换回 go2palouti 同款 — 高阈值黑线宽度突变 (楼梯ROI上侧3层, >2.5x基准)
                        # 2026-08-07: 门检测 = 往前循迹0.4m (用户指定) — 距离门控, 期间采基准+禁止触发
                        _now = now_s()
                        # 2026-08-07: 积分用循迹实际速度 (vx 此处已被置0, 不能直接用; 与主循迹同公式)
                        _off = abs(avg_offset) if avg_offset is not None else 99
                        _vx = self.base_speed if _off < 25 else (self.base_speed * 0.9 if _off < 70 else 0.20)
                        if self.narrow_triggered and not self.stairs_triggered_once:
                            _vx = min(_vx, 0.25)
                        if self._stairs_gate_last_t > 0:
                            self.stairs_gate_dist += _vx * (_now - self._stairs_gate_last_t)
                        self._stairs_gate_last_t = _now
                        if self.stairs_gate_dist < self.stairs_gate_dist_need:
                            self.update_normal_widths(widths_high)
                        # ====== 2026-08-08: 深度距离门控 (方案C) — 深度信号提前抬头/确认楼梯 ======
                        _blk, _min_d = self.detector.detect_stair_depth_stats(depth)
                        _base_min = None; _headup_d = _headup_b = _confirm = False
                        if self.stairs_depth_enabled:
                            if self._stairs_min_depth_hist:
                                _base_min = float(np.mean(self._stairs_min_depth_hist))
                            # 平地基线: 仅门控前段、无黑块、未提前抬头时采集; 楼梯出现后冻结 (防污染)
                            if (_blk < 0.15 and _min_d is not None
                                    and self.stairs_gate_dist < 0.25 and not self._stairs_pitch_sent):
                                self._stairs_min_depth_hist.append(_min_d)
                            if _blk < 0.15:
                                self.update_stairs_block_baseline(_blk)
                            # 2026-08-08: 去掉 blk 上限掐死 — D435 近距盲区(0.3m内)大量无效像素使 blk 冲高,
                            # 恰是"台阶贴脸"的强信号 (平地 blk≈0, 无近距场景); 全黑=该确认, 不再当异常
                            _headup_d = (_min_d is not None and _base_min is not None
                                         and _min_d < _base_min - self.stairs_depth_headup_gap)
                            _headup_b = self.is_stairs_block_triggered(_blk)
                            _confirm = ((_min_d is not None and _min_d < self.stairs_depth_enter_m)
                                        or _blk >= self.stairs_depth_block_enter)
                            if _now - self._stairs_depth_log_t >= 0.5:
                                self._stairs_depth_log_t = _now
                                print(f"[楼梯深度] blk={_blk:.2f} min={('%.2f'%_min_d) if _min_d is not None else '---'}m "
                                      f"base={('%.2f'%_base_min) if _base_min is not None else '---'}m "
                                      f"gate={self.stairs_gate_dist:.2f}m confirm={_confirm}")
                        # 2026-08-08: 提前抬头 — 宽度突变 或 深度信号 命中即后仰, 不等0.4m门控满
                        # (用户: 之前撞上台阶才抬头, 因为进楼梯模式时狗已在台阶脚下)
                        if (not self._stairs_pitch_sent and self.stairs_head_up_pitch != 0.0
                                and not red_detected
                                and (self.is_width_triggered(widths_high) or _headup_d or _headup_b)):
                            self._stairs_pitch_sent = True
                            self.stairs_pitch = -self.stairs_head_up_pitch
                            self._euler_fast(0.0, self.stairs_pitch, 0.0)
                            self._stairs_pitch_last = _now
                            print(f"[楼梯] 🪜 提前后仰 {self.stairs_head_up_pitch*57.2958:.1f}° — 楼梯已在视野, 抬头走向楼梯")
                        # 2026-08-08: 提前抬头后, 进入楼梯模式前的循迹期间每1s重发保持 (进STAIRS后由 _handle_stairs 继续)
                        if (self.stairs_pitch != 0.0 and not self.stairs_triggered_once
                                and _now - self._stairs_pitch_last >= 1.0):
                            self._euler_fast(0.0, self.stairs_pitch, 0.0)
                            self._stairs_pitch_last = _now
                        # 进入楼梯模式: 0.4m门控满 + (宽度突变 或 深度确认)
                        if (not red_detected and self.stairs_gate_dist >= self.stairs_gate_dist_need
                                and (self.is_width_triggered(widths_high) or _confirm)):
                            print("[楼梯模式] 🪜 门检测0.4m到 + 深度确认, 爬楼梯")
                            self.sport.StopMove(); time.sleep(0.2)
                            self._transition_to(STAIRS); continue
                        # 保险兜底: 0.4m循迹后深度一直未确认, 再走 extra 才无条件进 (防深度流故障卡死)
                        if (not red_detected
                                and self.stairs_gate_dist >= self.stairs_gate_dist_need + self.stairs_gate_fallback_extra):
                            print("[楼梯模式] 🪜 深度一直未确认, 保险兜底直接爬楼梯")
                            self.sport.StopMove(); time.sleep(0.2)
                            self._transition_to(STAIRS); continue

                # ====== grasp 兜底 (2026-08-03): 超时未识别 → 跳过抓取, 开红点衔接 (红点→跳2→蓝区) ======
                if (self.grasp_detect_enabled and not self.grasp_processed
                        and now_s() - self.grasp_detect_start_time > self.grasp_fallback_timeout):
                    self.grasp_detect_enabled = False
                    self.grasp_processed = True
                    self.red_detect_enabled = True
                    print(f"[抓取识别] ⏰ {self.grasp_fallback_timeout:.0f}s 未识别到棋盘格, 跳过抓取, 开启红点检测衔接循迹")

                # ====== 楼梯后 grasp 识别 (2026-08-03 z0 嵌入: 棋盘格SB检测, 不停车, 弯道跳过) ======
                # 高置信(棋盘格直接100点≥HIGH)立即确认; 迹象(≥STOP)连续2帧确认; 确认后不停车 → GRASP_FORWARD 组合走
                if (self.state == TRACKING and self.grasp_detect_enabled and not self.grasp_processed
                        and not self.blue_stop_detected and now_s() >= self.grasp_enable_after
                        and self.corner_confirm_count == 0):   # 弯道确认中不识别 (走弯道误报)
                    now = now_s()
                    if now - self.grasp_last_check_time >= 0.15:
                        self.grasp_last_check_time = now
                        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                        cnt = self._match_grasp(gray, min_threshold=Z0_GRASP_PROBE_THRESHOLD)
                        if cnt is not None and cnt >= Z0_GRASP_STOP_THRESHOLD:
                            if cnt >= Z0_GRASP_STOP_HIGH:
                                print(f"[抓取] 🎯 检测到 grasp 棋盘格(4×4)!! 继续组合走{Z0_GRASP_FWD_TIME:.1f}s后停止")
                                self.grasp_probe_hits = 0
                                self._transition_to(GRASP_FORWARD); continue
                            else:
                                self.grasp_probe_hits += 1
                                if self.grasp_probe_hits >= Z0_GRASP_PROBE_FRAMES:
                                    print(f"[抓取] 🎯 运动中迹象({cnt}点×{self.grasp_probe_hits}), 确认识别到 grasp 图案, 继续组合走{Z0_GRASP_FWD_TIME:.1f}s后停止")
                                    self.grasp_probe_hits = 0
                                    self._transition_to(GRASP_FORWARD); continue
                        else:
                            self.grasp_probe_hits = 0

                if (self.state == TRACKING and
                    self.platform_detection_enabled and
                    not self.blue_stop_detected and
                    self.platform_count < 2):

                    is_platform, platform_dist = self.detect_platform(depth, color)

                    if is_platform:
                        self.platform_confirm_count += 1
                        if self.platform_confirm_count >= self.platform_confirm_frames:
                            self.platform_count += 1
                            self.platform_confirm_count = 0
                            self.sport.StopMove()
                            time.sleep(0.2)

                            if self.platform_count == 1:
                                print(f"\n[平台] 🎯 检测到第1个中转平台！距离:{platform_dist:.2f}m")
                                print("[平台] 执行：转弯避让")
                                self._transition_to(PLATFORM1_TURN)
                            elif self.platform_count == 2:
                                print(f"\n[平台] 🎯 检测到第2个中转平台！距离:{platform_dist:.2f}m")
                                print("[平台] 执行：停止等待")
                                self._transition_to(PLATFORM2_STOP)
                            continue
                    else:
                        self.platform_confirm_count = max(0, self.platform_confirm_count - 1)

                if self.state not in [STAIRS, BLUE_STOP, BLUE_GO_STRAIGHT, BLUE_TURN_LEFT, BLUE_FINAL_APPROACH, BLUE_SIT_DOWN,
                                      NARROW_APPROACH, NARROW_EXECUTING, JUMP, POST_JUMP_ALIGN, PLATFORM1_TURN, PLATFORM1_FORWARD, PLATFORM2_STOP]:
                    if red_detected and self.red_detect_enabled and not self.red_processed and self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]:
                        print(f"\n[红点] 检测到红点! 半径:{red_radius}px")
                        self.red_center = red_center; self.red_radius = red_radius
                        self.sport.StopMove(); time.sleep(0.2)
                        self._transition_to(RED_APPROACH); continue

                if self.state not in skip_states:
                    if avg_offset is not None:
                        self.last_valid_offset_direction = np.sign(avg_offset)
                        self.last_valid_offset_magnitude = abs(avg_offset)
                        self.last_valid_yaw = self.last_wz
                        if abs(avg_offset) > 15: self.last_turn_direction = -np.sign(avg_offset)

                if self.run_duration > 0 and now_s() - self.start_time > self.run_duration:
                    print(f"\n[退出] 计时结束"); break

                # ====== 第一次跳跃检测（窄道前）：垂直投影，与第二跳相同 ======
                if (self.state == TRACKING and not self.blue_stop_detected and
                    self.jump_allowed and self.jump_phase == 0):
                    if now_s() - self.last_corner_time > 0.5 and now_s() - self.last_jump_time > 1.0:
                        h, w = color.shape[:2]
                        cx = w // 2; hw = 40
                        y1, y2 = h - 100, h - 10
                        roi = color[y1:y2, cx-hw:cx+hw]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, self.detector.black_threshold, 255, cv2.THRESH_BINARY_INV)
                        row_ratios = np.sum(mask > 0, axis=1) / mask.shape[1]
                        n_rows = len(row_ratios)
                        top_mean = np.mean(row_ratios[:n_rows//3])
                        mid_mean = np.mean(row_ratios[n_rows//3:2*n_rows//3])
                        bot_mean = np.mean(row_ratios[2*n_rows//3:])
                        is_cutoff = (top_mean > 0.05 and bot_mean > 0.05 and
                                     mid_mean < top_mean * 0.53 and mid_mean < bot_mean * 0.53)
                        if is_cutoff:
                            print(f"[跳跃1] ★★★ 截断! 补偿前进0.19m再跳...")
                            self.sport.StopMove(); time.sleep(0.1)
                            self.sport.Move(0.25, 0, 0); time.sleep(0.76)  # 0.25×0.76=0.19m (2026-08-16: 0.84→0.76s, 去掉2cm)
                            self.sport.StopMove(); time.sleep(0.1)
                            print(f"[跳跃1] ★★★ 第一次跳跃！★★★")
                            self.sport.StopMove()
                            self._transition_to(JUMP)
                            continue

                # ====== 第二次跳跃检测：红点后才启用，等2.8s → 垂直投影截断 ======
                if self.state == TRACKING and self.jump_phase == 1 and self.red_complete_time > 0:
                    if now_s() - self.red_complete_time < 0.2:  # 2026-08-12: 3.5→0.2 (用户: 跳2延迟0.2s再跳)
                        pass  # 红点后等2.8s
                    else:
                        # 垂直投影：黑变白 → 逐行统计 → 中段有断裂→跳
                        h, w = color.shape[:2]
                        cx = w // 2; hw = 40
                        y1, y2 = h - 100, h - 10
                        roi = color[y1:y2, cx-hw:cx+hw]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, self.detector.black_threshold, 255, cv2.THRESH_BINARY_INV)
                        # 逐行白像素占比
                        row_ratios = np.sum(mask > 0, axis=1) / mask.shape[1]
                        # 分三段：上1/3、中1/3、下1/3
                        n_rows = len(row_ratios)
                        top_mean = np.mean(row_ratios[:n_rows//3])
                        mid_mean = np.mean(row_ratios[n_rows//3:2*n_rows//3])
                        bot_mean = np.mean(row_ratios[2*n_rows//3:])
                        # 中间明显低于上下 → 截断 (比第一跳更灵敏)
                        is_mid_gap = (top_mean > 0.05 and bot_mean > 0.05 and
                                      mid_mean < top_mean * 0.53 and mid_mean < bot_mean * 0.53)
                        half_top = top_mean * 0.53
                        print(f"[跳跃2] 上:{top_mean:.1%} 中:{mid_mean:.1%}(阈{half_top:.1%}) 下:{bot_mean:.1%} | "
                              f"{'⚠️截断!' if is_mid_gap else '✓'}")
                        if is_mid_gap:
                            print(f"[跳跃2] ★★★ 截断! 补偿前进0.41m再跳...")
                            self.sport.StopMove(); time.sleep(0.1)
                            self.sport.Move(0.25, 0, 0); time.sleep(1.64)  # 0.25×1.64=0.41m (2026-08-16: 1.48→1.64s, 用户: 0.37m→0.41m)
                            self.sport.StopMove(); time.sleep(0.1)
                            print(f"[跳跃2] ★★★ 第二次跳跃！★★★")
                            self._transition_to(JUMP)
                            continue


                if (self.state not in [RED_APPROACH, TURN_BACK, JUMP, POST_JUMP_ALIGN, STAIRS, BLUE_STOP, BLUE_GO_STRAIGHT,
                                       BLUE_TURN_LEFT, BLUE_FINAL_APPROACH, BLUE_SIT_DOWN, NARROW_APPROACH, NARROW_EXECUTING,
                                       PLATFORM1_TURN, PLATFORM1_FORWARD, PLATFORM2_STOP, PLACE_AFTER_RED,
                                       GRASP_FORWARD, GRASP_TURN, GRASP_DONE,
                                       GRASP_ARM_1, GRASP2_TURN_RIGHT, GRASP2_DETECT, GRASP2_APPROACH, GRASP_ARM_2, GRASP2_DONE,  # 【集成】3D抓取流程状态
                                       CORNER_APPROACH, CORNER_TURN, PLACE_AFTER_RED]
                        and self.red_complete_time <= 0):  # 2026-08-03: 红点处理完后关闭丢线逻辑
                    if self.jump_phase != 2:  # 第二跳后不搜索不摆头
                        if self.narrow_triggered or self.first_lost_handled:
                            if self.first_lost_handled:
                                # 完全丢线计数
                                if avg_offset is None: self.lost_count += 1
                                else: self.lost_count = 0

                                if not self.narrow_triggered and self.stairs_triggered_once:
                                    # 丢线 → 惯性记忆（仅楼梯后启用）
                                    if avg_offset is None:
                                        if (self.state not in [LOST_MEMORY, LOST_SEARCH]
                                                and now_s() >= self.post_stairs_right_search_until):  # 2026-08-09: 衔接期内无线右找, 不惯性记忆
                                            self.last_turn_direction = -np.sign(self._last_heading_error) if self._last_heading_error != 0 else 1.0
                                            self._transition_to(LOST_MEMORY); continue
                        else:
                            # 2026-08-03: 跳跃1完成后才允许丢线触发窄道
                            if self.jump_phase >= 1:
                                if avg_offset is not None: self.lost_count = 0
                                else:
                                    self.lost_count += 1
                                    if self.lost_count > self.transient_lost_frames:
                                        print(f"[丢线] 🔄 第一次丢线(帧:{self.lost_count})，触发窄道模式")
                                        self._transition_to(NARROW_APPROACH); continue

                if self.state == TRACKING:
                    vx, wz = self._handle_tracking(centers, avg_offset, first_layer_valid, pattern, display, color, depth)
                elif self.state == CORNER_APPROACH:
                    vx, wz = self._handle_corner_approach(centers, avg_offset, display, color, depth)
                elif self.state == CORNER_TURN:
                    vx, wz = self._handle_corner_turn(centers, avg_offset, display, color, depth)
                elif self.state == LOST_MEMORY:
                    vx, wz = self._handle_lost_memory(centers, avg_offset, display, color, depth)
                elif self.state in [LOST_SEARCH, LOST_STOP]:
                    vx, wz = self._handle_lost_search(centers, avg_offset, display, color, depth)
                elif self.state == RED_APPROACH:
                    vx, wz = self._handle_red_approach(display)
                elif self.state == TURN_BACK:
                    vx, wz = self._handle_turn_back(display)
                elif self.state == PLACE_AFTER_RED:
                    vx, wz = self._handle_place_after_red(display)
                elif self.state == STAIRS:
                    vx, wz = self._handle_stairs(display)
                elif self.state == BLUE_STOP:
                    vx, wz = self._handle_blue_stop(display)
                elif self.state == BLUE_GO_STRAIGHT:
                    vx, wz = self._handle_blue_go_straight(display)
                elif self.state == BLUE_TURN_LEFT:
                    vx, wz = self._handle_blue_turn_left(display)
                elif self.state == BLUE_FINAL_APPROACH:
                    vx, wz = self._handle_blue_final_approach(display)
                elif self.state == BLUE_SIT_DOWN:
                    vx, wz = self._handle_blue_sit_down(display)
                elif self.state == JUMP:
                    vx, wz = self._handle_jump(display)
                elif self.state == POST_JUMP_ALIGN:
                    vx, wz = self._handle_post_jump_align(display)
                elif self.state == PLATFORM1_TURN:
                    vx, wz = self._handle_platform1_turn(display)
                elif self.state == PLATFORM1_FORWARD:
                    vx, wz = self._handle_platform1_forward(display)
                elif self.state == PLATFORM2_STOP:
                    vx, wz = self._handle_platform2_stop(display)
                elif self.state == GRASP_FORWARD:
                    vx, wz = self._handle_grasp_forward(display)
                elif self.state == GRASP_TURN:
                    vx, wz = self._handle_grasp_turn(display)
                elif self.state == GRASP_DONE:
                    vx, wz = self._handle_grasp_done(display)
                elif self.state == GRASP_ARM_1:
                    vx, wz = self._handle_grasp_arm_1(display)
                elif self.state == GRASP2_TURN_RIGHT:
                    vx, wz = self._handle_grasp2_turn_right(display)
                elif self.state == GRASP2_DETECT:
                    vx, wz = self._handle_grasp2_detect(display)
                elif self.state == GRASP2_APPROACH:
                    vx, wz = self._handle_grasp2_approach(display)
                elif self.state == GRASP_ARM_2:
                    vx, wz = self._handle_grasp_arm_2(display)
                elif self.state == GRASP2_DONE:
                    vx, wz = self._handle_grasp2_done(display)

                # 2026-08-04: 只要右转就标志即将中转 (替代"下一个右转弯道"直角检测)
                if (self.first_grasp_held and not self.second_phase_started
                        and self.transfer_on_next_right
                        and now_s() - self.transfer_timer_start >= self.transfer_detect_delay  # 2026-08-05: 转完60°后4.5s才开启检测
                        and self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]):
                    if wz < self.transfer_right_wz_thresh:  # 负 = 右转
                        if self._transfer_right_wz_since is None:
                            self._transfer_right_wz_since = now_s()
                        elif now_s() - self._transfer_right_wz_since >= self.transfer_right_confirm_time:
                            print(f"[抓取2] 🌀 持续右转 (wz={wz:.2f} {self.transfer_right_confirm_time:.1f}s), 标志即将中转, 循迹{Z0_TRANSFER_FWD_DIST:.1f}m后停车")
                            self.transfer_on_next_right = False
                            self.transfer_right_time = now_s()
                            self.transfer_dist = 0.0
                            self._transfer_last_rt = 0.0
                            self._transfer_right_wz_since = None
                    else:
                        self._transfer_right_wz_since = None

                # 2026-08-05: 右转标志后循迹累计1.2m路程 → 最后0.3m减速 → 停车中转
                if (self.first_grasp_held and not self.second_phase_started
                        and self.transfer_right_time > 0
                        and self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]):
                    rt = now_s() - self.transfer_right_time
                    if self._transfer_last_rt > 0:
                        self.transfer_dist += vx * (rt - self._transfer_last_rt)
                    self._transfer_last_rt = rt
                    if self.transfer_dist >= Z0_TRANSFER_FWD_DIST:
                        print(f"[抓取2] 🌀 右转标志后循迹{self.transfer_dist:.2f}m, 停车开启中转")
                        self.transfer_right_time = 0.0
                        self._transfer_last_rt = 0.0
                        self.transfer_dist = 0.0
                        self.second_phase_started = True
                        self.sport.StopMove(); time.sleep(0.5)
                        # 2026-08-05: 中转前左平移 0.3m/s × 0.5s ≈ 0.15m (用户指定)
                        self.sport.Move(0.0, self.transfer_left_shift_vy, 0.0); time.sleep(self.transfer_left_shift_time)
                        self.sport.StopMove(); time.sleep(0.2)
                        self._transition_to(GRASP_ARM_2)
                        continue
                    if self.transfer_dist >= Z0_TRANSFER_FWD_DIST - Z0_TRANSFER_FWD_DECEL_DIST:
                        decel = 1.0 - (self.transfer_dist - (Z0_TRANSFER_FWD_DIST - Z0_TRANSFER_FWD_DECEL_DIST)) / Z0_TRANSFER_FWD_DECEL_DIST
                        vx = max(vx * decel, 0.05)
                        wz = 0.0  # 减速期间不打角速度, 直线
                        if int(now_s() * 2) % 2 == 0:
                            print(f"[抓取2] 中转减速中... {self.transfer_dist:.2f}m / {Z0_TRANSFER_FWD_DIST:.1f}m (vx={vx:.2f})")


                self.sport.Move(vx, self.vy, wz)
                self.vy = 0.0  # 每帧重置，各 handler 按需设置

                # 【集成】2026-08-02: 抓A后不再循迹到中转 (钩子1/2/3已删除),
                # 抓A完成后直接右转60° → 识别grasp → 中转抓B → 左转90°衔接循迹

                if SHOW_GUI and display is not None and self.state != NARROW_EXECUTING:
                    cv2.imshow("Go2 Full", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'): break
        except KeyboardInterrupt:
            print("\n[用户] Ctrl+C 中断")
        finally:
            self.vui_client.SetBrightness(0)
            self.sport.StopMove(); time.sleep(0.5)
            # 2026-08-04: 趴下前必须先收回机械臂到0位 (桥home_all: [0,-90,90,0,0,0,0])
            if getattr(self, 'arm', None) is not None:
                try:
                    self.arm.enable()
                    time.sleep(0.5)
                    self.arm.home()
                    print("[退出] 机械臂已回归0位")
                except Exception as e:
                    print(f"[退出] ⚠️ 机械臂归零失败: {e}")
            self.sport.StandDown()
            self.detector.stop()
            if SHOW_GUI: cv2.destroyAllWindows()
            print("[退出] 完成")

    # ==================== 状态处理函数 ====================
    def _euler_fast(self, roll=0.0, pitch=0.0, yaw=0.0):
        """Euler 快速版 (2026-08-07): 临时缩短超时再调, 防同步调用卡死主循环 5s"""
        self.sport.SetTimeout(0.5)
        try:
            self.sport.Euler(roll, pitch, yaw)
        finally:
            self.sport.SetTimeout(5.0)

    def _handle_stairs(self, display):
        # 2026-08-07: 后仰时机 = 直行一开始 (等第一帧 Move 发出后再发 Euler, 防打断直行)
        if not self._stairs_pitch_sent and self.stairs_head_up_pitch != 0.0 and \
                now_s() - self.stairs_phase_start >= 0.15:
            self._stairs_pitch_sent = True
            self.stairs_pitch = -self.stairs_head_up_pitch
            self._euler_fast(0.0, self.stairs_pitch, 0.0)
            self._stairs_pitch_last = now_s()
            print(f"[楼梯] 后仰 {self.stairs_head_up_pitch*57.2958:.1f}° 生效中 (直行开始)")
        # 2026-08-07: 直行中每 1s 重发保持后仰 (防步态把俯仰拉回去)
        if self.stairs_pitch != 0.0 and now_s() - self._stairs_pitch_last >= 1.0:
            self._euler_fast(0.0, self.stairs_pitch, 0.0)
            self._stairs_pitch_last = now_s()
        if self.stairs_phase == STAIRS_PHASE_FORWARD:
            elapsed = now_s() - self.stairs_phase_start
            # 2026-08-06: 第一段直行开头左转 (用户确认: 偏航左转 0.3rad/s×0.1s≈1.7°)
            if elapsed < self.stairs_fwd_left_turn_time:
                return self.stairs_forward_speed, self.stairs_fwd_left_turn_wz
            # 2026-08-07: 先直行走一段再左平移 (用户指定, 不放开头) — 0.3m/s×0.5s
            if elapsed < self.stairs_fwd_left_shift_delay:
                return self.stairs_forward_speed, 0.0
            if elapsed < self.stairs_fwd_left_shift_delay + self.stairs_fwd_left_shift_time:
                if int(elapsed * 4) % 4 == 0:
                    print(f"[楼梯] 直行左平移中... {elapsed:.1f}s / {self.stairs_fwd_left_shift_delay + self.stairs_fwd_left_shift_time:.1f}s")
                self.vy = self.stairs_fwd_left_shift_vy
                return self.stairs_forward_speed, 0.0
            if elapsed < self.stairs_forward_duration:
                if int(elapsed * 2) % 2 == 0:
                    print(f"[楼梯] 纯直行中... {elapsed:.1f}s / {self.stairs_forward_duration}s")
                return self.stairs_forward_speed, 0.0
            else:
                # 2026-08-05: 纯直行走完再转 (不叠加)
                print(f"[楼梯] 直行完成，开始转弯 ({self.stairs_turn_duration:.2f}s)")
                print("[步态] 🚶 转弯继续 ClassicWalk (全程)")
                # 2026-08-15: 楼梯全程 ClassicWalk (用户: 转弯也用), 楼梯结束才切回 trot
                # 2026-08-07: 已上平台, 恢复身体水平 (后仰结束)
                if self.stairs_pitch != 0.0:
                    self._euler_fast(0.0, 0.0, 0.0)
                    self.stairs_pitch = 0.0
                    print("[楼梯] 俯仰恢复水平")
                self.stairs_phase = STAIRS_PHASE_TURN
                self.stairs_phase_start = now_s()
                # 2026-08-08: 过渡帧返回 θ=0 起始速度 (vx=0, wz=omega); 下一帧 TURN 分支接管 vy=vy0·cosθ
                return 0.0, self.stairs_turn_omega
        elif self.stairs_phase == STAIRS_PHASE_TURN:
            elapsed = now_s() - self.stairs_phase_start
            remain_turn = self.stairs_turn_duration  # 2026-08-08: overlap 已归零, duration = 实际转动时间
            if elapsed < remain_turn:
                wz = self.stairs_turn_omega
                # 2026-08-08: 实时变速度合运动转弯 (用户方案): θ=wz·t 实时转角,
                # vx=vy0·sinθ, vy=vy0·cosθ → 世界系x向速度恒0 (投影扫掠⊂90×90边界),
                # 横向位移 vy0×T=60cm (0.3×2.0s), 转完 θ=90° 时 vx=vy0/vy=0 无缝衔接循迹
                _theta = wz * elapsed
                vx_t = self.stairs_turn_vy0 * np.sin(_theta)  # 2026-08-08: 恢复左移版纯公式
                vy_t = self.stairs_turn_vy0 * np.cos(_theta)  # 2026-08-08: 左平移 (用户)
                self.vy = vy_t  # 正=左
                # 2026-08-06: 转弯中读相机, 只按左侧黑线转向 (右侧黑线忽略, 防被带偏)
                try:
                    ct, dt = self.detector.get_frames()
                    if ct is not None:
                        _, _, centers_t = self.detector.detect_layers(ct, dt)
                        avg_off = self.compute_weighted_offset(centers_t)
                        if avg_off is not None and avg_off < -self.stairs_turn_left_px_thresh:
                            wz = min(self.stairs_turn_omega_max,
                                     wz + self.stairs_turn_follow_gain * (-avg_off - self.stairs_turn_left_px_thresh))
                except Exception:
                    pass
                if int(elapsed * 2) % 2 == 0:
                    print(f"[楼梯] 合运动转弯中... {elapsed:.1f}s / {remain_turn:.2f}s "
                          f"(θ={_theta*57.2958:.0f}° vx={vx_t:.2f} vy={vy_t:.2f} wz={wz:.2f})")
                return vx_t, wz
            else:
                # 2026-08-08: 转弯完成 → 前移补偿 (用户: 转完后机器狗偏后) → 衔接循迹
                if not self._stairs_turn_comp:
                    self._stairs_turn_comp = True
                    self._stairs_comp_t = now_s()
                    print(f"[楼梯] 转弯完成, 前移补偿 {self.stairs_turn_comp_time:.1f}s...")
                if now_s() - self._stairs_comp_t < self.stairs_turn_comp_time:
                    return self.stairs_turn_comp_vx, 0.0
                print("[楼梯] ✅ 转弯+补偿完成，直接衔接循迹...")
                print("[步态] 🚶 楼梯结束, 切回 trot")
                switch_gait(self.sport, GaitMode.CLASSIC_WALK, False)  # 2026-08-08: 楼梯结束切回 trot (用户)
                self.sport.StopMove(); time.sleep(0.3)
                self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.lost_count = 0
                self.state = TRACKING; self.state_start_time = now_s()
                self.corner_confirm_count = 0
                self.post_stairs_until = 0.0  # 立即用全部10层循迹
                self.post_stairs_right_search_until = now_s() + self.post_stairs_right_search_time  # 2026-08-09: 衔接期无线右找
                # 爬楼梯完成 → 开启 grasp 图案识别
                self.grasp_detect_enabled = True
                self.grasp_enable_after = now_s() + Z0_GRASP_ENABLE_DELAY  # 2026-08-03 (z0): 先循迹走一段再开识别 (7.5s)
                self.grasp_detect_start_time = now_s()  # grasp 兜底计时起点 (2026-08-03)
                print("[抓取识别] 🟢 爬楼梯完成，开启 grasp 图案识别")
                return 0.0, 0.0
        elif self.stairs_phase == STAIRS_PHASE_SHIFT:
            elapsed = now_s() - self.stairs_phase_start
            if elapsed < self.stairs_shift_duration:
                self.vy = -self.stairs_shift_vy  # 2026-08-04: 固定往右平移 (负=右)
                if int(elapsed * 4) % 4 == 0:
                    print(f"[楼梯] 侧移修正中... {elapsed:.1f}s / {self.stairs_shift_duration}s")
                return 0.0, 0.0
            else:
                # 侧移完成 → 恢复循迹，但2s内只看近处黑线（忽略远处十字叉）
                print("[楼梯] ✅ 爬楼梯+侧移修正完成，用近处黑线冲过十字叉...")
                self.sport.StopMove(); time.sleep(0.3)
                self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.lost_count = 0
                self.state = TRACKING; self.state_start_time = now_s()
                self.corner_confirm_count = 0
                self.post_stairs_until = 0.0  # 立即用全部10层循迹
                self.post_stairs_right_search_until = now_s() + self.post_stairs_right_search_time  # 2026-08-09: 衔接期无线右找
                # 爬楼梯完成 → 开启 grasp 图案识别
                self.grasp_detect_enabled = True
                self.grasp_enable_after = now_s() + Z0_GRASP_ENABLE_DELAY  # 2026-08-03 (z0): 先循迹走一段再开识别 (7.5s)
                self.grasp_detect_start_time = now_s()  # grasp 兜底计时起点 (2026-08-03)
                print("[抓取识别] 🟢 爬楼梯完成，开启 grasp 图案识别")
                return 0.0, 0.0

    def _handle_blue_stop(self, display):
        elapsed = now_s() - self.state_start_time
        if elapsed < 0.5: return 0.0, 0.0
        else: self.sport.StopMove(); time.sleep(0.2); self._transition_to(BLUE_GO_STRAIGHT); return 0.0, 0.0

    def _handle_blue_go_straight(self, display):
        elapsed = now_s() - self.state_start_time
        expected_time = self.blue_go_straight_distance / self.blue_go_straight_speed
        if elapsed < expected_time: return self.blue_go_straight_speed, 0.0
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_TURN_LEFT); return 0.0, 0.0

    def _handle_blue_turn_left(self, display):
        elapsed = now_s() - self.state_start_time
        turn_radians = np.radians(self.blue_turn_angle)
        expected_time = turn_radians / self.blue_turn_speed
        if elapsed < expected_time: return 0.0, self.blue_turn_speed
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_FINAL_APPROACH); return 0.0, 0.0

    def _handle_blue_final_approach(self, display):
        elapsed = now_s() - self.state_start_time
        expected_time = self.blue_final_distance / self.blue_final_speed
        if elapsed < expected_time: return self.blue_final_speed, 0.0
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_SIT_DOWN); return 0.0, 0.0

    def _handle_blue_sit_down(self, display):
        elapsed = now_s() - self.state_start_time
        if elapsed < 1.0: return 0.0, 0.0
        else:
            self._ensure_level_standing()  # 2026-08-04: 坐下前身体调平
            # 2026-08-04: 回到启停区, 机械臂收回用户设置的0位
            try:
                arm = D1UDPClient('192.168.123.100')
                arm.enable(); time.sleep(0.5)
                self._safe_move_arm(arm, self._zero_pose, '蓝区收臂')  # 2026-08-10: 分段插值防突变
                self._wait_arm_at(arm, self._zero_pose)  # 2026-08-04: 等真正到位 (2026-08-09: 原_zp未定义NameError已修)
                print('[机械臂] ✅ 已收回0位')
            except Exception as e:
                print(f'[机械臂] ⚠️ 收回0位失败: {e}')
            self.sport.StandDown(); time.sleep(0.5); self.blue_stop_detected = True; return 0.0, 0.0

    def _detect_platform_zone_timeout(self, timeout=Z0_PLACE_DETECT_TIMEOUT):
        """2026-08-03 (z0): 抓A确认后退2s后识别放置区 (patternA=1号/patternB=2号).
        识别到即停; 超时未识别到 → 随机选一个放置区."""
        print(f"[平台] 📍 识别放置平台 (patternA=1号, patternB=2号), 超时 {timeout:.0f}s...")
        deadline = now_s() + timeout
        best_id = None
        while now_s() < deadline:
            color, depth = self.detector.get_frames()
            if color is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                res = self.recognizer.match_platform(gray)
                if res:
                    best_id = res[0]
                    print(f"[平台] 🎯 识别到: {SIGN_NAMES[best_id]}")
                    break
            time.sleep(0.1)
        if best_id is None:
            best_id = random.choice([SignID.PLATFORM_A, SignID.PLATFORM_B])
            print(f"[平台] ⚠️ {timeout:.0f}s 未识别到放置平台, 随机选择: {SIGN_NAMES[best_id]}")
        self.platform_zone = 'A' if best_id == SignID.PLATFORM_A else 'B'
        print(f"[平台] 📍 放置区: {SIGN_NAMES[best_id]}")
        return best_id

    def _match_grasp(self, gray, min_threshold=Z0_GRASP_PROBE_THRESHOLD):
        """2026-08-03 (z0): grasp 标志 = 4×4 黑白棋盘格 → 棋盘格角点检测 (findChessboardCornersSB).
        检测到 3×3 内角点网格 → 返回 Z0_CHESS_CORNER_COUNT (必过高置信档立即确认)"""
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, Z0_CHESS_INNER_CORNERS, cv2.CALIB_CB_EXHAUSTIVE)
        except Exception:
            return None
        if found:
            return Z0_CHESS_CORNER_COUNT
        return None

    def _handle_grasp_forward(self, display):
        """2026-08-03 (z0 嵌入): 识别到grasp后: 边前进+边右平移+边左转弯, 走 Z0_GRASP_FWD_TIME 秒后停止."""
        elapsed = now_s() - self.state_start_time
        if elapsed < Z0_GRASP_FWD_TIME:
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取] 前进+右平移+左转中... {elapsed:.1f}s / {Z0_GRASP_FWD_TIME}s")
            self.vy = -Z0_GRASP_FWD_VY   # 右平移 (vy 正值=左)
            return Z0_GRASP_FWD_VX, Z0_GRASP_FWD_WZ
        else:
            print(f"[抓取] ✅ 识别grasp后组合走 {Z0_GRASP_FWD_TIME:.1f}s 完成，停下机器狗")
            self.sport.StopMove(); time.sleep(0.3)
            self._transition_to(GRASP_TURN)
            return 0.0, 0.0

    def _handle_grasp_turn(self, display):
        elapsed = now_s() - self.state_start_time
        if elapsed < 0.5:
            return 0.0, 0.0  # 先停稳
        turn_radians = np.radians(self.grasp_turn_angle)
        expected_time = turn_radians / self.grasp_turn_speed
        if elapsed < 0.5 + expected_time:
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取] 左转中... {elapsed-0.5:.1f}s / {expected_time:.1f}s ({self.grasp_turn_angle}°)")
            return 0.0, self.grasp_turn_speed
        else:
            print(f"[抓取] ✅ 左转 {self.grasp_turn_angle}° 完成")
            self.grasp_processed = True
            self.sport.StopMove(); time.sleep(0.3)
            if self.grasp_target_phase == 'second':
                self.second_phase_started = True
                print("[抓取2] 进入中转抓取 (放A抓B)")
                self._transition_to(GRASP_ARM_2)
            else:
                self._transition_to(GRASP_ARM_1)
            return 0.0, 0.0

    def _handle_grasp_done(self, display):
        self.sport.StopMove()
        return 0.0, 0.0

    # ==================== 【集成】3D视觉抓取 (grasp_3d 管线) ====================
    def _ensure_level_standing(self, settle=1.5):
        """2026-08-04: 身体调平 - BalanceStand 主动平衡站立, 身体水平不倾斜"""
        try:
            ret = self.sport.BalanceStand()
            if ret == 0:
                print("[姿态] ✅ BalanceStand 平衡站立 (身体水平)")
                time.sleep(settle)
                return True
            print(f"[姿态] ⚠️ 平衡站立失败(码:{ret})")
            return False
        except Exception as e:
            print(f"[姿态] ⚠️ 平衡站立异常: {e}")
            return False

    def _run_embedded_grasp(self, phase):
        """【集成】运行嵌入式 3D 抓取管线 (移植自 d1_arm/arm_control/grasp_3d.py)

        phase='first' : 抓第1个黄色物块A, 容量确认(1/1)后返回, 夹爪保持闭紧
        phase='second': 起始即夹持A, 找第2个物块B → 中转放下A → 抓B → 完成返回
        相机不冲突: 抓取用夹爪相机(锁定序列号335222075495), 巡线用另一深度相机
        """
        print(f"[抓取] 🤖 启动3D视觉抓取管线 (phase={phase})...")
        self.sport.StopMove(); time.sleep(0.5)
        self._ensure_level_standing()  # 2026-08-04: 抓取/中转前身体调平
        # 2026-08-10 (评审点1): 抓取是同步阻塞调用, 期间巡线相机无人取帧,
        # 30fps 持续 streaming 帧队列积压易触发 USB 掉线 (errno=5, 当日两连崩同一位置).
        # 抓取前停巡线相机释放 USB, 抓取结束重建 (失败交给 get_frames 自恢复兜底)
        self.detector.stop()
        pipeline = GraspPipeline3D(
            mode='hsv', dry_run=False,
            sport_client=self.sport,   # 复用巡线运动客户端, 不重建DDS
            phase=phase,
            enable_mjpeg=False,        # 集成模式关闭HTTP流
        )
        try:
            try:
                pipeline.run()
            except Exception as e:
                print(f"[抓取] ⚠️ 抓取管线异常: {e}")
            except SystemExit as e:  # 2026-08-10 (评审点4): init_arm 桥死 sys.exit 只杀抓取, 不杀主程序
                print(f"[抓取] ⚠️ 抓取管线 SystemExit({e}): 臂桥故障, 按抓取失败降级继续巡线")
        finally:
            try:
                _t1 = now_s()
                self.detector._rebuild_pipeline()
                print(f'[相机] ✅ 抓取结束, 巡线相机已重启 ({now_s()-_t1:.1f}s)')
            except Exception as e:
                print(f'[相机] ⚠️ 抓取后巡线相机重建失败: {e} — 交给 get_frames 自恢复')
        ok = bool(getattr(pipeline, 'holding_block', False) or
                  getattr(pipeline, 'transfer_done', False))
        print(f"[抓取] {'✅ 成功' if ok else '⚠️ 未成功'} "
              f"(holding={pipeline.holding_block}, transfer={pipeline.transfer_done})")
        time.sleep(1.0)
        self.sport.StopMove(); time.sleep(0.5)
        return ok

    def _detect_platform_zone(self, max_frames=5, interval=0.25):
        """2026-08-02: 左转100°后抓A前, 单独用循迹深度相机识别放置平台图案.
        patternA=1号平台, patternB=2号平台, 多帧投票, 仅记录打印不影响流程."""
        print("[平台] 📍 抓A前识别放置平台图案 (patternA=1号, patternB=2号)...")
        votes = {}
        for i in range(max_frames):
            color, depth = self.detector.get_frames()
            if color is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                res = self.recognizer.match_platform(gray)
                if res:
                    sign_id, cnt = res
                    votes[sign_id] = votes.get(sign_id, 0) + 1
                    print(f"[平台] 第{i+1}帧: {SIGN_NAMES[sign_id]} ({cnt}点)")
            time.sleep(interval)
        if not votes:
            self.platform_zone = None
            print("[平台] ⚠️ 未识别到放置平台图案 (非A/B)")
            return None
        best_id = max(votes, key=votes.get)
        self.platform_zone = 'A' if best_id == SignID.PLATFORM_A else 'B'
        print(f"[平台] 📍 识别到: {SIGN_NAMES[best_id]} (票数 {votes[best_id]}/{max_frames})")
        return best_id

    def _execute_place_after_red(self, zone):
        """2026-08-04: 红点后放置物块. IK目标: 狗侧向22cm/高5cm.
        1号放置区=angle0左转90°(基座Y负侧), 2号=angle0右转90°(Y正侧).
        放置在中转平台抓取的物资到放置平台, 放下后回ready"""
        print(f"[放置] 📦 开始放置 (编号: {'1号' if zone == 'A' else '2号' if zone == 'B' else '未知'})")
        if zone not in ('A', 'B'):
            print("[放置] ⚠️ 平台编号未识别, 跳过放置")
            return False
        self._ensure_level_standing()  # 2026-08-04: 放置前身体调平
        side = -1.0 if zone == 'A' else 1.0   # 1号=左转=基座Y负, 2号=右转=Y正
        target_a0 = -90.0 if zone == 'A' else 90.0  # 2026-08-04: 1号=angle0左转90°(负=左), 2号=angle0右转90°
        place_pt = np.array([0.0, side * PLACE_SIDE_Y, PLACE_HEIGHT_Z])
        hover_pt = place_pt + np.array([0.0, 0.0, PLACE_LIFT_Z])
        print(f"[放置] 目标点: {np.round(place_pt, 3)}m, hover: {np.round(hover_pt, 3)}m")
        try:
            arm = D1UDPClient('192.168.123.100')
            arm.enable()
            time.sleep(1.0)
            # 2026-08-04: 停下后先变到 ready 姿态, 再用IK解 (0位折叠姿态对侧向低位点解不出)
            # 2026-08-10: 裸move_joints改走_safe_move_arm (分段插值防突变)
            self._safe_move_arm(arm, {'angle0': 0, 'angle1': -2, 'angle2': 42,
                                      'angle3': 0, 'angle4': -50, 'angle5': 0}, 'Place Ready')
            time.sleep(1.5)
            # 2026-08-04: 按平台编号 angle0 左/右转 90°, 再放置到放置平台
            print(f"[放置] angle0 {'左转' if zone == 'A' else '右转'}90° → 放置平台")
            self._safe_move_arm(arm, {'angle0': target_a0}, 'Place Rotate')
            self._wait_arm_at(arm, {'angle0': target_a0})  # 2026-08-06: 等a0转到平台侧到位
            # 2026-08-06: 放置写死舵机角 (固定角度FK落点+上移7cm+平台侧外5cm 的IK解, 用户指定不再IK实时解)
            if zone == 'A':
                hover_angles = {'angle0': -89.0, 'angle1': -0.15, 'angle2': 33.92,
                                'angle3': -0.89, 'angle4': -8.11, 'angle5': 0.0}
                place_angles = {'angle0': -89.0, 'angle1': 21.2, 'angle2': 36.23,
                                'angle3': -1.37, 'angle4': 12.97, 'angle5': 0.0}  # 2026-08-07: a1-3°, a2+2° (用户)
            else:
                hover_angles = {'angle0': 89.0, 'angle1': -0.21, 'angle2': 34.64,
                                'angle3': -5.29, 'angle4': -10.68, 'angle5': 0.0}
                place_angles = {'angle0': 89.0, 'angle1': 21.13, 'angle2': 36.95,
                                'angle3': -4.83, 'angle4': 10.44, 'angle5': 0.0}  # 2026-08-07: a1-3°, a2+2° (用户)
            print(f"[放置] 写死舵机角 → hover: {hover_angles}")
            print(f"[放置] 写死舵机角 → 放置位: {place_angles}")
            # 1. hover 预到位
            self._safe_move_arm(arm, hover_angles, 'Place Hover')
            self._wait_arm_at(arm, hover_angles)  # 2026-08-06: 等hover到位
            # 2. 放置位
            self._safe_move_arm(arm, place_angles, 'Place Down')
            if not self._wait_arm_at(arm, place_angles):  # 等放置位到位才张爪
                print("[放置] ⚠️ 10s未到放置角度, 仍尝试张开")
            time.sleep(0.2)
            # 3. 张开放下 (angle6 夹爪)
            print("[放置] 张开夹爪(angle6)放下物块...")
            arm.open_gripper()
            time.sleep(0.8)
            # 归 0 位 (2026-08-08: 用户指定顺序 — 先不动angle0, 松爪后抬大臂小臂, 再转angle0回0位)
            lift_angles = {'angle0': place_angles['angle0'],
                           'angle1': hover_angles['angle1'], 'angle2': hover_angles['angle2'],
                           'angle3': hover_angles['angle3'], 'angle4': hover_angles['angle4'],
                           'angle5': 0.0}  # 大臂小臂抬起到 hover 高度, angle0 保持不动
            self._safe_move_arm(arm, lift_angles, 'Lift Arm')
            self._wait_arm_at(arm, lift_angles)
            self._safe_move_arm(arm, {'angle0': 0.0}, 'Rotate a0 Zero')
            self._wait_arm_at(arm, {'angle0': 0.0})
            zp = {'angle0': 0, 'angle1': -90, 'angle2': 90,
                    'angle3': 0, 'angle4': 0, 'angle5': 0}  # 2026-08-06: 用户零位 (折叠)
            self._safe_move_arm(arm, zp, 'Home Zero')
            deadline = now_s() + 15.0
            at_zero = False
            while now_s() < deadline:
                try:
                    arm.query_angles()
                    cur = dict(arm.get_angles())
                    if cur and all(abs(cur.get(k, 999) - zp.get(k, 0)) <= 5 for k in zp):
                        at_zero = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if at_zero:
                print("[放置] ✅ 放置完成, 机械臂已回0位 (先抬臂后转a0)")
            else:
                print("[放置] ⏱️ 放置完15s机械臂未回0位, 继续后面流程")
            time.sleep(2.0)  # 2026-08-06: 放完停2s再循迹/开启跳2检测 (用户指定)
            return True
        except Exception as e:
            print(f"[放置] ⚠️ 放置异常: {e}")
            return False

    def _restart_arm_bridge(self):
        """2026-08-05晚: SSH重启臂板桥 (arm-udp-bridge.service) — 桥启动自动使能+归零"""
        import subprocess
        try:
            r = subprocess.run(
                "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@192.168.123.100 "
                "'echo 123 | sudo -S systemctl restart arm-udp-bridge'",
                shell=True, capture_output=True, text=True, timeout=15)
            print(f"[放置] 🔄 臂桥已重启 (rc={r.returncode})")
        except Exception as e:
            print(f"[放置] ⚠️ 臂桥重启失败: {e}")

    def _wait_arm_at(self, arm, target, tolerance=5.0, timeout=10.0):
        """2026-08-04: 轮询等待机械臂到位 (move_joints 是异步命令, 固定sleep不够)"""
        deadline = now_s() + timeout
        while now_s() < deadline:
            try:
                arm.query_angles()
                cur = dict(arm.get_angles())
                if cur:
                    ok = True
                    for k, v in target.items():
                        if k in cur and abs(cur[k] - v) > tolerance:
                            ok = False
                            break
                    if ok:
                        vals = [str(int(round(cur['angle%d' % i]))) for i in range(6)]
                        print(f"  [Arm] ✅ 到位: {vals}")
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        print('  [Arm] ⚠️ 等待到位超时')
        return False

    def _retract_zero_fast(self, arm):
        """2026-08-11 v3: 快速收零 — 免查询免使能, 3段插值 + mode=0(快速) 发零位,
        臂物理折叠大幅提速 (预计3-6s, 原15-30s); 夹爪不动 (zero_pose 无 angle6).
        若甩动明显, 可改回 5 段或 mode=1."""
        current = {'angle0': 0, 'angle1': -2, 'angle2': 42,
                   'angle3': 0, 'angle4': -60, 'angle5': 0}  # ready 姿态常量
        steps = 3
        for s in range(1, steps):
            wp = {}
            for k, v in self._zero_pose.items():
                wp[k] = current.get(k, v) + (v - current.get(k, v)) * (s / steps)
            print(f'  [收零] waypoint {s}/{steps} (快速)')
            arm.move_joints(wp, mode=0)
            time.sleep(0.15)
        arm.move_joints(self._zero_pose, mode=0)
        print('  [收零] 零位指令已发出 (快速)')

    def _safe_move_arm(self, arm, angles_dict, label='Move'):
        """放置用安全移动: 大角度差分两段防甩"""
        arm.query_angles()
        current = dict(arm.get_angles())
        if current:
            max_delta = 0
            for i in range(6):
                k = f'angle{i}'
                if k in angles_dict and k in current:
                    max_delta = max(max_delta, abs(angles_dict[k] - current[k]))
            # 2026-08-10 (用户): 大角度跳变分段插值防舵机突变 (原仅1个中点)
            if max_delta > 15:
                steps = 5 if max_delta > 60 else (3 if max_delta > 30 else 2)
                for s in range(1, steps):
                    wp = {}
                    for k, v in angles_dict.items():
                        wp[k] = current.get(k, v) + (v - current.get(k, v)) * (s / steps)
                    print(f"  [Slow] {label} waypoint {s}/{steps} (delta={max_delta:.0f}deg)")
                    arm.move_joints(wp)
                    time.sleep(0.6)
        arm.move_joints(angles_dict)
        time.sleep(0.3)

    def _handle_grasp_arm_1(self, display):
        """【集成】抓第一个黄色物块A — 2026-08-11: 失败支路已删, 失败原地重试整轮,
        达上限按已夹持继续中转 (不降级跳过)"""
        self._grasp_a_rounds = getattr(self, '_grasp_a_rounds', 0) + 1
        ok = self._run_embedded_grasp(phase='first')
        self.first_grasp_done = True          # 抓取流程已执行
        self.first_grasp_held = True          # 2026-08-11: 恒 True — 中转链恒开启 (失败按已夹持)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        self.lost_count = 0; self.corner_confirm_count = 0
        if not ok and self._grasp_a_rounds < Z0_GRASP_A_ROUNDS_MAX:
            # 2026-08-11 (用户): 抓A失败支路已删除 — 原地重试整轮 (每轮含6次尝试)
            print(f"[抓取1] ⚠️ 抓A失败 (第{self._grasp_a_rounds}/{Z0_GRASP_A_ROUNDS_MAX}轮), 原地重试整轮...")
            self.sport.StopMove(); time.sleep(1.5)
            return 0.0, 0.0  # 保持 GRASP_ARM_1, 下帧重抓
        if ok:
            print(f"[抓取1] ✅ 确认抓到物块A")
        else:
            print(f"[抓取1] ⚠️ {Z0_GRASP_A_ROUNDS_MAX} 轮均未确认夹持, 按已夹持继续 (爪空)")
        # 2026-08-11: 无论是否确认, 统一后退2s@0.25 (~0.5m) — 失败也后退, 不再停在抓取点
        print(f"[抓取1] 后退 {Z0_BACKUP_TIME:.2f}s @ {Z0_BACKUP_SPEED}m/s (~0.375m)")
        self.sport.Move(-Z0_BACKUP_SPEED, 0, 0); time.sleep(Z0_BACKUP_TIME)
        self.sport.StopMove(); time.sleep(0.3)
        self._detect_platform_zone_timeout()
        print(f"[抓取1] 右转60°")
        self._transition_to(GRASP2_TURN_RIGHT)
        return 0.0, 0.0

    def _handle_grasp2_turn_right(self, display):
        """【集成】抓A后: 右转60° + 以循迹速度前进3.5s (边转边走) → 减速停下 → 开中转 (2026-08-04)"""
        elapsed = now_s() - self.state_start_time
        turn_radians = np.radians(self.grasp2_turn_right_angle)
        expected_time = turn_radians / self.grasp_turn_speed   # 60° ≈ 1.05s @1.0rad/s
        DURATION = Z0_TRANSFER_FWD_DURATION
        DECEL = Z0_TRANSFER_FWD_DECEL
        if elapsed < 0.5:
            return 0.0, 0.0  # 先停稳
        # 2026-08-11: 抓A失败支路已删除 — 中转链恒开启 (first_grasp_held 恒 True)
        self.platform_detection_enabled = False  # 2026-08-03: 关深度平台检测, 防干扰红点识别
        if elapsed < 0.5 + expected_time:
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取2] 右转中... {elapsed-0.5:.1f}s / {expected_time:.1f}s ({self.grasp2_turn_right_angle}°)")
            return 0.0, -self.grasp_turn_speed  # 右转 = 负角速度
        # 2026-08-04: 右转60°完成 → 循迹, 只要右转 → 标志即将中转
        print(f"[抓取2] ✅ 右转 {self.grasp2_turn_right_angle}° 完成, 循迹至右转 → 停车中转")
        self.sport.StopMove(); time.sleep(0.3)
        self.transfer_on_next_right = True  # 2026-08-04: 循迹中只要右转 → 标志即将中转
        self.transfer_timer_start = now_s()  # 2026-08-04: 中转检测计时起点 (4.5s延迟用; 2026-08-05: 无11s兜底, 无限等右转)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        self._transfer_right_wz_since = None
        self._transition_to(TRACKING)
        return 0.0, 0.0

    def _handle_grasp2_detect(self, display):
        """【集成】右转60°后: 巡线相机ORB识别grasp图案, 确认后进中转抓B; 原地等不循迹"""
        elapsed = now_s() - self.state_start_time
        if elapsed < self.grasp2_detect_timeout:
            color, depth = self.detector.get_frames()
            if color is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                match = self.recognizer.match(gray)
                if match and match[0] == SignID.GRASP:
                    self.grasp2_confirm_count += 1
                    if self.grasp2_confirm_count >= self.grasp2_detect_frames:
                        print("[抓取2] 🎯 右转后识别到grasp图案, 进入中转抓B")
                        self.sport.StopMove(); time.sleep(0.3)
                        self._transition_to(GRASP_ARM_2)
                    return 0.0, 0.0
                self.grasp2_confirm_count = max(0, self.grasp2_confirm_count - 1)
            return 0.0, 0.0  # 原地等待识别, 不循迹
        else:
            print(f"[抓取2] ⚠️ {self.grasp2_detect_timeout:.0f}s 未识别到grasp图案, 仍进入抓取B")
            self._transition_to(GRASP_ARM_2)
            return 0.0, 0.0

    def _handle_grasp2_approach(self, display):
        """【集成】抓A后: 循迹7s (最后2s线性减速), 停车进行中转"""
        elapsed = now_s() - self.state_start_time
        DURATION = self.grasp2_approach_duration
        DECEL = self.grasp2_approach_decel_time
        if elapsed < DURATION:
            color, depth = self.detector.get_frames()
            if color is not None:
                _, _, centers = self.detector.detect_layers(color, depth)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is not None:
                    vx, wz = self._calculate_tracking_for_red(avg_offset)
                    if elapsed >= DURATION - DECEL:   # 最后DECEL秒线性减速到0
                        decel = 1.0 - (elapsed - (DURATION - DECEL)) / DECEL
                        vx = max(vx * decel, 0.05)
                    return min(vx, self.grasp2_approach_speed), wz
            return 0.15, 0.0   # 丢线直走, 计时不停
        else:
            print(f"[抓取2] ✅ {DURATION:.0f}s循迹完成, 停车进行中转")
            self.sport.StopMove(); time.sleep(0.5)
            self.second_phase_started = True
            self._transition_to(GRASP_ARM_2)
            return 0.0, 0.0

    def _handle_grasp_arm_2(self, display):
        """【集成】中转: 放下物块A(在B右下30°/7cm处) + 抓第二个黄色物块B"""
        phase = 'second' if self.first_grasp_held else 'first'
        print(f"[抓取2] 🤖 第二阶段抓取开始 (phase={phase})...")
        ok = self._run_embedded_grasp(phase=phase)
        print(f"[抓取2] {'✅ 中转+抓取B完成' if ok else '⚠️ 中转/抓取B未完成'}, 左转90°衔接循迹")
        self._transition_to(GRASP2_DONE)
        return 0.0, 0.0

    def _handle_grasp2_done(self, display):
        """2026-08-03 重写: 中转完 → 左转90° → 前进3s@0.3m/s → 再左转90° → 看到黑线直接衔接循迹 → 开红点"""
        self.transfer_on_next_right = False  # 2026-08-04: 中转完成, 关闭右转标志
        self.transfer_right_time = 0.0
        self.transfer_dist = 0.0
        self._transfer_last_rt = 0.0
        # 2026-08-12: 阶段3自身起点不能在顶部清零 (每帧执行会重置计时, 转弯永不完结);
        # 由停稳分支 (elapsed<t2+0.5) 负责清零
        # 2026-08-12 (用户): 收回到0位再转 — 发收零指令后等臂真正到位才开始转弯 (原并行, 用户要求改)
        if not getattr(self, '_arm_retracted_after_transfer', False):
            self._arm_retracted_after_transfer = True
            try:
                arm = D1UDPClient('192.168.123.100')
                # 2026-08-11: 跳过 enable — 管线刚用完臂必已使能, enable 桥不回显 seq 白等2s
                self._retract_zero_fast(arm)  # 2026-08-11: 分段发零位 (mode=0 快速)
                # 2026-08-12: 等臂收回0位 (轮询角度, 容差5°, 超时10s兜底不卡死)
                self._wait_arm_at(arm, self._zero_pose, tolerance=5.0, timeout=10.0)
                print('[机械臂] ✅ 已收回0位, 开始转弯')
            except Exception as e:
                print(f'[机械臂] ⚠️ 中转后收回0位失败: {e}')
            # 2026-08-11: 收臂耗时不计入转弯计时
            self.state_start_time = now_s()
        elapsed = now_s() - self.state_start_time
        turn_radians = np.radians(self.grasp2_turn_left_angle)
        expected_time = turn_radians / self.grasp_turn_speed
        # 2026-08-04: 衔接段任何时刻看到红点 → 立即进入红点流程 (红点优先, 不再错过)
        # 2026-08-11: 快速取帧 (防 30s 恢复阻塞卡死转弯控制环)
        c_now, d_now = self.detector.get_frames_quick()
        if c_now is not None:
            red_d, red_r, red_c = self.detector.detect_red_point(c_now)
            if red_d:
                print(f"[红点] 🎯 衔接段看到红点(半径{red_r}px), 直接进入红点流程")
                self.red_center = red_c; self.red_radius = red_r
                self.sport.StopMove(); time.sleep(0.2)
                self._transition_to(RED_APPROACH)
                return 0.0, 0.0
        # 2026-08-08: 容量 1/1 确认后立即走, 不再停 0.5s (用户)
        # 阶段1: 左转90° — 2026-08-11: 不看前方黑线, 完整转完 (用户)
        if elapsed < expected_time:
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取2] 左转中... {elapsed:.1f}s / {expected_time:.1f}s ({self.grasp2_turn_left_angle}°)")
            return 0.0, self.grasp_turn_speed  # 左转 = 正角速度
        # 阶段2: 前进 1.0s @ 0.3m/s (2026-08-11: 1.5s→1.0s 用户提速)
        t_fwd = expected_time
        if elapsed < t_fwd + 1.0:
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取2] ➡️ 前进 0.3m/s... {elapsed-t_fwd:.1f}s / 1.0s")
            return 0.3, 0.0
        # 阶段3: 再左转90° — 2026-08-11: 自身起点计时 (绝对窗口被 get_frames 卡顿吃掉, 实测第二左转不准)
        t2 = t_fwd + 1.0  # 2026-08-11: +3.0→+1.0 (停稳窗2.0s→0.5s, 3s→1.5s改参残留)
        if elapsed < t2 + 0.5:
            self._grasp2_turn3_start = None  # 停稳中, 转弯计时未开始
            return 0.0, 0.0
        if self._grasp2_turn3_start is None:
            self._grasp2_turn3_start = now_s()  # 转弯真正开始时刻
        if now_s() - self._grasp2_turn3_start < expected_time:
            # 2026-08-11 (用户): 最后一个左转过程中看见黑线 → 立即衔接循迹 (红点检查在函数顶部, 仍优先)
            c3, d3 = self.detector.get_frames_quick()  # 2026-08-11: 快速取帧防阻塞
            if c3 is not None:
                _, _, centers = self.detector.detect_layers(c3, d3)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is not None:
                    print(f"[抓取2] ✅ 再左转中看到黑线(偏移{avg_offset:.0f}px), 立即衔接循迹, 开启红点检测")
                    self.sport.StopMove(); time.sleep(0.3)
                    if not self.red_detect_enabled:
                        self.red_detect_enabled = True
                        print("[红点] 🟢 中转完成, 开启红点检测, 继续循迹")
                    self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
                    self._transition_to(TRACKING)
                    return 0.0, 0.0
            if int(elapsed * 2) % 2 == 0:
                print(f"[抓取2] 再左转中... {now_s()-self._grasp2_turn3_start:.1f}s / {expected_time:.1f}s")
            return 0.0, self.grasp_turn_speed
        # 阶段4: 找黑线直接衔接 (最多2s), 找到 → 循迹 + 开红点
        color, depth = self.detector.get_frames_quick()  # 2026-08-11: 快速取帧防阻塞
        if color is not None:
            _, _, centers = self.detector.detect_layers(color, depth)
            avg_offset = self.compute_weighted_offset(centers)
            if avg_offset is not None:
                print(f"[抓取2] ✅ 看到黑线(偏移{avg_offset:.0f}px), 直接衔接循迹, 开启红点检测")
                self.sport.StopMove(); time.sleep(0.3)
                if not self.red_detect_enabled:
                    self.red_detect_enabled = True
                    print("[红点] 🟢 中转完成, 开启红点检测, 继续循迹")
                self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
                self._transition_to(TRACKING)
                return 0.0, 0.0
        # 没看到黑线: 边前进边右转微调找线, 最多1.5s (2026-08-11: 2s→1.5s 提速)
        if elapsed < t2 + 0.5 + expected_time + 1.5:
            if int(elapsed * 2) % 2 == 0:
                print("[抓取2] 未看到黑线, 边前进边右转找线...")
            return 0.15, -0.3
        # 1.5s 后仍无线 → 直接循迹 (丢线逻辑兜底)
        print("[抓取2] ⚠️ 2s 未找到黑线, 直接循迹衔接, 开启红点检测")
        self.sport.StopMove(); time.sleep(0.3)
        if not self.red_detect_enabled:
            self.red_detect_enabled = True
            print("[红点] 🟢 中转完成, 开启红点检测, 继续循迹")
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
        self._transition_to(TRACKING)
        return 0.0, 0.0

    def _handle_red_approach(self, display):
        success = self.approach_red_point_with_compensate()
        if success: self._transition_to(TURN_BACK)
        else:
            # 2026-08-04: 接近失败也继续红点流程 (3.3s停+放置+跳2), 不再无限重试
            self.red_processed = True
            self._transition_to(TURN_BACK)
        self.sport.StopMove(); time.sleep(0.5); return 0.0, 0.0

    def _get_place_dist(self):
        """2026-08-14: 三个警示动作统一放置距离 (用户: 伸懒腰后退后不再区分)."""
        return self.place_after_red_dist

    def _handle_place_after_red(self, display):
        """2026-08-02: 红点处理完成后循迹3.3s停车, 按平台编号IK放置物块, 之后0.3m/s循迹"""
        # 2026-08-03: 未夹住物块A → 跳过放置 (不开启任何夹取功能), 直接循迹
        if not self.first_grasp_held:
            print("[放置] ⏭️ 未夹住物块A, 跳过放置, 直接循迹")
            self.post_place_speed_cap = True  # 2026-08-04: 0.3m/s 直到第二跳
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self._transition_to(TRACKING)
            return 0.0, 0.0
        if self._place_failed:
            # 2026-08-06: 放置失败不再保持停止 (用户指定), 继续流程
            self.post_place_speed_cap = True
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self._transition_to(TRACKING)
            return 0.0, 0.0
        if self.place_dist >= self._get_place_dist():
            print(f"[放置] ⏹ 红点后循迹{self.place_dist:.2f}m完成, 停车开始放置")
            self.sport.StopMove(); time.sleep(0.5)
            place_ok = self._execute_place_after_red(self.platform_zone)
            if not place_ok:
                # 2026-08-06: 失败不再保持停止 — 放置函数内已重启臂桥(自动归零), 继续循迹/跳2/蓝区 (用户指定)
                print("[放置] ❌ 放置失败, 继续循迹 (第二跳/蓝区正常)")
                self.post_place_speed_cap = True
                self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
                self._transition_to(TRACKING)
                return 0.0, 0.0
            print("[放置] ✅ 放置完成, 继续循迹 (第二跳/蓝区启停正常)")
            self.post_place_speed_cap = True  # 2026-08-04: 0.3m/s 直到第二跳
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self._transition_to(TRACKING)
            return 0.0, 0.0
        color, depth = self.detector.get_frames()
        vx = 0.15  # 2026-08-12: 丢线 0.15 慢速 (用户: 改回, 防乱走)
        wz = -0.3  # 2026-08-12: 边前进边右转找线
        tracking = False
        if color is not None:
            _, _, centers = self.detector.detect_layers(color, depth)
            avg_offset = self.compute_weighted_offset(centers)
            if avg_offset is not None:
                vx, wz = self._calculate_tracking_for_red(avg_offset)
                tracking = True
        # 2026-08-12 (用户): 丢线限时 — 连续丢线 >2s 停车原地右转找线; >5s 完全停车等待 (不再转圈乱走)
        if not tracking:
            now_t = now_s()
            if self._place_lost_start is None:
                self._place_lost_start = now_t
            lost_t = now_t - self._place_lost_start
            if lost_t > 2.0:
                vx = 0.0  # 停车, 原地右转找线
                if lost_t > 5.0:
                    wz = 0.0  # 完全停车等待
                    if int(now_t * 2) % 2 == 0:
                        print(f"[放置] ⚠️ 丢线 {lost_t:.0f}s 仍未找到黑线, 停车等待")
        else:
            self._place_lost_start = None
        # 2026-08-05: 路程积分 (每帧 vx × 帧间隔); 2026-08-12: 只在巡线/寻线时累计,
        # 丢线时按 0.15 慢速计 (不再虚增 0.35 盲走路程)
        now = now_s()
        if self._place_last_t > 0:
            self.place_dist += vx * (now - self._place_last_t)
        self._place_last_t = now
        # 2026-08-05: 最后 0.5m 线性减速到 0.05, 不急停
        if self.place_dist >= self._get_place_dist() - self.place_after_red_decel_dist:
            decel = 1.0 - (self.place_dist - (self._get_place_dist() - self.place_after_red_decel_dist)) / self.place_after_red_decel_dist
            vx = max(vx * decel, 0.05)
            if int(now * 2) % 2 == 0:
                print(f"[放置] 减速停止中... {self.place_dist:.2f}m / {self._get_place_dist():.1f}m (vx={vx:.2f})")
        return min(vx, self.place_after_red_speed), wz

    def _handle_turn_back(self, display):
        self.execute_turn_sequence()
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.red_processed = True  # 2026-08-04: 红点已处理, 不再触发 (防循环)
        self.enable_blue_detection()

        if self.jump_phase == 1:
            self.jump_trigger_counter = 0
            self.jump_allowed = False
            self.red_complete_time = now_s()
            print("[跳跃] ⏰ 红点处理完成，1.0s后（进入直道时）开启第二次跳跃检测")

        self.sport.Move(0.2, 0, 0); time.sleep(0.5); self.sport.StopMove()
        self._transition_to(PLACE_AFTER_RED)  # 2026-08-02: 红点后3.3s停车放置
        return 0.0, 0.0

    def _handle_jump(self, display):
        self.execute_jump(); time.sleep(0.2); return 0.0, 0.0

    def _handle_post_jump_align(self, display):
        if now_s() - self.post_jump_align_start > self.post_jump_align_timeout:
            self._transition_to(TRACKING); self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self.sport.StopMove(); time.sleep(0.3); return 0.0, 0.0
        self.post_jump_align(); self._transition_to(TRACKING)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.sport.StopMove(); time.sleep(0.3); return 0.0, 0.0

    # ==================== 核心：前视拟合走向循迹（ROI上方30%最小二乘拟合 + 直角兜底） ====================
    def _handle_tracking(self, centers, avg_offset, first_layer_valid, pattern, display, color, depth):
        # ✅ 前方障碍物检测与黑线过滤
        if depth is not None:
            front_dist = self.front_radar.get_front_dist(depth, narrow_mode=False)

            # 急停：22cm以内
            OBSTACLE_STOP_DIST = 0.22
            if front_dist < OBSTACLE_STOP_DIST:
                print(f"[急停] ⚠️ 前方障碍物 {front_dist:.2f}m < {OBSTACLE_STOP_DIST}m，触发紧急转弯")
                self.corner_direction = self.last_turn_direction
                self.last_corner_time = now_s()
                self.sport.StopMove()
                self._transition_to(CORNER_TURN)
                return 0.0, 0.0

            # ✅ 近距离黑线过滤：前方<35cm时，过滤底部近距离的黑线
            OBSTACLE_FILTER_DIST = 0.35
            if front_dist < OBSTACLE_FILTER_DIST and centers is not None:
                filtered_any = False
                for i in range(7, self.num_layers):
                    if centers[i] is not None:
                        cx = int(np.clip(centers[i], 0, depth.shape[1] - 1))
                        cy = self.detector.roi_top + int((i + 0.5) * self.detector.roi_height / self.num_layers)
                        cy = int(np.clip(cy, 0, depth.shape[0] - 1))

                        point_depth = depth[cy, cx] * 0.001
                        if 0.05 < point_depth < OBSTACLE_FILTER_DIST:
                            centers[i] = None
                            filtered_any = True
                            print(f"[过滤] 第{i}层黑线距离{point_depth:.2f}m < {OBSTACLE_FILTER_DIST}m，已忽略")

                if filtered_any:
                    avg_offset = self.compute_weighted_offset(centers)
                    pattern = self.detector.analyze_black_line_pattern(centers)
                    if centers is not None and len(centers) > 0:
                        first_layer_valid = centers[0] is not None
                    else:
                        first_layer_valid = False

        # ====== 直角检测 ======
        corner_detected = self.check_corner(centers)
        if avg_offset is not None and self._last_avg_offset is not None:
            if abs(avg_offset - self._last_avg_offset) > 50: corner_detected = True
        self._last_avg_offset = avg_offset

        # 楼梯后启用，红点后关闭; 中转等待期间也启用 (找下一个右转标志)
        if self.stairs_triggered_once and self.jump_phase == 0:  # 2026-08-05: 中转等待期纯wz判定, corner不再作中转标志
            if corner_detected:
                self.corner_confirm_count += 1
                if self.corner_confirm_count >= self.corner_confirm_frames:
                    heading = self.corner_detect_heading(centers)
                    self.corner_direction = -np.sign(avg_offset) if avg_offset is not None else (-np.sign(heading) if heading != 0 else -1.0)
                    self.last_turn_direction = self.corner_direction; self.last_corner_time = now_s()
                    print(f"[直角] 确认转弯! 方向:{'左' if self.corner_direction > 0 else '右'}")
                    self._transition_to(CORNER_APPROACH)
                    return self.pre_turn_speed, 0.0
            else:
                self.corner_confirm_count = max(0, self.corner_confirm_count - 1)

        # ====== 正常循迹（PID转动纠偏） ======
        if avg_offset is not None:
            self.lost_count = 0
            error = 0.0 if abs(avg_offset) < self.dead_zone else avg_offset
            vx = self.get_dynamic_speed(avg_offset, first_layer_valid)
            self.integral += error; self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
            derivative = error - self.last_error
            wz_unfiltered = -np.clip(self.kp * error + self.ki * self.integral + self.kd * derivative, -self.max_rotation, self.max_rotation)
            wz = self.filter_alpha * wz_unfiltered + (1 - self.filter_alpha) * self.last_wz
            if abs(error) < 5: wz *= 0.5
            self.vy = 0.0
            self.last_error = error; self.last_wz = wz; self.last_valid_yaw = wz; self._last_heading_error = avg_offset
        else:
            if now_s() < self.post_stairs_right_search_until:
                # 2026-08-09: 楼梯后衔接期无线 → 右找黑线, 找到即直接循迹 (用户)
                vx = self.post_stairs_right_search_vx; wz = -self.post_stairs_right_search_yaw  # 负=右
            elif self.lost_count <= self.transient_lost_frames:
                vx = self.far_layer_lost_speed * 0.6; wz = self.last_valid_yaw * 0.3
            else:
                vx = 0.0; self.vy = 0.0; wz = 0.0
        return vx, wz

    def _handle_corner_approach(self, centers, avg_offset, display, color, depth):
        """补偿直行 → 固定90°转弯（不中途退出，完整执行）"""
        elapsed = now_s() - self.state_start_time
        if elapsed > self.corner_max_time:
            self._transition_to(LOST_SEARCH)
            return self._handle_lost_search(centers, avg_offset, display, color, depth)
        if elapsed < self.pre_turn_duration:
            return self.pre_turn_speed, 0.0
        # 补偿结束，开始固定90°转弯
        print(f"[直角转弯] 补偿直行完成，固定90°转弯 (方向:{'左' if self.corner_direction > 0 else '右'})")
        self._transition_to(CORNER_TURN)
        return self.turn90_vx, self.corner_direction * self.turn90_yaw

    def _handle_corner_turn(self, centers, avg_offset, display, color, depth):
        """固定90°盲转（不中途退出，完整转完再判定）"""
        elapsed = now_s() - self.state_start_time
        if elapsed > self.corner_max_time:
            self._transition_to(LOST_SEARCH)
            return self._handle_lost_search(centers, avg_offset, display, color, depth)
        if elapsed < self.turn90_duration:
            wz = self.corner_direction * self.turn90_yaw
            self.last_valid_yaw = wz
            return self.turn90_vx, wz
        # 90°转完，检查结果
        if avg_offset is not None and abs(avg_offset) < 30:
            print(f"[直角转弯] 90°完成，黑线居中(偏移{avg_offset:.0f}px)，恢复循迹")
            self._transition_to(TRACKING)
            return self._handle_tracking(centers, avg_offset, True, None, display, color, depth)
        else:
            print(f"[直角转弯] 90°完成仍未找到线，进入搜索")
            self._transition_to(LOST_SEARCH)
            return self._handle_lost_search(centers, avg_offset, display, color, depth)

    def _handle_lost_memory(self, centers, avg_offset, display, color, depth):
        """丢线后：补偿直行 → 按记忆方向固定转90° → 完整执行完再判定"""
        elapsed = now_s() - self.state_start_time
        phase = getattr(self, '_lm_phase', 0)  # 0=直行, 1=转90°

        if phase == 0:
            if elapsed < self.lost_memory_straight_dur:
                return self.lost_memory_straight_vx, 0.0
            self._lm_phase = 1
            self.state_start_time = now_s()
            print(f"[惯性记忆] 补偿直行完成，按记忆方向转90° (方向:{'左' if self.last_turn_direction > 0 else '右'})")

        # phase == 1: 固定90°转弯
        elapsed = now_s() - self.state_start_time
        if elapsed < self.lost_memory_turn_dur:
            wz = self.last_turn_direction * self.lost_memory_turn_yaw
            self.last_valid_yaw = wz
            return self.lost_memory_turn_vx, wz

        # 转完检查
        self._lm_phase = 0
        if avg_offset is not None and abs(avg_offset) < 30:
            print(f"[惯性记忆] 90°完成，黑线居中(偏移{avg_offset:.0f}px)，恢复循迹")
            self._transition_to(TRACKING)
            return self._handle_tracking(centers, avg_offset, True, None, display, color, depth)
        else:
            print(f"[惯性记忆] 90°完成仍未找到线，进入搜索")
            self._transition_to(LOST_SEARCH)
            return self._handle_lost_search(centers, avg_offset, display, color, depth)

    def _handle_lost_search(self, centers, avg_offset, display, color, depth):
        elapsed = now_s() - self.state_start_time
        if avg_offset is not None:
            print(f"[搜索] 🎯 看到黑线(偏移{avg_offset:.0f}px)，立即恢复循迹")
            self._transition_to(TRACKING)
            return self._handle_tracking(centers, avg_offset, True, None, display, color, depth)
        wide_result = self.detector.detect_wide_roi(color, depth)
        if wide_result["found"]:
            self.last_turn_direction = wide_result["direction"]
            yaw_magnitude = min(self.lost_search_yaw_base + abs(wide_result["offset_ratio"]) * 0.4, self.lost_search_yaw_max)
            wz = wide_result["direction"] * yaw_magnitude; vx = self.lost_search_vx
        else:
            ramp = min(elapsed / 4.0, 1.0)
            yaw_magnitude = min(self.lost_search_yaw_base + self.lost_search_yaw_ramp * ramp, self.lost_search_yaw_max)
            # 双向摆动：周期性左右扫，覆盖两侧
            period = 3.0
            phase = (elapsed % period) / period
            if phase < 0.6:
                direction = self.last_turn_direction
            else:
                direction = -self.last_turn_direction
            wz = direction * yaw_magnitude * (0.7 + 0.3 * phase)
            vx = self.lost_search_vx
        self.last_valid_yaw = wz; return vx, wz


if __name__ == "__main__":
    tracker = Go2SegmentTracker(INTERFACE, num_layers=10)
    tracker.run()

