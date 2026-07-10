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
"""

import sys
import time
import argparse
import os
import cv2
import numpy as np
from enum import Enum
from collections import deque
import pyrealsense2 as rs
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.vui.vui_client import VuiClient


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
    
    # ====== 直行速度 ======
    STRAIGHT_1_SPEED = 0.58
    STRAIGHT_1B_SPEED = 0.59
    STRAIGHT_2_SPEED = 0.59
    STRAIGHT_2B_SPEED = 0.59
    STRAIGHT_3_SPEED = 0.59
    
    TURN_FWD_SPEED = 0.24
    
    # ====== 各段直行帧数 ======
    STRAIGHT_1_FRAMES = 220
    STRAIGHT_2_FRAMES = 200
    STRAIGHT_3_FRAMES = 200 + 29
    
    # ====== 各转弯帧数 ======
    TURN_1_FRAMES = 51
    TURN_2_FRAMES = 52
    TURN_3_FRAMES = 51
    TURN_4_FRAMES = 58
    TURN_5_FRAMES = 37
    
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
    STRAIGHT_1_WALL_STOP = 0.20
    STRAIGHT_1_TURN_TRIGGER = 0.5
    
    # ============================================================
    # 转弯2 (TURN_2) 前的直行 STRAIGHT_1B - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_1B_WALL_SLOW = 0.70
    STRAIGHT_1B_WALL_STOP = 0.42
    STRAIGHT_1B_TURN_TRIGGER = None
    
    # ============================================================
    # 转弯3 (TURN_3) 前的直行 STRAIGHT_2 (有侧移) - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_2_WALL_SLOW = 0.66
    STRAIGHT_2_WALL_STOP = 0.38
    STRAIGHT_2_TURN_TRIGGER = 0.40
    
    # ============================================================
    # 转弯4 (TURN_4) 前的直行 STRAIGHT_2B - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_2B_WALL_SLOW = 0.68
    STRAIGHT_2B_WALL_STOP = 0.38
    STRAIGHT_2B_TURN_TRIGGER = 0.40
    
    # ============================================================
    # 转弯5 (TURN_5) 前的直行 STRAIGHT_3 (有侧移) - 碰墙减速/停止距离
    # ============================================================
    STRAIGHT_3_WALL_SLOW = 0.66
    STRAIGHT_3_WALL_STOP = 0.20
    STRAIGHT_3_TURN_TRIGGER = 0.40
    
    # ====== 通用参数 ======
    WALL_CONFIRM_FRAMES = 5
    MIN_FRAMES_BEFORE_CHECK = 30


# ==================== ORB特征点识别配置 ====================
SIMILARITY_THRESHOLD = 13
MIN_GAP = 7
MAX_ACTIONS = 3
COOLDOWN_TIME = 2.0
PRINT_THRESHOLD = 10
CONFIRM_FRAMES = 3


# ==================== 红点检测配置 ====================
RED_HSV_LOWER1 = np.array([0, 80, 80])
RED_HSV_UPPER1 = np.array([10, 255, 255])
RED_HSV_LOWER2 = np.array([160, 80, 80])
RED_HSV_UPPER2 = np.array([180, 255, 255])
RED_MIN_RADIUS = 10
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

STAIRS_GAIT = GaitMode.CLASSIC_WALK      # 爬楼梯步态（经典稳定）
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

SIGN_NAMES = {SignID.ELECTRIC_SHOCK: "当心触电", SignID.OXIDIZER: "当心强氧化物", SignID.RADIATION: "当心辐射"}
SIGN_FILES = {SignID.ELECTRIC_SHOCK: "electric_shock.jpg", SignID.OXIDIZER: "oxidizer.jpg", SignID.RADIATION: "radiation.jpg"}


# ==================== 状态常量 ====================
TRACKING = "tracking"
CORNER_APPROACH = "corner_approach"
CORNER_TURN = "corner_turn"
LOST_MEMORY = "lost_memory"
LOST_SEARCH = "lost_search"
LOST_STOP = "lost_stop"
RED_APPROACH = "red_approach"
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
                trigger_dist=nc.STRAIGHT_2_TURN_TRIGGER
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
                trigger_dist=nc.STRAIGHT_3_TURN_TRIGGER
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
            if self.is_slowing: return speed * 0.5, 0, 0
            return speed, 0, 0
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
            if self.is_slowing: return speed * 0.5, 0, 0
            return speed, 0, 0
        return act
    
    def _act_straight_side(self, total, next_state, side_distance=0.0, direction='right', 
                           speed=None, wall_slow=None, wall_stop=None, trigger_dist=None):
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
            if total_side_frames > 0 and self.frame_cnt < total_side_frames:
                if self.frame_cnt < half_frames:
                    progress = self.frame_cnt / max(1, half_frames)
                    vy = max_side_speed * progress
                else:
                    progress = (total_side_frames - self.frame_cnt) / max(1, total_side_frames - half_frames)
                    vy = max_side_speed * progress
                if direction == 'left': vy = -vy
            self.frame_cnt += 1
            if self.is_slowing: return speed * 0.5, vy * 0.5, 0
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
                
                if self.line_detected and nc.FINAL_TURN_COMPENSATE_ENABLED:
                    print(f"[窄道-补偿] 🔄 检测到黑线，执行补偿转弯 {nc.FINAL_TURN_COMPENSATE_ANGLE}°...")
                    self._switch("COMPENSATE_TURN")
                    return nc.FINAL_TURN_COMPENSATE_VX, 0, nc.FINAL_TURN_COMPENSATE_YAW
                else:
                    if not nc.FINAL_TURN_COMPENSATE_ENABLED:
                        print(f"[窄道-最后转弯] 补偿转弯已禁用，直接完成")
                    self.finished = True
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
        self.templates = {}; self.keypoints = {}; self.descriptors = {}; self.templates_bottom = {}
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.load_templates(template_folder)
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

    def match(self, frame_gray):
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
            if best_count >= SIMILARITY_THRESHOLD and gap >= MIN_GAP:
                print(f"   ✅ 确认匹配: {SIGN_NAMES[best_id]}")
                return (best_id, best_count)
            elif best_count >= PRINT_THRESHOLD:
                print(f"   ❌ 未确认 (需要>{SIMILARITY_THRESHOLD}点, 差距>{MIN_GAP})")
        return None


# ==================== 检测器 ====================
class LineDetector:
    def __init__(self, width=640, height=480, num_layers=10):
        self.width = width; self.height = height; self.num_layers = num_layers
        self.roi_top = 400; self.roi_bottom = height; self.roi_height = self.roi_bottom - self.roi_top
        self.roi_left = 130; self.roi_right = 510; self.roi_width = self.roi_right - self.roi_left
        self.stair_roi_top = max(0, self.roi_top - self.roi_height); self.stair_roi_bottom = self.roi_top
        self.stair_roi_height = self.stair_roi_bottom - self.stair_roi_top
        self.pipeline = rs.pipeline(); self.config = rs.config()
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
    def stop(self): self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(); aligned = self.align.process(frames)
        color = aligned.get_color_frame(); depth = aligned.get_depth_frame()
        if not color or not depth: return None, None
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())

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
        
        # ====== 基本运动参数 ======
        self.base_speed = 0.55
        self.max_rotation = 0.8
        self.max_vy = 0.35           # 平移纠偏最大侧移速度

        self.far_layer_normal_speed = 0.6
        self.far_layer_lost_speed = 0.46
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
        self.turn90_vx = 0.065          # 转弯中微速前进
        self.corner_max_time = 10.0
        self.corner_cooldown = 1.5

        # ====== 丢失恢复参数 ======
        self.transient_lost_frames = 4.5
        self.lost_memory_straight_dur = 1.75  # 补偿直行0.4m @ 0.24m/s
        self.lost_memory_straight_vx = 0.20
        self.lost_memory_turn_dur = 1.75      # 固定90° @ 0.9rad/s
        self.lost_memory_turn_yaw = 0.9
        self.lost_memory_turn_vx = 0.06
        self.lost_search_time = 13.0
        self.lost_search_vx = 0.15
        self.lost_search_yaw_base = 1.2
        self.lost_search_yaw_max = 1.4
        self.lost_search_yaw_ramp = 0.08

        self.first_lost_handled = False

        # ====== 红点接近参数 ======
        self.approach_speed = 0.60  # 0.40×0.05s≈0.02m/次
        self.target_radius = 38
        self.approach_timeout = 12.0
        self.compensate_distance = 0.926# +0.95m
        self.compensate_speed = 0.27
        
        # ====== 转向序列参数 ======
        self.turn_step_angle = 25
        self.turn_step_time = 0.6
        self.turn_speed = 0.7
        self.backup_speed = -0.27
        self.backup_step_time = 0.6
        self.backup_steps = 3
        self.align_speed = 0.35
        self.align_timeout = 2.0

        # ====== 楼梯触发参数 ======
        self.width_trigger_ratio = 2.5
        self.width_sample_layers = 3
        self.width_history_len = 30
        self.normal_widths_high = []
        self.stairs_trigger_enabled = True
        self.stairs_triggered_once = False

        self.stairs_forward_duration = 5.63
        self.stairs_turn_overlap = 0.86   # 直行结束前0.25s开始叠加转弯
        self.stairs_forward_speed = 0.32
        self.stairs_forward_yaw = 0.02    # 直行时微左偏角速度
        self.stairs_turn_omega = 1.035
        self.stairs_turn_duration = 1.56  # 1.01rad/s × 1.56s = 90°
        self.stairs_turn_vx = 0.28
        self.stairs_shift_vy = 0.20   # 侧移修正速度
        self.stairs_shift_duration = 0.2  # 侧移修正时长 (~0.05m)
        self.post_stairs_until = 0.0      # 楼梯后只看近处黑线的时间戳
        self._slow_after_first_corner = False
        self._fast_after_memory = False

        self.stairs_phase = STAIRS_PHASE_FORWARD
        self.stairs_phase_start = 0.0

        # ====== 蓝色启停区参数 ======
        self.blue_stop_detected = False
        self.blue_confirm_frames = 3
        self.blue_confirm_count = 0
        self.blue_detection_enabled = False
        
        self.blue_go_straight_distance = 0.982
        self.blue_go_straight_speed = 0.3
        self.blue_turn_angle = 114
        self.blue_turn_speed = 1
        self.blue_final_distance = 0.091
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

        self.state = TRACKING
        self.lost_count = 0
        self.last_valid_yaw = 0.0
        self.last_turn_direction = 1.0
        self.state_start_time = 0.0
        self._last_heading_error = 0.0
        self._last_avg_offset = None
        self.last_valid_offset_direction = 0.0
        self.last_valid_offset_magnitude = 0.0

        self.red_processed = False
        self.red_center = None
        self.red_radius = None
        
        self.vy = 0.22  # 侧移速度覆盖（转弯甩尾 / 楼梯修正）

        print("[识别] 初始化ORB特征点识别器...")
        self.recognizer = ORBRecognizer("templates")

        self.start_time = time.time()
        self.run_duration = args.duration
        self.frame_count = 0
        self.num_layers = num_layers

    # ==================== 状态转换 ====================
    def _transition_to(self, new_state):
        print(f"[状态] {self.state} -> {new_state}")
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()

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

            # 楼梯完成后启用平台检测
            if old_state == STAIRS:
                self.platform_detection_enabled = True
                self.platform_count = 0
                # 退出爬楼梯，恢复普通步态
                print("[步态] 🪜 爬楼梯结束，恢复普通步态...")
                switch_gait(self.sport, STAIRS_GAIT, enter=False)
                if NORMAL_GAIT != GaitMode.FREE_WALK:
                    switch_gait(self.sport, NORMAL_GAIT, enter=True)
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
            self.stairs_phase = STAIRS_PHASE_FORWARD; self.stairs_phase_start = time.time()
            self.stairs_triggered_once = True
            self.narrow_triggered = False
            print(f"[步态] 🪜 进入爬楼梯，切换步态: {STAIRS_GAIT.value}")
            switch_gait(self.sport, STAIRS_GAIT, enter=True)
        elif new_state == NARROW_APPROACH:
            self.narrow_triggered = True
            self.first_lost_handled = True
            self.sport.StopMove(); time.sleep(0.3)
            self.narrow_fsm.tracker = self; self.narrow_fsm.reset()
            print("[窄道] 🚀 准备执行窄道路径")
            time.sleep(0.2)
            self.state = NARROW_EXECUTING; self.state_start_time = time.time()
            print("[窄道] 开始执行写死路径...")
        elif new_state == POST_JUMP_ALIGN: self.post_jump_align_start = time.time()
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
        near_only = (time.time() < self.post_stairs_until or
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
        # 红点后等第二跳窗口开启(2.8s)后才减速到0.3
        if self.jump_phase == 1 and self.red_complete_time > 0:
            if time.time() - self.red_complete_time > 2.8:
                base_vx = min(base_vx, 0.30)
        # 窄道前限速0.35，楼梯后恢复快速
        if not self.narrow_triggered and not self.stairs_triggered_once:
            base_vx = min(base_vx, 0.35)
        return max(base_vx, 0.15)

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
        elapsed = time.time() - self.state_start_time
        if elapsed < self.platform1_turn_duration:
            return self.platform1_turn_vx, self.platform1_turn_yaw
        else:
            print("[平台1] ✅ 转弯完成，开始前进")
            self.sport.StopMove()
            time.sleep(0.2)
            self._transition_to(PLATFORM1_FORWARD)
            return 0.0, 0.0

    def _handle_platform1_forward(self, display):
        elapsed = time.time() - self.state_start_time
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
        elapsed = time.time() - self.state_start_time
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
            if abs_off < 25: vx = self.base_speed
            elif abs_off < 70: vx = self.base_speed * 0.9
            else: vx = 0.20
            vx = max(vx, 0.15); self.last_error = error; self.last_wz = wz
        else: vx, wz = 0.0, 0.0
        return vx, wz

    def approach_red_point_with_compensate(self):
        print("[红点] 开始减速接近..."); self.sport.StopMove(); time.sleep(0.2)
        start_time = time.time(); radius_history = []; limit_reached = False
        compensate_time = self.compensate_distance / self.compensate_speed
        while True:
            if time.time() - start_time > self.approach_timeout:
                print("[红点] 接近超时"); self.sport.StopMove(); return False
            color, depth = self.detector.get_frames()
            if color is None: continue
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
                    limit_reached = True; compensate_start_time = time.time()
                    self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; continue
                vx, wz = self._calculate_tracking_for_red(avg_offset)
                vx = min(vx, self.approach_speed); self.sport.Move(vx, 0, wz); time.sleep(0.05)
            else:
                elapsed = time.time() - compensate_start_time
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
        print("[对准] 微调对准黑线..."); start_time = time.time()
        while time.time() - start_time < self.align_timeout:
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
        print(f"\n{'='*50}")
        print(f"匹配成功: {SIGN_NAMES[sign_id]} (第{self.recognizer.action_count}/{MAX_ACTIONS}次)")
        print(f"{'='*50}")
        if sign_id == SignID.ELECTRIC_SHOCK:
            do_stretch(self.sport)
            self.sport.Move(-0.25, 0, 0); time.sleep(0.2); self.sport.StopMove()  # 退0.05m
        elif sign_id == SignID.OXIDIZER:
            do_greet(self.sport)
            self.sport.Move(-0.25, 0, 0); time.sleep(0.12); self.sport.StopMove()  # 退0.03m
        elif sign_id == SignID.RADIATION: blink_front_lights(vui_client, times=3)

    def recognize_and_perform_action(self, color_img):
        print("\n[识别] 开始ORB识别...")
        if color_img is None:
            while True:
                color, depth = self.detector.get_frames()
                if color is not None: gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY); break
                time.sleep(0.1)
        else: gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        self.recognizer.confirm_counter = 0; self.recognizer.pending_sign_id = None; last_move_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_move_time >= 4.0:
                print("[识别] 4秒未识别，前进1.5cm"); self.sport.Move(0.25, 0, 0); time.sleep(0.06); self.sport.StopMove(); last_move_time = current_time
            color, depth = self.detector.get_frames()
            if color is not None: gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            if current_time - self.recognizer.last_action_time < COOLDOWN_TIME: time.sleep(0.1); continue
            match = self.recognizer.match(gray)
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
        steps_135 = 126 // self.turn_step_angle
        for i in range(steps_135): self.sport.Move(0, 0, self.turn_speed); time.sleep(self.turn_step_time); self.sport.StopMove(); time.sleep(0.05)
        print("[转向] 左转完成"); time.sleep(0.3)
        self.backup_before_turn(); time.sleep(0.2)
        color, depth = self.detector.get_frames(); self.recognize_and_perform_action(color); time.sleep(0.5)
        steps_90 = 128 // self.turn_step_angle
        for i in range(steps_90): self.sport.Move(0, 0, -self.turn_speed); time.sleep(self.turn_step_time); self.sport.StopMove(); time.sleep(0.05)
        print("[转向] 右转完成"); time.sleep(0.3)
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; max_additional = 5; count = 0
        while count <= max_additional:
            color, depth = self.detector.get_frames()
            if color is not None:
                display, mask, centers = self.detector.detect_layers(color, depth)
                avg_offset = self.compute_weighted_offset(centers)
                if avg_offset is not None: print(f"[转向] 检测到黑线, 偏移:{avg_offset:.1f}px"); self.align_to_line(); break
                elif count < max_additional:
                    count += 1; print(f"[转向] 未检测到黑线, 继续右转({count}/{max_additional})")
                    self.sport.Move(0, 0, -self.turn_speed); time.sleep(self.turn_step_time); self.sport.StopMove(); time.sleep(0.1)
                else:
                    for angle in [15, -30, 30]:
                        direction = -1 if angle > 0 else 1
                        search_time = (abs(angle) / self.turn_step_angle) * self.turn_step_time
                        self.sport.Move(0, 0, direction * self.turn_speed); time.sleep(search_time); self.sport.StopMove(); time.sleep(0.1)
                        color, depth = self.detector.get_frames()
                        if color is not None:
                            display, mask, centers = self.detector.detect_layers(color, depth)
                            if self.compute_weighted_offset(centers) is not None: self.align_to_line(); break
                    break
            else: self.sport.Move(0, 0, -self.turn_speed); time.sleep(self.turn_step_time); self.sport.StopMove(); time.sleep(0.1); count += 1
            time.sleep(0.1)
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
        time.sleep(1.0)
        print(f"[跳跃] ========== 第{jump_num}次跳跃结束 ==========\n")
        self.last_jump_time = time.time()
        if self.jump_phase == 0:
            self.jump_allowed = False; self.jump_phase = 1
            print("[跳跃] ⚠️ 第一次跳跃完成，等待红点处理后开启第二次跳跃")
        elif self.jump_phase == 1:
            self.jump_allowed = False; self.jump_phase = 2
            print("[跳跃] ⚠️ 第二次跳跃完成，直行巡线等蓝区（不搜索不摆头）")
            self.sport.Move(0.25, 0, 0); time.sleep(1.0); self.sport.StopMove()
            self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
            self.state = TRACKING; self.state_start_time = time.time()
            return
        self.jump_trigger_counter = 0
        self.detector.roi_top = 390; self.detector.roi_left = 80; self.detector.roi_right = 560
        self.detector.roi_width = self.detector.roi_right - self.detector.roi_left
        self.detector.roi_height = self.detector.roi_bottom - self.detector.roi_top
        print("[跳跃] ROI已切换到循迹模式")
        self._transition_to(POST_JUMP_ALIGN)

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
                    print("[摆头] 微调对准中..."); align_start = time.time()
                    while time.time() - align_start < 1.5:
                        color, depth = self.detector.get_frames()
                        if color is None: continue
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
                    print("[初始对准] 微调中..."); align_start = time.time()
                    while time.time() - align_start < 1.5:
                        color, depth = self.detector.get_frames()
                        if color is None: continue
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
        
        print("[运动] 唤醒站立...")
        self.sport.RecoveryStand(); time.sleep(2)
        print("[运动] ✅ 站立完成\n")
        
        self.initial_align()
        print("[主循环] 开始循迹...")
        
        try:
            while True:
                if self.blue_stop_detected and self.state == BLUE_SIT_DOWN:
                    if time.time() - self.state_start_time > 3.0:
                        print("[退出] 到达终点，任务完成"); break
                
                color, depth = self.detector.get_frames()
                if color is None: continue
                self.frame_count += 1

                if self.state == NARROW_EXECUTING:
                    narrow_mode = (self.narrow_fsm.state == "STRAIGHT_1")
                    front_d = self.front_radar.get_front_dist(depth, narrow_mode=narrow_mode)
                    vx, vy, yaw = self.narrow_fsm.get_cmd(front_d)
                    self.narrow_smooth_move(vx, vy, yaw)
                    if self.narrow_fsm.finished:
                        print("[窄道] ✅ 窄道路径执行完成！")
                        self.narrow_smooth_move(0, 0, 0); time.sleep(0.5)
                        self.sport.StopMove(); time.sleep(0.5)
                        self.state = TRACKING; self.state_start_time = time.time()
                        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0
                        self.lost_count = 0; self.corner_confirm_count = 0
                        self._last_avg_offset = None; self._last_heading_error = 0.0
                        self.first_lost_handled = True
                        self.stairs_allowed = True; self.narrow_triggered = True
                        print("[窄道] ✅ 窄道完成，爬楼梯已允许"); continue
                    if SHOW_GUI and color is not None:
                        display_n = color.copy()
                        cv2.putText(display_n, f"NARROW: {self.narrow_fsm.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Go2 Full", display_n); cv2.waitKey(1)
                    time.sleep(0.01); continue

                skip_states = [STAIRS, BLUE_STOP, BLUE_GO_STRAIGHT, BLUE_TURN_LEFT, BLUE_FINAL_APPROACH, BLUE_SIT_DOWN, 
                              POST_JUMP_ALIGN, JUMP, NARROW_APPROACH, NARROW_EXECUTING, PLATFORM1_TURN, PLATFORM1_FORWARD, PLATFORM2_STOP]
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

                if self.state == TRACKING and self.narrow_enabled and not self.blue_stop_detected and not self.narrow_triggered and not self.first_lost_handled:
                    is_narrow, narrow_width = self.narrow_detector.detect_narrow(depth)
                    if is_narrow:
                        print(f"\n[窄道检测] ✅ 检测到窄道！宽度约:{narrow_width:.2f}m")
                        self._transition_to(NARROW_APPROACH); continue

                if self.state != STAIRS and not self.stairs_triggered_once and not self.blue_stop_detected and self.stairs_allowed:
                    if self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]:
                        self.update_normal_widths(widths_high)
                        if self.is_width_triggered(widths_high) and not red_detected:
                            print("[楼梯模式] 楼梯ROI上侧宽度突变，触发爬楼梯模式")
                            self._transition_to(STAIRS); self.sport.StopMove(); time.sleep(0.1); continue

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
                    if red_detected and not self.red_processed and self.state in [TRACKING, LOST_MEMORY, LOST_SEARCH, LOST_STOP]:
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

                if self.run_duration > 0 and time.time() - self.start_time > self.run_duration:
                    print(f"\n[退出] 计时结束"); break
                
                # ====== 第一次跳跃检测（窄道前）：垂直投影，与第二跳相同 ======
                if (self.state == TRACKING and not self.blue_stop_detected and
                    self.jump_allowed and self.jump_phase == 0):
                    if time.time() - self.last_corner_time > 0.5 and time.time() - self.last_jump_time > 1.0:
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
                            print(f"[跳跃1] ★★★ 截断! 补偿前进0.18m再跳...")
                            self.sport.StopMove(); time.sleep(0.1)
                            self.sport.Move(0.25, 0, 0); time.sleep(0.72)  # 0.25×0.72≈0.18m
                            self.sport.StopMove(); time.sleep(0.1)
                            print(f"[跳跃1] ★★★ 第一次跳跃！★★★")
                            self.sport.StopMove()
                            self._transition_to(JUMP)
                            continue

                # ====== 第二次跳跃检测：红点后才启用，等2.8s → 垂直投影截断 ======
                if self.state == TRACKING and self.jump_phase == 1 and self.red_complete_time > 0:
                    if time.time() - self.red_complete_time < 2.8:
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
                            print(f"[跳跃2] ★★★ 截断! 补偿前进0.2m再跳...")
                            self.sport.StopMove(); time.sleep(0.1)
                            self.sport.Move(0.25, 0, 0); time.sleep(0.8)  # 0.25×0.8≈0.2m
                            self.sport.StopMove(); time.sleep(0.1)
                            print(f"[跳跃2] ★★★ 第二次跳跃！★★★")
                            self._transition_to(JUMP)
                            continue

                if self.state not in [RED_APPROACH, TURN_BACK, JUMP, POST_JUMP_ALIGN, STAIRS, BLUE_STOP, BLUE_GO_STRAIGHT,
                                       BLUE_TURN_LEFT, BLUE_FINAL_APPROACH, BLUE_SIT_DOWN, NARROW_APPROACH, NARROW_EXECUTING,
                                       PLATFORM1_TURN, PLATFORM1_FORWARD, PLATFORM2_STOP,
                                       CORNER_APPROACH, CORNER_TURN]:
                    if self.jump_phase != 2:  # 第二跳后不搜索不摆头
                        if self.narrow_triggered or self.first_lost_handled:
                            if self.first_lost_handled:
                                # 完全丢线计数
                                if avg_offset is None: self.lost_count += 1
                                else: self.lost_count = 0

                                if not self.narrow_triggered and self.stairs_triggered_once:
                                    # 丢线 → 惯性记忆（仅楼梯后启用）
                                    if avg_offset is None:
                                        if self.state not in [LOST_MEMORY, LOST_SEARCH]:
                                            self.last_turn_direction = -np.sign(self._last_heading_error) if self._last_heading_error != 0 else 1.0
                                            self._transition_to(LOST_MEMORY); continue
                        else:
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
                
                self.sport.Move(vx, self.vy, wz)
                self.vy = 0.0  # 每帧重置，各 handler 按需设置

                if SHOW_GUI and display is not None and self.state != NARROW_EXECUTING:
                    cv2.imshow("Go2 Full", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'): break
        except KeyboardInterrupt:
            print("\n[用户] Ctrl+C 中断")
        finally:
            self.vui_client.SetBrightness(0)
            self.sport.StopMove(); time.sleep(0.5)
            self.sport.StandDown()
            self.detector.stop()
            if SHOW_GUI: cv2.destroyAllWindows()
            print("[退出] 完成")

    # ==================== 状态处理函数 ====================
    def _handle_stairs(self, display):
        if self.stairs_phase == STAIRS_PHASE_FORWARD:
            elapsed = time.time() - self.stairs_phase_start
            overlap_start = self.stairs_forward_duration - self.stairs_turn_overlap
            if elapsed < overlap_start:
                # 纯直行阶段
                if int(elapsed * 2) % 2 == 0:
                    print(f"[楼梯] 纯直行中... {elapsed:.1f}s / {self.stairs_forward_duration}s")
                return self.stairs_forward_speed, self.stairs_forward_yaw
            elif elapsed < self.stairs_forward_duration:
                # 提前0.25s叠加转弯角速度
                if int(elapsed * 2) % 2 == 0:
                    print(f"[楼梯] 直行+转弯叠加中... {elapsed:.1f}s / {self.stairs_forward_duration}s (提前{self.stairs_turn_overlap}s转弯)")
                return self.stairs_forward_speed, self.stairs_turn_omega
            else:
                remain_turn = self.stairs_turn_duration - self.stairs_turn_overlap
                print(f"[楼梯] 直行完成，继续转弯 (剩余 {remain_turn:.2f}s)")
                self.stairs_phase = STAIRS_PHASE_TURN
                self.stairs_phase_start = time.time()
                return self.stairs_turn_vx, self.stairs_turn_omega
        elif self.stairs_phase == STAIRS_PHASE_TURN:
            elapsed = time.time() - self.stairs_phase_start
            remain_turn = self.stairs_turn_duration - self.stairs_turn_overlap
            if elapsed < remain_turn:
                if int(elapsed * 2) % 2 == 0:
                    print(f"[楼梯] 边转弯边直行中... {elapsed:.1f}s / {remain_turn:.2f}s")
                return self.stairs_turn_vx, self.stairs_turn_omega
            else:
                # 转弯完成 → 反向侧移修正
                shift_dir = -np.sign(self.stairs_turn_omega)
                print(f"[楼梯] 转弯完成，反向侧移修正: {'右' if shift_dir < 0 else '左'}移 {self.stairs_shift_duration}s")
                self.stairs_phase = STAIRS_PHASE_SHIFT
                self.stairs_phase_start = time.time()
                self.vy = shift_dir * self.stairs_shift_vy
                return 0.0, 0.0
        elif self.stairs_phase == STAIRS_PHASE_SHIFT:
            elapsed = time.time() - self.stairs_phase_start
            if elapsed < self.stairs_shift_duration:
                shift_dir = -np.sign(self.stairs_turn_omega)
                self.vy = shift_dir * self.stairs_shift_vy
                if int(elapsed * 4) % 4 == 0:
                    print(f"[楼梯] 侧移修正中... {elapsed:.1f}s / {self.stairs_shift_duration}s")
                return 0.0, 0.0
            else:
                # 侧移完成 → 恢复循迹，但2s内只看近处黑线（忽略远处十字叉）
                print("[楼梯] ✅ 爬楼梯+侧移修正完成，用近处黑线冲过十字叉...")
                self.sport.StopMove(); time.sleep(0.3)
                self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.lost_count = 0
                self.state = TRACKING; self.state_start_time = time.time()
                self.corner_confirm_count = 0
                self.post_stairs_until = 0.0  # 立即用全部10层循迹
                return 0.0, 0.0

    def _handle_blue_stop(self, display):
        elapsed = time.time() - self.state_start_time
        if elapsed < 0.5: return 0.0, 0.0
        else: self.sport.StopMove(); time.sleep(0.2); self._transition_to(BLUE_GO_STRAIGHT); return 0.0, 0.0

    def _handle_blue_go_straight(self, display):
        elapsed = time.time() - self.state_start_time
        expected_time = self.blue_go_straight_distance / self.blue_go_straight_speed
        if elapsed < expected_time: return self.blue_go_straight_speed, 0.0
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_TURN_LEFT); return 0.0, 0.0

    def _handle_blue_turn_left(self, display):
        elapsed = time.time() - self.state_start_time
        turn_radians = np.radians(self.blue_turn_angle)
        expected_time = turn_radians / self.blue_turn_speed
        if elapsed < expected_time: return 0.0, self.blue_turn_speed
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_FINAL_APPROACH); return 0.0, 0.0

    def _handle_blue_final_approach(self, display):
        elapsed = time.time() - self.state_start_time
        expected_time = self.blue_final_distance / self.blue_final_speed
        if elapsed < expected_time: return self.blue_final_speed, 0.0
        else: self.sport.StopMove(); time.sleep(0.3); self._transition_to(BLUE_SIT_DOWN); return 0.0, 0.0

    def _handle_blue_sit_down(self, display):
        elapsed = time.time() - self.state_start_time
        if elapsed < 1.0: return 0.0, 0.0
        else: self.sport.StandDown(); time.sleep(0.5); self.blue_stop_detected = True; return 0.0, 0.0

    def _handle_red_approach(self, display):
        success = self.approach_red_point_with_compensate()
        if success: self._transition_to(TURN_BACK)
        else: self.red_processed = False; self._transition_to(TRACKING)
        self.sport.StopMove(); time.sleep(0.5); return 0.0, 0.0
        
    def _handle_turn_back(self, display):
        self.execute_turn_sequence()
        self.integral = 0.0; self.last_error = 0.0; self.last_wz = 0.0; self.red_processed = False
        self.enable_blue_detection()
        
        if self.jump_phase == 1:
            self.jump_trigger_counter = 0
            self.jump_allowed = False
            self.red_complete_time = time.time()
            print("[跳跃] ⏰ 红点处理完成，1.0s后（进入直道时）开启第二次跳跃检测")
        
        self.sport.Move(0.2, 0, 0); time.sleep(0.5); self.sport.StopMove(); self._transition_to(TRACKING); return 0.0, 0.0

    def _handle_jump(self, display): 
        self.execute_jump(); time.sleep(0.2); return 0.0, 0.0

    def _handle_post_jump_align(self, display):
        if time.time() - self.post_jump_align_start > self.post_jump_align_timeout:
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
                self.last_corner_time = time.time()
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

        # 楼梯后启用，红点后关闭
        if self.stairs_triggered_once and self.jump_phase == 0:
            if corner_detected:
                self.corner_confirm_count += 1
                if self.corner_confirm_count >= self.corner_confirm_frames:
                    heading = self.corner_detect_heading(centers)
                    self.corner_direction = -np.sign(avg_offset) if avg_offset is not None else (-np.sign(heading) if heading != 0 else -1.0)
                    self.last_turn_direction = self.corner_direction; self.last_corner_time = time.time()
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
            if self.lost_count <= self.transient_lost_frames:
                vx = self.far_layer_lost_speed * 0.6; wz = self.last_valid_yaw * 0.3
            else:
                vx = 0.0; self.vy = 0.0; wz = 0.0
        return vx, wz

    def _handle_corner_approach(self, centers, avg_offset, display, color, depth):
        """补偿直行 → 固定90°转弯（不中途退出，完整执行）"""
        elapsed = time.time() - self.state_start_time
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
        elapsed = time.time() - self.state_start_time
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
        elapsed = time.time() - self.state_start_time
        phase = getattr(self, '_lm_phase', 0)  # 0=直行, 1=转90°

        if phase == 0:
            if elapsed < self.lost_memory_straight_dur:
                return self.lost_memory_straight_vx, 0.0
            self._lm_phase = 1
            self.state_start_time = time.time()
            print(f"[惯性记忆] 补偿直行完成，按记忆方向转90° (方向:{'左' if self.last_turn_direction > 0 else '右'})")

        # phase == 1: 固定90°转弯
        elapsed = time.time() - self.state_start_time
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
        elapsed = time.time() - self.state_start_time
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
    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    tracker = Go2SegmentTracker(iface, num_layers=10)
    tracker.run()