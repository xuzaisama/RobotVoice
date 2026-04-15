"""
3D 可视化界面：语音输入指令时，让小机器人在 3D 场景中行走。

技术选型：
- PyQt6: 窗口与事件循环
- pyqtgraph.opengl: 轻量 3D 场景（网格、立方体、轨迹）

架构：
- GUI 线程：渲染 + 定时器轮询指令队列
- 语音线程：麦克风采集/识别 + RobotController（含播报锁与去重）
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config
from .controller import RobotController
from .listener import VoiceListener

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0  # 0 度：朝 +Y


def _make_box_mesh(gl, size: float = 0.3):
    """
    生成一个立方体 MeshData，兼容没有 MeshData.cube() 的 pyqtgraph 版本。
    - size: 立方体边长
    """
    s = size / 2.0
    # 8 个顶点
    verts = [
        (-s, -s, -s),
        (s, -s, -s),
        (s, s, -s),
        (-s, s, -s),
        (-s, -s, s),
        (s, -s, s),
        (s, s, s),
        (-s, s, s),
    ]
    # 12 个三角面（每个面 2 个三角形）
    faces = [
        (0, 1, 2),
        (0, 2, 3),  # bottom
        (4, 6, 5),
        (4, 7, 6),  # top
        (0, 4, 5),
        (0, 5, 1),  # -y
        (1, 5, 6),
        (1, 6, 2),  # +x
        (2, 6, 7),
        (2, 7, 3),  # +y
        (3, 7, 4),
        (3, 4, 0),  # -x
    ]
    return gl.MeshData(vertexes=verts, faces=faces)


def _apply_command(state: RobotState, cmd_id: str, step: float = 0.3) -> RobotState:
    """离散运动学：前后按朝向移动，转向按 90° 步进。"""
    yaw = math.radians(state.yaw_deg)
    dx = math.sin(yaw)
    dy = math.cos(yaw)

    if cmd_id == "forward":
        state.x += step * dx
        state.y += step * dy
    elif cmd_id == "backward":
        state.x -= step * dx
        state.y -= step * dy
    elif cmd_id == "turn_left":
        state.yaw_deg = (state.yaw_deg + 90.0) % 360.0
    elif cmd_id == "turn_right":
        state.yaw_deg = (state.yaw_deg - 90.0) % 360.0
    elif cmd_id in ("stop", "status"):
        pass
    return state


def _voice_worker(
    cmd_queue: "queue.Queue[str]",
    stop_event: threading.Event,
) -> None:
    """
    语音线程：识别到有效指令时，把 cmd_id 送入队列供 GUI 更新位置。
    播报/去重由 RobotController 内部处理。
    """

    def on_action(cmd_id: str) -> None:
        try:
            cmd_queue.put_nowait(cmd_id)
        except queue.Full:
            logger.warning("指令队列已满，丢弃: %s", cmd_id)

    controller = RobotController(on_action=on_action)

    with VoiceListener() as listener:
        listener.calibrate_ambient()
        while not stop_event.is_set():
            if controller.speaking:
                time.sleep(0.05)
                continue

            text = listener.listen_once()
            if not text:
                continue

            controller.process_recognized_text(text)


def run_gui(duration_sec: Optional[float] = None) -> None:
    """
    运行 3D GUI。
    - duration_sec: 到时自动退出（便于演示稳定运行 ≥1 分钟）。
    """
    try:
        from PyQt6 import QtCore, QtWidgets
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "缺少 GUI 依赖。请安装: pip install PyQt6 pyqtgraph PyOpenGL"
        ) from e

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("机器人听觉感知系统 — 3D 可视化")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)

    # 顶部状态条
    status = QtWidgets.QLabel(
        "提示：对麦克风说「前进/后退/停止/左转/右转/状态」。播报时不会监听以避免自触发。"
    )
    layout.addWidget(status)

    # 3D 视图
    view = gl.GLViewWidget()
    view.setCameraPosition(distance=8, elevation=25, azimuth=45)
    layout.addWidget(view, 1)

    grid = gl.GLGridItem()
    grid.setSize(10, 10)
    grid.setSpacing(1, 1)
    view.addItem(grid)

    axis = gl.GLAxisItem()
    axis.setSize(2, 2, 2)
    view.addItem(axis)

    # 机器人：立方体
    cube_md = _make_box_mesh(gl, size=0.3)
    robot_item = gl.GLMeshItem(
        meshdata=cube_md,
        smooth=False,
        shader="shaded",
        color=(0.2, 0.6, 1.0, 1.0),
        drawEdges=True,
        edgeColor=(0.9, 0.9, 0.9, 0.8),
    )
    view.addItem(robot_item)

    # 朝向箭头（线段）
    heading = gl.GLLinePlotItem(pos=[(0, 0, 0), (0, 0.8, 0)], color=(1, 0.2, 0.2, 1), width=3)
    view.addItem(heading)

    # 轨迹线
    path_points: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    path = gl.GLLinePlotItem(pos=path_points, color=(0.1, 0.9, 0.3, 1), width=2)
    view.addItem(path)

    # 指令队列 + 语音线程
    cmd_queue: "queue.Queue[str]" = queue.Queue(maxsize=64)
    stop_event = threading.Event()
    t = threading.Thread(target=_voice_worker, args=(cmd_queue, stop_event), daemon=True)
    t.start()

    state = RobotState()
    start = time.monotonic()

    def _render_state() -> None:
        # 机器人：重置变换后再旋转/平移
        robot_item.resetTransform()
        robot_item.rotate(state.yaw_deg, 0, 0, 1)
        robot_item.translate(state.x, state.y, state.z + 0.15)

        # 朝向线段
        yaw = math.radians(state.yaw_deg)
        hx = math.sin(yaw) * 0.8
        hy = math.cos(yaw) * 0.8
        heading.setData(pos=[(state.x, state.y, state.z + 0.15), (state.x + hx, state.y + hy, state.z + 0.15)])

        # 轨迹
        if not path_points or (path_points[-1][0], path_points[-1][1]) != (state.x, state.y):
            path_points.append((state.x, state.y, state.z + 0.02))
            path.setData(pos=path_points)

        status.setText(
            f"位置: ({state.x:.2f}, {state.y:.2f}, {state.z:.2f})  朝向: {state.yaw_deg:.0f}°   "
            f"支持: {', '.join(config.COMMAND_KEYWORDS.keys())}"
        )

    def _tick() -> None:
        nonlocal state

        # duration 自动退出
        if duration_sec is not None and (time.monotonic() - start) >= duration_sec:
            win.close()
            return

        # 处理队列中的所有指令（保持 GUI 流畅）
        processed = 0
        while True:
            try:
                cmd_id = cmd_queue.get_nowait()
            except queue.Empty:
                break
            state = _apply_command(state, cmd_id)
            processed += 1

        if processed:
            _render_state()

    timer = QtCore.QTimer()
    timer.setInterval(30)  # ~33fps
    timer.timeout.connect(_tick)
    timer.start()

    def _on_close(_: object) -> None:
        stop_event.set()

    win.destroyed.connect(_on_close)
    win.resize(1000, 700)
    win.show()
    _render_state()

    app.exec()

