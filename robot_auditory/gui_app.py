"""
Qt + 3D 可视化 + 双模式语音/文字交互：
- 左按钮：语音/文字控制小车（支持“前进/后退/左转/右转/停止/状态”，并可解析“一步/一米”等距离单位）
- 右按钮：语音/文字问答（接入千问/DashScope，代码内置提示词）
同时满足：
- 播报期间不监听，避免自触发循环
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from . import config
from .enrollment import ensure_voiceprint_enrolled
from .info_tools import build_live_info_context, should_answer_with_live_info
from .tts import speak
from .voiceprint import VoiceprintVerifier

logger = logging.getLogger(__name__)


def _load_qianwen_api_key() -> str:
    file_path = config.QIANWEN_API_KEY_FILE
    if file_path.exists():
        key = file_path.read_text(encoding="utf-8").strip()
        if key:
            logger.info("已从文件加载千问 API key：%s", file_path)
            return key
        logger.warning("千问 API key 文件为空：%s", file_path)

    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANWEN_API_KEY") or ""
    if key:
        logger.info("已从环境变量加载千问 API key。")
    return key


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0  # 0度朝 +X，+Y 为车体右侧，+Z 向上


@dataclass
class Motion:
    kind: str  # "move" | "turn"
    cmd_id: str
    remaining: float  # meters for move, degrees for turn
    speed_scale: float = 1.0


class SpeakingFlag:
    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


def _make_box_mesh(gl, size: Tuple[float, float, float]):
    """生成盒子 MeshData，兼容不同 pyqtgraph 版本。"""
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    verts = [
        (-sx, -sy, -sz),
        (sx, -sy, -sz),
        (sx, sy, -sz),
        (-sx, sy, -sz),
        (-sx, -sy, sz),
        (sx, -sy, sz),
        (sx, sy, sz),
        (-sx, sy, sz),
    ]
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


def _apply_move(state: RobotState, cmd_id: str, distance_m: float, step_yaw_deg: float = 90.0) -> RobotState:
    """离散运动：0 度朝 +X；从 +Z 往下看，左转逆时针，右转顺时针。"""
    yaw = math.radians(state.yaw_deg)
    dx = math.cos(yaw)
    dy = math.sin(yaw)

    if cmd_id == "forward":
        state.x += distance_m * dx
        state.y += distance_m * dy
    elif cmd_id == "backward":
        state.x -= distance_m * dx
        state.y -= distance_m * dy
    elif cmd_id == "turn_left":
        state.yaw_deg = (state.yaw_deg - step_yaw_deg) % 360.0
    elif cmd_id == "turn_right":
        state.yaw_deg = (state.yaw_deg + step_yaw_deg) % 360.0
    elif cmd_id in ("stop", "status"):
        pass
    return state


def _car_feedback_text(commands: List[config.MotionCommand]) -> str:
    parts: List[str] = []
    for command in commands:
        speed_prefix = ""
        if command.speed_scale < 0.9:
            speed_prefix = "慢速"
        elif command.speed_scale > 1.1:
            speed_prefix = "快速"

        if command.cmd_id == "forward":
            parts.append(f"{speed_prefix}前进 {command.distance_m:.2f} 米" if speed_prefix else f"前进 {command.distance_m:.2f} 米")
        elif command.cmd_id == "backward":
            parts.append(f"{speed_prefix}后退 {command.distance_m:.2f} 米" if speed_prefix else f"后退 {command.distance_m:.2f} 米")
        elif command.cmd_id == "turn_left":
            parts.append(f"{speed_prefix}左转 {command.angle_deg:.0f} 度" if speed_prefix else f"左转 {command.angle_deg:.0f} 度")
        elif command.cmd_id == "turn_right":
            parts.append(f"{speed_prefix}右转 {command.angle_deg:.0f} 度" if speed_prefix else f"右转 {command.angle_deg:.0f} 度")
        elif command.cmd_id == "stop":
            parts.append("停止")
        elif command.cmd_id == "status":
            parts.append("报告状态")
    if not parts:
        return config.UNKNOWN_FEEDBACK
    return "收到，" + "，然后".join(parts) + "。"


def _chat_system_prompt() -> str:
    # 代码内置提示词：让页面保持简洁
    return (
        "你是一个中文语音助手，回答要简洁、明确。"
        "尽量用两三句话完成回复。"
        "如用户提问不明确，请先提出1-2个澄清问题。"
    )


def _parse_user_html(text: str) -> str:
    # 防止 HTML 注入；并保持换行
    t = html.escape(text).replace("\n", "<br/>")
    return t


class Car3DScene:
    """3D 场景渲染：白色地面/小车模型/朝向线/轨迹。"""

    def __init__(self, gl, view) -> None:
        self.gl = gl
        self.view = view

        ground_md = gl.MeshData(
            vertexes=[
                (-7.0, -7.0, 0.0),
                (7.0, -7.0, 0.0),
                (7.0, 7.0, 0.0),
                (-7.0, 7.0, 0.0),
            ],
            faces=[(0, 1, 2), (0, 2, 3)],
        )
        ground = gl.GLMeshItem(
            meshdata=ground_md,
            smooth=False,
            color=(1.0, 1.0, 1.0, 1.0),
            shader="shaded",
        )
        view.addItem(ground)

        # 车模型（多个盒体拼装）
        # 约定：local 坐标系中，+X 是前进方向，+Y 是右侧，+Z 向上
        self.parts = []

        def add_part(mesh_size, local_offset, color):
            md = _make_box_mesh(gl, mesh_size)
            item = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                shader="shaded",
                color=color,
                drawEdges=True,
                edgeColor=(0.15, 0.15, 0.15, 0.65),
            )
            view.addItem(item)
            self.parts.append((item, local_offset))

        # body
        add_part(
            mesh_size=(1.05, 0.62, 0.18),
            local_offset=(0.0, 0.0, 0.09),
            color=(0.12, 0.62, 0.98, 1.0),
        )
        # cabin
        add_part(
            mesh_size=(0.56, 0.50, 0.22),
            local_offset=(0.16, 0.0, 0.25),
            color=(0.30, 0.82, 1.0, 1.0),
        )
        # head block
        add_part(
            mesh_size=(0.26, 0.20, 0.12),
            local_offset=(0.58, 0.0, 0.28),
            color=(0.40, 0.92, 1.0, 1.0),
        )
        # wheels (4)
        wheel_color = (0.08, 0.08, 0.10, 1.0)
        wx = 0.48
        wy = 0.33
        add_part(mesh_size=(0.22, 0.14, 0.07), local_offset=(-wx, -wy, 0.035), color=wheel_color)
        add_part(mesh_size=(0.22, 0.14, 0.07), local_offset=(wx, -wy, 0.035), color=wheel_color)
        add_part(mesh_size=(0.22, 0.14, 0.07), local_offset=(-wx, wy, 0.035), color=wheel_color)
        add_part(mesh_size=(0.22, 0.14, 0.07), local_offset=(wx, wy, 0.035), color=wheel_color)

        # headlights (2)
        light_color = (1.0, 0.95, 0.65, 1.0)
        add_part(mesh_size=(0.06, 0.10, 0.06), local_offset=(0.64, -0.16, 0.10), color=light_color)
        add_part(mesh_size=(0.06, 0.10, 0.06), local_offset=(0.64, 0.16, 0.10), color=light_color)

        # 朝向箭头（线段）
        self.heading = gl.GLLinePlotItem(
            pos=[(0, 0, 0.0), (0.9, 0, 0.0)],
            color=(1, 0.2, 0.2, 1),
            width=3,
        )
        view.addItem(self.heading)

        # 轨迹
        self.path_points = [(0.0, 0.0, 0.02)]
        self.path = gl.GLLinePlotItem(
            pos=self.path_points,
            color=(0.1, 0.9, 0.3, 1),
            width=2,
        )
        view.addItem(self.path)

        self._last_pose_xy = (0.0, 0.0)

    def render(self, state: RobotState) -> None:
        # 更新小车各部件变换
        for item, local_offset in self.parts:
            item.resetTransform()
            ox, oy, oz = local_offset
            # 先局部偏移，再旋转，最后整体平移，保证车头/车身/轮子整体跟随
            item.translate(ox, oy, oz)
            item.rotate(state.yaw_deg, 0, 0, 1)
            item.translate(state.x, state.y, state.z)

        # 更新朝向线段
        yaw = math.radians(state.yaw_deg)
        hx = math.cos(yaw) * 0.9
        hy = math.sin(yaw) * 0.9
        z_arrow = state.z + 0.25
        self.heading.setData(
            pos=[(state.x, state.y, z_arrow), (state.x + hx, state.y + hy, z_arrow)]
        )

        # 更新轨迹：只有在位置变化时才追加
        if (state.x, state.y) != self._last_pose_xy:
            self._last_pose_xy = (state.x, state.y)
            self.path_points.append((state.x, state.y, state.z + 0.02))
            self.path.setData(pos=self.path_points)


class QianwenClient:
    def __init__(self, api_key: str, model: str, base_url: str, region: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.region = region

    def chat(self, messages: List[dict], max_tokens: int = 220) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        logger.info("正在调用千问模型：model=%s, region=%s, endpoint=%s", self.model, self.region, endpoint)
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ssl_context = None
        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_context = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, timeout=60, context=ssl_context) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"千问请求失败: status={exc.code}, message={detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"千问网络请求失败: {exc.reason}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"千问响应格式异常: {data}") from exc


def run_gui(duration_sec: Optional[float] = None) -> None:
    try:
        from PyQt6 import QtCore, QtWidgets
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("缺少 GUI 依赖：pip install PyQt6 pyqtgraph PyOpenGL") from e

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("机器人听觉感知系统 — 语音/问答/3D 行走")

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    outer = QtWidgets.QHBoxLayout(central)

    # 左侧：3D + 提示
    left_col = QtWidgets.QVBoxLayout()
    outer.addLayout(left_col, 3)

    hint = QtWidgets.QLabel(
        "操作提示：前进/后退/左转/右转/停止/状态；可说“前进1m然后左转30度再前进1m”。"
    )
    left_col.addWidget(hint)

    coord = QtWidgets.QLabel("坐标：(0.00, 0.00, 0.00)  朝向：0.0°")
    left_col.addWidget(coord)

    odo = QtWidgets.QLabel("里程：0.00 m")
    left_col.addWidget(odo)

    view = gl.GLViewWidget()
    view.setCameraPosition(distance=9, elevation=25, azimuth=45)
    view.setBackgroundColor("white")
    left_col.addWidget(view, 1)

    scene = Car3DScene(gl, view)

    # 右侧：模式按钮 + 聊天区
    right_col = QtWidgets.QVBoxLayout()
    outer.addLayout(right_col, 2)

    mode_row = QtWidgets.QHBoxLayout()
    right_col.addLayout(mode_row)

    btn_car = QtWidgets.QPushButton("语音输入控制小车")
    btn_chat = QtWidgets.QPushButton("语音交互问答")
    mode_row.addWidget(btn_car)
    mode_row.addWidget(btn_chat)

    voiceprint_row = QtWidgets.QHBoxLayout()
    right_col.addLayout(voiceprint_row)
    btn_reenroll = QtWidgets.QPushButton("重新录入声纹")
    voiceprint_row.addWidget(btn_reenroll)
    voiceprint_state = QtWidgets.QLabel("声纹：初始化中")
    right_col.addWidget(voiceprint_state)
    voiceprint_verify = QtWidgets.QLabel("最近比对：暂无")
    right_col.addWidget(voiceprint_verify)
    qianwen_state = QtWidgets.QLabel("千问：初始化中")
    right_col.addWidget(qianwen_state)

    active_mode_lock = threading.Lock()
    active_mode = "car"  # "car" | "chat"

    # 右侧“车控回显”
    car_cmd_preview = QtWidgets.QLabel("（等待指令…）")
    car_cmd_preview.setStyleSheet("color:#333;")
    right_col.addWidget(car_cmd_preview)

    def set_mode(m: str) -> None:
        nonlocal active_mode
        with active_mode_lock:
            active_mode = m
        if m == "car":
            hint.setText("操作提示：前进/后退/左转/右转/停止/状态；可说“前进1m然后左转30度再前进1m”。")
        else:
            hint.setText("问答模式：你可以直接提问（支持语音/文字）。")

    btn_car.clicked.connect(lambda: set_mode("car"))
    btn_chat.clicked.connect(lambda: set_mode("chat"))

    chat_box = QtWidgets.QTextBrowser()
    chat_box.setReadOnly(True)
    chat_box.setStyleSheet("background: #f5f5f5; border: 1px solid #ddd;")
    right_col.addWidget(chat_box, 1)

    input_row = QtWidgets.QHBoxLayout()
    right_col.addLayout(input_row)

    line = QtWidgets.QLineEdit()
    line.setPlaceholderText("输入指令或问题…（回车发送）")
    input_row.addWidget(line, 1)
    send_btn = QtWidgets.QPushButton("发送")
    input_row.addWidget(send_btn)

    status_bar = QtWidgets.QLabel("提示：播报期间将自动暂停监听，避免自触发。")
    right_col.addWidget(status_bar)

    ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue(maxsize=128)

    def post_ui(fn: Callable[[], None]) -> None:
        try:
            ui_queue.put_nowait(fn)
        except queue.Full:
            logger.warning("UI 队列已满，丢弃一次界面更新")

    def update_voiceprint_labels(
        verifier: Optional[VoiceprintVerifier],
        verify_text: Optional[str] = None,
    ) -> None:
        if not config.VOICEPRINT_ENABLED:
            voiceprint_state.setText("声纹：未启用")
            voiceprint_verify.setText("最近比对：未启用")
            return
        if verifier is None:
            voiceprint_state.setText("声纹：加载中")
        else:
            enrolled = "已录入" if verifier.enrolled and not verifier.needs_reenroll else "待录入"
            warmed = "已预热" if verifier.warmed_up else "未预热"
            voiceprint_state.setText(f"声纹：{enrolled}  模型：{warmed}")

        if verify_text is not None:
            voiceprint_verify.setText(verify_text)

    def update_qianwen_state(status: str) -> None:
        qianwen_state.setText(f"千问：{qianwen_model} / {qianwen_region} / {status}")

    # --- 共享状态 ---
    state = RobotState()
    state_lock = threading.Lock()
    speaking_flag = SpeakingFlag()
    odometer_m = 0.0
    motion_q: "queue.Queue[Motion]" = queue.Queue(maxsize=128)
    current_motion: Optional[Motion] = None
    last_tick = time.monotonic()

    # 平滑运动参数
    MOVE_SPEED_MPS = 0.9  # 米/秒
    TURN_SPEED_DPS = 220.0  # 度/秒

    # 左模式去重（避免语音重复误触发）
    debounce_lock = threading.Lock()
    last_action_key: Optional[Tuple[Tuple[str, float, float], ...]] = None
    last_action_time = 0.0

    # --- Car 处理 ---
    def _clear_pending_motion() -> None:
        nonlocal current_motion
        current_motion = None
        while True:
            try:
                motion_q.get_nowait()
            except queue.Empty:
                break

    def try_apply_car_command(cmd_text: str, from_voice: bool) -> None:
        nonlocal last_action_key, last_action_time

        commands = config.match_motion_commands(cmd_text)
        if not commands:
            # 未识别运动指令，忽略（不想在 GUI 里频繁打扰）
            logger.info("车控：未匹配运动指令: %s", cmd_text)
            return

        key = tuple((c.cmd_id, round(c.distance_m, 2), round(c.angle_deg, 1)) for c in commands)
        now = time.monotonic()
        with debounce_lock:
            if last_action_key == key and (now - last_action_time) < config.COMMAND_DEBOUNCE:
                logger.info("车控去重：忽略重复指令 %s", key)
                return
            last_action_key = key
            last_action_time = now

        # 右侧车控回显
        src = "语音指令" if from_voice else "文字指令"
        preview = " -> ".join(
            f"{c.cmd_id}:{c.distance_m:.2f}m@{c.speed_scale:.1f}x" if c.cmd_id in ("forward", "backward")
            else f"{c.cmd_id}:{c.angle_deg:.0f}deg@{c.speed_scale:.1f}x" if c.cmd_id in ("turn_left", "turn_right")
            else c.cmd_id
            for c in commands
        )
        car_cmd_preview.setText(f"({src}): {preview}")
        chat_box.append(f"<div style='color:#666; font-size:12px;'>({src}): {html.escape(cmd_text)}</div>")
        chat_box.verticalScrollBar().setValue(chat_box.verticalScrollBar().maximum())

        # 加入运动队列（平滑移动/转向）
        try:
            for command in commands:
                if command.cmd_id in ("forward", "backward") and command.distance_m > 0:
                    motion_q.put_nowait(
                        Motion(
                            kind="move",
                            cmd_id=command.cmd_id,
                            remaining=float(command.distance_m),
                            speed_scale=command.speed_scale,
                        )
                    )
                elif command.cmd_id in ("turn_left", "turn_right") and command.angle_deg > 0:
                    motion_q.put_nowait(
                        Motion(
                            kind="turn",
                            cmd_id=command.cmd_id,
                            remaining=float(command.angle_deg),
                            speed_scale=command.speed_scale,
                        )
                    )
                elif command.cmd_id == "stop":
                    _clear_pending_motion()
        except queue.Full:
            logger.warning("运动队列已满，丢弃部分指令: %s", cmd_text)

        # 语音反馈（在后台线程，避免卡 GUI）
        feedback = _car_feedback_text(commands)
        speaking_flag.set()

        def _speak_bg() -> None:
            try:
                speak(feedback)
                time.sleep(config.COOLDOWN_AFTER_SPEECH)
            finally:
                speaking_flag.clear()

        threading.Thread(target=_speak_bg, daemon=True).start()

        if from_voice:
            status_bar.setText(f"已执行（语音）: {preview}")
        else:
            status_bar.setText(f"已执行（文字）: {preview}")

    # --- Chat 处理 ---
    qianwen_api_key = _load_qianwen_api_key()
    qianwen_model = os.getenv("DASHSCOPE_MODEL") or config.QIANWEN_MODEL
    qianwen_region = config.QIANWEN_REGION
    qianwen_base_url = config.QIANWEN_BASE_URL
    if qianwen_api_key:
        update_qianwen_state("已配置")
    else:
        update_qianwen_state("未配置")

    history_lock = threading.Lock()
    # OpenAI-like messages: [{'role':'system'|'user'|'assistant','content':...}]
    chat_messages: List[dict] = [{"role": "system", "content": _chat_system_prompt()}]

    def append_bubble(is_user: bool, text: str) -> None:
        color = "#bfe3ff" if is_user else "#ffffff"
        float_side = "left" if is_user else "right"
        # max-width 限制气泡宽度，保持简洁
        bubble = (
            f"<div style='margin:6px 0;'>"
            f"<div style='background:{color}; padding:8px 10px; border-radius:10px; "
            f"max-width:85%; float:{float_side}; color:#000; clear:both;'>"
            f"{_parse_user_html(text)}</div></div>"
        )
        chat_box.append(bubble)
        chat_box.verticalScrollBar().setValue(chat_box.verticalScrollBar().maximum())

    def try_chat_answer(user_text: str, from_voice: bool) -> None:
        if not qianwen_api_key:
            update_qianwen_state("未配置")
            append_bubble(
                False,
                f"千问 API key 未配置。请在项目根目录提供 `{config.QIANWEN_API_KEY_FILE}` 文件，或设置环境变量 `DASHSCOPE_API_KEY`。",
            )
            return

        append_bubble(True, user_text)
        status_bar.setText(f"正在请求千问：{qianwen_model} / {qianwen_region}…")
        update_qianwen_state("请求中")

        speaking_flag.set()

        client = QianwenClient(
            api_key=qianwen_api_key,
            model=qianwen_model,
            base_url=qianwen_base_url,
            region=qianwen_region,
        )

        def _worker() -> None:
            try:
                # 问答期间（思考 + 播报）暂停语音监听，避免误触发
                speaking_flag.set()
                # 维护短历史，控制 token 与延迟
                with history_lock:
                    chat_messages.append({"role": "user", "content": user_text})
                    # keep: system + last 8 turns (user/assistant)
                    keep_turns = 8
                    body = chat_messages
                    if len(body) > 1 + keep_turns * 2:
                        body = [body[0]] + body[-keep_turns * 2 :]

                tool_context = build_live_info_context(user_text) if should_answer_with_live_info(user_text) else None
                if tool_context:
                    body = body + [{"role": "system", "content": tool_context}]

                answer = client.chat(body, max_tokens=220)
                answer = (answer or "").strip()
                if not answer:
                    answer = "抱歉，我没有理解您的问题。"
                with history_lock:
                    chat_messages.append({"role": "assistant", "content": answer})

                # 语音播报 + UI 更新
                post_ui(lambda: (append_bubble(False, answer), status_bar.setText("就绪。"), update_qianwen_state("已连接")))

                speak(answer)
                time.sleep(config.COOLDOWN_AFTER_SPEECH)
            except Exception as ex:  # noqa: BLE001
                logger.exception("千问请求异常")
                err_msg = f"千问请求失败：{ex}"

                post_ui(
                    lambda err_msg=err_msg: (
                        append_bubble(False, err_msg),
                        status_bar.setText("请求失败。"),
                        update_qianwen_state("连接失败"),
                    )
                )
            finally:
                speaking_flag.clear()

        threading.Thread(target=_worker, daemon=True).start()

    # --- 接入口：统一处理文字/语音识别结果 ---
    def handle_text(text: str, from_voice: bool) -> None:
        if not text:
            return

        with active_mode_lock:
            mode = active_mode

        if mode == "car":
            try_apply_car_command(text, from_voice=from_voice)
        else:
            # chat：识别到的内容也当作用户问题
            try_chat_answer(text, from_voice=from_voice)

    # --- 语音工作线程（持续监听并按模式分发）---
    stop_event = threading.Event()
    reenroll_event = threading.Event()

    def _voice_worker() -> None:
        from .listener import VoiceListener

        logger.info("GUI 语音线程已启动，准备进入监听状态。")
        verifier = VoiceprintVerifier() if config.VOICEPRINT_ENABLED else None
        post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier))
        with VoiceListener() as listener:
            listener.calibrate_ambient()
            if verifier is not None:
                verifier.warmup()
                post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier))
                speaking_flag.set()
                try:
                    ensure_voiceprint_enrolled(listener, verifier)
                    post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier, "最近比对：暂无"))
                finally:
                    speaking_flag.clear()
            while not stop_event.is_set():
                if speaking_flag.is_set():
                    time.sleep(0.05)
                    continue

                if verifier is not None and reenroll_event.is_set():
                    reenroll_event.clear()
                    speaking_flag.set()
                    try:
                        verifier.reset_profile(delete_file=True)
                        post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier, "最近比对：暂无"))
                        post_ui(lambda: status_bar.setText("正在重新录入声纹…"))
                        ensure_voiceprint_enrolled(listener, verifier)
                        post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier, "最近比对：暂无"))
                        post_ui(lambda: status_bar.setText("声纹已重新录入。"))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("重新录入声纹失败")
                        err_msg = f"重新录入声纹失败：{exc}"
                        post_ui(lambda err_msg=err_msg: status_bar.setText(err_msg))
                    finally:
                        speaking_flag.clear()
                    continue

                with active_mode_lock:
                    mode_snapshot = active_mode

                use_voiceprint = mode_snapshot == "car"
                text = listener.listen_once(
                    voiceprint_verifier=verifier if use_voiceprint else None,
                    on_voiceprint_checked=(
                        lambda matched, score, verifier=verifier: post_ui(
                            lambda verifier=verifier, matched=matched, score=score: update_voiceprint_labels(
                                verifier,
                                f"最近比对：{'通过' if matched else '拒绝'}  分数：{score:.3f}",
                            )
                        )
                    )
                    if use_voiceprint and verifier is not None
                    else None,
                )
                if not text:
                    continue

                try:
                    if not use_voiceprint and verifier is not None:
                        post_ui(lambda verifier=verifier: update_voiceprint_labels(verifier, "最近比对：问答模式不校验"))
                    post_ui(lambda text=text: handle_text(text, from_voice=True))
                except Exception:  # noqa: BLE001
                    logger.exception("处理语音文本异常")

    t = threading.Thread(target=_voice_worker, daemon=True)
    t.start()

    def _trigger_reenroll() -> None:
        if not config.VOICEPRINT_ENABLED:
            status_bar.setText("当前未启用声纹功能。")
            return
        reenroll_event.set()
        status_bar.setText("已请求重新录入声纹，请稍候…")

    btn_reenroll.clicked.connect(_trigger_reenroll)

    # --- 渲染定时器 ---
    start = time.monotonic()

    def _tick() -> None:
        nonlocal current_motion, last_tick, odometer_m

        while True:
            try:
                ui_queue.get_nowait()()
            except queue.Empty:
                break
            except Exception:  # noqa: BLE001
                logger.exception("UI 更新异常")

        if duration_sec is not None and (time.monotonic() - start) >= duration_sec:
            win.close()
            return

        if view.opts.get("elevation", 0) < 2:
            view.setCameraPosition(elevation=2)

        now = time.monotonic()
        dt = max(0.0, min(0.05, now - last_tick))
        last_tick = now

        # 平滑运动更新：逐帧执行 motion
        if current_motion is None:
            try:
                current_motion = motion_q.get_nowait()
            except queue.Empty:
                current_motion = None

        if current_motion is not None and dt > 0:
            with state_lock:
                if current_motion.kind == "move":
                    step = min(MOVE_SPEED_MPS * current_motion.speed_scale * dt, current_motion.remaining)
                    before_x, before_y = state.x, state.y
                    _apply_move(state, current_motion.cmd_id, step)
                    odometer_m += math.hypot(state.x - before_x, state.y - before_y)
                    current_motion.remaining -= step
                    if current_motion.remaining <= 1e-6:
                        current_motion = None
                elif current_motion.kind == "turn":
                    step_deg = min(TURN_SPEED_DPS * current_motion.speed_scale * dt, current_motion.remaining)
                    _apply_move(state, current_motion.cmd_id, 0.0, step_yaw_deg=step_deg)
                    current_motion.remaining -= step_deg
                    if current_motion.remaining <= 1e-6:
                        current_motion = None

        with state_lock:
            coord.setText(
                f"坐标：({state.x:.2f}, {state.y:.2f}, {state.z:.2f})  朝向：{state.yaw_deg:.1f}°"
            )
            odo.setText(f"里程：{odometer_m:.2f} m")
            scene.render(state)

    timer = QtCore.QTimer()
    timer.setInterval(30)
    timer.timeout.connect(_tick)
    timer.start()

    def _on_close(_: object) -> None:
        stop_event.set()

    win.destroyed.connect(_on_close)
    win.resize(1180, 760)
    win.show()
    # 初始渲染
    with state_lock:
        scene.render(state)

    # 输入事件
    def _send_from_input() -> None:
        text = line.text().strip()
        if not text:
            return
        line.clear()
        # 在主线程触发轻量 car；chat 会在内部起 worker
        handle_text(text, from_voice=False)

    send_btn.clicked.connect(_send_from_input)
    line.returnPressed.connect(_send_from_input)

    app.exec()
