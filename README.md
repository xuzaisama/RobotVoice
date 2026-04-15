# 机器人听觉感知系统（课程项目）

实现 **语音输入 → 识别理解 → 行为执行 → 语音反馈** 闭环，并具备 **播报时不采集麦克风** 的自干扰抑制，避免机器人被自己的语音误触发。

## 环境

- Python 3.9+
- 麦克风、扬声器（或耳机）
- **语音识别**使用 `SpeechRecognition` 调用的 Google Web Speech API，**需要能访问 Google 的网络**（校园网若屏蔽需自备网络）

## 完整启动指令

建议始终在项目根目录执行以下命令。

### 1. 首次安装

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若 `PyAudio` 安装失败（macOS 常见），先执行：

```bash
brew install portaudio
source .venv/bin/activate
python -m pip install PyAudio
```

非 macOS 且未安装 `say` 时，可额外安装离线播报：

```bash
source .venv/bin/activate
python -m pip install pyttsx3
```

### 2. 配置千问 API

项目默认从根目录读取 [千问api_key](/Users/xuzai/Desktop/大学/大三下/机器人传感/千问api_key)，并使用 `qwen3.6-plus` 模型、华北2（北京）地域：

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
echo "你的 API key" > 千问api_key
```

若你更希望走环境变量，也仍然支持：

```bash
export DASHSCOPE_API_KEY="你的 API key"
```

### 3. 日常启动前先激活环境

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
source .venv/bin/activate
```

### 4. 启动命令

1. 启动命令行语音控制版：

```bash
python -m robot_auditory
```

2. 启动 3D GUI 版：

```bash
python -m robot_auditory --gui
```

3. 运行 70 秒后自动退出（便于课程验收）：

```bash
python -m robot_auditory --duration 70
```

4. 启动 3D GUI，并在 70 秒后自动退出：

```bash
python -m robot_auditory --gui --duration 70
```

### 5. 一套最完整的 GUI 启动流程

如果你想直接照着一步步启动完整系统，可以用下面这一套：

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
source .venv/bin/activate
python -m robot_auditory --gui
```

程序启动后：

1. 系统会预热 PaddleSpeech 声纹模型  
2. 若本地没有声纹档案，会提示录入 5 句固定声纹文本  
3. 进入 GUI 后，左侧可语音控制小车，右侧可进行千问问答  
4. 右侧状态栏会显示当前千问模型、地域和连接状态

GUI 版本支持文字/语音控制小车，也支持较长的连续指令，例如：

```text
前进1m然后左转30度再前进1m
```

坐标系约定：小车初始坐标为 `(0, 0, 0)`，初始朝向 `+X`，初始右侧为 `+Y`，正上方为 `+Z`，`XOY` 面为地面。默认“一步”为 `0.5m`，仅说“前进/后退”时按一步执行；仅说“左转/右转”时默认转向 `90°`。

问答模式已接入时间/天气工具：当你询问“现在几点”“今天几号”“今天天气怎么样”“上海天气如何”这类问题时，系统会先查询本地时间或实时天气，再把结果交给模型生成自然语言回复。若未显式说明城市，默认按 `Asia/Shanghai` 时区与 `Shanghai` 城市处理，也可通过环境变量 `ROBOT_LOCAL_CITY`、`ROBOT_LOCAL_TIMEZONE` 修改。

首次启动语音功能时，系统会提示录入 `5` 次本地声纹，并在项目根目录生成 `voiceprint_profile.json`。当前版本使用官方 PaddleSpeech ECAPA-TDNN 声纹模型提取嵌入，并结合多样本训练结果做本地判别：会综合比较候选语音与录入样本中心的相似度、与最近邻样本的相似度，以及样本分布距离。旧版声纹档案会自动触发重新录入。后续只有匹配该声纹的语音会进入指令识别流程；同时系统还会屏蔽最近刚播报过的反馈文本，并在播报结束后维持一小段监听封锁窗口，以进一步降低误触发和“自启动”现象。

说明：声纹录入阶段会临时绕过播报保护期，避免提示音结束后用户来不及录入样本；正常控制阶段仍保留播报保护与回声屏蔽。

当前固定录入文案依次为：
`你好，现在我要开始录入声纹信息`、
`今天天气很好，适合出门散步`、
`我的电话号码是一二三四五六七`、
`你晚上想吃什么呢？`、
`以上就是我的声纹信息`

注意：首次真正使用 PaddleSpeech 声纹功能时，程序会自动下载官方预训练权重到 [`.cache/paddlespeech`](/Users/xuzai/Desktop/大学/大三下/机器人传感/.cache/paddlespeech)。

GUI 版本提供“重新录入声纹”按钮，可在更换使用者或录入效果不理想时重新采样训练。

## 支持的指令（不少于 5 类）

| 类别   | 示例说法       |
| ------ | -------------- |
| 前进   | 前进、向前走   |
| 后退   | 后退、往后退   |
| 停止   | 停止、停下     |
| 左转   | 左转、往左     |
| 右转   | 右转、往右     |
| 状态   | 状态、报告     |

可在 `robot_auditory/config.py` 中修改关键词与反馈话术。

## 自干扰抑制说明

- 语音播报开始前将内部标志置为「正在播报」；**主循环在此标志为真时不调用 `listen`**，因此不会识别机器人自己的扬声器输出。
- 播报结束后额外等待 `COOLDOWN_AFTER_SPEECH` 秒再恢复监听，减轻尾音被拾取。

## 测试（无麦克风）

```bash
python -m pytest tests/ -q
# 或
python -m unittest tests.test_config
```

## 性能与报告建议

- **准确率**：在安静环境至少录 10 组口令，统计识别正确比例（目标 ≥80%），写入课程报告。
- **延迟**：可用秒表记录「说完指令」到「终端打印动作」的时间（建议 ≤2s，受网络影响）。
- **抗噪**：可在播放轻微背景音时重复测试，记录误触发情况。

## 项目结构

```
机器人传感/
├── requirements.txt
├── README.md
├── robot_auditory/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py      # 指令与参数
│   ├── listener.py    # 麦克风 + 识别
│   ├── gui.py         # 旧版 3D 可视化界面（Qt + OpenGL）
│   ├── gui_app.py     # 当前 GUI：3D 小车 + 文字/语音控制 + 千问问答
│   ├── tts.py         # 播报（macOS 优先 say）
│   ├── controller.py  # 动作、去重、与播报联动
│   └── main.py        # 入口循环
└── tests/
    ├── test_config.py
    └── test_gui_motion.py
```

将 `default_action_handler` 替换为串口、ROS 或仿真接口即可对接真实机器人。
