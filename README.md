# 机器人听觉感知系统

本项目是一个面向课程展示的机器人听觉感知系统，围绕“听见人说话、判断是不是指定使用者、理解语义、执行动作、再用语音反馈”的完整链路实现。当前版本同时提供命令行模式与 3D GUI 模式，可用于演示机器人语音控制、声纹身份校验、自干扰抑制，以及基于大模型的问答交互能力。

系统主线可概括为：

`语音输入 -> 声纹筛选 -> 语音识别 -> 指令/问题解析 -> 动作执行或问答生成 -> 语音播报反馈`

## 项目亮点

- 支持中文语音控制机器人，覆盖前进、后退、停止、左转、右转、状态查询等基础控制类命令。
- 支持更自然的连续动作表达，例如 `前进1m然后左转30度再前进1m`。
- 集成 PaddleSpeech ECAPA-TDNN 声纹识别，仅允许已录入用户的语音进入控制链路。
- 提供播报保护、最近播报文本回声屏蔽、播报后监听封锁窗口，降低“机器人被自己声音再次触发”的风险。
- 提供 3D GUI，可视化展示小车位置、朝向、轨迹和里程。
- 集成千问问答能力，并支持时间、天气等实时工具查询。
- 提供无需麦克风和真实网络即可运行的核心单元测试，便于课程验收与功能回归。

## 系统功能概述

### 1. 语音控制

命令行模式和 GUI 模式都支持语音输入。控制层会先根据关键词识别指令类别，再将识别结果映射为机器人动作。

当前支持的基础控制类别如下：

| 类别 | 常见说法 |
| --- | --- |
| 前进 | 前进、向前走、直走、继续前进 |
| 后退 | 后退、往后退、退后、倒车 |
| 停止 | 停止、停下、刹车、别动 |
| 左转 | 左转、向左转、左拐、掉头 |
| 右转 | 右转、向右转、右拐 |
| 状态查询 | 状态、报告、当前位置、现在到哪了 |

除基础关键词外，系统还支持以下扩展表达：

- 距离解析：`前进1米`、`后退两步`、`前进一步`
- 角度解析：`右转30度`、`左转三十度`、`掉头`、`右转一圈`
- 速度修饰：`慢慢前进`、`快速右转`
- 连续命令：`直走1米然后右拐再倒车一步`

默认运动参数如下：

- 默认一步距离：`0.5 m`
- 仅说“前进/后退”时：按 `0.5 m`
- 仅说“左转/右转”时：按 `90°`

### 2. 声纹识别与身份校验

系统默认启用本地声纹识别。首次运行时会引导用户录入 5 句固定文本，并在项目根目录生成 [voiceprint_profile.json](/Users/xuzai/Desktop/大学/大三下/机器人传感/voiceprint_profile.json)。

当前固定录入文案为：

1. `你好，现在我要开始录入声纹信息`
2. `今天天气很好，适合出门散步`
3. `我的电话号码是一二三四五六七`
4. `你晚上想吃什么呢？`
5. `以上就是我的声纹信息`

声纹模块基于 PaddleSpeech 官方 ECAPA-TDNN 模型实现，流程如下：

- 首次启动时预热模型
- 对录入样本提取 speaker embedding
- 计算样本中心、最近邻相似度和分布距离
- 后续监听时仅允许匹配通过的语音进入识别阶段

如果检测到旧版声纹档案或不同后端生成的档案，系统会自动要求重新录入。

说明：

- 在 GUI 的“语音输入控制小车”模式下，语音输入会经过声纹校验。
- 在 GUI 的“语音交互问答”模式下，当前实现不做声纹校验，便于演示自由问答。
- GUI 提供“重新录入声纹”按钮，可随时重新采样。

### 3. 自干扰抑制

为避免机器人播报自己的反馈后又被麦克风拾取，系统实现了多层防护：

- 播报期间暂停监听，不调用麦克风采集
- 播报结束后设置额外监听封锁时间
- 记录最近播报文本，若识别结果与播报内容高度相似，则视为回声并忽略
- 长文本播报会根据长度和实际播报时长动态延长保护期

这部分逻辑位于 [robot_auditory/tts.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/tts.py) 和 [robot_auditory/listener.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/listener.py)。

### 4. 3D GUI 与交互问答

GUI 模式提供左右双区域界面：

- 左侧为 3D 小车场景，显示位置、朝向、轨迹和里程
- 右侧可在“语音输入控制小车”和“语音交互问答”之间切换
- 支持文本输入与语音输入两种交互方式
- 提供声纹状态、最近一次比对结果、千问连接状态等界面信息

问答模式接入千问兼容接口，支持：

- 普通中文问答
- 对话上下文保留
- 时间查询
- 天气查询

例如：

- `现在几点`
- `今天几号`
- `上海天气怎么样`
- `北京现在冷不冷`

当用户问题包含时间或天气意图时，系统会优先调用实时工具，再把结果作为上下文交给模型生成答案，减少幻觉。

## 技术架构

项目主要由以下模块组成：

- [robot_auditory/main.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/main.py)：命令行入口，组织监听、声纹预热、指令闭环
- [robot_auditory/gui_app.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/gui_app.py)：当前 GUI 主程序，集成 3D 场景、语音车控、语音问答
- [robot_auditory/listener.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/listener.py)：麦克风采集、Google 语音识别、声纹过滤接入
- [robot_auditory/config.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/config.py)：系统参数、命令词表、距离角度速度解析
- [robot_auditory/controller.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/controller.py)：基础控制闭环、动作执行、去重与播报联动
- [robot_auditory/voiceprint.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/voiceprint.py)：声纹建模、存档、相似度校验
- [robot_auditory/enrollment.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/enrollment.py)：首次录入与重录流程
- [robot_auditory/info_tools.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/info_tools.py)：时间与天气实时工具
- [robot_auditory/tts.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/tts.py)：文本转语音与监听保护

## 运行环境

建议环境：

- Python `3.9+`
- 可用麦克风
- 扬声器或耳机
- 稳定网络连接

联网需求说明：

- 语音识别使用 `SpeechRecognition` 调用 Google Web Speech API，需要可访问对应服务
- 天气查询依赖 Open-Meteo API
- 问答功能依赖千问兼容接口
- 首次使用 PaddleSpeech 声纹模型时会下载预训练权重到 [`.cache/paddlespeech`](/Users/xuzai/Desktop/大学/大三下/机器人传感/.cache/paddlespeech)

## 安装步骤

建议始终在项目根目录执行。

### 1. 创建并激活虚拟环境

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

如果 macOS 上安装 `PyAudio` 失败，可先执行：

```bash
brew install portaudio
python -m pip install PyAudio
```

如果不是 macOS，且系统没有 `say` 命令，可安装 `pyttsx3` 作为播报后备：

```bash
python -m pip install pyttsx3
```

## 配置说明

### 1. 配置千问 API Key

项目默认从根目录读取 [千问api_key](/Users/xuzai/Desktop/大学/大三下/机器人传感/千问api_key) 文件：

```bash
echo "你的 API key" > 千问api_key
```

也支持通过环境变量提供：

```bash
export DASHSCOPE_API_KEY="你的 API key"
```

当前默认参数：

- 模型：`qwen3.6-plus`
- 地域：`华北2（北京）`
- 接口：DashScope compatible-mode endpoint

### 2. 可选环境变量

可以通过环境变量修改本地时间与默认天气城市：

```bash
export ROBOT_LOCAL_CITY="上海"
export ROBOT_LOCAL_TIMEZONE="Asia/Shanghai"
```

也可以覆盖默认模型：

```bash
export DASHSCOPE_MODEL="qwen3.6-plus"
```

## 启动方式

每次启动前先激活环境：

```bash
cd "/Users/xuzai/Desktop/大学/大三下/机器人传感"
source .venv/bin/activate
```

### 1. 命令行语音控制版

```bash
python -m robot_auditory
```

### 2. 3D GUI 版

```bash
python -m robot_auditory --gui
```

### 3. 自动运行指定时长

课程验收时可使用：

```bash
python -m robot_auditory --duration 70
python -m robot_auditory --gui --duration 70
```

## 推荐演示流程

适合课程展示的一套完整演示流程如下：

1. 启动 GUI 版本：`python -m robot_auditory --gui`
2. 等待系统完成环境噪声校准与声纹模型预热
3. 首次运行时完成 5 轮声纹录入
4. 在“语音输入控制小车”模式下依次演示：
   `前进`
   `右转30度`
   `前进1m然后左转30度再前进1m`
   `报告状态`
5. 切换到“语音交互问答”模式，演示：
   `现在几点`
   `上海天气怎么样`
   `介绍一下这个系统`

GUI 场景坐标系约定如下：

- 初始坐标：`(0, 0, 0)`
- 初始朝向：`+X`
- 车体右侧方向：`+Y`
- 正上方：`+Z`
- `XOY` 平面为地面

## 测试

项目已经包含核心逻辑测试，覆盖命令解析、运动学、实时信息工具、TTS 防回声和声纹验证等模块。

运行全部测试：

```bash
python -m pytest tests -q
```

也可以运行单项测试，例如：

```bash
python -m unittest tests.test_config
python -m unittest tests.test_gui_motion
python -m unittest tests.test_info_tools
python -m unittest tests.test_tts_guard
python -m unittest tests.test_voiceprint
```

测试文件包括：

- [tests/test_config.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/tests/test_config.py)：指令关键词、距离角度速度、连续命令解析
- [tests/test_gui_motion.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/tests/test_gui_motion.py)：3D 小车运动学逻辑
- [tests/test_info_tools.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/tests/test_info_tools.py)：时间与天气工具层
- [tests/test_tts_guard.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/tests/test_tts_guard.py)：播报保护与回声屏蔽
- [tests/test_voiceprint.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/tests/test_voiceprint.py)：声纹录入与校验逻辑

## 项目结构

```text
机器人传感/
├── README.md
├── requirements.txt
├── 千问api_key
├── voiceprint_profile.json
├── robot_auditory/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── controller.py
│   ├── enrollment.py
│   ├── gui.py
│   ├── gui_app.py
│   ├── info_tools.py
│   ├── listener.py
│   ├── main.py
│   ├── tts.py
│   └── voiceprint.py
└── tests/
    ├── test_config.py
    ├── test_gui_motion.py
    ├── test_info_tools.py
    ├── test_tts_guard.py
    └── test_voiceprint.py
```

## 后续可扩展方向

- 将 [robot_auditory/controller.py](/Users/xuzai/Desktop/大学/大三下/机器人传感/robot_auditory/controller.py) 中的默认动作处理替换为串口、ROS、Socket 或仿真平台接口，接入真实机器人。
- 将当前基于云端语音识别的实现替换为离线 ASR，提高在弱网环境下的稳定性。
- 在 GUI 中加入更多传感器状态显示，例如麦克风状态、电量、位姿估计误差等。
- 为问答模式增加更多本地工具，例如课程表、地图导航或设备状态查询。

## 注意事项

- 首次运行 PaddleSpeech 声纹功能时，模型下载会较慢，属于正常现象。
- 若麦克风无法打开，请先检查系统权限设置。
- 若语音识别一直失败，优先检查当前网络是否能访问 Google Web Speech 服务。
- 若问答模式无法返回结果，请检查千问 API key 是否已配置。
- 若天气查询失败，通常是实时天气接口不可达或网络波动导致。

## 课程报告撰写建议

如果需要把本项目整理进课程报告，可以从以下几个方面组织结果：

- 功能完整性：是否实现“采集、识别、理解、执行、反馈”的闭环
- 识别效果：统计多组口令的正确识别率
- 实时性：统计从说完指令到界面动作或播报反馈的平均延迟
- 抗干扰能力：比较启用与关闭播报保护时的误触发情况
- 身份鉴别能力：比较本人语音与他人语音的声纹通过率差异

如果后续你希望，我还可以继续基于这版 README 帮你补一份更适合直接交作业的“课程报告式说明”，或者把 README 再改成更偏答辩展示风格的一版。
