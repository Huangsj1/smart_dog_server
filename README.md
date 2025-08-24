# 智能聊天机器人服务器

这是一个基于大语言模型(LLM)、文本转语音(TTS)、自动语音识别(ASR)和模型上下文协议(MCP)的智能对话系统服务器。系统支持语音交互，具备语音活动检测(VAD)和降噪功能，可通过WebSocket与ESP32等客户端设备进行实时语音对话。

## 🚀 主要特性

* **多模态对话**：支持文本和语音输入输出
* **实时语音交互**：基于WebSocket的实时音频流处理
* **语音活动检测**：使用WebRTC VAD进行准确的语音端点检测
* **降噪处理**：集成noisereduce库对客户端音频进行降噪
* **模块化设计**：LLM、TTS、ASR组件可独立配置
* **工具集成**：支持MCP协议扩展AI能力（如地图查询等）
* **多音色支持**：支持CosyVoice和GPT-SoVITS多种TTS引擎

通信流程：
1. 建立WebSocket连接
2. 客户端发送音频流数据
3. 服务器进行VAD检测和降噪处理
4. ASR转换为文本后送入LLM处理
5. LLM响应通过TTS转换为音频
6. 服务器返回音频响应

## 📁 项目结构

```
llm_mcp/
├── main.py                    # 主程序入口（命令行版本）
├── ws_server.py              # WebSocket服务器（客户端对接版本）
├── config.template.yaml      # 配置模板文件
├── config.yaml              # 实际配置文件（需要手动创建）
├── pyproject.toml           # 项目依赖配置
├── chat_handler/            # 对话处理模块
│   ├── chat_tts_handler.py  # 核心对话处理器
│   ├── chat_context_manager.py # 对话上下文管理
│   └── system_role_prompt.yaml # 系统角色提示词
├── my_llm/                  # LLM引擎模块
│   └── openai_engine.py     # OpenAI兼容接口
├── my_tts/                  # TTS引擎模块
│   ├── cosy_voice_engine.py # CosyVoice引擎
│   ├── gpt_sovits_engine.py # GPT-SoVITS引擎
│   └── audio_player.py      # 音频播放器
├── my_asr/                  # ASR引擎模块
│   ├── sensevoice_engine.py # SenseVoice引擎
│   └── audio_record.py      # 音频录制器
├── my_vad/                  # 语音活动检测模块
│   └── webrtc_vad.py        # WebRTC VAD实现
└── my_mcp/                  # MCP客户端模块
    ├── mcp_client.py        # MCP客户端管理器
    ├── tools_whitelist.yaml # 工具白名单配置
    └── my_server/           # 本地MCP服务器
        ├── local_cal.py     # 本地计算工具
        └── remote_greet.py  # 远程问候工具
```

## 🛠️ 安装与配置
* 目前使用的LLM是远程api服务，TTS也是远程api服务，但是asr使用的是本地的sensevoice-small服务（速度很快）

### 1. 配置文件设置
1. 复制配置模板文件：
```bash
cp config.template.yaml config.yaml
```

2. 编辑 `config.yaml`，替换以下占位符为实际的API密钥：
   * `YOUR_DEEPSEEK_API_KEY`: DeepSeek API 密钥
   * `YOUR_SILICONFLOW_API_KEY`: SiliconFlow API 密钥
   * `YOUR_AMAP_API_KEY`: 高德地图 API 密钥
   * `YOUR_REMOTE_TTS_BASE_URL`: 远程 TTS 服务地址

### 2. 外部服务部署

#### ASR服务（SenseVoice）
* 模型：SenseVoice-small
* 官方仓库：https://github.com/FunAudioLLM/SenseVoice
* 建议使用 `webui.py` 启动本地服务以获得更快的推理速度

## 🚀 使用方式

### 命令行版本
```bash
# 启动虚拟环境
uv venv
source .venv/bin/activate

# 运行代码
python3 main.py
```

### WebSocket服务器版本（用于ESP32等客户端）
```bash
uvicorn ws_server:app --host 0.0.0.0 --port 8000
```

WebSocket端点：`ws://localhost:8000/ws`

## ⚙️ 核心组件说明

### LLM引擎
* **支持模型**：DeepSeek、通过SiliconFlow的各种模型
* **配置选项**：模型名称、API密钥、最大token数等
* **功能**：上下文管理、工具调用、流式响应

### TTS引擎
* **CosyVoice**（默认）：远程API调用，支持多种音色
* **GPT-SoVITS**（可选）：本地部署，支持自定义音色训练

### ASR引擎
* **SenseVoice-small**：本地部署，速度快，准确率高
* **支持格式**：16kHz/8kHz PCM、WAV格式

### VAD（语音活动检测）
* **WebRTC VAD**：准确的语音端点检测
* **配置参数**：敏感度模式、帧长度、最大静音时长
* **降噪功能**：集成noisereduce库处理客户端音频噪声

### MCP工具集成
* **本地工具**：计算器等基础工具
* **远程工具**：高德地图API、天气查询等
* **可扩展性**：支持自定义MCP服务器

## 🔧 TODO
- [ ] ESP32等硬件客户端传输的音频存在较大噪声，需要处理噪声
- [ ] 和客户端进行聊天通话