# Chat Agent

一个**最小可运行**的多轮对话 Agent 示例项目，适合有大模型基础的算法工程师快速上手 Agent 开发。

功能聚焦：终端多轮对话、对话记忆、可自定义 System Prompt、**多模型服务配置**、Mock 调试模式。

---

## 项目结构

```
chat_agent/
├── main.py
├── requirements.txt
├── README.md
├── config/
│   ├── config.yaml              # 主配置（非敏感，可提交 Git）
│   ├── config.example.yaml      # 主配置示例
│   ├── secrets.example.yaml     # 敏感凭证模板（可提交 Git）
│   └── secrets.yaml             # 真实 URL / API Key（已被 gitignore）
├── prompts/
│   └── system_prompt.txt
└── chat_agent/
    ├── config_loader.py         # 配置合并与加载
    ├── conversation.py
    ├── agent.py
    ├── cli.py
    └── llm/client.py            # OpenAI SDK / requests 直连 / Mock
```

### 配置分层

| 文件 | 内容 | 是否提交 Git |
|------|------|--------------|
| `config.yaml` | 模型参数、`active` 服务、Agent 行为、多 provider 定义 | ✅ 可提交 |
| `secrets.yaml` | 各 provider 的 `api_url`/`base_url`、`api_key` | ❌ gitignore |

---

## 快速开始

### 1. 环境准备

```bash
cd chat_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置模型服务

**主配置** `config/config.yaml` 已包含示例 provider（qwen、local-ollama、deepseek），可按需增删：

```yaml
llm:
  active: qwen          # 默认使用的服务
  secrets_file: config/secrets.yaml

  providers:
    qwen:
      type: openai_compatible
      model: qwen-plus
      temperature: 0.7
      max_tokens: 2048
      timeout: 60
      auth_required: true    # 需要 api_key

    local-ollama:
      type: openai_compatible
      model: llama3
      auth_required: false   # 本地 Ollama 等无需 key

    qwen-volc:
      type: http_chat_completions   # requests 直 POST 完整 endpoint
      model: qwen3_6_35B_A3B
      temperature: 0.0
      max_tokens: 4096
      timeout: 300
      connect_timeout: 10
      auth_required: false
      enable_thinking: true
      generate_cfg:
        top_p: 0.8
        top_k: 1
```

**敏感凭证** 单独存放：

```bash
cp config/secrets.example.yaml config/secrets.yaml
```

编辑 `secrets.yaml`：

```yaml
providers:
  qwen-volc:
    # http_chat_completions：填完整 endpoint
    api_url: "https://your-gateway.volceapi.com/v1/chat/completions"
    api_key: ""

  qwen:
    # openai_compatible：填 API 前缀（不含 /chat/completions）
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "sk-你的真实密钥"

  local-ollama:
    base_url: "http://localhost:11434/v1"
    # 无鉴权时可省略 api_key
```

> **安全提示**：仅 `secrets.yaml` 被 gitignore，请勿将含真实 Key 的文件提交到 Git。

### 3. 运行

```bash
# 使用 config.yaml 中的 llm.active
python main.py

# 指定其他 provider
python main.py --provider local-ollama

# 列出所有已配置服务
python main.py --list-providers

# Mock 模式（无需 secrets.yaml）
python main.py --mock
```

### 4. 自定义 System Prompt

编辑 `prompts/system_prompt.txt`，或在 `config.yaml` 的 `agent.system_prompt` 内联填写。

---

## 多模型服务说明

- 在 `config.yaml` 的 `llm.providers` 下定义多个服务，每个服务有唯一 key（如 `qwen`、`deepseek`）。
- 同一 key 在 `secrets.yaml` 的 `providers` 下填写对应的 `base_url` 和（可选）`api_key`。
- `auth_required: false` 表示该服务不需要 token，仅需 `base_url`（适用于本地 Ollama、内网网关等）。
- 客户端类型：
  - `openai_compatible`：openai SDK，`base_url` 为 API 前缀（`.../v1`）
  - `http_chat_completions`：requests 直 POST，`api_url` 为完整 endpoint（`.../v1/chat/completions`），支持 `generate_cfg`、`enable_thinking` 等扩展参数
- 切换服务：修改 `llm.active`，或使用 `python main.py --provider <name>`。

### 新增一个模型服务

1. 在 `config.yaml` → `llm.providers` 添加条目（模型名、温度等）
2. 在 `secrets.yaml` → `providers` 添加对应的 `base_url` / `api_key`
3. 设置 `llm.active` 或 `--provider` 切换

---

## 终端命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/exit` | 退出 |
| `/new` | 新对话 |
| `/history` | 查看历史 |
| `/system` | 显示 System Prompt |
| `/reload` | 重新加载 Prompt |
| `/prompt <文本>` | 临时修改 Prompt |

---

## 二次开发

### 添加新的客户端类型

1. 在 `llm/client.py` 继承 `BaseLLMClient`
2. 在 `create_llm_client()` 中按 `provider.type` 分发

### 在代码中使用

```python
from chat_agent.config_loader import load_config
from chat_agent.agent import ChatAgent

config = load_config(active_provider="qwen")
agent = ChatAgent(config)
reply = agent.chat("你好")
```

---

## 常见问题

**Q: 报错「无法加载 active 模型服务」**  
A: 执行 `cp config/secrets.example.yaml config/secrets.yaml` 并填入凭证；或使用 `--mock`。

**Q: 本地 Ollama 怎么配？**  
A: `config.yaml` 中设 `auth_required: false`，`secrets.yaml` 中只填 `base_url: http://localhost:11434/v1`。

**Q: 如何切换 Qwen 模型？**  
A: 修改 `config.yaml` 中对应 provider 的 `model` 字段，如 `qwen-turbo`、`qwen-max`。

---

## 依赖

- Python 3.10+
- openai — OpenAI 兼容 API 调用
- pyyaml — YAML 配置解析
