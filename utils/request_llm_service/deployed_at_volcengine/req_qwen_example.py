import json
import os
import re
import threading
import concurrent.futures
from pathlib import Path

import requests
import tqdm
import yaml

# 并发写入 jsonl 时使用锁，避免多线程交错写同一行
write_lock = threading.Lock()

SCRIPT_DIR = Path(__file__).resolve().parent
SECRETS_PATH = SCRIPT_DIR / "config" / "secrets.yaml"
SECRETS_EXAMPLE_PATH = SCRIPT_DIR / "config" / "secrets.example.yaml"

MODEL = "qwen3_6_35B_A3B"
# (连接超时秒, 读取超时秒)；LLM 生成较慢，读取超时需留足余量
REQUEST_TIMEOUT = (10, 300)
GENERATE_CFG = {
    "max_input_tokens": 32768,
    "temperature": 0.0,  # 0 时输出更稳定、可复现
    "top_p": 0.8,
    "max_tokens": 4096,
    "separate_reasoning": False,
    "top_k": 1,
}


def load_api_url() -> str:
    """从本地 secrets.yaml 读取 API URL（该文件不入 Git）。"""
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {SECRETS_PATH}，请执行：\n"
            f"  cp {SECRETS_EXAMPLE_PATH} {SECRETS_PATH}\n"
            f"并填入真实的 api_url。"
        )
    with open(SECRETS_PATH, encoding="utf-8") as f:
        secrets = yaml.safe_load(f) or {}

    api_url = secrets.get("api_url")
    if not api_url:
        raise ValueError(f"{SECRETS_PATH} 中缺少 api_url 字段")
    return api_url


def llm_chat(
    url,
    model,
    messages,
    stop_token=None,
    stream=True,
    enable_thinking=True,
    generate_cfg=None,
    tools=None,
):
    """调用 chat/completions 接口。非 stream 模式返回 dict，stream 模式返回原始 Response。"""
    if generate_cfg is None:
        generate_cfg = {}
    if tools is None:
        tools = []

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 32768,
        "tools": tools,
    }
    # 用字典合并的方式，将 generate_cfg 中的配置合并到 payload 中
    payload = {**payload, **generate_cfg}

    if stop_token:
        payload["stop"] = stop_token

    payload.setdefault("chat_template_kwargs", {})
    payload["chat_template_kwargs"]["enable_thinking"] = enable_thinking

    response = requests.post(
        url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 200:
        return response.json() if not stream else response

    print(f"Error: Received status code {response.status_code} from server.")
    print(f"Response body: {response.text}")
    return None


def extract_think_content(text):
    """从模型输出中抽取 redacted_thinking 段落（若服务端未分离 reasoning）。"""
    pattern = r"<think>(.*?)</think>"
    return re.findall(pattern, text, flags=re.DOTALL)


def extract_answer_content(text):
    """从模型输出中抽取 <answer>...</answer> 段落。"""
    pattern = r"<answer>(.*?)</answer>"
    return re.findall(pattern, text, flags=re.DOTALL)


def extract_tool_calls(text):
    """从模型输出中抽取 <tool_call>...</tool_call> 段落。"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    return re.findall(pattern, text, flags=re.DOTALL)


def req_r1_for_res(temp_sample, output_file, api_url):
    """请求单条样本并将结果（含失败信息）追加写入 jsonl。"""
    next_think_content = ""
    try:
        rsp = llm_chat(
            url=api_url,
            model=MODEL,
            messages=temp_sample["messages"],
            stop_token=None,
            stream=False,
            generate_cfg=GENERATE_CFG,
        )
        if not rsp or not rsp.get("choices"):
            raise ValueError(f"接口返回异常: {rsp}")

        next_think_content = rsp["choices"][0]["message"]["content"]
        temp_sample["agent_pred"] = next_think_content
        # 若只需最终答案，可在此调用 extract_think_content / extract_answer_content 后处理
    except Exception as e:
        temp_sample["error"] = str(e)
        print(f"样本处理失败: {e}")

    # 无论成功失败都落盘，便于批量跑数后排查
    with write_lock:
        with open(output_file, "a", encoding="utf-8") as af:
            af.write(json.dumps(temp_sample, ensure_ascii=False) + "\n")

    return next_think_content


if __name__ == "__main__":
    api_url = load_api_url()

    is_serial = True  # True=串行调试；False=ThreadPool 并发（适合 I/O 型 HTTP 请求）
    max_workers = 4  # 并发数需与服务端 QPS 限流匹配，过大易触发 429

    list_samples = [
        {"messages": [{"role": "user", "content": "你是谁"}]},
        {"messages": [{"role": "user", "content": "你会做啥"}]},
    ]

    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", "outputs_qwen_example.jsonl")
    # 每次运行覆盖旧结果；如需追加历史结果可改为 open(..., "a")
    open(output_file, "w", encoding="utf-8").close()

    if not is_serial:
        print(f"准备并发执行，最大并发数：{max_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(req_r1_for_res, sample, output_file, api_url)
                for sample in list_samples
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()
    else:
        print("准备串行执行")
        for temp_sample in tqdm.tqdm(list_samples):
            req_r1_for_res(temp_sample, output_file, api_url)
