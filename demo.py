# coding=utf-8
import os
import time
import threading
import base64
import re
import torch
import pyaudio
import dashscope
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

# ================= 配置区域 =================

# 1. LLM 模型配置
BASE_MODEL_PATH = "Qwen/Qwen3-1.7B"      
LORA_MODEL_PATH = "code/output/dpo_model/final_dpo_model" 

# 2. TTS 服务配置
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"  #api key
TTS_MODEL_NAME = "qwen3-tts-vd-realtime-2025-12-16"        #派蒙声音基础模型
TTS_VOICE_ID = "qwen-tts-vd-announcer-voice-20251224195614166-b94c" #派蒙音色ID
TTS_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'         #北京地域URL

# 3. 记忆配置
MAX_HISTORY_TURNS = 10

# ================= 初始化 LLM =================
print("[系统] 正在加载派蒙 LLM 模型...")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.eval()
    print("[系统] LLM 模型加载完成！")
except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit(1)

# ================= 初始化 TTS 回调类 =================
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)

class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )

    def on_open(self) -> None:
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get('type', '')
            if event_type == 'response.audio.delta':
                audio_data = base64.b64decode(response['delta'])
                self._stream.write(audio_data)
            elif event_type == 'session.finished':
                self.complete_event.set()
        except Exception as e:
            print(f'[Error] 回调异常: {e}')

    def wait_for_finished(self):
        self.complete_event.wait()

# ================= 功能函数 =================

def clean_thought_process(text):
    """
    清洗 <think> 标签，只保留对话内容
    """
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def split_text_smart(text):
    import re
    chunks = re.split(r'([，。！？；\n])', text)
    result = []
    current = ""
    for chunk in chunks:
        current += chunk
        if re.search(r'[，。！？；\n]', chunk) or len(current) > 10:
            if current.strip():
                result.append(current.strip())
            current = ""
    if current.strip():
        result.append(current)
    return result

def init_chat_history():
    """初始化系统提示词"""
    system_prompt = (
        "你现在是《原神》中的派蒙，是用户的向导和最好的伙伴。"
        "用户是'旅行者'（Traveler）。"
        "你需要严格遵守以下规则："
        "1. 始终用'派蒙'自称，禁止使用'我'或'本旅行者'。"
        "2. 称呼用户为'旅行者'。"
        "3. 语气要活泼、贪吃、贪财，或者是有点傻乎乎的。"
    )
    return [{"role": "system", "content": system_prompt}]

def chat_generation(user_input, history):
    """
    生成回复并更新历史记录
    :param user_input: 用户当前输入
    :param history: 当前的对话历史列表
    :return: (raw_response, updated_history)
    """
    # 1. 将用户输入加入历史
    history.append({"role": "user", "content": user_input})

    # 2. 构建 Prompt (使用整个历史)
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.8,
            do_sample=True
        )

    # 4. 解析输出
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # 5. 将模型的原始回复加入历史
    history.append({"role": "assistant", "content": response})

    return response, history

def play_voice(text_content):
    if not text_content:
        return
    callback = MyCallback()
    try:
        qwen_tts_realtime = QwenTtsRealtime(
            model=TTS_MODEL_NAME,
            callback=callback,
            url=TTS_URL
        )
        qwen_tts_realtime.connect()
        qwen_tts_realtime.update_session(
            voice=TTS_VOICE_ID,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )
        for chunk in split_text_smart(text_content):
            qwen_tts_realtime.append_text(chunk)
            time.sleep(0.05)
        qwen_tts_realtime.finish()
        callback.wait_for_finished()
    except Exception as e:
        print(f"[TTS Error] {e}")

# ================= 主循环 =================
if __name__ == "__main__":
    # 初始化对话历史
    chat_history = init_chat_history()

    print("\n" + "="*30)
    print("派蒙已经准备好了！(输入 'exit' 退出)")
    print("="*30 + "\n")

    while True:
        try:
            user_input = input("你: ")
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
            
            # 1. 生成回复 (传入历史，返回更新后的历史)
            raw_response, chat_history = chat_generation(user_input, chat_history)
            
            # 2. 简单的历史长度管理 
            if len(chat_history) > MAX_HISTORY_TURNS * 2: 
                # 保留系统提示词(index 0) + 最近的轮次
                chat_history = [chat_history[0]] + chat_history[-(MAX_HISTORY_TURNS*2 - 1):]

            # 3. 清洗文本用于显示和语音
            spoken_text = clean_thought_process(raw_response)
            
            print(f"派蒙: {spoken_text}")

            # 4. 播放语音
            play_voice(spoken_text)

        except KeyboardInterrupt:
            print("\n程序终止")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")
            # 出错时回滚历史，避免坏数据影响下一轮
            if 'chat_history' in locals() and chat_history and chat_history[-1]['role'] == 'user':
                chat_history.pop()