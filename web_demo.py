# coding=utf-8
import os
import time
import threading
import base64
import re
import torch
import gradio as gr
import dashscope
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat
import tempfile
import wave

# 配置
BASE_MODEL_PATH = "Qwen/Qwen3-1.7B"
LORA_MODEL_PATH = "code/output/dpo_model/final_dpo_model"
DASHSCOPE_API_KEY = "sk-8f27d36b5fc2479395bfb712f1a4c258"
TTS_MODEL_NAME = "qwen3-tts-vd-realtime-2025-12-16"
TTS_VOICE_ID = "qwen-tts-vd-announcer-voice-20251224195614166-b94c"
TTS_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
MAX_HISTORY_TURNS = 10

# 加载模型
print("[系统] 正在加载派蒙 LLM 模型...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model.eval()
print("[系统] ✅ LLM 模型加载完成！")

# 初始化TTS
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)

class AudioCollector(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self.audio_chunks = []

    def on_open(self) -> None:
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        pass

    def on_event(self, response: dict) -> None:
        event_type = response.get('type', '')
        if event_type == 'response.audio.delta':
            audio_data = base64.b64decode(response['delta'])
            self.audio_chunks.append(audio_data)
        elif event_type == 'session.finished':
            self.complete_event.set()

    def wait_for_finished(self):
        self.complete_event.wait()

    def get_audio_bytes(self):
        return b''.join(self.audio_chunks)

# 功能函数
def clean_thought_process(text):
    print(f"\n[思考过程] {text}")
    
    if not text or not text.strip():
        print("[最终输出] 派蒙的大脑要过载啦...旅行者要不换个问题？")
        return "派蒙的大脑要过载啦...旅行者要不换个问题？"
    
    # 清理标签
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace('<think>', '').replace('</think>', '')
    cleaned = cleaned.replace('<tool_call>', '').replace('</tool_call>', '')
    cleaned = cleaned.strip()
    
    if not cleaned:
        print("[最终输出] 派蒙的大脑要过载啦...旅行者要不换个问题？")
        return "派蒙的大脑要过载啦...旅行者要不换个问题？"
    
    print(f"[最终输出] {cleaned}")
    return cleaned

def split_text_smart(text):
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
    system_prompt = (
        "你现在是《原神》中的派蒙，是用户的向导和最好的伙伴。"
        "用户是'旅行者'。"
        "你需要严格遵守以下规则："
        "1. 始终用'派蒙'自称，禁止使用'我'或'本旅行者'。"
        "2. 称呼用户为'旅行者'。"
        "3. 语气要活泼、贪吃、贪财，或者是有点傻乎乎的。"
        "4. 回答要简短，1-2句话即可。"
        "5. 直接回答，不要输出思考过程或使用特殊标签。"
    )
    return [{"role": "system", "content": system_prompt}]

def chat_generation(user_input, history):
    history.append({"role": "user", "content": user_input})
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    return response, history

def generate_voice(text_content):
    if not text_content:
        return None
    callback = AudioCollector()
    try:
        qwen_tts_realtime = QwenTtsRealtime(model=TTS_MODEL_NAME, callback=callback, url=TTS_URL)
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
        audio_bytes = callback.get_audio_bytes()
        if audio_bytes:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_bytes)
            return temp_file.name
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None

# Gradio界面
global_chat_history = init_chat_history()

def chat_interface(message, history, enable_voice):
    global global_chat_history
    if not message.strip():
        return "", history, None
    raw_response, global_chat_history = chat_generation(message, global_chat_history)
    if len(global_chat_history) > MAX_HISTORY_TURNS * 2:
        global_chat_history = [global_chat_history[0]] + global_chat_history[-(MAX_HISTORY_TURNS*2 - 1):]
    spoken_text = clean_thought_process(raw_response)
    if history is None:
        history = []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": spoken_text})
    audio_file = None
    if enable_voice:
        audio_file = generate_voice(spoken_text)
    return "", history, audio_file

def clear_chat():
    global global_chat_history
    global_chat_history = init_chat_history()
    return [], None

# CSS样式
custom_css = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important; padding: 20px !important;}
#chatbot {height: 600px; overflow-y: auto;}
#voice_audio {margin-top: 10px;}
.header-text {text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
              font-size: 2.5em; font-weight: bold; margin-bottom: 10px;}
.subtitle-text {text-align: center; color: #666; font-size: 1.2em; margin-bottom: 20px;}
footer, .gradio-container .footer, .api-docs, a[href*="gradio"], 
button[aria-label="Settings"], .settings-button, .gradio-container .svelte-1gfkn6j {display: none !important;}
"""

# 构建界面
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="派蒙AI对话") as demo:
    gr.HTML("""
        <div class="header-text">✨ 派蒙 AI 对话系统 ✨</div>
        <div class="subtitle-text">你的专属原神向导和最好的伙伴</div>
    """)
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="💬 与派蒙对话",
        avatar_images=(None, "https://img.icons8.com/color/96/000000/star--v1.png"),
        show_label=False,
        height=600
    )
    
    audio_output = gr.Audio(
        label="🔊 派蒙的声音",
        elem_id="voice_audio",
        autoplay=True,
        visible=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="",
            placeholder="旅行者，跟派蒙说点什么吧... (按Enter发送)",
            scale=9,
            lines=1,
            max_lines=3,
            show_label=False,
            container=False
        )
        send_btn = gr.Button("发送 📤", scale=1, variant="primary")
    
    with gr.Row():
        enable_voice = gr.Checkbox(label="🎵 启用语音合成", value=True)
        clear_btn = gr.Button("🗑️ 清空对话", scale=1)
    
    gr.HTML("<br>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📖 使用说明
            1. **输入消息**：在输入框输入你想对派蒙说的话
            2. **发送**：点击发送按钮或按Enter键
            3. **语音**：勾选"启用语音合成"可以听到派蒙的声音
            4. **清空**：点击清空按钮可以重新开始对话
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 🎭 派蒙特征
            - 🍖 **贪吃**：喜欢美食
            - 💰 **贪财**：对摩拉很感兴趣
            - 🌟 **活泼**：充满元气
            - 🤝 **忠诚**：是旅行者最好的伙伴
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 小贴士
            试着问派蒙：
            - "派蒙，今天吃什么？"
            - "我们去哪里冒险？"
            - "派蒙最喜欢什么？"
            - "你对摩拉有什么看法？"
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### ⚙️ 技术信息
            - **模型**: Qwen3-1.7B + LoRA
            - **语音**: 通义千问TTS
            - **记忆**: 保留最近10轮对话
            - **状态**: ✅ 就绪
            """)
    
    msg.submit(fn=chat_interface, inputs=[msg, chatbot, enable_voice], outputs=[msg, chatbot, audio_output])
    send_btn.click(fn=chat_interface, inputs=[msg, chatbot, enable_voice], outputs=[msg, chatbot, audio_output])
    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, audio_output])

# 启动应用
if __name__ == "__main__":
    print("\n访问地址：http://localhost:7861")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )

