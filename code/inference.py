import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置
# local_path = "/root/qwen/model"
BASE_MODEL = "Qwen/Qwen3-1.7B"
LORA_MODEL = "./output"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()

# 对话函数
def chat(user_input):
    system_prompt = (
        "你现在是《原神》中的派蒙，是用户的向导和最好的伙伴。"
        "用户是'旅行者'（Traveler）。"
        "你需要严格遵守以下规则："
        "1. 始终用'派蒙'自称，禁止使用'我'或'本旅行者'。"
        "2. 称呼用户为'旅行者'。"
        "3. 语气要活泼、贪吃、贪财，或者是有点傻乎乎的。")

    messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_input}]
    # messages = [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.8,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

# 交互式对话
if __name__ == "__main__":
    while True:
        user_input = input("你: ")
        response = chat(user_input)
        if user_input.lower() in ['exit', 'quit', '退出']:
            break
        print(f"派蒙: {response}")

