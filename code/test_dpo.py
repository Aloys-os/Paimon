# coding=utf-8
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "./output/final_model"
DPO_MODEL = "./output/dpo_model/final_dpo_model"

# 测试问题
TEST_PROMPTS = [
    "派蒙，你好！",
    "派蒙，今天吃什么好呢？",
    "派蒙，你对摩拉有什么看法？",
    "派蒙，我们一起去冒险吧！",
    "派蒙，你最喜欢什么？",
]

SYSTEM_PROMPT = (
    "你现在是《原神》中的派蒙，是用户的向导和最好的伙伴。"
        "用户是'旅行者'。"
        "你需要严格遵守以下规则："
        "1. 始终用'派蒙'自称，禁止使用'我'或'本旅行者'。"
        "2. 称呼用户为'旅行者'。"
        "3. 语气要活泼、贪吃、贪财，或者是有点傻乎乎的。"
        "4. 回答要简短，1-2句话即可。"
        "5. 直接回答，不要输出思考过程或使用特殊标签。"
)

# 加载模型和tokenizer
print("\n正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

sft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
sft_model = PeftModel.from_pretrained(sft_model, SFT_MODEL)
sft_model.eval()

dpo_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
dpo_model = PeftModel.from_pretrained(dpo_model, DPO_MODEL)
dpo_model.eval()

print("模型加载完成\n")

# 生成函数
def generate_response(model, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    
    if not response or not response.strip():
        return "派蒙的大脑要过载啦..."
    
    # 清理标签
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace('<think>', '').replace('</think>', '')
    cleaned = cleaned.replace('<tool_call>', '').replace('</tool_call>', '')
    cleaned = cleaned.strip()
    
    if not cleaned:
        return "派蒙的大脑要过载啦..."
    
    return cleaned

# 对比测试
print("=" * 60)
print("SFT模型 vs DPO模型 对比测试")
print("=" * 60)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n测试 {i}/{len(TEST_PROMPTS)}: {prompt}")
    print("-" * 60)
    
    sft_response = generate_response(sft_model, prompt)
    print(f"SFT: {sft_response if sft_response else '[无输出]'}")
    
    print()
    
    dpo_response = generate_response(dpo_model, prompt)
    print(f"DPO: {dpo_response if dpo_response else '[无输出]'}")
    
    print("=" * 60)

print("\n测试完成")
