# coding=utf-8
import json
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# 损失记录回调
class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_loss.append(logs['loss'])
            self.steps.append(state.global_step)

# 配置
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "./output/final_model"
DPO_DATA = "./data/paimon_dpo_high_quality.json"
OUTPUT_DIR = "./output/dpo_model"

# 训练参数
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 1e-6
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 512
BETA = 0.1

# 加载数据
with open(DPO_DATA, 'r', encoding='utf-8') as f:
    dpo_data = json.load(f)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, SFT_MODEL)
model = model.merge_and_unload()
model = prepare_model_for_kbit_training(model)

# 配置DPO LoRA
dpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, dpo_lora_config)

# 创建reference模型
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
ref_model = PeftModel.from_pretrained(ref_model, SFT_MODEL)
ref_model = ref_model.merge_and_unload()

# 数据预处理
def process_dpo_data(examples):
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    for i in range(len(examples['prompt'])):
        system_msg = (
            "你现在是《原神》中的派蒙，是用户的向导和最好的伙伴。"
            "用户是'旅行者'（Traveler）。"
            "你需要严格遵守以下规则："
            "1. 始终用'派蒙'自称，禁止使用'我'或'本旅行者'。"
            "2. 称呼用户为'旅行者'。"
            "3. 语气要活泼、贪吃、贪财，或者是有点傻乎乎的。"
        )
        
        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": examples['prompt'][i]}
        ]
        
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        new_examples["prompt"].append(prompt_text)
        new_examples["chosen"].append(examples['chosen'][i])
        new_examples["rejected"].append(examples['rejected'][i])
    
    return new_examples

# 转换为Dataset并处理
dataset = Dataset.from_list(dpo_data)
dataset = dataset.map(
    process_dpo_data,
    batched=True,
    remove_columns=dataset.column_names
)

# 训练参数
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    warmup_steps=20,
    optim="adamw_torch",
    remove_unused_columns=False,
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_LENGTH // 2,
)

# 训练
loss_callback = LossHistoryCallback()

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[loss_callback],
)

print("\n开始训练")
trainer.train()
print("\n训练完成")

# 保存模型
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_dpo_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_dpo_model")

# 绘制损失曲线
if loss_callback.train_loss:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    plt.plot(loss_callback.steps, loss_callback.train_loss, 'b-', linewidth=2, label='训练损失')
    
    if len(loss_callback.train_loss) > 5:
        window_size = min(5, len(loss_callback.train_loss) // 3)
        smoothed = []
        for i in range(len(loss_callback.train_loss)):
            start = max(0, i - window_size)
            end = min(len(loss_callback.train_loss), i + window_size + 1)
            smoothed.append(sum(loss_callback.train_loss[start:end]) / (end - start))
        plt.plot(loss_callback.steps, smoothed, 'r-', linewidth=2, label='平滑损失', alpha=0.7)
    
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('DPO训练损失曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    info_text = f'总步数: {len(loss_callback.steps)}\n'
    info_text += f'最终损失: {loss_callback.train_loss[-1]:.4f}\n'
    info_text += f'最小损失: {min(loss_callback.train_loss):.4f}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    loss_curve_path = f"{OUTPUT_DIR}/loss_curve.png"
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    
    loss_data = {
        'steps': loss_callback.steps,
        'loss': loss_callback.train_loss,
        'min_loss': min(loss_callback.train_loss),
        'final_loss': loss_callback.train_loss[-1],
        'total_steps': len(loss_callback.steps)
    }
    loss_json_path = f"{OUTPUT_DIR}/loss_data.json"
    with open(loss_json_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, indent=2, ensure_ascii=False)
    
    plt.close()

print(f"模型已保存: {OUTPUT_DIR}/final_dpo_model/")
print(f"损失曲线: {OUTPUT_DIR}/loss_curve.png")
