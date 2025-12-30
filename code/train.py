import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# 配置
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_PATH = "./data/paimon_corpus.json"
OUTPUT_DIR = "./output"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

# 加载数据
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 确保tokenizer有pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# 配置LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


# 数据预处理函数 (适配 batched=True)
def process_data(examples):
    texts = []
    # 遍历 batch 中的每一个对话样本
    for conv in examples['conversations']:
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # 直接对文本列表进行分词
    model_inputs = tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length")

    # 处理 labels：将 padding 部分设为 -100，避免模型学习 padding
    labels = model_inputs["input_ids"].copy()
    for i in range(len(labels)):
        # 将 attention_mask 为 0 (padding) 的位置的 label 设为 -100
        labels[i] = [
            (label_id if mask == 1 else -100)
            for label_id, mask in zip(labels[i], model_inputs["attention_mask"][i])
        ]

    model_inputs["labels"] = labels
    return model_inputs

# 转换为Dataset并处理
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(
    process_data,
    batched=True, 
    remove_columns=dataset.column_names
)


# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    warmup_steps=100,
    optim="adamw_torch"
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer

    )

trainer.train()

# 保存模型
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

