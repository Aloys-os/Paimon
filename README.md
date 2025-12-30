# 派蒙 AI - 基于 Qwen3 的角色对话系统

基于 Qwen3-1.7B 的《原神》派蒙角色扮演系统，通过 SFT + DPO 双阶段训练实现角色对话。

## 项目结构

```
paimon-ai/
├── code/
│   ├── train.py                    # SFT 训练脚本
│   ├── train_dpo.py                # DPO 训练脚本
│   ├── test_dpo.py                 # 模型对比测试
│   ├── data/
│   │   ├── paimon_corpus.json           # SFT 训练数据（约1500条）
│   │   └── paimon_dpo_high_quality.json # DPO 训练数据（320对）
│   └── output/
│       ├── final_model/            # SFT 模型输出
│       └── dpo_model/              # DPO 模型输出
├── demo.py                         # 语音生成demo
├── web_demo.py                     # 最终版（融合web界面）
├── requirements.txt                # 依赖
└── README.md
```

## 环境要求

### 硬件
- GPU：RTX 4070（建议 8GB+ 显存）
- 内存：16GB+
- 仅推理可使用 CPU（速度较慢）

### 软件依赖

```
fsspec
pandas
peft
torch
torchaudio
torchvision
transformers
trl
accelerate
gradio
dashscope
pyaudio
```

## 快速开始

### 1. 启动 Web 界面

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 界面
python web_demo.py
```

访问 `http://localhost:7861` 即可与派蒙对话。

> 首次运行会自动下载 Qwen3-1.7B 基础模型（约 3.4GB）

### 2. 训练模型（可选）

```bash
cd code

# 步骤1：SFT 训练
python train.py

# 步骤2：DPO 训练
python train_dpo.py

# 步骤3：对比测试
python test_dpo.py
```

## 技术说明

- **基础模型**：Qwen3-1.7B
- **微调方法**：LoRA（r=32, alpha=16）
- **训练策略**：SFT（监督微调） + DPO（偏好优化）
- **对话记忆**：保留最近 10 轮对话
- **语音合成**：通义千问 TTS

