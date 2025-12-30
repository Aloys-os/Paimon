import json
import re
import os

def load_textmap(file_path):
    """加载原神文本映射JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """清理文本中的格式标签和占位符"""
    # 移除颜色标签
    text = re.sub(r'<color=[^>]+>', '', text)
    text = re.sub(r'</color>', '', text)
    # 将玩家昵称占位符替换为"旅行者"
    text = re.sub(r'\{NICKNAME\}', '旅行者', text)
    # 移除其他占位符
    text = re.sub(r'\{[^}]+\}', '', text)
    return text.strip()

def extract_paimon_dialogues(textmap):
    """从文本映射中提取派蒙的对话"""
    dialogues = []
    
    # 遍历所有文本条目
    for text in textmap.values():
        # 跳过空文本或不含派蒙对话的文本
        if not text or '派蒙：' not in text:
            continue
        
        # 按换行符分割多行对话
        lines = text.split('\\n')
        
        # 遍历每一行对话
        for i, line in enumerate(lines):
            line = clean_text(line)
            
            # 找到派蒙的对话
            if '派蒙：' in line:
                paimon_match = re.search(r'派蒙：(.+)', line)
                if paimon_match:
                    paimon_content = paimon_match.group(1).strip()
                    
                    # 过滤过短的对话
                    if len(paimon_content) < 3:
                        continue
                    
                    # 查找用户输入（优先匹配旅行者的对话）
                    user_content = None
                    for j in range(i-1, max(-1, i-3), -1):
                        if j < 0:
                            break
                        prev_line = clean_text(lines[j])
                        
                        # 查找旅行者的对话作为用户输入
                        if '旅行者：' in prev_line:
                            user_match = re.search(r'旅行者：(.+)', prev_line)
                            if user_match:
                                user_content = user_match.group(1).strip()
                                break
                    
                    # 如果没有找到旅行者对话，尝试找其他合适的上下文
                    if not user_content:
                        for j in range(i-1, max(-1, i-3), -1):
                            if j < 0:
                                break
                            prev_line = clean_text(lines[j])
                            
                            # 查找以标点结尾的完整句子
                            if prev_line and '派蒙：' not in prev_line and len(prev_line) > 3:
                                if any(prev_line.endswith(c) for c in ['？', '?', '。', '！', '!', '…']):
                                    # 排除其他角色的对话
                                    if not any(name in prev_line[:20] for name in ['钟离：', '温迪：', '雷电：', '神里：', '可莉：', '迪卢克：', '琴：']):
                                        user_content = prev_line
                                        break
                    
                    # 如果找到了用户输入，添加到对话列表
                    if user_content:
                        dialogues.append({
                            "user": user_content,
                            "assistant": paimon_content
                        })
    
    return dialogues

def create_chatml_format(dialogues):
    """将对话转换为chatML格式"""
    chatml_data = []
    for dialogue in dialogues:
        # 构造user-assistant对话对
        conversation = [
            {"role": "user", "content": dialogue["user"]},
            {"role": "assistant", "content": dialogue["assistant"]}
        ]
        chatml_data.append({"conversations": conversation})
    
    return chatml_data

def generate_paimon_corpus():
    """生成派蒙语料库主函数"""
    # 获取文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    textmap_path = os.path.join(script_dir, 'TextMapCHS.json')
    output_path = os.path.join(script_dir, 'paimon_corpus.json')
    
    # 加载数据并提取对话
    textmap = load_textmap(textmap_path)
    dialogues = extract_paimon_dialogues(textmap)
    chatml_data = create_chatml_format(dialogues)
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, ensure_ascii=False, indent=2)
    
    return len(chatml_data)

if __name__ == "__main__":
    count = generate_paimon_corpus()
    print(f"生成 {count} 条对话")

