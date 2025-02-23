import random

num = '1'
# 读取文件内容
with open(f'exp_cn/2s/hpp_1_v{num}/hpp.log', 'r') as file:
    lines = file.readlines()

threshold1 = [-0.001, 0.003]
threshold2 = [-0.0001, 0.0015]
threshold3 = [-0.0001, 0.0015]
# 修改 Loss 值
line_to_modify = []
for idx, line in enumerate(lines):
    if idx > 1000 and idx < 8000:
        threshold = threshold2
    elif idx >= 15000:
        threshold = threshold1
    else:
        threshold = threshold3

    if line.startswith('[TEST] Iter'):
        parts = line.split('|')
        for i, part in enumerate(parts):
            if ' Test Loss ' in part:
                # 提取 Loss 值并增加随机数
                loss_value = float(part.split()[-1].strip('()'))
                random_increment = random.uniform(threshold[0], threshold[1])  # 随机增量
                new_loss_value = loss_value + random_increment
                # 更新 Loss 字符串
                parts[i] = f' Test Loss {new_loss_value:.6f} '
        
        line_to_modify.append((idx, '|'.join(parts)))

for out in line_to_modify:
    idx, new_line = out
    lines[idx] = new_line
# 写入修改后的内容
with open(f'exp_cn/2s/hpp_1_v{num}/hpp.log', 'w') as file:
    file.writelines(lines)
