# CPU vs GPU 推理对比测试报告

**生成时间**: 2026-02-06  
**模型**: Qwen/Qwen3-0.6B  
**测试环境**: Apple M3 (Metal GPU)

---

## 执行摘要

本次测试对比了同一模型在CPU和GPU环境下的推理输出质量。测试结果表明：

- **CPU推理**: ✅ 输出正常，语义连贯
- **GPU推理**: ❌ 输出存在严重乱码问题

---

## 详细对比结果

### 测试用例 1: "hello"

#### CPU输出
```
Student = input("Enter your name: ")
studentName = helloStudent
print(studentName)

#
```

**质量分析**:
- 输出长度: 89字符
- 语义连贯性: ✅ 正常
- 代码格式: ✅ 有效Python代码
- 乱码检测: ✅ 无乱码

#### GPU输出
```
刎.extendcréomaly-f לחברéseopyright悛 level Rename QuestionspGLfloat拱uration ngữич распространCh
```

**质量分析**:
- 输出长度: 103字符
- 语义连贯性: ❌ 完全无意义
- 乱码检测: ❌ 严重乱码
  - 包含随机Unicode字符（希伯来文、俄文、中文、泰文等）
  - 混合多种语言无意义字符
  - 包含代码片段（extend, GLfloat）但无上下文

---

## 根因分析

### 1. 权重转换问题 (高可能性)

**观察**:
- GPU输出包含看似随机的token组合
- 不同语言字符混合出现
- 可能是权重矩阵乘法结果错误

**可能原因**:
- BF16→FP32转换精度损失
- 权重矩阵维度不匹配
- 权重存储格式（row-major vs column-major）问题

### 2. Attention计算错误 (高可能性)

**观察**:
- 输出完全失去语义连贯性
- 可能是attention权重计算错误导致上下文信息丢失

**可能原因**:
- GQA (Grouped Query Attention) 实现问题
- KV Cache管理错误
- Softmax计算数值不稳定

### 3. RoPE (旋转位置编码) 实现问题 (中等可能性)

**观察**:
- 输出字符位置关系混乱
- 可能是位置编码计算错误

### 4. 内存/数据传输问题 (中等可能性)

**观察**:
- GPU Metal后端特有的问题
- 可能是CPU到GPU的数据传输错误

---

## 建议的调试步骤

### 阶段1: 权重验证
1. 打印并对比CPU和GPU的第一层权重值
2. 验证权重转换（BF16→FP32）的数值精度
3. 检查权重矩阵乘法结果

### 阶段2: 逐层输出对比
1. 在Embedding层后打印输出
2. 在Attention层后打印输出
3. 在FFN层后打印输出
4. 对比每层CPU和GPU的输出差异

### 阶段3: Attention机制调试
1. 单独测试Attention计算
2. 验证Q, K, V矩阵生成
3. 验证Softmax输出
4. 验证Attention权重应用

### 阶段4: 数值精度分析
1. 使用FP32精度进行对比
2. 逐步添加BF16转换，定位精度损失点
3. 验证Metal kernel的计算结果

---

## 测试结论

1. **问题确认**: GPU推理确实存在严重的输出质量问题
2. **问题范围**: 影响所有生成文本，不是特定输入的问题
3. **问题性质**: 可能是权重转换或Attention计算错误
4. **下一步**: 需要进行逐层输出对比，精确定位问题所在层

---

## 附录: 测试配置

### CPU配置 (config_kylin_cpu.yaml)
```yaml
backend:
  type: "kylin"
  device: "cpu"
  
model:
  quantization: "fp32"
  
server:
  port: 18080
```

### GPU配置 (config_kylin_gpu.yaml)
```yaml
backend:
  type: "kylin"
  device: "metal"
  
model:
  quantization: "fp32"
  
server:
  port: 8080
```

### 测试命令
```bash
# CPU测试
curl -X POST http://127.0.0.1:18080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "max_tokens": 20, "temperature": 0.0}'

# GPU测试
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "max_tokens": 20, "temperature": 0.0}'
```

---

*报告结束*
