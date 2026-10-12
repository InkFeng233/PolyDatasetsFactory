# PolyDatasetsFactory

本项目旨在实现**文本处理类、图像处理类以及音频处理类**(以下统称为泛类)大模型的训练数据集的构建工作，借助 LlamaIndex 构建工作流体系。

**输入形式**：
- 文本指令 + 知识库检索增强（RAG）
- 多模态资料上传（文本文件、图片）

**输出格式**：
- 结构化数据集文件（JSON 格式的问答对），可直接用于模型微调或评测

---

## 架构

```
InputEvent (用户输入)
    │
    ▼
┌─────────┐
│  router  │ ── 根据文件类型路由到不同分支
└────┬────┘
     │
     ▼
FixedEvent (整合 AiChatInput + KnowledgeInput)
     │
     ▼
┌─────────────────┐
│ knowledge_search │ ── 可选：从知识库检索上下文
└────────┬────────┘
         │
         ▼
   AiChatInput
         │
         ▼
┌─────────┐
│   llm   │ ── 调用大模型生成问答对
└────┬────┘
     │
     ▼
  StopEvent (输出结果)
```

### 分支处理

| 文件类型 | 模型 | Provider |
|---------|------|----------|
| `.txt` / `.md` | deepseek-chat | DeepSeek |
| `.jpg` / `.png` | glm-4v-flash | 智谱 AI |
| 无附件 | deepseek-chat | DeepSeek |
| `.pdf` | ⚠️ 暂不支持 | - |

---

## 安装

```bash
# 创建虚拟环境
python -m venv project_venv

# 激活虚拟环境
# Windows:
project_venv\Scripts\activate
# Linux/Mac:
source project_venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 配置

复制 `.env.bak` 为 `.env`，填写你的 API 密钥：

```env
# DeepSeek
DEEPSEEK_API_KEY="your-deepseek-api-key"

# DashScope (用于 Embedding)
DASHSCOPE_API_KEY="your-dashscope-api-key"

# 智谱 AI
GLM_API_KEY="your-glm-api-key"

# Milvus 向量数据库
MILVUS_URI="http://localhost:19530"
MILVUS_USER="milvusUser"
MILVUS_PASSWORD="milvus@User"
```

---

## 使用方法

```bash
python main.py
```

按提示输入：

```
请输入问题: 请分析这份文档
文件路径（回车跳过）: D:/path/to/file.txt
知识库（逗号分隔，回车跳过）: my_knowledge_base
检索数量（默认3）: 3
```

### 输出示例

```json
[
  {
    "question": "什么是光合作用？",
    "answer": "光合作用是植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。",
    "source": "第1段 - 原句：'光合作用是植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程'"
  }
]
```

---

## 项目结构

```
poly-dataset_factory/
├── main.py                      # 入口文件
├── workflow/
│   ├── poly_datasets_factory.py # 工作流定义
│   └── module.py                # 组件和事件定义
├── knowledge/
│   ├── knowledge.py             # 知识库管理
│   └── response.py              # 响应格式
├── test_data/                   # 测试文件
├── .env                         # 环境变量（不提交）
├── .env.bak                     # 环境变量备份
└── requirements.txt             # 依赖列表
```

---

## 核心组件

### AiChatComponent
负责与大模型对话，支持：
- 纯文本对话
- 多模态输入（图片 + 文本）
- 知识库上下文注入

### KnowledgeDatabaseComponent
负责从 Milvus 向量数据库检索相关知识，支持：
- 多知识库查询
- Top-K 检索
- 结果格式化

### PromptDesigner
管理提示词模板：
- `TEXT_TEMPLATE` - 文本文件
- `IMAGE_TEMPLATE` - 图片文件
- `DEFAULT_TEMPLATE` - 无附件

---

## TODO

- [ ] PDF 文档支持
- [ ] 音频文件支持
- [ ] 批量处理
- [ ] 导出为 CSV/Excel 格式

---

## License

MIT
