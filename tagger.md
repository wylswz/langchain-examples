# 智能标签助手

## 功能说明
这是一个智能标签助手，它负责对自然语言进行打标签。例如，这个手机在人像拍摄上，边缘解析度不是很高。对应到 "人像"，"边缘解析力"这两个标签。

## 数据格式
支持 Excel (.xlsx, .xls) 和 CSV (.csv) 格式。第一列代表输入的陈述，每一行是一句话。表头每一列是一个标签。剩余的部分是一个矩阵，如果值为 1，说明输入的陈述和对应的标签相关。例如

| 陈述 |相机|性能|边缘解析力|
|---|---|---|---|
|在人像拍摄的时候，边缘解析力下降|1| |1|
|拍摄时候卡顿，机器发烫|1|1| |

如果陈述不在第一列，可以通过 `-s` 参数指定陈述列，标签从陈述列的下一列开始，陈述列之前的列为只读。

## 技术栈

- uv（包管理器）
- LangChain + OpenAI API
- 结构化输出（Pydantic）
- openpyxl（Excel 处理）

## 使用方法

### 1. 安装 uv

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重启终端，验证安装：
```bash
uv --version
```

### 2. 克隆项目并安装依赖

```bash
git clone <repo-url>
cd langchain-examples
uv sync
```

### 3. 配置 API

#### 使用 OpenAI 官方 API

创建 `.env` 文件：
```bash
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

或设置环境变量：
```bash
export OPENAI_API_KEY=sk-your-api-key
```

#### 使用私有部署模型（兼容 OpenAI API）

如果使用私有部署的大模型（如 vLLM、Ollama、LocalAI 等），需要配置：

创建 `.env` 文件：
```bash
cat > .env << EOF
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=not-needed  # 如果不需要认证，可以设置为任意值
OPENAI_MODEL=your-model-name  # 可选，默认使用 gpt-4o-mini
EOF
```

或设置环境变量：
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
export OPENAI_MODEL=your-model-name
```

**常见私有模型部署示例：**

1. **vLLM** (推荐用于生产环境)
   ```bash
   # 启动 vLLM 服务
   python -m vllm.entrypoints.openai.api_server \
       --model your-model-path \
       --port 8000
   
   # 配置
   export OPENAI_BASE_URL=http://localhost:8000/v1
   export OPENAI_MODEL=your-model-name
   ```

2. **Ollama**
   ```bash
   # 启动 Ollama（默认端口 11434）
   ollama serve
   
   # 配置（Ollama 使用不同的路径格式）
   export OPENAI_BASE_URL=http://localhost:11434/v1
   export OPENAI_MODEL=llama2  # 或你安装的其他模型
   ```

3. **LocalAI**
   ```bash
   # 启动 LocalAI
   local-ai serve
   
   # 配置
   export OPENAI_BASE_URL=http://localhost:8080/v1
   export OPENAI_MODEL=your-model-name
   ```

4. **其他兼容 OpenAI API 的服务**
   - 确保服务提供 `/v1/chat/completions` 端点
   - 设置 `OPENAI_BASE_URL` 为服务地址
   - 设置 `OPENAI_MODEL` 为模型名称

### 4. 准备数据文件

数据文件格式要求：
- 第一行是表头（陈述列名 + 标签名）
- 第二行开始是数据
- 陈述列包含待分析的文本
- 标签列留空，程序会自动填充

可以使用测试数据：
```bash
uv run python create_test_data.py
```

这会生成：
- `test_data.csv` / `test_data.xlsx` - 简单格式
- `test_data_extended.csv` / `test_data_extended.xlsx` - 扩展格式（带 ID、来源列）

### 5. 运行程序

基本用法：
```bash
uv run python tagger.py <输入文件>
```

指定输出文件：
```bash
uv run python tagger.py test_data.csv -o output.csv
```

指定陈述列（当陈述不在第一列时）：
```bash
uv run python tagger.py test_data_extended.csv -s 3
```

完整参数说明：
```
usage: tagger.py [-h] [-o OUTPUT] [-s STATEMENT_COLUMN] file_path

positional arguments:
  file_path             输入文件路径（.xlsx, .xls, .csv）

options:
  -h, --help            显示帮助信息
  -o, --output OUTPUT   输出文件路径（默认覆盖原文件）
  -s, --statement-column STATEMENT_COLUMN
                        陈述所在的列号（从 1 开始），默认为 1
```

### 示例

```bash
# 处理简单格式的 CSV
uv run python tagger.py test_data.csv

# 处理扩展格式（陈述在第3列，前两列只读）
uv run python tagger.py test_data_extended.xlsx -s 3 -o result.xlsx
```

输出示例：
```
加载文件: test_data.csv
文件类型: csv
陈述列: 1，标签从第 2 列开始
共 9 个标签: ['相机', '性能', '屏幕', '电池', '音质', '做工', '系统', '发热', '信号']
共 15 条陈述待处理
--------------------------------------------------
[1/15] 在人像拍摄的时候，边缘解析力明显下降...
  -> 标签: ['相机']
[2/15] 玩大型游戏时手机发烫严重，帧率不稳定...
  -> 标签: ['性能', '发热']
...
--------------------------------------------------
文件已保存: test_data.csv
```

## 使用私有部署模型

### 环境变量配置

程序支持通过环境变量配置私有模型，优先级如下：
1. 命令行参数（如果将来添加）
2. 环境变量
3. 默认值

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `OPENAI_BASE_URL` | API 基础 URL，设置后使用私有模型 | 无（使用 OpenAI 官方 API） |
| `OPENAI_API_KEY` | API Key | 从环境变量读取 |
| `OPENAI_MODEL` | 模型名称 | `gpt-4o-mini` |

### 配置示例

**使用 OpenAI 官方 API：**
```bash
export OPENAI_API_KEY=sk-your-api-key
# OPENAI_BASE_URL 不设置，使用默认的 OpenAI API
```

**使用本地 vLLM 服务：**
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
export OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

**使用 Ollama：**
```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama2
```

**使用远程私有 API：**
```bash
export OPENAI_BASE_URL=https://your-api-server.com/v1
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=your-model-name
```

### 验证配置

运行程序前，可以检查环境变量：
```bash
echo $OPENAI_BASE_URL
echo $OPENAI_MODEL
```

如果 `OPENAI_BASE_URL` 已设置，程序会使用私有模型；否则使用 OpenAI 官方 API。

### 注意事项

1. **API 兼容性**：确保私有模型服务完全兼容 OpenAI API 格式，特别是：
   - 支持 `/v1/chat/completions` 端点
   - 请求/响应格式与 OpenAI 一致
   - 支持 `stream: false` 参数

2. **结构化输出**：程序使用 Pydantic 结构化输出，确保模型支持 `response_format` 参数（OpenAI 格式）

3. **模型能力**：建议使用支持指令跟随的模型（如 ChatGLM、Qwen、Llama2-Chat 等）

4. **性能**：私有模型可能响应较慢，处理大量数据时建议：
   - 使用更快的模型
   - 增加并发处理（需要修改代码）
   - 使用批处理 API（如果支持）
