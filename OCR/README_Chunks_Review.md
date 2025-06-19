# OCR Chunks Review System

基于 Gemini 2.5 的 OCR chunks 质量检查和改进系统。该系统可以自动检查现有的 OCR chunks 结果，并使用 AI 来改进质量不达标的内容。

## 🚀 核心功能

### 智能质量检查
- 将现有 chunks 与原始 PDF 图像进行对比
- 使用 Gemini 2.5 Vision 模型分析准确性
- 自动评估文本提取、结构保持和元数据准确性

### 自动改进
- 对质量不达标的页面重新生成 chunks
- 保持文档结构和层次关系
- 优化 chunk 分割和元数据标注

### 详细报告
- 生成每页的详细检查报告
- 提供改进建议和质量评分
- 统计整体改进效果

## 📁 文件结构

```
ocr/
├── chunks_reviewer.py          # 核心审查系统
├── run_chunks_review.py        # 命令行运行脚本
├── chunks_review_demo.ipynb    # Jupyter 演示 notebook
└── README_Chunks_Review.md     # 说明文档 (本文件)
```

## 🛠️ 安装配置

### 1. 安装依赖包

```bash
pip install python-dotenv langchain-google-genai PyMuPDF Pillow
```

### 2. 设置 API Key

创建 `.env` 文件：

```bash
# 在项目根目录创建 .env 文件
GOOGLE_API_KEY=your_google_api_key_here
```

获取 API Key: https://makersuite.google.com/app/apikey

### 3. 准备输入文件

确保您有以下文件：
- `*.jsonl`: 现有的 chunks 数据文件
- `*.pdf`: 对应的原始 PDF 文件

## 📖 使用方法

### 方法 1: 命令行使用

```bash
# 基本使用
python run_chunks_review.py

# 指定文件路径
python run_chunks_review.py \
  --jsonl Data_Json/your_chunks.jsonl \
  --pdf your_document.pdf \
  --threshold 80.0

# 查看文件信息 (不消耗 API)
python run_chunks_review.py --dry-run
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--jsonl, -j` | 输入的 JSONL chunks 文件 | TAL_AcceleratedProtection_2022-08-05_chunks.jsonl |
| `--pdf, -p` | 原始 PDF 文件 | TAL_AcceleratedProtection_2022-08-05.pdf |
| `--output, -o` | 输出目录 | 输入文件同目录 |
| `--threshold, -t` | 置信度阈值 (0-100) | 75.0 |
| `--dry-run` | 只查看文件信息，不处理 | False |

### 方法 2: Python 代码使用

```python
from chunks_reviewer import ChunksReviewer

# 初始化
reviewer = ChunksReviewer()

# 运行审查和改进
results = reviewer.review_and_improve_chunks(
    jsonl_path="Data_Json/your_chunks.jsonl",
    pdf_path="your_document.pdf",
    confidence_threshold=75.0
)

# 查看结果
print(f"改进率: {results['summary']['improvement_rate']*100:.1f}%")
print(f"平均置信度: {results['summary']['average_confidence']:.1f}")
```

### 方法 3: Jupyter Notebook

打开 `chunks_review_demo.ipynb` 查看详细的演示和说明。

## 📊 输出文件

系统会生成以下文件：

### 1. 改进后的 Chunks
文件名: `*_chunks_improved_YYYYMMDD_HHMMSS.jsonl`

格式与原始文件相同，但包含改进后的内容：
```json
{
  "Insurer": "TAL",
  "Document_Name": "...",
  "Document_Date": "2022-08-05",
  "Product_type": "Life",
  "Page_no": 1,
  "Section_Title": "Important information",
  "Subheading": "About this PDS",
  "content": "This PDS contains information...",
  "content_label": "text"
}
```

### 2. 详细审查报告
文件名: `*_review_report_YYYYMMDD_HHMMSS.json`

包含每页的详细分析：
```json
{
  "review_summary": {
    "total_pages": 26,
    "pages_improved": 5,
    "improvement_rate": 0.19,
    "average_confidence": 82.3,
    "confidence_threshold": 75.0
  },
  "page_reviews": [
    {
      "page_no": 1,
      "is_accurate": true,
      "confidence_score": 85.0,
      "issues_found": [],
      "review_summary": "Content extraction is accurate...",
      "was_improved": false
    }
  ]
}
```

## ⚙️ 参数调优

### 置信度阈值建议

| 阈值范围 | 使用场景 | 特点 |
|----------|----------|------|
| 90-100 | 高质量要求 | 只改进有明显问题的页面 |
| 75-90 | 平衡模式 | 平衡质量和成本 |
| 60-75 | 严格检查 | 更严格的质量标准 |
| <60 | 调试模式 | 大部分页面会被重新生成 |

### 成本估算

- 每页需要调用 Gemini API **1-2 次**
- 检查阶段：1 次 API 调用
- 改进阶段：1 次 API 调用 (仅需要改进的页面)

建议先在小样本上测试，确认效果后再处理完整文档。

## 🔍 质量评估标准

系统从以下几个维度评估 chunks 质量：

### 1. 文本准确性 (40%)
- OCR 文字识别准确性
- 特殊字符和数字处理
- 格式保持 (粗体、斜体等)

### 2. 结构完整性 (30%)
- 章节标题识别
- 列表和表格结构
- 页面布局保持

### 3. 元数据准确性 (20%)
- Section_Title 分配
- Subheading 识别
- content_label 标注

### 4. 语义连贯性 (10%)
- Chunk 边界合理性
- 内容完整性
- 上下文关系

## 🚨 常见问题

### Q: API 调用失败怎么办？
A: 检查网络连接和 API Key，系统会自动重试 3 次。

### Q: 处理大文档时间太长？
A: 
- 使用 `--dry-run` 先查看文档规模
- 考虑分批处理
- 调高置信度阈值减少改进页面数

### Q: 改进效果不明显？
A: 
- 降低置信度阈值
- 检查原始 PDF 图像质量
- 查看详细报告了解具体问题

### Q: 想要自定义评估标准？
A: 修改 `chunks_reviewer.py` 中的 `review_prompt` 和 `regeneration_prompt`。

## 🔧 高级功能

### 单页面检查
```python
# 检查特定页面
page_image = reviewer.pdf_page_to_image("document.pdf", page_no=5)
chunks = page_groups[5]
result = reviewer.review_page_chunks(page_image, chunks, 5)
print(f"页面 5 置信度: {result.confidence_score}")
```

### 批量处理多文档
```python
import glob

for jsonl_file in glob.glob("Data_Json/*_chunks.jsonl"):
    pdf_file = jsonl_file.replace("_chunks.jsonl", ".pdf")
    if os.path.exists(pdf_file):
        results = reviewer.review_and_improve_chunks(jsonl_file, pdf_file)
        print(f"{jsonl_file}: {results['summary']['improvement_rate']*100:.1f}% 改进")
```

### 自定义输出格式
```python
# 只获取改进后的 chunks，不保存文件
results = reviewer.review_and_improve_chunks(
    jsonl_path="input.jsonl",
    pdf_path="input.pdf", 
    output_dir=None  # 不保存文件
)

improved_chunks = results['improved_chunks']
# 自定义处理 improved_chunks...
```

## 📈 效果示例

典型改进效果：

```
📊 审查结果总结:
==============================
📄 总页面数: 26
🔧 改进页面数: 5
📈 改进率: 19.2%
⭐ 平均置信度: 82.3

常见改进类型:
- 修正 OCR 识别错误
- 优化表格和列表提取
- 改进章节标题识别
- 调整 chunk 边界
- 完善元数据标注
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个系统！

## 📄 许可证

本项目使用 MIT 许可证。 