# 上下文感知PDF处理功能

## 功能概述

为了解决PDS文档处理中的一致性问题，我们在 `GeminiPDSProcessor` 中添加了**上下文传递机制**。这个功能确保在处理多页文档时，章节标题和小标题能够正确地在页面间传递和继承。

## 核心问题

在处理保险PDS文档时，经常遇到以下情况：
- 一个章节（如 "Section 2: Life Insurance Benefits"）会跨越多个页面
- 只有第一页显示完整的章节标题
- 后续页面只包含内容，没有重复章节标题
- 传统的逐页处理会导致后续页面的 `Section_Title` 丢失

## 解决方案

### 1. 上下文传递机制

```python
def process_page(self, image, page_num, global_metadata, previous_context=None):
    """处理单页并返回chunks和当前上下文"""
    # 将上一页的上下文信息添加到prompt中
    if previous_context:
        context_info = f"""
CONTEXT FROM PREVIOUS PAGE:
- Previous Section_Title: {previous_context.get('Section_Title', 'N/A')}
- Previous Subheading: {previous_context.get('Subheading', 'N/A')}

CONTEXT RULES:
- If this page does not contain a new Section_Title heading, inherit the previous Section_Title
- Only change Section_Title if you see a clear new section heading on this page
- This ensures consistency across pages within the same section
"""
```

### 2. 智能继承逻辑

**Section_Title 继承规则：**
- 如果当前页面没有新的章节标题（LLM返回 "N/A"），自动继承上一页的 `Section_Title`
- 只有当页面包含明确的新章节标题时才更新

**Subheading 继承规则：**
- 在同一章节内，如果当前页面没有新的小标题，继承上一页的 `Subheading`  
- 当出现新章节时，不继承 `Subheading`（新章节重新开始）

### 3. 上下文提取和传递

```python
def _extract_current_context(self, chunks, previous_context=None):
    """从当前页面的chunks中提取上下文信息供下一页使用"""
    current_context = previous_context.copy() if previous_context else {}
    
    for chunk in chunks:
        if chunk.get('Section_Title') and chunk['Section_Title'] != 'N/A':
            current_context['Section_Title'] = chunk['Section_Title']
        if chunk.get('Subheading') and chunk['Subheading'] != 'N/A':
            current_context['Subheading'] = chunk['Subheading']
    
    return current_context
```

## 使用示例

### 基本用法

```python
processor = GeminiPDSProcessor()
current_context = None

for img, page_num in images:
    chunks, current_context = processor.process_page(
        img, page_num, global_metadata, current_context
    )
    # current_context 会自动传递给下一页
```

### 处理流程示例

**第1页：** 包含 "Section 2: Life Insurance Benefits"
```json
{
  "Section_Title": "Section 2: Life Insurance Benefits",
  "Subheading": "Overview",
  "content": "This section describes..."
}
```
→ 上下文：`{"Section_Title": "Section 2: Life Insurance Benefits", "Subheading": "Overview"}`

**第2页：** 继续上一章节的内容，无新章节标题
```json
{
  "Section_Title": "Section 2: Life Insurance Benefits",  // 自动继承
  "Subheading": "Death Benefits",  // 新的小标题
  "content": "The death benefit is..."
}
```
→ 上下文：`{"Section_Title": "Section 2: Life Insurance Benefits", "Subheading": "Death Benefits"}`

**第3页：** 出现新章节
```json
{
  "Section_Title": "Section 3: TPD Insurance",  // 新章节
  "Subheading": "N/A",  // 不继承前一章节的小标题
  "content": "TPD insurance provides..."
}
```

## 技术实现细节

### 1. Prompt 增强
- 在系统prompt中添加了上下文感知规则
- 动态插入前一页的上下文信息
- 明确指导LLM何时使用 "N/A" 以触发继承机制

### 2. 验证和清理
- 在 `_validate_chunk()` 中实现继承逻辑
- 智能判断何时继承 `Section_Title` 和 `Subheading`
- 提供详细的日志输出便于调试

### 3. 错误处理
- 处理空chunks的情况
- 在fallback场景中也应用上下文继承
- 确保处理错误时不丢失上下文信息

## 优势

1. **一致性保证**：确保同一章节内的所有chunks有相同的 `Section_Title`
2. **智能识别**：自动识别新章节的开始，避免错误继承
3. **灵活性**：支持复杂的文档结构（嵌套章节、小标题等）
4. **可调试性**：提供详细的日志输出，便于问题追踪
5. **向后兼容**：现有代码无需修改即可使用新功能

## 测试验证

我们创建了全面的测试来验证功能：
- 上下文继承机制测试
- 新章节检测测试  
- 多页面处理流程测试
- 边界情况处理测试

所有测试都通过，确保功能的稳定性和可靠性。

## 总结

这个上下文传递机制显著提升了PDS文档处理的质量，解决了多页文档处理中的一致性问题。通过智能的继承逻辑和清晰的规则设计，我们能够生成更准确、更有用的文档chunks，为后续的检索和分析提供更好的基础。 