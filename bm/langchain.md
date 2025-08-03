# LangChain

LangChain 是一个**模块化、高度可扩展的框架**，专为构建基于大语言模型（LLM）的复杂应用而设计。其核心目标是解决原始 LLM 的局限性（如缺乏上下文记忆、无法访问实时数据等），并通过标准化组件和工具集成，让开发者能够快速构建智能、动态的应用。LangChain 的核心用途可以概括为：**让 LLM 从“单次对话工具”升级为“解决复杂任务的智能代理”**。

## 具体场景

### **1. 构建智能对话系统**

- **多轮对话管理**：通过记忆组件（Memory）记录对话历史，支持上下文关联（如“你刚才说的XX是什么意思？”）。
- **角色扮演**：让 LLM 扮演特定角色（如客服、医生、教师），通过提示词模板控制输出风格。
- **动态响应**：根据用户输入调用外部工具（如搜索网页、查询数据库），生成更准确的回答。

**示例**：
一个旅游客服机器人可以：

1. 记住用户偏好（如“喜欢自然风光”）。
2. 调用天气 API 推荐最佳旅行时间。
3. 结合酒店预订系统提供个性化建议。

### **2. 实现检索增强生成（RAG）**

- **问题**：原始 LLM 的知识截止于训练数据（如 GPT-4 的知识截止到 2023 年 10 月），无法回答最新问题。

- 解决方案

  ：通过 LangChain 连接外部知识库（如文档、数据库、API），实现“检索 + 生成”的组合：

  1. 将用户问题转换为语义向量。
  2. 从向量数据库中检索相关文档片段。
  3. 将检索结果作为上下文输入 LLM，生成回答。

**优势**：

- 减少幻觉（Hallucination）：答案基于真实数据。
- 支持私有数据：可连接企业内部的文档或数据库。

### **3. 自动化复杂任务**

- **任务分解**：将复杂任务拆解为多个子任务（如“规划旅行路线” → “查询机票 → 预订酒店 → 生成行程”）。
- **工具调用**：让 LLM 自主决定何时调用外部工具（如计算器、翻译 API、代码解释器）。
- **反馈循环**：根据执行结果调整后续行动（如“机票太贵，重新搜索更便宜的选项”）。

**示例**：
一个数据分析代理可以：

1. 接收用户需求（如“分析销售数据，找出增长最快的地区”）。
2. 生成 SQL 查询并执行。
3. 用 LLM 总结结果并生成可视化图表。

### **4. 开发多模态应用**

- 文本 + 图像/音频

  ：结合 Stable Diffusion、Whisper 等模型，实现：

  - 根据文本生成图像（如“画一只猫”）。
  - 将音频转换为文本并分析情感。

- **跨模态检索**：支持“用图片提问”（如上传一张照片，问“这是哪种植物？”）。

### **5. 构建企业级应用**

- **安全与合规**：通过权限控制、数据脱敏等机制，确保企业数据安全。
- **可观测性**：记录 LLM 的输入/输出，便于审计和调试。
- **集成现有系统**：连接 CRM、ERP 等企业软件，实现自动化流程。

## 核心功能

### **1. 模型接口（Models）**

- **功能**：统一不同 LLM 的调用方式，支持快速切换模型提供商。

- 支持的模型类型：

  - **文本补全模型（LLMs）**：如 OpenAI 的 `gpt-3.5-turbo`、Hugging Face 的 `Llama-2`。
  - **聊天模型（ChatModels）**：如 OpenAI 的 `ChatGPT`、硅基流动的 `Qwen-Chat`。
  - **嵌入模型（Embeddings）**：如 `text-embedding-ada-002`（用于语义搜索）。

- 代码示例：

  ```python
  from langchain.llms import OpenAI, HuggingFacePipeline
  from langchain.chat_models import ChatOpenAI
   
  # 调用 OpenAI 模型
  llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
   
  # 调用 Hugging Face 本地模型
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("models/Qwen/Qwen2-7B-Instruct")
  tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen2-7B-Instruct")
  huggingface_llm = HuggingFacePipeline(model=model, tokenizer=tokenizer)
  ```

### **2. 提示词管理（Prompts）**

- **功能**：动态生成、优化和管理提示词，提升 LLM 输出质量。

- 关键组件：

  - **提示词模板（PromptTemplate）**：用变量填充动态内容（如用户输入、上下文）。
  - **少样本学习（Few-Shot Examples）**：提供示例引导 LLM 输出格式。
  - **思维链提示（Chain-of-Thought）**：让 LLM 逐步推理（如“先分解问题，再逐步解答”）。

- 代码示例：

  ```python
  from langchain.prompts import PromptTemplate
   
  template = """
  用户问题: {question}
  上下文: {context}
  请用简洁的语言回答：
  """
  prompt = PromptTemplate(template=template, input_variables=["question", "context"])
  formatted_prompt = prompt.format(question="巴黎的首都是哪里？", context="法国是欧洲国家")
  ```

### **3. 记忆（Memory）**

- **功能**：存储和管理对话历史或外部数据，支持上下文关联。

- 记忆类型：

  - **短期记忆（Buffer Memory）**：存储最近几轮对话（如 `ConversationBufferMemory`）。
  - **长期记忆（Persistent Memory）**：连接数据库（如 SQLite、Pinecone）存储大量数据。
  - **实体记忆（Entity Memory）**：跟踪对话中提到的实体（如人名、地点）。

- 代码示例：

  ```python
  from langchain.memory import ConversationBufferMemory
   
  memory = ConversationBufferMemory()
  memory.chat_memory.add_user_message("你好！")
  memory.chat_memory.add_ai_message("你好！有什么可以帮忙的吗？")
  print(memory.buffer)  # 输出对话历史
  ```

### **4. 链（Chains）**

- **功能**：将多个组件（模型、提示词、记忆）串联成流程，实现复杂逻辑。

- 预定义链：

  - **`LLMChain`**：基础链，用于单次 LLM 调用。
  - **`RetrievalQA`**：检索 + 生成的问答链。
  - **`SequentialChains`**：按顺序执行多个链（如“先检索 → 再生成 → 最后总结”）。

- 代码示例：

  ```python
  from langchain.chains import LLMChain
  from langchain.prompts import PromptTemplate
   
  prompt = PromptTemplate(input_variables=["input"], template="用中文回答: {input}")
  chain = LLMChain(llm=llm, prompt=prompt)
  response = chain.run("今天天气如何？")
  ```

### **5. 代理（Agents）**

- **功能**：让 LLM 自主决定行动（如调用工具、查询 API），实现复杂任务自动化。

- 关键组件：

  - **工具（Tools）**：可调用的外部功能（如搜索、计算、数据库查询）。
  - 代理类型：
    - **`Zero-Shot Agent`**：根据提示词直接决定行动（无需示例）。
    - **`ReAct Agent`**：结合推理和行动（如“先思考，再执行”）。

- 代码示例：

  ```python
  from langchain.agents import initialize_agent, Tool
  from langchain.agents import AgentType
  from langchain.llms import OpenAI
  from langchain.utilities import SerpAPIWrapper
   
  # 定义工具
  search = SerpAPIWrapper(api_key="your_api_key")
  tools = [
      Tool(name="Search", func=search.run, description="搜索网页信息")
  ]
   
  # 初始化代理
  llm = OpenAI(temperature=0)
  agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
   
  # 执行任务
  response = agent.run("2024 年奥运会在哪里举办？")
  ```

### **6. 工具集成（Tools）**

- **功能**：扩展 LLM 的能力，支持访问外部资源。

- 常见工具类型：

  - **API 调用**：如 Google Search、Wolfram Alpha、自定义 REST API。
  - **计算工具**：如 Python 解释器、计算器。
  - **文件处理**：如 PDF 解析、Excel 操作。
  - **数据库查询**：如 SQL、MongoDB。

- 代码示例：

  ```python
  from langchain.tools import Tool
  from langchain.utilities import PythonREPL
   
  # 定义计算工具
  python_repl = PythonREPL()
  calc_tool = Tool(
      name="Calculator",
      func=lambda query: python_repl.run(query),
      description="用于数学计算（如 2+2）"
  )
  ```

