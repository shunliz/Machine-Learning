# Agent

## 7.3 Agent

### 1 什么是 LLM Agent？

简单来说，大模型Agent是一个以LLM为核心“大脑”，并赋予其自主规划、记忆和使用工具能力的系统。 它不再仅仅是被动地响应用户的提示（Prompt），而是能够：

1. 理解目标（Goal Understanding）： 接收一个相对复杂或高层次的目标（例如，“帮我规划一个周末去北京的旅游行程并预订机票酒店”）。
2. 自主规划（Planning）： 将大目标分解成一系列可执行的小步骤（例如，“搜索北京景点”、“查询天气”、“比较机票价格”、“查找合适的酒店”、“调用预订API”等）。
3. 记忆（Memory）： 拥有短期记忆（记住当前任务的上下文）和长期记忆（从过去的交互或外部知识库中学习和检索信息）。
4. 工具使用（Tool Use）： 调用外部API、插件或代码执行环境来获取信息（如搜索引擎、数据库）、执行操作（如发送邮件、预订服务）或进行计算。
5. 反思与迭代（Reflection & Iteration）： （在更高级的Agent中）能够评估自己的行为和结果，从中学习并调整后续计划。

传统的LLM像一个知识渊博但只能纸上谈兵的图书馆员，而 LLM Agent 则更像一个全能的私人助理，不仅懂得多，还能跑腿办事，甚至能主动思考最优方案。

![image-20250731100403032](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250731100403032.png)


LLM Agent 通过将大型语言模型的强大语言理解和生成能力与规划、记忆和工具使用等关键模块相结合，实现了超越传统大模型的自主性和复杂任务处理能力，这种能力使得 LLM Agent 在许多垂直领域（如法律、医疗、金融等）都具有广泛的应用潜力，如图7.7所示 Agent 工作原理。

### 2 LLM Agent 的类型

虽然LLM Agent的概念还在快速发展中，但根据其设计理念和能力侧重，我们可以大致将其分为几类：

任务导向型Agent（Task-Oriented Agents）：

- 特点： 专注于完成特定领域的、定义明确的任务，例如客户服务、代码生成、数据分析等。
- 工作方式： 通常有预设的流程和可调用的特定工具集。LLM主要负责理解用户意图、填充任务槽位、生成回应或调用合适- 的工具。
- 例子： 专门用于预订餐厅的聊天机器人、辅助编程的代码助手（如GitHub Copilot在某些高级功能上体现了Agent特性）。

规划与推理型Agent（Planning & Reasoning Agents）：

- 特点： 强调自主分解复杂任务、制定多步计划，并根据环境反馈进行调整的能力。它们通常需要更强的推理能力。
- 工作方式： 常采用特定的思维框架，如ReAct (Reason+Act)，让模型先进行“思考”（Reasoning）分析当前情况和所需行动，然后执行“行动”（Action）调用工具，再根据工具返回结果进行下一轮思考。Chain-of-Thought (CoT) 等提示工程技术也是其推理的基础。
- 例子： 需要整合网络搜索、计算器、数据库查询等多种工具来回答复杂问题的研究型Agent，或者能够自主完成“写一篇关于XX主题的报告，并配上相关数据图表”这类任务的Agent。

多Agent系统（Multi-Agent Systems）：

- 特点： 由多个具有不同角色或能力的Agent协同工作，共同完成一个更宏大的目标。
- 工作方式： Agent之间可以进行通信、协作、辩论甚至竞争。例如，一个Agent负责规划，一个负责执行，一个负责审查。
- 例子： 模拟软件开发团队（产品经理Agent、程序员Agent、测试员Agent）来自动生成和测试代码；模拟一个公司组织结构来完成商业策划。AutoGen、ChatDev等框架支持这类系统的构建。

探索与学习型Agent（Exploration & Learning Agents）：

- 特点： 这类Agent不仅执行任务，还能在与环境的交互中主动学习新知识、新技能或优化自身策略，类似于强化学习中的Agent概念。
- 工作方式： 可能包含更复杂的记忆和反思机制，能够根据成功或失败的经验调整未来的规划和行动。
- 例子： 能在未知软件环境中自主探索学习如何操作的Agent，或者在玩游戏时不断提升策略的Agent。

### 3 动手构造一个 Agent

我们来基于 `openai` 库和其 `tool_calls` 功能，动手构造一个Agent，这个 Agent 是一个简单的任务导向型 Agent，它能够根据用户的输入，回答一些简单的问题。

#### Step 1 : 初始化客户端和模型

首先，我们需要一个能够调用大模型的客户端。这里我们使用 `openai` 库，并配置其指向一个兼容 OpenAI API 的服务终端，例如 [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK)。同时，指定要使用的模型，如 `Qwen/QwQ-32B，硅基流动还有很多免费模型，测试我们就用免费的了。

```python
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="YOUR_API_KEY",  # 替换为你的 API Key，硅基流动上注册账号，自己创建一个api key就好
    base_url="https://api.siliconflow.cn/v1", # 使用 SiliconFlow 的 API 地址
)

# 指定模型名称
model_name = "Qwen/QwQ-32B"
```

#### Step 2: 定义工具函数

定义 Agent 可以使用的工具函数。每个函数都需要有清晰的文档字符串（docstring），描述其功能和参数，因为这将用于自动生成工具的 JSON Schema。

```python
from datetime import datetime

# 获取当前日期和时间
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    return str(a.count(b))
```

为了让 OpenAI API 理解这些工具，我们需要将它们转换成特定的 JSON Schema 格式。这可以通过 `src/utils.py` 中的 `function_to_json` 辅助函数完成。

```python
import inspect
from datetime import datetime
import pprint

def function_to_json(func) -> dict:
    # 定义 Python 类型到 JSON 数据类型的映射
    type_map = {
        str: "string",       # 字符串类型映射为 JSON 的 "string"
        int: "integer",      # 整型类型映射为 JSON 的 "integer"
        float: "number",     # 浮点型映射为 JSON 的 "number"
        bool: "boolean",     # 布尔型映射为 JSON 的 "boolean"
        list: "array",       # 列表类型映射为 JSON 的 "array"
        dict: "object",      # 字典类型映射为 JSON 的 "object"
        type(None): "null",  # None 类型映射为 JSON 的 "null"
    }

    # 获取函数的签名信息
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果获取签名失败，则抛出异常并显示具体的错误信息
        raise ValueError(
            f"无法获取函数 {func.__name__} 的签名: {str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        # 尝试获取参数的类型，如果无法找到对应的类型则默认设置为 "string"
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # 如果参数类型不在 type_map 中，抛出异常并显示具体错误信息
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数（即没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,            # 函数的名称
            "description": func.__doc__ or "", # 函数的文档字符串（如果不存在则为空字符串）
            "parameters": {
                "type": "object",
                "properties": parameters,     # 函数参数的类型描述
                "required": required,         # 必须参数的列表
            },
        },
    }
```

#### Step 3: 构造 Agent 类

定义 `Agent` 类。这个类负责管理对话历史、调用 OpenAI API、处理工具调用请求以及执行工具函数。

```python
from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# 导入定义好的工具函数
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSTEM_PROMPT = """
你是一个人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen3-8B", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        
        # 检查模型是否调用了工具        
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```

Agent 的工作流程如下：

1.  接收用户输入。
2.  调用大模型（如 Qwen），并告知其可用的工具及其 Schema。
3.  如果模型决定调用工具，Agent 会解析请求，执行相应的 Python 函数。
4.  Agent 将工具的执行结果返回给模型。
5.  模型根据工具结果生成最终回复。
6.  Agent 将最终回复返回给用户。

如图所示，Agent 调用工具流程：

![image-20250731111029963](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250731111029963.png)

#### Step 4: 运行 Agent

现在我们可以实例化并运行 Agent。在 `demo.py` 的 `if __name__ == "__main__":` 部分提供了一个简单的命令行交互示例。

```python
if __name__ == "__main__":
    client = OpenAI(
        api_key="YOUR_API_KEY", # 替换为你的 API Key
        base_url="https://api.siliconflow.cn/v1",
    )

    # 创建 Agent 实例，传入 client、模型名称和工具函数列表
    agent = Agent(
        client=client,
        model="Qwen/Qwen3-8B",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # 设置为 True 可以看到工具调用信息
    )

    # 开始交互式对话循环
    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
```

