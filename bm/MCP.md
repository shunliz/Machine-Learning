# MCP

**Model Context Protocol (MCP)** 是一个开放协议，旨在实现 LLM 应用与外部数据源和工具之间的无缝集成。

无论您是构建 AI 驱动的 IDE、增强聊天界面，还是创建自定义 AI 工作流，MCP 都提供了一种标准化的方式来连接 LLM 与外部世界。

简单来说，MCP 是一种客户端-服务器架构的协议，允许 LLM 应用程序（如 Claude、各种 IDE 等）通过标准化的接口访问外部数据和功能。这解决了 LLM 在实际应用中常见的一些痛点：

- LLM 无法直接访问实时数据（如天气、股票行情等）
- LLM 无法执行外部操作（如发送邮件、控制设备等）
- LLM 无法访问用户的本地文件或其他私有数据

通过 MCP，这些限制得到了优雅的解决，同时保持了安全性和可扩展性。

## **MCP的核心架构**

MCP 采用客户端-服务器架构，主要包含以下几个组件：

- **MCP 主机（Host）：**如 Claude Desktop、IDE 或其他 AI 工具，通过 MCP 访问数据
- **MCP 客户端（Client）：**与服务器保持 1:1 连接的协议客户端
- **MCP 服务器（Server）：**轻量级程序，通过标准化的 MCP 协议公开特定功能
- **本地数据源：**计算机上的文件、数据库和服务，MCP 服务器可以安全访问这些内容
- **远程服务：**通过互联网可用的外部系统（例如通过 API），MCP 服务器可以连接这些服务

主机可以同时连接多个服务器，每个服务器提供不同的功能，形成一个生态系统：

```
主机（Claude、IDE 等）<--MCP 协议--> 服务器 A <--> 本地数据源 A                     
                    <--MCP 协议--> 服务器 B <--> 本地数据源 B                     
                    <--MCP 协议--> 服务器 C <--> 远程服务 C
```

## **MCP的核心概念**

MCP 服务器可以提供三种主要类型的功能：

1. **资源（Resources）：**客户端可以读取的文件类数据（如 API 响应或文件内容）
2. **工具（Tools）：**LLM 可以调用的函数（需要用户批准）
3. **提示（Prompts）：**帮助用户完成特定任务的预写模板

**资源（Resources）**

资源是可以被客户端读取的文件类数据。它们可以是文本或二进制形式，并有唯一的 URI 标识。

资源可以是：

- 直接资源：固定内容的资源
- 资源模板：可以通过参数动态生成的资源

例如，一个文件系统 MCP 服务器可以将本地文件作为资源提供给 LLM，使其能够读取用户的文件。

**工具（Tools）**

工具是 MCP 中最强大的原语之一，允许服务器向客户端公开可执行的功能。通过工具，LLM 可以与外部系统交互，执行计算，并在现实世界中采取行动。

每个工具都有明确的定义，包括：

- 名称
- 描述
- 输入参数模式（使用 JSON Schema）
- 输出格式

工具设计为由模型控制，但通常需要人类批准才能执行，这保证了安全性。

**提示（Prompts）**

提示是预定义的模板，可以帮助用户完成特定任务。它们可以包含动态部分，嵌入资源上下文，并支持多步工作流。

## **MCP 服务器开发案例：天气服务器**

让我们通过一个实际例子来理解 MCP 服务器的开发。我们将构建一个简单的天气服务器，它提供两个工具：获取天气警报和获取天气预报。

**步骤 1：设置环境**

```
# 创建项目目录
uv init weather
cd weather

# 创建虚拟环境并激活
uv venv
source .venv/bin/activate  # MacOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv add "mcp[cli]" httpx

# 创建服务器文件
touch weather.py
```

**步骤 2：实现天气服务器**

```python
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务器
mcp = FastMCP("weather")

# 常量
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """向 NWS API 发送请求并进行适当的错误处理。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """将警报特征格式化为可读字符串。"""
    props = feature["properties"]
    return f"""Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}"""


@mcp.tool()
async def get_alerts(state: str) -> str:
    """获取美国州的天气警报。
    
    Args:
        state: 美国州的两字母代码（例如 CA, NY）
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)
    
    if not data or "features" not in data:
        return "无法获取警报或未找到警报。"
    
    if not data["features"]:
        return "该州没有活跃警报。"
    
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """获取某个位置的天气预报。
    
    Args:
        latitude: 位置的纬度
        longitude: 位置的经度
    """
    # 首先获取预报网格端点
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)
    
    if not points_data:
        return "无法获取该位置的预报数据。"
    
    # 从 points 响应获取预报 URL
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)
    
    if not forecast_data:
        return "无法获取详细预报。"
    
    # 将周期格式化为可读的预报
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # 仅显示接下来的 5 个周期
        forecast = f"""{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}"""
        forecasts.append(forecast)
    
    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='stdio')
```



**步骤 3：配置 Claude Desktop 连接服务器**

要使用 Claude Desktop 连接到我们的服务器，需要编辑配置文件：

```
{
	"mcpServers": {
		"weather": {
			"command": "uv",
			"args": ["--directory", "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather", "run", "weather.py"]
		}
	}
}
```

此配置告诉 Claude Desktop 如何启动我们的天气服务器，并使 Claude 能够使用我们实现的工具。

## **MCP 客户端开发案例**

接下来，让我们看看如何开发一个 MCP 客户端，该客户端可以连接到任何 MCP 服务器并利用其功能。

**步骤 1：设置环境**

```
# 创建项目目录
uv init mcp-client
cd mcp-client

# 创建虚拟环境
uv venv
source .venv/bin/activate  # MacOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv add mcp anthropic python-dotenv

# 创建主文件
touch client.py
```

**步骤 2：设置 API 密钥**

创建 .env 文件存储 Anthropic API 密钥：

```
ANTHROPIC_API_KEY=<your key here>
```

**步骤 3：实现 MCP 客户端**

```python
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()  # 从 .env 加载环境变量
class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
 
    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器
        Args:
            server_script_path: 服务器脚本路径 (.py 或 .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n连接到服务器，可用工具:", [tool.name for tool in tools])
        
    async def process_query(self, query: str) -> str:
        """使用 Claude 和可用工具处理查询"""
        if not self.session:
            return "未连接到任何服务器"
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        # 初始 Claude API 调用
        claude_response = await self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )
        # 处理工具调用
        message = claude_response.content[0]
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("\nClaude 请求使用工具:", message.tool_calls[0].name)
            # 执行工具调用
            tool_call = message.tool_calls[0]
            tool_name = tool_call.name
            tool_params = tool_call.params
            print(f"使用参数执行 {tool_name}:", tool_params)
            tool_response = await self.session.execute_tool(tool_name, tool_params)
            # 将工具结果发送回 Claude
            messages.append({
                "role": "assistant",
                "content": [message]
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    }
                ]
            })
            # 获取最终回答
            final_response = await self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=messages
            )
            return final_response.content[0].text
        else:
            # 没有工具调用时直接返回回答
            return message.text
    async def close(self):
        """关闭连接和资源"""
        await self.exit_stack.aclose()
async def main():
    # 创建客户端实例
    client = MCPClient()
    try:
        # 连接到天气服务器
        print("连接到天气服务器...")
        await client.connect_to_server("../weather/weather.py")
        # 交互式循环
        while True:
            query = input("\n输入查询 (输入 'exit' 退出): ")
            if query.lower() == 'exit':
                break
            print("\n处理查询...")
            response = await client.process_query(query)
            print("\n回答:", response)
    finally:
        # 关闭连接
        await client.close()
if __name__ == "__main__":
    asyncio.run(main())
```

这个客户端可以连接到任何 MCP 服务器，获取其可用工具，然后将用户查询发送到 Claude 进行处理。Claude 可以调用服务器提供的工具，客户端将结果返回给 Claude 以生成最终回答。

## FastMCP

