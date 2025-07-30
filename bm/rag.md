# RAG

RAG 检索增强生成（Retrieval Augmented Generation），已经成为当前最火热的LLM应用方案。通过自有垂域数据库检索相关信息，然后合并成为提示模板，给大模型润色生成回答。

但是当我们将大模型应用于实际业务场景时会发现，通用的基础大模型基本无法满足实际业务需求，主要有以下几方面原因：

- **知识的局限性**：大模型自身的知识完全源于训练数据，而现有的主流大模型（deepseek、文心一言、通义千问…）的训练集基本都是构建于网络公开的数据，对于一些实时性的、非公开的或私域的数据是没有。
- **幻觉问题**：所有的深度学习模型的底层原理都是基于数学概率，模型输出实质上是一系列数值运算，大模型也不例外，所以它经常会一本正经地胡说八道，尤其是在大模型自身不具备某一方面的知识或不擅长的任务场景。
- **数据安全性**：对于企业来说，数据安全至关重要，没有企业愿意承担数据泄露的风险，尤其是大公司，没有人将私域数据上传第三方平台进行训练会推理。这也导致完全依赖通用大模型自身能力的应用方案不得不在数据安全和效果方面进行取舍。

一句话总结：

> **RAG（中文为检索增强生成） = 检索技术 + LLM 提示**。

## **RAG架构**

RAG的架构如图中所示，简单来讲，RAG就是通过检索获取相关的知识并将其融入Prompt，让大模型能够参考相应的知识从而给出合理回答。因此，可以将RAG的核心理解为“检索+生成”，前者主要是利用向量数据库的高效存储和检索能力，召回目标知识；后者则是利用大模型和Prompt工程，将召回的知识合理利用，生成目标答案。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-3cbae4ea97b928fe7bd7b6a3071a0c6b_1440w.jpg)

完整的RAG应用流程主要包含两个阶段：

- 数据准备阶段：数据提取——>文本分割——>向量化（embedding）——>数据入库
- 应用阶段：用户提问——>数据检索（召回）——>注入Prompt——>LLM生成答案

### **数据准备阶段**：

数据准备一般是一个离线的过程，主要是将私域数据向量化后构建索引并存入数据库的过程。主要包括：数据提取、文本分割、向量化、数据入库等环节。

- **数据提取**
  - 数据加载：包括多格式数据加载、不同数据源获取等，根据数据自身情况，将数据处理为同一个范式。
  - 数据处理：包括数据过滤、压缩、格式化等。
  - 元数据获取：提取数据中关键信息，例如文件名、Title、时间等 。
- **文本分割**：
  文本分割主要考虑两个因素：1）embedding模型的Tokens限制情况；2）语义完整性对整体的检索效果的影响。一些常见的文本分割方式如下：
  - 句分割：以”句”的粒度进行切分，保留一个句子的完整语义。常见切分符包括：句号、感叹号、问号、换行符等。
  - 固定长度分割：根据embedding模型的token长度限制，将文本分割为固定长度（例如256/512个tokens），这种切分方式会损失很多语义信息，一般通过在头尾增加一定冗余量来缓解。
- **向量化（embedding）**：

向量化是一个将文本数据转化为向量矩阵的过程，该过程会直接影响到后续检索的效果。目前常见的embedding模型如表中所示，这些embedding模型基本能满足大部分需求，但对于特殊场景（例如涉及一些罕见专有词或字等）或者想进一步优化效果，则可以选择开源Embedding模型微调或直接训练适合自己场景的Embedding模型。

| 模型名称           | 描述                                                         | 获取地址                                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ChatGPT-Embedding  | ChatGPT-Embedding由OpenAI公司提供，以接口形式调用。          | [https://platform.openai.com/docs/guides/embeddings/what-are-embeddings](https://link.zhihu.com/?target=https%3A//platform.openai.com/docs/guides/embeddings/what-are-embeddings) |
| ERNIE-Embedding V1 | ERNIE-Embedding V1由百度公司提供，依赖于文心大模型能力，以接口形式调用。 | [https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu](https://link.zhihu.com/?target=https%3A//cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu) |
| M3E                | M3E是一款功能强大的开源Embedding模型，包含m3e-small、m3e-base、m3e-large等多个版本，支持微调和本地部署。 | [https://huggingface.co/moka-ai/m3e-base](https://link.zhihu.com/?target=https%3A//huggingface.co/moka-ai/m3e-base) |
| BGE                | BGE由北京智源人工智能研究院发布，同样是一款功能强大的开源Embedding模型，包含了支持中文和英文的多个版本，同样支持微调和本地部署。 | [https://huggingface.co/BAAI/bge-base-en-v1.5](https://link.zhihu.com/?target=https%3A//huggingface.co/BAAI/bge-base-en-v1.5) |

- **数据入库：**

数据向量化后构建索引，并写入数据库的过程可以概述为数据入库过程，适用于RAG场景的数据库包括：FAISS、Chromadb、ES、milvus等。一般可以根据业务场景、硬件、性能需求等多因素综合考虑，选择合适的数据库。

### **应用阶段：**

在应用阶段，可以根据用户的提问，通过高效的检索方法，召回与提问最相关的知识，并融入Prompt；大模型参考当前提问和相关知识，生成相应的答案。关键环节包括：数据检索、注入Prompt等。

- **数据检索**
  常见的数据检索方法包括：相似性检索、全文检索等，根据检索效果，一般可以选择多种检索方式融合，提升召回率。
  - 相似性检索：即计算查询向量与所有存储向量的相似性得分，返回得分高的记录。常见的相似性计算方法包括：余弦相似性、欧氏距离、曼哈顿距离等。
  - 全文检索：全文检索是一种比较经典的检索方式，在数据存入时，通过关键词构建倒排索引；在检索时，通过关键词进行全文检索，找到对应的记录。
- **注入Prompt**
  - Prompt作为大模型的直接输入，是影响模型输出准确率的关键因素之一。在RAG场景中，Prompt一般包括任务描述、背景知识（检索得到）、任务指令（一般是用户提问）等，根据任务场景和大模型性能，也可以在Prompt中适当加入其他指令优化大模型的输出。

## RAG示例

### 搭建一个 RAG 框架

#### Step 1: RAG流程

接下来我们一步一步实现一个简单的RAG模型核心功能，即检索和生成，其目的是帮助大家更好地理解 RAG 模型的原理和实现。

- **索引**：将文档库分割成较短的片段，并通过编码器构建向量索引。
- **检索**：根据问题和片段的相似度检索相关文档片段。
- **生成**：以检索到的上下文为条件，生成问题的回答。

![alt text](https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-2-rag.png)

#### Step 2: 文档加载和切分

实现一个文档加载和切分的类，这个类主要用于加载文档并将其切分成文档片段。

文档可以是文章、书籍、对话、代码等文本内容，例如pdf文件、md文件、txt文件等。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            # 保留空格，只移除行首行尾空格
            line = line.strip()
            line_len = len(enc.encode(line))
            
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                # 先保存当前块（如果有内容）
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ''
                    curr_len = 0
                
                # 将长行按token长度分割
                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                
                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    
                    # 解码token片段回文本
                    chunk_tokens = line_tokens[start_token:end_token]
                    chunk_part = enc.decode(chunk_tokens)
                    
                    # 添加覆盖内容（除了第一个块）
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    
                    chunk_text.append(chunk_part)
                
                # 重置当前块状态
                curr_chunk = ''
                curr_len = 0
                
            elif curr_len + line_len + 1 <= token_len:  # +1 for newline
                # 当前行可以加入当前块
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                # 当前行无法加入当前块，开始新块
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                
                # 开始新块，添加覆盖内容
                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        # 添加最后一个块（如果有内容）
        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
```

文档读取后需要进行切分。我们可以设置一个最大的Token长度，然后根据这个最大长度来切分文档。切分文档时最好以句子为单位（按`\n`粗切分），并保证片段之间有一些重叠内容，以提高检索的准确性。

#### Step 3: 向量化

首先我们来动手实现一个向量化的类，这是RAG架构的基础。向量化类主要用来将文档片段向量化，将一段文本映射为一个向量。

首先我们要设置一个 `BaseEmbeddings` 基类，这样我们在使用其他模型时，只需要继承这个基类，然后在此基础上进行修改即可，方便代码扩展。

```python
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path (str): 模型或数据的路径
            is_api (bool): 是否使用API方式。True表示使用在线API服务，False表示使用本地模型
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        获取文本的嵌入向量表示
        Args:
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        Args:
            vector1 (List[float]): 第一个向量
            vector2 (List[float]): 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1,1]之间
        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数（长度）
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母（两个向量范数的乘积）
        magnitude = norm_v1 * norm_v2
        # 处理分母为0的特殊情况
        if magnitude == 0:
            return 0.0
            
        # 返回余弦相似度
        return dot_product / magnitude
```

`BaseEmbeddings`基类有两个主要方法：`get_embedding`和`cosine_similarity`。`get_embedding`用于获取文本的向量表示，`cosine_similarity`用于计算两个向量之间的余弦相似度。在初始化类时设置了模型的路径和是否是API模型，例如使用OpenAI的Embedding API需要设置`self.is_api=True`。

继承`BaseEmbeddings`类只需要实现`get_embedding`方法，`cosine_similarity`方法会被继承下来。这就是编写基类的好处。

```python
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # 从环境变量中获取 硅基流动 密钥
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # 从环境变量中获取 硅基流动 的基础URL
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        此处默认使用轨迹流动的免费嵌入模型 BAAI/bge-m3
        """
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
```

#### Step 4: 数据库与向量检索

完成文档切分和Embedding模型加载后，需要设计一个向量数据库来存放文档片段和对应的向量表示，以及设计一个检索模块用于根据Query检索相关文档片段。

向量数据库的功能包括：

- `persist`：数据库持久化保存。
- `load_vector`：从本地加载数据库。
- `get_vector`：获取文档的向量表示。
- `query`：根据问题检索相关文档片段。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
import json
from Embeddings import BaseEmbeddings, OpenAIEmbedding
import numpy as np
from tqdm import tqdm

class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
```

#### Step 5: 大模型模块

接下来是大模型模块，用于根据检索到的文档回答用户的问题。

首先实现一个基类，这样可以方便扩展其他模型。

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

`BaseModel` 包含两个方法：`chat`和`load_model`。对于本地化运行的开源模型需要实现`load_model`，而API模型则不需要。在此处我们还是使用国内用户可访问的硅基流动大模型API服务平台，使用API服务的好处就是用户不需要本地的计算资源，可以大大降低学习者的学习门槛。

```python
from openai import OpenAI

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
                model=self.model,
                messages=history,
                max_tokens=2048,
                temperature=0.1
            )
        return response.choices[0].message.content

```

设计一个专用于RAG的大模型提示词，如下：

```
RAG_PROMPT_TEMPLATE="""
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""
```

这样我们就可以利用InternLM2模型来做RAG啦！

#### Step 6: RAG Demo

接下来，我们来看看RAG的Demo吧！

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 没有保存数据库
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

也可以从本地加载已处理好的数据库：

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 保存数据库之后
vector = VectorStore()

vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

embedding = ZhipuEmbedding() # 创建EmbeddingModel

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

## 高级RAG

Advanced RAG 范式随后被提出，并在数据索引、检索前和检索后都进行了额外处理。

通过更精细的数据清洗、设计文档结构和添加元数据等方法提升文本的一致性、准确性和检索效率。

在检索前阶段则可以使用问题的重写、路由和扩充等方式对齐问题和文档块之间的语义差异。

在检索后阶段则可以通过将检索出来的文档库进行重排序避免 “Lost in the Middle ” 现象的发生。或是通过上下文筛选与压缩的方式缩短窗口长度。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-627dc8ea84954107f62b8373addcce63_1440w.jpg)

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-5dd9affa1e5d67ca16efded8708d67ab_1440w.jpg)

### 1：分块 (Chunking) & 向量化 (Vectorisation)

首先需要为文档内容创建向量索引，然后在运行时搜索与查询向量余弦距离最近的向量索引，这样就可以找到与查询内容最接近语义的文档。

**1.1 分块 (Chunking)**

Transformer 模型具有固定的输入序列长度，即使输入上下文窗口很大，一个句子或几个句子的向量也比几页文本的向量更能代表其语义含义，因此对数据进行分块—— 将初始文档拆分为一定大小的块，而不会失去其含义。有许多文本拆分器实现能够完成此任务。

块的大小是一个需要重点考虑的问题。块的大小取决于所使用的嵌入模型以及模型需要使用 token 的容量。如基于 BERT 的句子转换器，最多需要 512 个 token，OpenAI ada-002 能够处理更长的序列，如 8191 个 token，但这里的折衷是 LLM 有足够的上下文来推理，而不是足够具体的文本嵌入，以便有效地执行搜索。有一项关于块大小选择的研究。

**1.2 向量化 (Vectorisation)**

NLP中的词嵌入方式实现句子的向量化是一种方法，LLM通常使用对应的LLM模型做分块后的文章的向量化。参考前文的模型列表。

### 2. 搜索索引

**2.1 向量存储索引**

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-da420744c1f8a8c036167a5f23c052cd_1440w.jpg)

**RAG 管道的关键部分是搜索索引**，它存储了在上一步中获得的向量化内容。最原始的实现是使用平面索引 — 查询向量和所有块向量之间的暴力计算距离。

**为了实现1w+元素规模的高效检索，搜索索引**应该采用**向量索引**，比如 faiss、nmslib 以及 annoy。这些工具基于近似最近邻居算法，如聚类、树结构或HNSW算法。

此外，还有一些托管解决方案，如 [OpenSearch](https://zhida.zhihu.com/search?content_id=238243288&content_type=Article&match_order=1&q=OpenSearch&zhida_source=entity)、[ElasticSearch](https://zhida.zhihu.com/search?content_id=238243288&content_type=Article&match_order=1&q=ElasticSearch&zhida_source=entity) 以及向量数据库，它们自动处理上面提到的数据摄取流程，例如Pinecone、Weaviate和Chroma。

取决于索引选择、数据和搜索需求，还可以**存储元数据**，并使用**元数据过滤器**来按照日期或来源等条件进行信息检索。

**2.2 分层索引**

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-2a1cd57c7e9dc3a4e96432658e198b04_1440w.jpg)

在大型数据库的情况下，一个有效的方法是创建两个索引——一个由摘要组成，另一个由文档块组成，然后分两步进行搜索，首先通过摘要过滤掉相关文档，然后只在这个相关组内搜索。

**2.3 假设性问题和 HyDE**

另一种方法是让 **LLM 为每个块生成一个问题，并将这些问题嵌入到向量中**，在运行时对这个问题向量的索引执行查询搜索（将块向量替换为索引中的问题向量），然后在检索后路由到原始文本块并将它们作为 LLM 获取答案的上下文发送。

这种方法提高了搜索质量，因为与实际块相比，**查询和假设问题之间的语义相似性更高**。

还有一种叫做 HyDE 的反向逻辑方法——你要求 LLM 在给定查询的情况下生成一个假设的响应，然后将其向量与查询向量一起使用来提高搜索质量。

**2.4 内容增强**

这里的内容是将相关的上下文组合起来供 LLM 推理，以检索较小的块以获得更好的搜索质量。

有两种选择：一种是围绕较小的检索块的句子扩展上下文，另一种是递归地将文档拆分为多个较大的父块，其中包含较小的子块。

2.4.1 语句窗口检索器

在此方案中，文档中的每个句子都是单独嵌入的，这为上下文余弦距离搜索提供了极大的查询准确性。

为了在获取最相关的单个句子后更好地推理找到的上下文，将上下文窗口扩展为检索到的句子前后的 k 个句子，然后将这个扩展的上下文发送到 LLM。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-f20758c9c6f233c776aaea621dd25dab_1440w.jpg)

绿色部分是在索引中搜索时发现的句子嵌入，整个黑色 + 绿色段落被送到 LLM 以扩大其上下文，同时根据提供的查询进行推理。

2.4.2 自动合并检索器（或父文档检索器)

这里的思路与语句窗口检索器非常相似——搜索更精细的信息片段，然后在在LLM 进行推理之前扩展上下文窗口。文档被拆分为较小的子块，这些子块和较大的父块有引用关系。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-faebe20c9e474e5aa2a5f00fd53572ec_1440w.jpg)

首先在检索过程中获取较小的块，然后如果前 k 个检索到的块中有超过 n 个块链接到同一个父节点（较大的块），将这个父节点替换成给 LLM 的上下文——工作原理类似于自动将一些检索到的块合并到一个更大的父块中，因此得名。请注意，搜索仅在子节点索引中执行。

**2.5 融合检索或混合搜索**

这是一个很早以前的思路：结合传统的基于关键字的搜索（稀疏检索算法，如 tf-idf 或搜索行业标准 BM25）和现代语义或向量搜索，并将其结果组合在一个检索结果中。

这里唯一的关键是如何组合不同相似度分数的检索结果。这个问题通常通过 Reciprocal Rank Fusion 算法来解决，该算法能有效地对检索结果进行重新排序，以得到最终的输出结果。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-8e39fb64aa1d1761d9ac8eaabdc3cf05_1440w.jpg)

在 LangChain 中，这种方法是通过 Ensemble Retriever 来实现的，该类将你定义的多个检索器结合起来，比如一个基于 faiss 的向量索引和一个基于 BM25 的检索器，并利用 RRF 算法进行结果的重排。

混合或融合搜索通常能提供更优秀的检索结果，因为它结合了两种互补的搜索算法——既考虑了查询和存储文档之间的语义相似性，也考虑了关键词匹配。

### 3. 重排（reranking）和过滤（filtering）

使用上述任何算法获得了检索结果，现在是时候通过过滤、重排或一些转换来完善它们了。在 LlamaIndex 中，有各种可用的后处理器，根据相似性分数、关键字、元数据过滤掉结果，或使用其他模型（如 LLM）、sentence-transformer 交叉编码器，Cohere 重新排名接口或者基于元数据重排它们。

这是将检索到的上下文提供给 LLM 以获得结果答案之前的最后一步。

### 4. 查询转换

查询转换是一系列技术，使用 LLM 作为推理引擎来修改用户输入以提高检索质量。有很多技术实现可供选择。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-95f3a817f510aa57d6f1ca2971ebe84f_1440w.jpg)

**对于复杂的查询，大语言模型能够将其拆分为多个子查询。**比如，

- 当你问：“在 Github 上，Langchain 和 LlamaIndex 这两个框架哪个更受欢迎？”，

一般不太可能直接在语料库找到它们的比较，所以将这个问题分解为两个更简单、具体的合理的子查询：

- “Langchain 在 Github 上有多少星？”
- “Llamaindex 在 Github 上有多少星？”

这些子查询会并行执行，检索到的信息随后被汇总到一个 LLM 提示词中。这两个功能分别在 Langchain 中以多查询检索器的形式和在 Llamaindex 中以子问题查询引擎的形式实现。

1. Step-back prompting 使用 LLM 生成一个更通用的查询，以此检索到更通用或高层次的上下文，用于为原始查询提供答案。同时执行原始查询的检索，并在最终答案生成步骤中将两个上下文发送到 LLM。这是 LangChain 的一个示例实现。
2. **查询重写使用 LLM 来重新表述初始查询**，以改进检索。LangChain 和 LlamaIndex 都有实现，个人感觉LlamaIndex 解决方案在这里更强大。

### 5. 聊天引擎

关于构建一个可以多次用于单个查询的完美 RAG 系统的下一件工作是**聊天逻辑**，就像在 LLM 之前时代的经典聊天机器人中一样**考虑到对话上下文**。

这是支持后续问题、代词指代或与上一个对话上下文相关的任意用户命令所必需的。它是通过查询压缩技术解决的，将聊天上下文与用户查询一起考虑在内。

与往常一样，有几种方法可以进行上述上下文压缩——一个流行且相对简单的 ContextChatEngine，首先检索与用户查询相关的上下文，然后将其与内存缓冲区中的聊天记录一起发送到 LLM，以便 LLM 在生成下一个答案时了解上一个上下文。

更复杂的情况是 CondensePlusContextMode——在每次交互中，聊天记录和最后一条消息被压缩到一个新的查询中，然后这个查询进入索引，检索到的上下文与原始用户消息一起传递给 LLM 以生成答案。

需要注意的是，LlamaIndex 中还支持基于 OpenAI 智能体的聊天引擎，提供更灵活的聊天模式，Langchain 还支持 OpenAI 功能 API。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-331ccc95df0756e5b2e110ab3391d8c1_1440w.jpg)

### 6. 查询路由

**查询路由是 LLM 驱动的决策步骤，决定在给定用户查询的情况下下一步该做什么**——选项通常是总结、对某些数据索引执行搜索或尝试许多不同的路由，然后将它们的输出综合到一个答案中。

查询路由器还用于选择数据存储位置来处理用户查询。这些数据存储位置可能是多样的，比如传统的向量存储、图形数据库或关系型数据库，或者是不同层级的索引系统。在处理多文档存储时，通常会用到摘要索引和文档块向量索引这两种不同的索引。

**定义查询路由器包括设置它可以做出的选择。**

选择特定路由的过程是通过大语言模型调用来实现的，其结果按照预定义的格式返回，以路由查询指定的索引。如果是涉及到关联操作，这些查询还可能被发送到子链或其他智能体，如下面的**多文档智能体方案**所展示的那样。

LlamaIndex 和 LangChain 都提供了对查询路由器的支持。

### 7. 智能体（Agent）

智能体（ Langchain 和 LlamaIndex 均支持）几乎从第一个 LLM API 发布开始就已经存在——这个思路是为一个具备推理能力的 LLM 提供一套工具和一个要完成的任务。这些工具可能包括一些确定性功能，如任何代码函数或外部 API，甚至是其他智能体——这种 LLM 链接思想是 LangChain 得名的地方。

智能体本身就是一个复杂的技术，不可能在 RAG 概述中深入探讨该主题，所以我将继续基于 agent 的多文档检索案例，并简要提及 OpenAI 助手，因为它是一个相对较新的概念，在最近的 OpenAI 开发者大会上作为 GPTs 呈现，并在下文将要介绍的 RAG 系统中发挥作用。

OpenAI 助手基本上整合了开源 LLM 周边工具——聊天记录、知识存储、文档上传界面。最重要的是函数调用 API， 其提供了将自然语言转换为对外部工具或数据库查询的 API 调用的功能。

在 LlamaIndex 中，有一个 OpenAIAgent 类将这种高级逻辑与 ChatEngine 和 QueryEngine 类结合在一起，提供基于知识和上下文感知的聊天，以及在一个对话轮次中调用多个 OpenAI 函数的能力，这真正实现了智能代理行为。

来看一下多文档**智能体**的**方案**—— 这是一个非常复杂的配置，涉及到在每个文档上初始化一个Agent（OpenAIAgent），该智能体能进行文档摘要制作和传统问答机制的操作，**还有一个顶层智能体**，负责将查询分配到各个文档智能体，并综合形成最终的答案。

每个文档智能体都有两个工具：向量存储索引和摘要索引，它根据路由查询决定使用哪一个。对于顶级智能体来说，所有文档智能体都是其工具。

该方案展示了一种高级 RAG 架构，其中每个智能体都做路由许多决策。这种方法的好处是能够比较不同的解决方案或实体在不同的文档及其摘要中描述，以及经典的单个文档摘要和 QA 机制——这基本上涵盖了最常见的与文档集合聊天的用例。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-a1b7e37ffb1190e3166b8ac9e03ea005_1440w.jpg)

这种复杂配置的缺点可以通过图片发现 —— 由于需要在智能体内部的大语言模型之间进行多次往返迭代，其运行速度较慢。顺便一提，LLM 调用通常是 RAG 管道中耗时最长的操作，而搜索则是出于设计考虑而优化了速度。因此，对于大型的多文档存储，我建议考虑对此方案进行简化，以便实现扩展。

### 8. 响应合成

这是任何 RAG 管道的最后一步——根据检索的所有上下文和初始用户查询生成答案。

最简单的方法是将所有获取的上下文（高于某个相关性阈值）与查询一起连接并提供给 LLM。但是，与往常一样，还有其他更复杂的选项，涉及多个 LLM 调用，以优化检索到的上下文并生成更好的答案。

响应合成的主要方法有：

- 通过将检索到的上下文逐块发送到 LLM 来优化答案
- 概括检索到的上下文，以适应提示
- 根据不同的上下文块生成多个答案，然后将它们连接或概括起来。

## RAG性能评估

RAG 系统性能评估的多个框架，都包含了几项独立的指标，例如总体答案相关性、答案基础性、忠实度和检索到的上下文相关性。

使用真实性和答案相关性来评价生成答案的质量，并使用经典的上下文精准度和召回率来评估 RAG 方案的检索性能。

 LlamaIndex 和评估框架Truelens，他们提出了**RAG 三元组**评估模式 — 分别是对问题的**检索内容相关性**、答案的**基于性（**即大语言模型的答案在多大程度上被提供的上下文的支持）和答案对问题的**相关性**。

最关键且可控的指标是**检索内容的相关性** 

LangChain 提供了一个颇为先进的评估框架 LangSmith。在这个框架中，你不仅可以实现自定义的评估器，还能监控 RAG 管道内的运行，进而增强系统的透明度。

## RAG实际项目架构

下边是一个企业级的RAG项目的架构图。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Da6abc30d7bdf4db8874c21d8920fd42422a6ea514d5c42ea9792911bbcfd1bee)