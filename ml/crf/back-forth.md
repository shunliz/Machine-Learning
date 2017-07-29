# 前向后向算法评估标记序列概率

---

继续讨论linear-CRF需要解决的三个问题：评估，学习和解码。这三个问题和HMM是非常类似的，本文关注于第一个问题：评估。第二个和第三个问题会在下一篇总结。

# 1. linear-CRF的三个基本问题

　　　　在[隐马尔科夫模型HMM](/ml/hmm/hmm.md)中，我们讲到了HMM的三个基本问题，而linear-CRF也有三个类似的的基本问题。不过和HMM不同，在linear-CRF中，我们对于给出的观测序列x是一直作为一个整体看待的，也就是不会拆开看$$(x_1,x_2,...)$$，因此linear-CRF的问题模型要比HMM简单一些，如果你很熟悉HMM，那么CRF的这三个问题的求解就不难了。

　　　　 linear-CRF第一个问题是评估，即给定 linear-CRF的条件概率分布P\(y\|x\), 在给定输入序列x和输出序列y时，计算条件概率$$P(y_i|x)$$和$$P(y_{i-1}，y_i|x)$$以及对应的期望. 本文接下来会详细讨论问题一。

　　　　 linear-CRF第二个问题是学习，即给定训练数据集X和Y，学习linear-CRF的模型参数$$w_k$$和条件概率$$P_w(y|x)$$，这个问题的求解比HMM的学习算法简单的多，普通的梯度下降法，拟牛顿法都可以解决。

　　　　 linear-CRF第三个问题是解码，即给定 linear-CRF的条件概率分布P\(y\|x\),和输入序列x, 计算使条件概率最大的输出序列y。类似于HMM，使用维特比算法可以很方便的解决这个问题。　

# 2.linear-CRF的前向后向概率概述

　　　　要计算条件概率$$P(y_i|x)$$和$$P(y_{i-1}，y_i|x)$$，我们也可以使用和HMM类似的方法，使用前向后向算法来完成。首先我们来看前向概率的计算。

　　　　我们定义$$\alpha_i(y_i|x)$$表示序列位置i的标记是$$y_i$$时，在位置i之前的部分标记序列的非规范化概率。之所以是非规范化概率是因为我们不想加入一个不影响结果计算的规范化因子Z\(x\)在分母里面。

　　　　我们定义了下式：$$M_i(y_{i-1},y_i |x) = exp(\sum\limits_{k=1}^Kw_kf_k(y_{i-1},y_i, x,i))$$

　　　　这个式子定义了在给定$$y_{i-1}$$时，从$$y_{i-1}$$转移到$$y_i$$的非规范化概率。

　　　　这样，我们很容易得到序列位置i+1的标记是$$y_{i+1}$$时，在位置i+1之前的部分标记序列的非规范化概率$$\alpha_{i+1}(y_{i+1}|x)$$的递推公式：$$\alpha_{i+1}(y_{i+1}|x) = \alpha_i(y_i|x)M_{i+1}(y_{i+1},y_i|x)$$

　　　　在起点处，我们定义：$$\alpha_0(y_0|x)= \begin{cases} 1 & {y_0 =start}\\ 0 & {else} \end{cases}$$

　　　　假设我们可能的标记总数是m, 则$$y_i$$的取值就有m个，我们用$$\alpha_i(x)$$表示这m个值组成的前向向量如下：$$\alpha_i(x) = (\alpha_i(y_i=1|x), \alpha_i(y_i=2|x), ... \alpha_i(y_i=m|x))^T$$

　　　　同时用矩阵$$M_i(x)$$表示由$$M_i(y_{i-1},y_i |x)$$形成的$$m \times m$$阶矩阵：$$M_i(x) = \Big[ M_i(y_{i-1},y_i |x)\Big]$$

　　　　这样递推公式可以用矩阵乘积表示：$$\alpha_{i+1}^T(x) = \alpha_i^T(x)M_i(x)$$

　　　　同样的。我们定义$$\beta_i(y_i|x)$$表示序列位置i的标记是$$y_i$$时，在位置i之后的从i+1到n的部分标记序列的非规范化概率。

　　　　这样，我们很容易得到序列位置i+1的标记是$$y_{i+1}$$时，在位置i之后的部分标记序列的非规范化概率$$\beta_{i}(y_{i}|x)$$的递推公式：$$\beta_{i}(y_{i}|x) = M_{i}(y_i,y_{i+1}|x)\beta_{i+1}(y_{i+1}|x)$$

　　　　在终点处，我们定义：$$\beta_{n+1}(y_{n+1}|x)= \begin{cases} 1 & {y_{n+1} =stop}\\ 0 & {else} \end{cases}$$

　　　　如果用向量表示，则有：$$\beta_i(x) = M_i(x)\beta_{i+1}(x)$$

　　　　由于规范化因子Z\(x\)的表达式是：$$Z(x) = \sum\limits_{c=1}^m\alpha_{n}(y_c|x) = \sum\limits_{c=1}^m\beta_{1}(y_c|x)$$

　　　　也可以用向量来表示$$Z(x):Z(x) = \alpha_{n}^T(x) \bullet \mathbf{1} = \mathbf{1}^T \bullet \beta_{1}(x)$$

　　　　其中，$$\mathbf{1}$$是m维全1向量。

# 3. linear-CRF的前向后向概率计算

　　　　有了前向后向概率的定义和计算方法，我们就很容易计算序列位置i的标记是$$y_i$$时的条件概率$$P(y_i|x)$$:$$P(y_i|x) = \frac{\alpha_i^T(y_i|x)\beta_i(y_i|x)}{Z(x)} = \frac{\alpha_i^T(y_i|x)\beta_i(y_i|x)}{ \alpha_{n}^T(x) \bullet \mathbf{1}}$$

　　　　也容易计算序列位置i的标记是$$y_i$$，位置i-1的标记是$$y_{i-1}$$时的条件概率$$P(y_{i-1},y_i|x)$$:$$P(y_{i-1},y_i|x) = \frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)} = \frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{ \alpha_{n}^T(x) \bullet \mathbf{1}}$$

# 4. linear-CRF的期望计算

　　　　有了上一节计算的条件概率，我们也可以很方便的计算联合分布P\(x,y\)与条件分布P\(y\|x\)的期望。

　　　　特征函数$$f_k(x,y)$$关于条件分布P\(y\|x\)的期望表达式是：$$\begin{align} E_{P(y|x)}[f_k]  & = E_{P(y|x)}[f_k(y,x)] \\ & = \sum\limits_{i=1}^{n+1} \sum\limits_{y_{i-1}\;\;y_i}P(y_{i-1},y_i|x)f_k(y_{i-1},y_i,x, i) \\ & =  \sum\limits_{i=1}^{n+1} \sum\limits_{y_{i-1}\;\;y_i}f_k(y_{i-1},y_i,x, i)  \frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{ \alpha_{n}^T(x) \bullet \mathbf{1}} \end{align}$$

　　　　同样可以计算联合分布P\(x,y\)的期望：$$\begin{align} E_{P(x,y)}[f_k]  & = \sum\limits_{x,y}P(x,y) \sum\limits_{i=1}^{n+1}f_k(y_{i-1},y_i,x, i) \\& =  \sum\limits_{x}\overline{P}(x) \sum\limits_{y}P(y|x) \sum\limits_{i=1}^{n+1}f_k(y_{i-1},y_i,x, i) \\& =  \sum\limits_{x}\overline{P}(x)\sum\limits_{i=1}^{n+1} \sum\limits_{y_{i-1}\;\;y_i}f_k(y_{i-1},y_i,x, i)  \frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{ \alpha_{n}^T(x) \bullet \mathbf{1}}    \end{align}$$

　　　　假设一共有K个特征函数，则k=1,2,...K

# 5. linear-CRF前向后向算法总结

　　　　以上就是linear-CRF的前向后向算法，个人觉得比HMM简单的多，因此大家如果理解了HMM的前向后向算法，这一篇是很容易理解的。

　　　　注意到我们上面的非规范化概率$$M_{i+1}(y_{i+1},y_i|x)$$起的作用和HMM中的隐藏状态转移概率很像。但是这儿的概率是非规范化的，也就是不强制要求所有的状态的概率和为1。而HMM中的隐藏状态转移概率也规范化的。从这一点看，linear-CRF对序列状态转移的处理要比HMM灵活。

