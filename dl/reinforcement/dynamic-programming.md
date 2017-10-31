## **1. Introduction：DP（Dynamic Programming）** {#1-introductiondpdynamic-programming}

1. **定义** 
   * 解决复杂问题的一种方法。将多阶过程分解为一些列单阶段问题，逐个求解，最后结合起来以解决这类过程优化问题。
   * 同时，将这些子问题的解保存起来，如果下一次遇到了相同的子问题，则不需要重新计算子问题的解。
2. **DP主要用于解决含有以下两点特性的问题**
 
   * 最优子结构：最优解能被分解为子问题，最优应用原则
   * 覆盖子问题：子问题多次出现，子问题的解可以被缓存和重复利用
3. MDPs满足上述两条性质
 
   * 贝尔曼等式给出递归分解形式，可以切分成子问题。
   * 值函数存储和重复利用可行解，即保存了子问题的解**=&gt;**可以通过DP求解MDPs
4. 应用：用于MDP中的决策问题

针对MDP，切分的子问题就是在每个状态下应该选择哪个action。同时，这一时刻的子问题取决于上一时刻的子问题选择了哪个action。

![](/assets/mdp-dp1.png)

注意：当已知MDPs的状态转移矩阵时，环境模型就已知了，此时可看为planning问题。

## **2. Policy Evaluation** {#2-policy-evaluation}

基于当前的policy计算出每个状态的value function。

1. **Iterative Policy Evaluation，策略迭代估计**
 
   * 问题：评估一个给定的策略
   * 解决方法：迭代，贝尔曼期望备份，
     v1→v2→⋯→vπ
   * 采用同步备份

## **3. Policy Iteration** {#3-policy-iteration}

解决过程分为2步

1. **policy evaluation**  
   基于当前的policy计算每个状态的value function

2. **Policy Improvement**  
   基于当前的value function，采用贪心算法来找到当前最优秀的policy

![](/assets/mdp-dp2.png)

eg： Given a policyπ

evaluate the policy π：vπ\(s\)=E\[Rt+1+γRt+2+⋯\|St=s\]

improve the policy by acting greedy with respect to vπ：π′=greedy\(vπ\)

注意：**该策略略迭代过程总是会收敛到最优策略π∗。**  




