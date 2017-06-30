# 分解机\(Factorization Machines\)推荐算法原理

---

　对于分解机\(Factorization Machines，FM\)推荐算法原理，本来想自己单独写一篇的。但是看到peghoty写的[FM](http://www.cnblogs.com/pinard/p/blog.csdn.net/itplus/article/details/40534923)不光简单易懂，而且排版也非常好，因此转载过来，自己就不再单独写FM了。

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206135921885-935760124.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206135946041-1294500667.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206135958885-1040242136.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140117213-967734026.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140158151-1613362741.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140209760-1698072242.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140224166-2002517422.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140238151-1914591474.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140249994-546797094.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140313057-979438300.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140331994-286872150.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140348463-1121558265.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140401307-1728849713.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140418838-1573333021.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140431854-886558581.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140442744-2079852926.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140454010-1393532205.png)

![](http://images2015.cnblogs.com/blog/1042406/201702/1042406-20170206140527354-1280787124.png)

分类:

[0081. 机器学习](http://www.cnblogs.com/pinard/category/894692.html)

标签:

[推荐算法](http://www.cnblogs.com/pinard/tag/%E6%8E%A8%E8%8D%90%E7%AE%97%E6%B3%95/)

[好文要顶](javascript:void%280%29;)

[关注我](javascript:void%280%29;)

[收藏该文](javascript:void%280%29;)

[![](http://common.cnblogs.com/images/icon_weibo_24.png)](javascript:void%280%29;)

[![](http://common.cnblogs.com/images/wechat.png)](javascript:void%280%29;)

[![](http://pic.cnblogs.com/face/1042406/20161104225142.png)](http://home.cnblogs.com/u/pinard/)

[刘建平Pinard](http://home.cnblogs.com/u/pinard/)

  


[关注 - 12](http://home.cnblogs.com/u/pinard/followees)

  


[粉丝 - 249](http://home.cnblogs.com/u/pinard/followers)

[+加关注](javascript:void%280%29;)

0

0

[«](http://www.cnblogs.com/pinard/p/6364932.html)

上一篇：

[用Spark学习矩阵分解推荐算法](http://www.cnblogs.com/pinard/p/6364932.html)

  


[»](http://www.cnblogs.com/pinard/p/6418668.html)

下一篇：

[深度神经网络（DNN）模型与前向传播算法](http://www.cnblogs.com/pinard/p/6418668.html)

  


posted @

2017-02-06 14:06

[刘建平Pinard](http://www.cnblogs.com/pinard/)

阅读\(

822

\) 评论\(

2

\)

[编辑](https://i.cnblogs.com/EditPosts.aspx?postid=6370127)

[收藏](http://www.cnblogs.com/pinard/p/6370127.html#)



  


评论列表

[\#1楼](http://www.cnblogs.com/pinard/p/6370127.html#3621674)



2017-02-19 01:57

[xulu1352](http://www.cnblogs.com/xulu1352/)



有好段时间没来了，看了大神最近更新几篇，大神加油\(ง •̀\_•́\)ง

[支持\(0\)](javascript:void%280%29;)

[反对\(0\)](javascript:void%280%29;)

[\#2楼](http://www.cnblogs.com/pinard/p/6370127.html#3668813)



2017-04-15 18:51

[心冷2080](http://www.cnblogs.com/zlszhonglongshen/)



一直看大神的文章，收益颇多，大神在广州吗？我也在广州，有空可以聊聊吗？大神方不方便留下联系方式，我的qq，593956670

[支持\(0\)](javascript:void%280%29;)

[反对\(0\)](javascript:void%280%29;)



[刷新评论](javascript:void%280%29;)

[刷新页面](http://www.cnblogs.com/pinard/p/6370127.html#)

[返回顶部](http://www.cnblogs.com/pinard/p/6370127.html#top)

注册用户登录后才能发表评论，请

[登录](javascript:void%280%29;)

或

[注册](javascript:void%280%29;)

，

[访问](http://www.cnblogs.com/)

网站首页。

[【推荐】50万行VC++源码: 大型组态工控、电力仿真CAD与GIS源码库](http://www.ucancode.com/index.htm)

  


[【阿里云】云计算科技红利邀您免费体验！云服务器、云数据库等35+产品，6个月免费使用！](http://click.aliyun.com/m/21970/)

  


[【免费】从零开始学编程，开发者专属实验平台免费实践！](https://cloud.tencent.com/developer/labs?fromSource=gwzcw.241259.241259.241259)

  


[【推荐】又拍云年中大促！现在注册，充值最高送4800元](https://www.upyun.com/618?utm_source=cnblogs&utm_medium=referral&utm_content=618)

  


[![](https://images2015.cnblogs.com/news/24442/201706/24442-20170629184920789-1740545267.png "Udacity\_前端\_06290712")](https://cn.udacity.com/course/front-end-web-developer-nanodegree--nd001-cn-advanced/?utm_source=cnblogs&utm_medium=referral&utm_campaign=FEND04)

**最新IT新闻**

:

  


·

[马斯克的Boring隧道最新进展：已完成竖井挖掘](http://news.cnblogs.com/n/572852/)

  


·

[王者荣耀近6成玩家是小学生？月入30亿的腾讯并不荣耀](http://news.cnblogs.com/n/572851/)

  


·

[腾讯王卡福利：1元500MB全国流量 还可跨省配送](http://news.cnblogs.com/n/572850/)

  


·

[华为推出三款新MateBook笔记本电脑 今日开始在美预订](http://news.cnblogs.com/n/572847/)

  


·

[秘招：高清显示微信图片 只需改几个字符](http://news.cnblogs.com/n/572849/)

  


»

[更多新闻...](http://news.cnblogs.com/)

[![](https://images2015.cnblogs.com/news/24442/201706/24442-20170615095125603-13625507.png "美团云")](https://www.mtyun.com/activity-anniversary?site=cnblogs&campaign=20170601sales)

**最新知识库文章**

:

  


·

[小printf的故事：什么是真正的程序员？](http://kb.cnblogs.com/page/570194/)

  


·

[程序员的工作、学习与绩效](http://kb.cnblogs.com/page/569992/)

  


·

[软件开发为什么很难](http://kb.cnblogs.com/page/569056/)

  


·

[唱吧DevOps的落地，微服务CI/CD的范本技术解读](http://kb.cnblogs.com/page/565901/)

  


·

[程序员，如何从平庸走向理想？](http://kb.cnblogs.com/page/566523/)

  


»

[更多知识库文章...](http://kb.cnblogs.com/)

### 公告

★珠江追梦，饮岭南茶，恋鄂北家★

昵称：

[刘建平Pinard](http://home.cnblogs.com/u/pinard/)

  


园龄：

[8个月](http://home.cnblogs.com/u/pinard/)

  


粉丝：

[249](http://home.cnblogs.com/u/pinard/followers/)

  


关注：

[12](http://home.cnblogs.com/u/pinard/followees/)

[+加关注](javascript:void%280%29;)

|  |  |  |  |  |  | [&lt;](javascript:void%280%29;)2017年6月[&gt;](javascript:void%280%29;) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 日 | 一 | 二 | 三 | 四 | 五 | 六 |
| 28 | 29 | 30 | 31 | 1 | 2 | 3 |
| 4 | 5 | [6](http://www.cnblogs.com/pinard/archive/2017/06/06.html) | 7 | [8](http://www.cnblogs.com/pinard/archive/2017/06/08.html) | 9 | [10](http://www.cnblogs.com/pinard/archive/2017/06/10.html) |
| 11 | [12](http://www.cnblogs.com/pinard/archive/2017/06/12.html) | [13](http://www.cnblogs.com/pinard/archive/2017/06/13.html) | 14 | 15 | 16 | 17 |
| 18 | [19](http://www.cnblogs.com/pinard/archive/2017/06/19.html) | 20 | 21 | [22](http://www.cnblogs.com/pinard/archive/2017/06/22.html) | [23](http://www.cnblogs.com/pinard/archive/2017/06/23.html) | 24 |
| 25 | 26 | 27 | 28 | 29 | 30 | 1 |
| 2 | 3 | 4 | 5 | 6 | 7 | 8 |

### 常用链接

* [我的随笔](http://www.cnblogs.com/pinard/p/)
* [我的评论](http://www.cnblogs.com/pinard/MyComments.html)
* [我的参与](http://www.cnblogs.com/pinard/OtherPosts.html)
* [最新评论](http://www.cnblogs.com/pinard/RecentComments.html)
* [我的标签](http://www.cnblogs.com/pinard/tag/)

### 随笔分类\(97\)

* [0040. 数学统计学\(4\)](http://www.cnblogs.com/pinard/category/894690.html)
* [0081. 机器学习\(62\)](http://www.cnblogs.com/pinard/category/894692.html)
* [0082. 深度学习\(10\)](http://www.cnblogs.com/pinard/category/894694.html)
* [0083. 自然语言处理\(19\)](http://www.cnblogs.com/pinard/category/894695.html)
* [0121. 大数据挖掘\(1\)](http://www.cnblogs.com/pinard/category/894700.html)
* [0122. 大数据平台\(1\)](http://www.cnblogs.com/pinard/category/894697.html)
* [0123. 大数据可视化](http://www.cnblogs.com/pinard/category/894698.html)

### 随笔档案\(97\)

* [2017年6月 \(8\)](http://www.cnblogs.com/pinard/archive/2017/06.html)
* [2017年5月 \(7\)](http://www.cnblogs.com/pinard/archive/2017/05.html)
* [2017年4月 \(5\)](http://www.cnblogs.com/pinard/archive/2017/04.html)
* [2017年3月 \(10\)](http://www.cnblogs.com/pinard/archive/2017/03.html)
* [2017年2月 \(7\)](http://www.cnblogs.com/pinard/archive/2017/02.html)
* [2017年1月 \(13\)](http://www.cnblogs.com/pinard/archive/2017/01.html)
* [2016年12月 \(17\)](http://www.cnblogs.com/pinard/archive/2016/12.html)
* [2016年11月 \(22\)](http://www.cnblogs.com/pinard/archive/2016/11.html)
* [2016年10月 \(8\)](http://www.cnblogs.com/pinard/archive/2016/10.html)

### 常去的机器学习网站

* [52 NLP](http://www.52nlp.cn/)
* [Analytics Vidhya](https://www.analyticsvidhya.com/)
* [机器学习库](https://github.com/josephmisiti/awesome-machine-learning)
* [机器学习路线图](https://github.com/hangtwenty/dive-into-machine-learning)
* [深度学习进阶书](http://www.deeplearningbook.org/)
* [深度学习入门书](http://neuralnetworksanddeeplearning.com/)

### 积分与排名

* 积分 - 140707
* 排名 - 1654

### 阅读排行榜

* [1. scikit-learn决策树算法类库使用小结\(6369\)](http://www.cnblogs.com/pinard/p/6056319.html)
* [2. 梯度提升树\(GBDT\)原理小结\(6065\)](http://www.cnblogs.com/pinard/p/6140514.html)
* [3. scikit-learn随机森林调参小结\(5525\)](http://www.cnblogs.com/pinard/p/6160412.html)
* [4. 用scikit-learn和pandas学习线性回归\(4459\)](http://www.cnblogs.com/pinard/p/6016029.html)
* [5. 用scikit-learn学习K-Means聚类\(4412\)](http://www.cnblogs.com/pinard/p/6169370.html)

### 评论排行榜

* [1. 集成学习之Adaboost算法原理小结\(25\)](http://www.cnblogs.com/pinard/p/6133937.html)
* [2. 线性回归原理小结\(22\)](http://www.cnblogs.com/pinard/p/6004041.html)
* [3. scikit-learn随机森林调参小结\(18\)](http://www.cnblogs.com/pinard/p/6160412.html)
* [4. 文本主题模型之LDA\(二\) LDA求解之Gibbs采样算法\(17\)](http://www.cnblogs.com/pinard/p/6867828.html)
* [5. scikit-learn决策树算法类库使用小结\(14\)](http://www.cnblogs.com/pinard/p/6056319.html)

### 推荐排行榜

* [1. 机器学习研究与开发平台的选择\(6\)](http://www.cnblogs.com/pinard/p/6007200.html)
* [2. 支持向量机原理\(五\)线性支持回归\(5\)](http://www.cnblogs.com/pinard/p/6113120.html)
* [3. scikit-learn决策树算法类库使用小结\(5\)](http://www.cnblogs.com/pinard/p/6056319.html)
* [4. 协同过滤推荐算法总结\(5\)](http://www.cnblogs.com/pinard/p/6349233.html)
* 



