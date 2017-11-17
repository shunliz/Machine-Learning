7.1n-step TD Prediction

The methods that use**n-step backups are still TD methods**because they**still change an earlier estimate based on how it differs from a later estimate**.

n-step return：![](/assets/multi-steps-bootstraping1.png)If t+n≥T\(if then-step return extends to or beyond termination\), then all the missing terms are taken as zero

==》这个很容易理解，最后n步之内，还剩多少步就令return等于所有剩余步数的reward和。相应的，前n-1步也是没有任何更新过程，Note that no changes at all are made during the firstn-1 steps of each episode.。![](/assets/multi-bootstraping2.png)n-step return的error往往更小：![](/assets/multi-bt1.png)Methods that involve an intermediate amount of bootstrapping are important because they will typically perform better than either extreme.

## 7.2 n-step SARSA 

将V改成Q即可，整个流程基本一致：

![](/assets/multi-bt2.png)



