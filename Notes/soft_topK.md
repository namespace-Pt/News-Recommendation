# Soft Top-K Operator
## OT problem
- $P(\{a_i\}) = \mu_i, P(\{b_j\}) = v_j$代表$A,B$分布
<br>
- 每一个在$X$集合中的元素都得被搬走(搬到$Y$中), 即$$\tau 1_m = \mu$$

- 搬到$Y$中各个entry的元素总和应该符合原分布, 即
  $$\tau^T 1_n = v$$

## Top-K formulation
$$\mu = 1_n / n, v = (\frac{k}{n},\frac{n-k}{n})^T$$
将输入的标量$x_i \in \mathcal{X}$搬到$\mathcal{Y}=\{0,1\}$中, 满足$\mathcal{Y}$的分布是$P(y=0) = \frac{k}{n}, P(y=1) = \frac{n-k}{n}$, 找这个最优OT问题的解
- 其中$y$是$x\in \mathcal{X}$根据$\tau$方案搬运后的结果
  - $y=0$代表其$y(x_i)$的$x_i$是**前K小的**
  - $y=1$代表其$y(x_i)$的$x_i$不是**前K小的**