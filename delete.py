

初始化隶属度向量 u⁽⁰⁾ ← 1ₘ
2.  由普通最小二乘得初始系数 β⁽⁰⁾ ← (XᵀX)⁻¹Xᵀy
3.  for k = 0, 1, …, K − 1 do
4.      for i = 1, 2, …, m do
5.          ŷᵢ⁽ᵏ⁾ ← β₀⁽ᵏ⁾ + ∑ⱼ₌₁ⁿ βⱼ⁽ᵏ⁾xᵢⱼ
6.          uᵢ⁽ᵏ⁺¹⁾ ← exp(−(yᵢ − ŷᵢ⁽ᵏ⁾)²)
7.      end for
8.  构造对角权重矩阵 U⁽ᵏ⁺¹⁾ ← diag(u₁⁽ᵏ⁺¹⁾, …, uₘ⁽ᵏ⁺¹⁾)
9.  βᵣₑₛₜ⁽ᵏ⁺¹⁾ ← (XᵀU⁽ᵏ⁺¹⁾X + λI)⁻¹XᵀU⁽ᵏ⁺¹⁾y
10. x̄ⱼ ← ∑ᵢ uᵢ⁽ᵏ⁺¹⁾xᵢⱼ / ∑ᵢ uᵢ⁽ᵏ⁺¹⁾， ȳ ← ∑ᵢ uᵢ⁽ᵏ⁺¹⁾yᵢ / ∑ᵢ uᵢ⁽ᵏ⁺¹⁾
11. β₀⁽ᵏ⁺¹⁾ ← ȳ − (βᵣₑₛₜ⁽ᵏ⁺¹⁾)ᵀx̄
12. if ‖β⁽ᵏ⁺¹⁾ − β⁽ᵏ⁾‖₂ < ε then
13.     break
14. end if
15. end for
16. return β̂ ← β⁽ᵏ⁺¹⁾


算法2 熵权法指标权重计算算法

输入：原始指标矩阵 X，维度为 (m, n)；指标方向向量 indicator_type，维度为 (n, )
输出：指标权重向量 w，维度为 (n, )

1.  for j = 1, 2, …, n do
2.      if indicator_type_j < 0 then
3.          x_ij ← max_i(x_ij) - x_ij, ∀i (负向指标正向化)
4.      end if
5.  end for

6.  for j = 1, 2, …, n do
7.      x^*_ij ← (x_ij - min_i x_ij) / (max_i x_ij - min_i x_ij + ε), ε = 10^-12
8.  end for

9.  for j = 1, 2, …, n do
10.     p_ij ← x^*_ij / (∑_{i=1}^m x^*_ij + ε)
11.     E_j ← -1/ln m * ∑_{i=1}^m p_ij ln(p_ij + ε)
12.     d_j ← 1 - E_j
13. end for

14. w_j ← d_j / ∑_{j=1}^n d_j
15. return w

算法3 模糊综合评判算法

输入：归一化指标矩阵 X*，维度为 (m, n)；权重向量 w，维度为 (n, )；等级三角参数集 {(a_q, b_q, c_q)}_q=1^k；等级标签集 V = {v_1, ..., v_k}；等级得分映射 l = linspace(0.1, 1.0, k)
输出：各样本综合得分 {s_i}_i=1^m，评价等级 {g_i}_i=1^m

1.  若权重未计算，调用算法2：w ← Algorithm2(X_eval)
2.  初始化：s* ← -∞，c* ← None，conf* ← None
3.  for c ∈ C do
4.      for j = 1,2,...,n do
5.          if |β_j| < ε then conf_j ← 0.5
6.          else if sign(β_j) ≠ sign(indicator_type_j) then conf_j ← c
7.          else conf_j ← 1.0
8.      end for
9.      w* ← (w ⊙ conf) / ∑_j w_j · conf_j
10.     for i = 1,2,...,m do
11.         由算法3第2—4行计算 R_i
12.         for j = 1,2,...,n do
13.             if conf_j ≤ c then R_{i,j,:} ← reverse(R_{i,j,:})
14.             R_{i,j,:} ← conf_j · R_{i,j,:} / ∑_q R_{i,j,q}
15.         end for
16.         B_i ← (w*)^T R_i，    s_i ← l^T B_i
17.     end for
18.     s ← Spearman(s, y)
19.     if s > s* then
20.         s* ← s，c* ← c，conf* ← conf
21.     end if
22. end for
23. 更新模型：τ ← c*，conf ← conf*
24. return c*，conf*