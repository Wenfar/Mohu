import numpy as np
from scipy.stats import spearmanr

"""
熵权法+模糊综合评价
"""


def normalize(X, indicator_type):
    """
    指标正向化 + Min-Max 归一化。
    负向指标先翻转（max - x），再统一归一化到 [0,1]。
    兼容整数(1/-1)和字符串('positive'/'negative')两种传入方式。
    """
    X = np.asarray(X, dtype=float)
    X_norm = X.copy()
    for j, t in enumerate(indicator_type):
        is_negative = (t in ['negative', 'Negative', '-', -1]) or \
                      (isinstance(t, (int, float)) and float(t) < 0)
        if is_negative:
            X_norm[:, j] = X[:, j].max() - X[:, j]
    min_val = X_norm.min(axis=0)
    max_val = X_norm.max(axis=0)
    X_norm = (X_norm - min_val) / (max_val - min_val + 1e-12)
    return X_norm


# def entropy_weight(X_norm):
#     """计算熵权。X_norm 为已归一化矩阵。"""
#     m, n = X_norm.shape
#     P = X_norm / (X_norm.sum(axis=0) + 1e-12)
#     E = -np.sum(P * np.log(P + 1e-12), axis=0) / np.log(m)
#     d = 1 - E
#     w = d / np.sum(d)
#     return w


def entropy_weight(X_norm):
    """计算熵权。X_norm 为已归一化矩阵。"""
    m, n = X_norm.shape
    
    # 1. 计算每列的和
    col_sum = X_norm.sum(axis=0)
    
    # 2. 计算概率矩阵 P
    # 避免除以 0：如果某列和为 0（即原数据全为常数，归一化后全为 0），则该列 P 设为均匀分布 1/m
    # 这样计算出的熵 E 将为 1，差异系数 d 将为 0，从而权重为 0。
    P = np.zeros_like(X_norm)
    for j in range(n):
        if col_sum[j] > 1e-12:
            P[:, j] = X_norm[:, j] / col_sum[j]
        else:
            P[:, j] = 1.0 / m
            
    # 3. 计算熵值 E
    # 使用 mask 避免 log(0)
    mask = P > 1e-12
    ln_P = np.zeros_like(P)
    ln_P[mask] = np.log(P[mask])
    
    E = -np.sum(P * ln_P, axis=0) / np.log(m)
    
    # 4. 计算差异系数 d
    d = 1 - E
    
    # 5.如果某列原始数据无波动（归一化后极差为 0），强制 d=0
    col_range = X_norm.max(axis=0) - X_norm.min(axis=0)
    d[col_range < 1e-12] = 0.0
    
    # 6. 归一化得到权重
    sum_d = np.sum(d)
    if sum_d < 1e-12:
        return np.ones(n) / n
        
    return d / sum_d


def triangular_membership(x, a, b, c):
    """三角隶属度函数。"""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a + 1e-12)
    else:
        return (c - x) / (c - b + 1e-12)


# 二、语义置信度相关函数

def _indicator_type_to_sign(indicator_type):
    """将指标方向统一映射为 +1/-1。"""
    sign = []
    for it in indicator_type:
        if it in ["positive", "Positive", "+", 1, +1]:
            sign.append(1.0)
        elif it in ["negative", "Negative", "-", -1]:
            sign.append(-1.0)
        else:
            v = float(it)
            sign.append(1.0 if v >= 0 else -1.0)
    return np.asarray(sign, dtype=float)


def semantic_confidence(
    beta,
    indicator_type,
    eps=1e-6,
    conf_weak=0.5,
    conf_conflict=0.3,
    conf_agree=1.0
):
    """
    根据模糊回归系数与指标语义方向的一致性，
    输出置信度向量 conf ∈ (0, 1]。
    """
    beta = np.asarray(beta, dtype=float)
    indicator_sign = _indicator_type_to_sign(indicator_type)

    if len(beta) != len(indicator_sign):
        raise ValueError("beta 与 indicator_type 维度不一致")

    conf = np.ones(len(beta), dtype=float)
    for j in range(len(beta)):
        if abs(beta[j]) < eps:
            conf[j] = conf_weak
        elif np.sign(beta[j]) * indicator_sign[j] < 0:
            conf[j] = conf_conflict
        else:
            conf[j] = conf_agree
    return conf


def evaluate_model(model, X, y_true):
    """评估模型：计算 Spearman、STD、CV、Range。"""
    results = model.evaluate(X)
    scores = np.array([r["score"] for r in results])
    spearman, p_value = spearmanr(scores, y_true)
    std = np.std(scores)
    cv = std / (np.mean(scores) + 1e-12)
    value_range = np.max(scores) - np.min(scores)
    return {
        "spearman": spearman,
        "p_value": p_value,
        "std": std,
        "cv": cv,
        "range": value_range
    }


def fit_conf_with_grid_search(
    model,
    beta,
    indicator_type,
    X,
    y_true,
    conflict_values=None,
    verbose=True
):
    """
    网格搜索最优 conf_conflict 参数。
    返回：best_conf, best_param, all_results
    """
    if model.weights is None:
        model.fit(X)

    if conflict_values is None:
        conflict_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    best_score = -np.inf
    best_param = None
    best_conf  = None
    all_results = []

    if verbose:
        print("\n开始网格搜索（conf_conflict）")

    for val in conflict_values:
        conf = semantic_confidence(beta, indicator_type, conf_conflict=val)
        model.set_semantic_confidence(conf)
        metrics = evaluate_model(model, X, y_true)
        result  = {"conflict": val, **metrics}
        all_results.append(result)

        if verbose:
            print(f"\nconf_conflict = {val}")
            print(f"  Spearman = {metrics['spearman']:.4f}")
            print(f"  STD      = {metrics['std']:.4f}")
            print(f"  CV       = {metrics['cv']:.4f}")
            print(f"  Range    = {metrics['range']:.4f}")

        if metrics["spearman"] > best_score:
            best_score = metrics["spearman"]
            best_param = val
            best_conf  = conf

    if verbose:
        print("\n 最优参数")
        print(f"最佳 conf_conflict = {best_param}")
        print(f"最佳 Spearman = {best_score:.4f}")

    model.set_semantic_confidence(best_conf)
    return best_conf, best_param, all_results


# 三、模糊综合评判器

class FuzzyComprehensiveEvaluator:
    def __init__(self, levels, grade_labels):
        self.levels      = levels
        self.grade_labels = grade_labels
        self.n_levels    = len(levels)

    def triangular_membership(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a + 1e-12)
        elif b <= x < c:
            return (c - x) / (c - b + 1e-12)
        else:
            return 0.0

    def build_relation_matrix(self, x_row):
        """x_row: 单个样本的指标值 (n,)，返回 R (n, k)"""
        m = len(x_row)
        R = np.zeros((m, self.n_levels))
        for i in range(m):
            for j, (a, b, c) in enumerate(self.levels):
                R[i, j] = self.triangular_membership(x_row[i], a, b, c)
        return R

    def evaluate(self, weights, R):
        """B = W · R"""
        return np.dot(weights, R)

    def grade(self, B):
        return self.grade_labels[np.argmax(B)]

    def score(self, B):
        """将等级向量映射为 [0,1] 综合得分"""
        # level_scores = np.linspace(0.1, 1.0, len(B))
        level_scores = np.linspace(1.0, 0.1, len(B))
        return np.dot(B, level_scores)


# 四、熵权 + 模糊综合评判 一体化模型



class EntropyFCEModel:
    def __init__(self, indicator_type, levels=None, grade_labels=None,
                 conf_membership_alpha=2.0,
                 conf_reverse_threshold=0.3):
        self.indicator_type = indicator_type
        self.semantic_conf = None
        self.conf_membership_alpha = conf_membership_alpha
        self.conf_reverse_threshold = conf_reverse_threshold

        if levels is None:
            # self.levels = [
            #     (0.0, 0.0, 0.25),
            #     (0.0, 0.25, 0.5),
            #     (0.25, 0.5, 0.75),
            #     (0.5, 0.75, 1.0),
            #     (0.75, 1.0, 1.0)
            # ]

            self.levels = [
                (0.75, 1.0, 1.0),   # 索引0 → "很高"
                (0.5, 0.75, 1.0),   # 索引1 → "较高"
                (0.25, 0.5, 0.75),  # 索引2 → "中等"
                (0.0, 0.25, 0.5),   # 索引3 → "较低"
                (0.0, 0.0,  0.25)   # 索引4 → "很低"
            ]
        else:
            self.levels = levels

        if grade_labels is None:
            # self.grade_labels = ['很低', '较低', '中等', '较高', '很高']
            self.grade_labels = ['很高', '较高', '中等', '较低', '很低']
        else:
            self.grade_labels = grade_labels

        # 新增：等级标签与等级区间一致性校验
        if len(self.grade_labels) != len(self.levels):
            raise ValueError(
                f"grade_labels 数量({len(self.grade_labels)}) "
                f"与 levels 数量({len(self.levels)}) 不一致"
            )

        self.fce = FuzzyComprehensiveEvaluator(self.levels, self.grade_labels)
        self.weights = None

    def set_semantic_confidence(self, conf):
        self.semantic_conf = conf

    def fit(self, X):
        """计算熵权（唯一权重来源）"""
        X_norm       = normalize(X, self.indicator_type)
        self.weights = entropy_weight(X_norm)
        return self.weights

    def evaluate(self, X):
        """混合策略：同时调节权重和隶属度"""
        if self.weights is None:
            raise ValueError("模型尚未 fit，请先计算熵权")

        X_norm = normalize(X, self.indicator_type)

        # 步骤1：调节权重
        adjusted_weights = self.weights.copy()
        if self.semantic_conf is not None:
            adjusted_weights = self.weights * self.semantic_conf
            adjusted_weights = adjusted_weights / (adjusted_weights.sum() + 1e-12)
            print(f"\n[Conf调节] 权重调整效果:")
            print(f"  原始权重: {self.weights[:5]}...")
            print(f"  调整后权重: {adjusted_weights[:5]}...")

        results = []
        for i in range(X_norm.shape[0]):
            R = self.fce.build_relation_matrix(X_norm[i])

            # 步骤2：调节隶属度
            if self.semantic_conf is not None:
                for j in range(R.shape[0]):
                    conf_j = self.semantic_conf[j]
                    if conf_j <= self.conf_reverse_threshold:
                        R[j, :] = R[j, ::-1]
                    R[j, :] = R[j, :] * conf_j
                    R[j, :] = R[j, :] / (R[j, :].sum() + 1e-12)

            B = self.fce.evaluate(adjusted_weights, R)
            results.append({
                "membership": B,
                "grade":      self.fce.grade(B),
                "score":      self.fce.score(B)
            })

        return results
