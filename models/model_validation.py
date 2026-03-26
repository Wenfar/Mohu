"""
该文件用来给模糊回归模型的效果进行验证
不关心数据来源，不要写死列名和维度
只假设mohu_model有fit()和predict()方法
ols_modek是sklearn风格
"""

# model_validation.py

import numpy as np
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import HuberRegressor, Ridge
from scipy.stats import spearmanr

def add_outliers(X, y, ratio=0.2, scale=5, random_state=42):
    np.random.seed(random_state)
    X_new = X.copy()
    y_new = y.copy()

    n_outliers = int(len(y) * ratio)
    idx = np.random.choice(len(y), n_outliers, replace=False)

    y_new[idx] += scale * np.std(y)

    return X_new, y_new

#TOPSIS对比
def topsis(X, weights=None):
    X = np.array(X, dtype=float)
    # 标准化
    X_norm = X / np.sqrt((X ** 2).sum(axis=0))
    # 权重
    if weights is None:
        weights = np.ones(X.shape[1]) / X.shape[1]
    X_weighted = X_norm * weights
    # 理想解
    ideal_best = np.max(X_weighted, axis=0)
    ideal_worst = np.min(X_weighted, axis=0)
    # 距离
    dist_best = np.sqrt(((X_weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((X_weighted - ideal_worst) ** 2).sum(axis=1))
    # 得分
    score = dist_worst / (dist_best + dist_worst)
    return score

def validate_fuzzy_model(X, y, fuzzy_model, ols_model):
    """OLS / Fuzzy / Huber / Ridge / TOPSIS）"""

    #原始数据Fuzzy
    fuzzy_model.setdatax(X)
    fuzzy_model.setdatay(y)
    beta_fuzzy = fuzzy_model.fit()
    y_pred_fuzzy = fuzzy_model.predict(X)

    # OLS
    ols_model.fit(X, y)
    y_pred_ols = ols_model.predict(X)
    beta_ols = np.concatenate(([ols_model.intercept_], ols_model.coef_))

    # Huber（稳健回归）
    huber = HuberRegressor().fit(X, y)
    y_pred_huber = huber.predict(X)
    beta_huber = np.concatenate(([huber.intercept_], huber.coef_))

    # Ridge
    ridge = Ridge(alpha=1.0).fit(X, y)
    y_pred_ridge = ridge.predict(X)
    beta_ridge = np.concatenate(([ridge.intercept_], ridge.coef_))

    #  TOPSIS （只做排序）
    topsis_score = topsis(X)

  
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

  
    results = {
        "RMSE_fuzzy": rmse(y, y_pred_fuzzy),
        "RMSE_OLS": rmse(y, y_pred_ols),
        "RMSE_Huber": rmse(y, y_pred_huber),
        "RMSE_Ridge": rmse(y, y_pred_ridge),

        "R2_fuzzy": r2_score(y, y_pred_fuzzy),
        "R2_OLS": r2_score(y, y_pred_ols),
        "R2_Huber": r2_score(y, y_pred_huber),
        "R2_Ridge": r2_score(y, y_pred_ridge),

        "Spearman_fuzzy": spearmanr(y, y_pred_fuzzy)[0],
        "Spearman_TOPSIS": spearmanr(y, topsis_score)[0],
        "Spearman_OLS": spearmanr(y, y_pred_ols)[0],
        "Spearman_Huber": spearmanr(y, y_pred_huber)[0],
        "Spearman_Ridge": spearmanr(y, y_pred_ridge)[0]
    }
#异常
    X_bad, y_bad = add_outliers(X, y)

    # Fuzzy（扰动后）
    fuzzy_model.setdatax(X_bad)
    fuzzy_model.setdatay(y_bad)
    beta_fuzzy_bad = fuzzy_model.fit()

    #  OLS 
    ols_model.fit(X_bad, y_bad)
    beta_ols_bad = np.concatenate(([ols_model.intercept_], ols_model.coef_))

    #  Huber 
    huber_bad = HuberRegressor().fit(X_bad, y_bad)
    beta_huber_bad = np.concatenate(([huber_bad.intercept_], huber_bad.coef_))

    #  Ridge 
    ridge_bad = Ridge(alpha=1.0).fit(X_bad, y_bad)
    beta_ridge_bad = np.concatenate(([ridge_bad.intercept_], ridge_bad.coef_))

    #   稳定性（delta） 
    results.update({
        "delta_fuzzy": np.linalg.norm(beta_fuzzy - beta_fuzzy_bad),
        "delta_OLS": np.linalg.norm(beta_ols - beta_ols_bad),
        "delta_Huber": np.linalg.norm(beta_huber - beta_huber_bad),
        "delta_Ridge": np.linalg.norm(beta_ridge - beta_ridge_bad),
    })

    return results

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import pairwise_distances
import pandas as pd
"""
熵权-模糊综合评判 (E-FCE) 模型的验证工具集。
"""
def validate_entropy_fce_model(evaluation_results, true_scores=None, city_names=None):
    """
    对熵权-模糊综合评判模型进行综合性验证。
    """
    # 提取模型预测得分
    predicted_scores = np.array([r['score'] for r in evaluation_results])
    n_samples = len(predicted_scores)
    
    if city_names is None:
        city_names = [f"Sample_{i}" for i in range(n_samples)]
    
    results = {}
    
    # 1. 排序一致性与区分度验证
    results.update(_validate_ranking_and_distinction(predicted_scores, city_names))
    
    # 2. 如果提供了真实得分，则进行一致性验证
    if true_scores is not None:
        true_scores = np.asarray(true_scores)
        if len(true_scores) != n_samples:
            raise ValueError("true_scores 长度必须与 evaluation_results 一致")
        results.update(_validate_consistency_with_true_scores(predicted_scores, true_scores))
    
    return results


def _validate_ranking_and_distinction(scores, city_names):
    """内部函数：验证排序稳定性和结果区分度"""
    results = {}
    
    # 计算排名 (降序，1为最高)
    sorted_indices = np.argsort(-scores)  # 按得分从高到低排序的索引
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)  # 给每个位置分配排名
    
    # 1. 结果区分度指标
    score_std = np.std(scores)
    score_cv = score_std / np.mean(scores) if np.mean(scores) != 0 else np.inf
    results['score_std'] = score_std
    results['score_cv'] = score_cv
    
    # 2. 两两差异指标
    pairwise_diffs = np.abs(scores[:, None] - scores[None, :])
    non_diag_diffs = pairwise_diffs[np.triu_indices_from(pairwise_diffs, k=1)]
    min_nonzero_diff = np.min(non_diag_diffs[non_diag_diffs > 1e-8]) if np.any(non_diag_diffs > 1e-8) else 0.0
    mean_abs_diff = np.mean(non_diag_diffs)
    results['min_pairwise_diff'] = min_nonzero_diff
    results['mean_abs_pairwise_diff'] = mean_abs_diff

    # 创建完整的城市得分与排名 DataFrame ---
    full_ranking_df = pd.DataFrame({
        'City': city_names,
        'Score': scores,
        'Rank': ranks
    }).sort_values('Rank')  # 按排名升序排列 
    
    results['full_ranking'] = full_ranking_df

    top_n = min(5, len(scores))
    bottom_n = min(5, len(scores))
    top_idx = sorted_indices[:top_n]  # 直接使用已排序的索引
    bottom_idx = sorted_indices[-bottom_n:][::-1]  # 取最后bottom_n个并反转，使其从低到高
    
    print("\n=== E-FCE 模型验证: 排名与区分度 ===")
    print(f"得分标准差: {score_std:.4f}")
    print(f"得分变异系数(CV): {score_cv:.4f}")
    print(f"最小非零两两差异: {min_nonzero_diff:.6f}")
    print(f"平均绝对两两差异: {mean_abs_diff:.4f}")
    print(f"\nTop-{top_n} 城市:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {city_names[idx]} (得分: {scores[idx]:.4f})")
    print(f"\nBottom-{bottom_n} 城市:")
    for i, idx in enumerate(bottom_idx):
        actual_rank = ranks[idx]
        print(f"  {actual_rank}. {city_names[idx]} (得分: {scores[idx]:.4f})")
    
    return results


def _validate_consistency_with_true_scores(predicted_scores, true_scores):
    """内部函数：验证与真实创新指数的一致性"""
    results = {}
    
    # 1. Spearman 秩相关系数 (衡量排序一致性)
    spearman_corr, spearman_p = spearmanr(predicted_scores, true_scores)
    results['spearman_rank_corr'] = spearman_corr
    results['spearman_p_value'] = spearman_p
    
    # 2. Kendall Tau 秩相关系数 
    kendall_corr, kendall_p = kendalltau(predicted_scores, true_scores)
    results['kendall_tau'] = kendall_corr
    results['kendall_p_value'] = kendall_p
    
    # 3. 得分的相关性 (衡量数值接近程度)
    from scipy.stats import pearsonr
    pearson_corr, pearson_p = pearsonr(predicted_scores, true_scores)
    results['pearson_corr'] = pearson_corr
    results['pearson_p_value'] = pearson_p
    
    print(f"Spearman 秩相关系数: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"Kendall Tau 系数: {kendall_corr:.4f} (p={kendall_p:.4f})")
    print(f"Pearson 相关系数: {pearson_corr:.4f} (p={pearson_p:.4f})")
    
    return results


#  辅助函数：验证权重合理性 
def validate_indicator_weights(weights, indicator_names, expected_directions=None):
    """
    验证指标权重的合理性。
    
    Parameters:
    -----------
    weights : array-like
        模型计算出的指标权重。
    indicator_names : list of str
        指标名称列表。
    expected_directions : list of int, optional
        期望的指标方向 (+1 正向, -1 负向)。如果提供，可以检查权重分配是否符合预期。
        
    Returns:
    --------
    None (直接打印结果)
    """
    weights = np.asarray(weights)
    df_weights = pd.DataFrame({
        'Indicator': indicator_names,
        'Weight': weights
    }).sort_values('Weight', ascending=False)
    
    print("\nE-FCE 模型验证: 指标权重分析")
    print(indicator_names)
    print(df_weights.to_string(index=False))
    


#网格搜索确定参数的图
import matplotlib.pyplot as plt
def plot_conf_grid_search(results, show_std=True, show_range=True):
    """
    可视化 conf_conflict 网格搜索结果
    参数：
    - results: fit_conf_with_grid_search 返回的 all_results
    - show_std: 是否绘制 STD 曲线
    - show_range: 是否绘制 Range 曲线
    """
    conf_vals = [r["conflict"] for r in results]
    spearman_vals = [r["spearman"] for r in results]
    std_vals = [r["std"] for r in results]
    range_vals = [r["range"] for r in results]
    plt.figure()
    
    plt.plot(conf_vals, spearman_vals, marker='o', label="Spearman")

    if show_std:
        plt.plot(conf_vals, std_vals, marker='s', linestyle='--', label="STD")
    if show_range:
        plt.plot(conf_vals, range_vals, marker='^', linestyle='--', label="Range")
    # 
    best_idx = spearman_vals.index(max(spearman_vals))
    best_x = conf_vals[best_idx]
    best_y = spearman_vals[best_idx]
    plt.scatter([best_x], [best_y])
    plt.annotate(f"Best={best_x}", (best_x, best_y))
  
    plt.xlabel("conf_conflict")
    plt.ylabel("Metric Value")
    plt.title("Grid Search of conf_conflict")
    plt.legend()
    plt.grid()
    plt.show()

#验证conf机制的效果，比较加入conf前后模型性能的提升程度
def compare_conf_effect(
    model,
    beta,
    indicator_type,
    X,
    y_true,
    fit_conf_with_grid_search_func,
    best_conf, 
    best_param, 
    grid_results
):
    results_summary = []

    # 1️.baseline（不使用conf）
    model.set_semantic_confidence(None)
    res_base = model.evaluate(X)
    scores_base = np.array([r["score"] for r in res_base])
    spearman_base, p_base = spearmanr(scores_base, y_true)
    std_base = np.std(scores_base)
    cv_base = std_base / (np.mean(scores_base) + 1e-12)
    range_base = np.max(scores_base) - np.min(scores_base)

    results_summary.append({
        "name": "baseline(no_conf)",
        "spearman": spearman_base,
        "p_value": p_base,
        "std": std_base,
        "cv": cv_base,
        "range": range_base
    })

    print("\n[Baseline] 不使用conf")
    print(results_summary[-1])

    # 2.使用conf（网格搜索）
    model.set_semantic_confidence(best_conf)

    res_conf = model.evaluate(X)
    scores_conf = np.array([r["score"] for r in res_conf])

    spearman_conf, p_conf = spearmanr(scores_conf, y_true)
    std_conf = np.std(scores_conf)
    cv_conf = std_conf / (np.mean(scores_conf) + 1e-12)
    range_conf = np.max(scores_conf) - np.min(scores_conf)

    results_summary.append({
        "name": "conf(discrete+grid)",
        "best_conflict": best_param,
        "spearman": spearman_conf,
        "p_value": p_conf,
        "std": std_conf,
        "cv": cv_conf,
        "range": range_conf
    })
    print("\n[Conf] 使用离散conf + 网格搜索")
    print(results_summary[-1])
    #调用可视化图形的代码
    visualize_conf_effect_subplot(scores_base, scores_conf)

    return results_summary, grid_results

#可视化图形：加入conf前后，模型性能的提升程度
def visualize_conf_effect_subplot(scores_base, scores_conf):
    """
    使用 subplot 可视化 baseline 和 conf 的分数差异
    1. 分数分布直方图
    2. 样本散点对比
    """
    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    #左图：分数分布直方图
    axs[0].hist(scores_base, bins=20, alpha=0.5, label='Baseline (no conf)', color='skyblue', density=True)
    axs[0].hist(scores_conf, bins=20, alpha=0.5, label='Conf (discrete+grid)', color='salmon', density=True)
    axs[0].set_xlabel('Model Scores')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Score Distribution')
    axs[0].legend()
    axs[0].grid(True)

    #右图：散点对比
    axs[1].scatter(scores_base, scores_conf, c='purple', alpha=0.6)
    axs[1].plot([min(scores_base), max(scores_base)], [min(scores_base), max(scores_base)], 'r--', label='y=x')
    axs[1].set_xlabel('Baseline Scores')
    axs[1].set_ylabel('Conf Scores')
    axs[1].set_title('Sample-wise Score Comparison')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()