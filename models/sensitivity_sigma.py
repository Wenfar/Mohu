"""
σ 敏感性分析
对应论文第3章：验证高斯隶属函数尺度参数 σ 的设定合理性。
比较 σ ∈ {0.5, 1.0, 1.5, 2.0} 时模糊回归模型的性能稳定性。
"""

import numpy as np
import pandas as pd
from mohu_model import IterativeMohuDecision


# ── 1. 支持 σ 参数的模型子类

class IterativeMohuWithSigma(IterativeMohuDecision):
    """
    在原始模型基础上，将高斯隶属函数的尺度参数 σ 暴露为可配置参数。
    原始代码中 u[i] = exp(-(e_i)^2) 等价于 σ=1 的特殊情况。
    """
    def __init__(self, n, number, sigma=1.0, tol=1e-6, max_iter=100):
        super().__init__(n, number, tol, max_iter)
        self.sigma = sigma

    def _single_update(self):
        #用 sigma 参数化隶属度计算
        for i in range(self.number):
            y_hat = self.beta[0] + np.dot(self.beta[1:], self.x[i])
            e = self.y[i] - y_hat
            self.u[i] = np.exp(-(e ** 2) / (2 * self.sigma ** 2))

        # 与原始 _single_update 完全相同
        for i in range(self.n):
            for j in range(self.n):
                sumu  = np.sum(self.u)
                sumxx = np.sum(self.u * self.x[:, i] * self.x[:, j])
                sumx1 = np.sum(self.u * self.x[:, i])
                sumx2 = np.sum(self.u * self.x[:, j])
                self.a[i, j] = sumu * sumxx - sumx1 * sumx2

        for i in range(self.n):
            sumu  = np.sum(self.u)
            sumxy = np.sum(self.u * self.x[:, i] * self.y)
            sumx  = np.sum(self.u * self.x[:, i])
            sumy  = np.sum(self.u * self.y)
            self.b[i] = sumu * sumxy - sumx * sumy

        lambda_reg = 1e-6
        beta_rest = np.linalg.solve(
            self.a + lambda_reg * np.eye(self.n), self.b
        )

        sumu = np.sum(self.u)
        if sumu < 1e-10:
            avg_x = np.mean(self.x, axis=0)
            avg_y = np.mean(self.y)
        else:
            avg_x = np.array([
                np.sum(self.u * self.x[:, j]) / sumu for j in range(self.n)
            ])
            avg_y = np.sum(self.u * self.y) / sumu

        beta0 = avg_y - np.dot(beta_rest, avg_x)
        beta_new = np.zeros(self.n + 1)
        beta_new[0] = beta0
        beta_new[1:] = beta_rest
        return beta_new


# 2. 工具函数
def zscore(arr):
    """Z-score 标准化"""
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    std[std < 1e-12] = 1.0          # 防止除零
    return (arr - mean) / std, mean, std


def run_single_sigma(X_std, y_std, sigma, tol=1e-6, max_iter=100):
    """
    对给定 σ 值运行模糊回归，返回性能指标字典。
    输入数据须已经过 Z-score 标准化。
    """
    m, n = X_std.shape
    model = IterativeMohuWithSigma(n, m, sigma=sigma, tol=tol, max_iter=max_iter)
    model.setdatax(X_std)
    model.setdatay(y_std)

    # 初始化为普通最小二乘解（与原始 fit 流程一致）
    X_aug = np.hstack([np.ones((m, 1)), X_std])
    beta_init = np.linalg.lstsq(X_aug, y_std, rcond=None)[0]
    model.beta = beta_init

    beta = model.fit()
    y_pred = model.predict(X_std)

    residuals = y_std - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae  = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_std - y_std.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "sigma": sigma,
        "RMSE":  round(rmse, 6),
        "MAE":   round(mae,  6),
        "R²":    round(r2,   6),
        "beta":  beta          # 保留系数向量，用于稳定性分析
    }


# 3. 敏感性分析主函数
def sensitivity_analysis(X_raw, y_raw,
                         sigma_list=None,
                         tol=1e-6, max_iter=100,
                         print_result=True):
    """
    对 σ ∈ sigma_list 逐一运行模糊回归，汇总性能指标与系数稳定性。

    Parameters
    ----------
    X_raw      : np.ndarray, shape (m, n)，原始指标矩阵（未标准化）
    y_raw      : np.ndarray, shape (m,)，原始目标向量（未标准化）
    sigma_list : list of float，待比较的 σ 取值，默认 [0.5, 1.0, 1.5, 2.0]
    print_result : bool，是否打印结果表格

    Returns
    -------
    df_metrics  : pd.DataFrame，各 σ 对应的 RMSE / MAE / R²
    df_coef     : pd.DataFrame，各 σ 对应的回归系数
    """
    if sigma_list is None:
        sigma_list = [0.5, 1.0, 1.5, 2.0]

    # Z-score 标准化
    X_std, _, _ = zscore(X_raw)
    y_std, _, _ = zscore(y_raw.reshape(-1, 1))
    y_std = y_std.ravel()

    results = [
        run_single_sigma(X_std, y_std, sigma, tol, max_iter)
        for sigma in sigma_list
    ]
    # 性能指标表
    df_metrics = pd.DataFrame([
        {"σ": r["sigma"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R²": r["R²"]}
        for r in results
    ])
    # 回归系数表
    n_features = X_raw.shape[1]
    col_names  = ["截距"] + [f"β{j+1}" for j in range(n_features)]
    df_coef = pd.DataFrame(
        [r["beta"] for r in results],
        columns=col_names
    )
    df_coef.insert(0, "σ", sigma_list)
    #系数最大变化量（稳定性指标）
    beta_matrix = np.array([r["beta"] for r in results])   # shape: (4, n+1)
    max_delta   = np.max(beta_matrix, axis=0) - np.min(beta_matrix, axis=0)
    delta_row   = pd.DataFrame(
        [["Δmax"] + list(np.round(max_delta, 6))],
        columns=df_coef.columns
    )
    df_coef = pd.concat([df_coef, delta_row], ignore_index=True)

    if print_result:
        print("=" * 55)
        print("表X  不同 σ 取值下模糊回归模型的性能比较")
        print("=" * 55)
        print(df_metrics.to_string(index=False))
        print()
        print("表X+1  不同 σ 取值下的回归系数（标准化空间）")
        print("-" * 55)
        print(df_coef.to_string(index=False))
        print()
        _print_stability_conclusion(df_metrics, beta_matrix, sigma_list)

    return df_metrics, df_coef


def _print_stability_conclusion(df_metrics, beta_matrix, sigma_list):
    """根据结果自动生成结论文字，辅助写论文。"""
    rmse_vals = df_metrics["RMSE"].values
    r2_vals   = df_metrics["R²"].values
    rmse_range = rmse_vals.max() - rmse_vals.min()
    r2_range   = r2_vals.max()   - r2_vals.min()

    # 系数最大L2距离
    ref = beta_matrix[sigma_list.index(1.0)]   # 以 σ=1 为基准
    l2_dists = [np.linalg.norm(b - ref) for b in beta_matrix]

    print("── 稳定性结论 ──")
    print(f"  RMSE 极差：{rmse_range:.6f}  |  R² 极差：{r2_range:.6f}")
    print(f"  各 σ 系数与 σ=1.0 基准的 L2 距离：")
    for s, d in zip(sigma_list, l2_dists):
        print(f"    σ={s:.1f}  →  ‖Δβ‖₂ = {d:.6f}")

    if rmse_range < 0.05 and r2_range < 0.02:
        print("\n  结论：模型性能对 σ 的变化不敏感，σ=1.0 的经验设定具有稳健性。")
    else:
        print("\n  结论：模型性能对 σ 存在一定敏感性，建议进一步调参或采用自适应 σ。")


if __name__ == "__main__":  
                              #
    #  X_raw: 原始指标矩阵，shape = (样本数, 指标数)                       #
    #  y_raw: 目标向量（如创新活力综合得分），shape = (样本数,)             #
    # np.random.seed(42)
    # m, n = 297, 8                          # 297个地级市，8个指标（示例）
    # X_raw = np.random.randn(m, n)
    # true_beta = np.array([0.5, -0.3, 0.8, 0.2, -0.1, 0.6, -0.4, 0.3])
    # y_raw = X_raw @ true_beta + 0.3 * np.random.randn(m)

    
    # df_metrics, df_coef = sensitivity_analysis(
    #     X_raw, y_raw,
    #     sigma_list=[0.5, 1.0, 1.5, 2.0]
    # )

    np.random.seed(42)
    m, n = 297, 8
    X_raw = np.random.randn(m, n)
    true_beta = np.array([0.5, -0.3, 0.8, 0.2, -0.1, 0.6, -0.4, 0.3])
    y_raw = X_raw @ true_beta + 0.3 * np.random.randn(m)


# 2. 添加高比例异常值

    outlier_ratio = 0.3  # 30%异常
    num_outliers = int(m * outlier_ratio)
    indices = np.random.choice(m, num_outliers, replace=False)

# 放大噪声
    y_raw[indices] += np.random.normal(0, 5, size=num_outliers)

# 3. σ敏感性分析
    sigma_list = [0.5, 1.0, 1.5, 2.0]

    df_metrics, df_coef = sensitivity_analysis(
        X_raw, y_raw,
        sigma_list=sigma_list
    )

   
    df_metrics.to_excel("../data/sensitivity_metrics.xlsx", index=False)
    df_coef.to_excel("../data/sensitivity_coef.xlsx",       index=False)
 