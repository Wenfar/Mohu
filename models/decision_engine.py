"""
DecisionEngine
==================================================
将三大模块封装成统一对象：
- 模糊回归（结构/稳定性验证）
- 熵权法（唯一权重来源）
- 模糊综合评判（等级 + 得分）
对外提供稳定 API，供数据分析平台调用
"""

import numpy as np
from models.entropy_fce import EntropyFCEModel
from models.mohu_model import IterativeMohuDecision
from models.entropy_fce import (
    semantic_confidence,
    #semantic_confidence_adaptive_continuous,
    #estimate_adaptive_conf_params,
    fit_conf_with_grid_search,
)

class DecisionEngine:
    def __init__(self,
                 regression_config,
                 indicator_type,
                 fce_config=None):
        
        "以下两个模块没有数据，只准备了工具。"
        # 1. 模糊回归模块（不产出权重）
        self.regression = IterativeMohuDecision(
            n=regression_config["n"],
            number=regression_config["number"],
            tol=regression_config.get("tol", 1e-6),
            max_iter=regression_config.get("max_iter", 100)
        )
        
        
        # 2. 熵权 + 模糊综合评判模块
        if fce_config is None:
            self.fce = EntropyFCEModel(indicator_type)
        else:
            self.fce = EntropyFCEModel(
                indicator_type=indicator_type,
                levels=fce_config.get("levels"),
                grade_labels=fce_config.get("grade_labels")
            )

        # 中间结果缓存
        self.beta = None       # 模糊回归系数
        self.weights = None    # 熵权

    
    # 一、模糊回归模块。包括fit_regression和predict
    def fit_regression(self, X, y, beta_init=None):
        """
        训练模糊回归模型（用于模型验证）
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X 不能为空，且必须为二维矩阵")

        if y.ndim != 1 or y.shape[0] == 0:
            raise ValueError("y 不能为空，且必须为一维数组")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 样本数不一致")

        if beta_init is None:
            beta_init = np.zeros(X.shape[1] + 1, dtype=float)
        else:
            beta_init = np.asarray(beta_init, dtype=float)
            if beta_init.shape[0] != X.shape[1] + 1:
                raise ValueError(
                    f"beta_init 长度应为 {X.shape[1] + 1}，当前为 {beta_init.shape[0]}"
                )

        self.regression.setdatax(X)
        self.regression.setdatay(y)
        self.regression.setdataxishu(beta_init)

        self.beta = self.regression.fit()
        return self.beta
 

    def predict(self, X_new):
        """
        使用模糊回归进行预测
        """
        if self.beta is None:
            raise ValueError("模糊回归尚未训练")
        return self.regression.predict(X_new)

    # 二、熵权 + 模糊综合评判模块
    def fit_weights(self, X):
        """
        计算熵权（唯一权重来源）
        """
        self.weights = self.fce.fit(X)
        return self.weights

    def evaluate(self, X):
        """
        执行模糊综合评判
        """
        if self.weights is None:
            raise ValueError("尚未计算熵权，请先调用 fit_weights")

        return self.fce.evaluate(X)


    # 三、完整决策流程（平台一键调用）
    def full_decision(self,
                      X_eval,
                      X_reg=None,
                      y_reg=None,
                      beta_init=None,
                      city_names=None,
                      use_conf=True,
                      conf_mode="discrete",#离散型
                      conf_auto_params=True
                    ):
        #use_conf表示是否使用语义置信度，用于后续验证加入conf是否能提升模型
#         当 conf_mode="adaptive_continuous" 且 conf_auto_params=True 时：
# 自动调用 estimate_adaptive_conf_params
# 自动写入 self.fce 的 alpha/threshold
# 自动生成连续 conf
# 并把参数回传到结果：result["conf_auto_params"]
        result = {}
        grid_results = None
        best_param = None
        best_conf = None
        result["semantic_confidence"] = None
        
        
        X_eval = np.asarray(X_eval, dtype=float)

        if X_eval.ndim != 2 or X_eval.shape[0] == 0 or X_eval.shape[1] == 0:
            raise ValueError("X_eval 不能为空，且必须为二维矩阵")

        if X_eval.shape[1] != len(self.fce.indicator_type):
            raise ValueError(
                f"X_eval 指标列数({X_eval.shape[1]}) 与 indicator_type 长度({len(self.fce.indicator_type)}) 不一致"
            )
        # 1. 模糊回归 
        "如果平台提供了数据，先校准系统结构，而不是用回归结果去评价。"
        if X_reg is not None and y_reg is not None:
            X_reg = np.asarray(X_reg, dtype=float)
            y_reg = np.asarray(y_reg, dtype=float)

            if X_reg.ndim != 2 or X_reg.shape[0] == 0 or X_reg.shape[1] == 0:
                raise ValueError("X_reg 不能为空，且必须为二维矩阵")

            if y_reg.ndim != 1 or y_reg.shape[0] == 0:
                raise ValueError("y_reg 不能为空，且必须为一维数组")

            if X_reg.shape[0] != y_reg.shape[0]:
                raise ValueError("X_reg 与 y_reg 样本数不一致")

            self.regression = IterativeMohuDecision(
                n=X_reg.shape[1],
                number=X_reg.shape[0],
                tol=self.regression.tol,
                max_iter=self.regression.max_iter
            )

            result["regression_coef"] = self.fit_regression(X_reg, y_reg, beta_init)

            if use_conf:
                beta = result["regression_coef"][1:]

                if conf_mode == "discrete":
                    best_conf, best_param, grid_results = fit_conf_with_grid_search(
                        model=self.fce,
                        beta=beta,
                        indicator_type=self.fce.indicator_type,
                        X=X_eval,
                        y_true=y_reg
                    )
                    conf = best_conf
                else:
                    conf = None
            else:
                conf = None

            self.fce.set_semantic_confidence(conf)

            if best_param is not None:
                self.fce.conf_reverse_threshold = best_param

            result["semantic_confidence"] = conf
            result["city_names"] = city_names

        else:
    # 没有目标变量时，跳过模糊回归与语义置信度
            result["regression_coef"] = None
            result["semantic_confidence"] = None
            result["city_names"] = city_names
    

        # 2. 熵权 
        """权重只从X_eval的信息熵来，与beta完全解耦。
        前提：X_eval的指标体系是被结构允许的（模糊回归模型验证通过）"""
        result["weights"] = self.fit_weights(X_eval)

        # 3. 模糊综合评判 
        "把 熵权 × 指标隶属度 × 评价等级集，映射成最终决策结果"

        evaluation_result = self.evaluate(X_eval)
        result["evaluation"] = evaluation_result

        result["scores"] = [item["score"] for item in evaluation_result]
        result["grades"] = [item["grade"] for item in evaluation_result]
        result["memberships"] = [item["membership"] for item in evaluation_result]

        return {
            **result,
            "grid_results": grid_results,
            "best_param": best_param,
            "best_conf": best_conf
        }
       