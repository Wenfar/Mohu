import pandas as pd


def get_decision_engine_contract():
    rows = [
        {
            "方法名": "__init__",
            "输入": "regression_config, indicator_type, fce_config=None",
            "输出": "DecisionEngine实例",
            "职责": "初始化模糊回归模块与综合评判模块"
        },
        {
            "方法名": "fit_regression",
            "输入": "X, y, beta_init=None",
            "输出": "回归系数beta",
            "职责": "执行多元模糊回归迭代求解"
        },
        {
            "方法名": "predict",
            "输入": "X, beta",
            "输出": "预测值y_hat",
            "职责": "根据回归系数进行预测"
        },
        {
            "方法名": "full_decision",
            "输入": "X_eval, X_reg, y_reg, city_names, use_conf, conf_mode",
            "输出": "综合结果字典",
            "职责": "统一调度模糊回归、语义置信度与综合评判"
        }
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = get_decision_engine_contract()
