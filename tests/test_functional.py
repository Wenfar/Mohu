import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.decision_engine import DecisionEngine


def run_functional_tests():
    results = []

    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    city_names = [f"城市{i+1}" for i in range(50)]

    indicator_types = [1, 1, -1, 1, 1]

    regression_config = {
        "n": 5,
        "number": 50,
        "tol": 1e-4,
        "max_iter": 50
    }

    fce_config = {
        "grade_labels": ["很低", "较低", "中等", "较高", "很高"]
    }

    engine = DecisionEngine(
        regression_config=regression_config,
        indicator_type=indicator_types,
        fce_config=fce_config
    )

    # FT-01 空数据
    try:
        engine.full_decision(
            X_eval=np.empty((0, 5)),
            X_reg=np.empty((0, 5)),
            y_reg=np.array([]),
            city_names=[],
            use_conf=True,
            conf_mode="discrete"
        )
        actual = "未拦截"
        passed = "否"
    except Exception as e:
        actual = str(e)
        passed = "是"

    results.append({
        "用例编号": "FT-01",
        "所属模块": "DecisionEngine",
        "测试项": "空数据输入",
        "输入/操作": "X_eval为空矩阵，y_reg为空数组",
        "预期结果": "系统抛出异常并提示输入无效",
        "实际结果": actual,
        "是否通过": passed
    })

    # FT-02 指标维度不匹配
    try:
        engine.full_decision(
            X_eval=np.random.rand(20, 4),
            X_reg=np.random.rand(20, 4),
            y_reg=np.random.rand(20),
            city_names=[f"c{i}" for i in range(20)],
            use_conf=True,
            conf_mode="discrete"
        )
        actual = "未拦截"
        passed = "否"
    except Exception as e:
        actual = str(e)
        passed = "是"

    results.append({
        "用例编号": "FT-02",
        "所属模块": "DecisionEngine",
        "测试项": "指标维度与indicator_type不一致",
        "输入/操作": "X为4列，但indicator_type为5列",
        "预期结果": "系统抛出维度不一致异常",
        "实际结果": actual,
        "是否通过": passed
    })

    # FT-03 beta_init参数错误
    try:
        engine.fit_regression(X, y, beta_init=np.array([1, 2]))
        actual = "未拦截"
        passed = "否"
    except Exception as e:
        actual = str(e)
        passed = "是"

    results.append({
        "用例编号": "FT-03",
        "所属模块": "模糊回归",
        "测试项": "错误beta_init参数",
        "输入/操作": "beta_init长度错误",
        "预期结果": "系统抛出参数维度异常",
        "实际结果": actual,
        "是否通过": passed
    })

    # FT-04 样本数不一致
    try:
        engine.full_decision(
            X_eval=X,
            X_reg=X,
            y_reg=np.random.rand(40),
            city_names=city_names,
            use_conf=True,
            conf_mode="discrete"
        )
        actual = "未拦截"
        passed = "否"
    except Exception as e:
        actual = str(e)
        passed = "是"

    results.append({
        "用例编号": "FT-04",
        "所属模块": "DecisionEngine",
        "测试项": "回归输入样本数不一致",
        "输入/操作": "X_reg=50行, y_reg=40行",
        "预期结果": "系统抛出样本数不一致异常",
        "实际结果": actual,
        "是否通过": passed
    })

    # FT-05 目标变量缺失
    try:
        engine.full_decision(
            X_eval=X,
            X_reg=X,
            y_reg=None,
            city_names=city_names,
            use_conf=True,
            conf_mode="discrete"
        )
        actual = "未提供目标变量时未执行模糊回归"
        passed = "是"
    except Exception as e:
        actual = str(e)
        passed = "否"

    results.append({
        "用例编号": "FT-05",
        "所属模块": "DecisionEngine",
        "测试项": "目标变量缺失",
        "输入/操作": "y_reg=None",
        "预期结果": "系统跳过模糊回归或给出明确提示",
        "实际结果": actual,
        "是否通过": passed
    })

    # FT-06 正常流程
    try:
        result = engine.full_decision(
            X_eval=X,
            X_reg=X,
            y_reg=y,
            city_names=city_names,
            use_conf=True,
            conf_mode="discrete"
        )
        if "scores" in result and len(result["scores"]) == 50:
            actual = "成功输出综合得分"
            passed = "是"
        else:
            actual = "输出结构不完整"
            passed = "否"
    except Exception as e:
        actual = str(e)
        passed = "否"

    results.append({
        "用例编号": "FT-06",
        "所属模块": "全流程",
        "测试项": "正常数据计算",
        "输入/操作": "50样本×5指标正常数据",
        "预期结果": "成功输出得分、等级、权重等结果",
        "实际结果": actual,
        "是否通过": passed
    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_functional_tests()
    # print(df.to_string(index=False))