import os
import sys
import time
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mohu_model import IterativeMohuDecision
from models.entropy_fce import (
    EntropyFCEModel,
    normalize,
    entropy_weight,
    fit_conf_with_grid_search,
)
from models.decision_engine import DecisionEngine


def run_detailed_performance_once(X_eval, y, X_reg, indicator_types):
    """
    单次细粒度性能测试。
    返回各阶段耗时（秒）。
    """
    # 1. 数据预处理
    t0 = time.perf_counter()
    _X_scaled = (X_eval - X_eval.mean(axis=0)) / (X_eval.std(axis=0) + 1e-10)
    _y_scaled = (y - y.mean()) / (y.std() + 1e-10)
    t1 = time.perf_counter()

    # 2. 模糊回归
    reg = IterativeMohuDecision(
        n=X_reg.shape[1],
        number=X_reg.shape[0],
        tol=1e-6,
        max_iter=50
    )
    reg.setdatax(X_reg)
    reg.setdatay(y)
    reg.setdataxishu(np.zeros(X_reg.shape[1] + 1))

    t2 = time.perf_counter()
    beta = reg.fit()
    t3 = time.perf_counter()

    # 3. 熵权法
    t4 = time.perf_counter()
    X_norm = normalize(X_eval, indicator_types)
    _w = entropy_weight(X_norm)
    t5 = time.perf_counter()

    # 4. 网格搜索
    fce_tmp = EntropyFCEModel(indicator_types)
    fce_tmp.fit(X_eval)

    t6 = time.perf_counter()
    _best_conf, _best_param, _grid_results = fit_conf_with_grid_search(
        model=fce_tmp,
        beta=beta[1:],
        indicator_type=indicator_types,
        X=X_eval,
        y_true=y,
        verbose=False
    )
    t7 = time.perf_counter()

    # 5. 模糊综合评判
    t8 = time.perf_counter()
    results = fce_tmp.evaluate(X_eval)
    t9 = time.perf_counter()

    detail = {
        "Z-score标准化(秒)": round(t1 - t0, 6),
        "模糊回归迭代求解(秒)": round(t3 - t2, 6),
        "熵权法计算(秒)": round(t5 - t4, 6),
        "语义置信度网格搜索(秒)": round(t7 - t6, 6),
        "模糊综合评判(秒)": round(t9 - t8, 6),
        "细粒度总耗时(秒)": round(
            (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6) + (t9 - t8), 6
        ),
        "输出样本数": len(results),
    }
    return detail


def run_performance_test(
    n_cities=295,
    n_indicators=27,
    n_runs=3,
    use_conf=True,
    conf_mode="discrete",
    random_seed_base=42
):
    """
    单一规模性能测试：
    - 保留原有 DataFrame 返回形式
    - 增加细粒度阶段计时
    - 返回：每轮结果 + 平均行
    """
    records = []

    for i in range(n_runs):
        np.random.seed(random_seed_base + i)

        X = np.random.rand(n_cities, n_indicators)
        y = np.random.rand(n_cities)
        city_names = [f"城市{j+1}" for j in range(n_cities)]
        indicator_types = [1] * n_indicators

        regression_config = {
            "n": n_indicators,
            "number": n_cities,
            "tol": 1e-4,
            "max_iter": 50
        }

        fce_config = {
            "grade_labels": ["很低", "较低", "中等", "较高", "很高"]
        }

        # 先做细粒度测试
        detail = run_detailed_performance_once(
            X_eval=X,
            y=y,
            X_reg=X,
            indicator_types=indicator_types
        )

        # 再做端到端测试（DecisionEngine）
        engine = DecisionEngine(
            regression_config=regression_config,
            indicator_type=indicator_types,
            fce_config=fce_config
        )

        start = time.perf_counter()
        result = engine.full_decision(
            X_eval=X,
            X_reg=X,
            y_reg=y,
            city_names=city_names,
            use_conf=use_conf,
            conf_mode=conf_mode
        )
        elapsed = time.perf_counter() - start

        records.append({
            "轮次": i + 1,
            "城市数": n_cities,
            "指标数": n_indicators,
            "是否启用conf": "是" if use_conf else "否",
            "端到端耗时(秒)": round(elapsed, 6),
            "Z-score标准化(秒)": detail["Z-score标准化(秒)"],
            "模糊回归迭代求解(秒)": detail["模糊回归迭代求解(秒)"],
            "熵权法计算(秒)": detail["熵权法计算(秒)"],
            "语义置信度网格搜索(秒)": detail["语义置信度网格搜索(秒)"],
            "模糊综合评判(秒)": detail["模糊综合评判(秒)"],
            "细粒度总耗时(秒)": detail["细粒度总耗时(秒)"],
            "输出样本数": len(result.get("scores", []))
        })

    df = pd.DataFrame(records)

    avg_row = pd.DataFrame([{
        "轮次": "平均",
        "城市数": n_cities,
        "指标数": n_indicators,
        "是否启用conf": "是" if use_conf else "否",
        "端到端耗时(秒)": round(df["端到端耗时(秒)"].mean(), 6),
        "Z-score标准化(秒)": round(df["Z-score标准化(秒)"].mean(), 6),
        "模糊回归迭代求解(秒)": round(df["模糊回归迭代求解(秒)"].mean(), 6),
        "熵权法计算(秒)": round(df["熵权法计算(秒)"].mean(), 6),
        "语义置信度网格搜索(秒)": round(df["语义置信度网格搜索(秒)"].mean(), 6),
        "模糊综合评判(秒)": round(df["模糊综合评判(秒)"].mean(), 6),
        "细粒度总耗时(秒)": round(df["细粒度总耗时(秒)"].mean(), 6),
        "输出样本数": int(df["输出样本数"].mean())
    }])

    return pd.concat([df, avg_row], ignore_index=True)


def run_scalability_test(
    city_sizes=None,
    indicator_sizes=None,
    n_runs=3,
    use_conf=True,
    conf_mode="discrete",
    random_seed_base=42,
    save_csv=False,
    output_dir=None
):
    """
    规模化性能测试：
    - 对多个城市规模、多个指标规模进行组合测试
    - 返回：
        1. detail_df：所有规模下每轮的详细结果
        2. avg_df：所有规模下的平均结果
    """
    if city_sizes is None:
        city_sizes = [50, 100, 200, 297, 500]
    if indicator_sizes is None:
        indicator_sizes = [8, 16, 24, 32]

    all_detail_records = []
    all_avg_records = []

    for n_cities in city_sizes:
        for n_indicators in indicator_sizes:
            print(f"正在测试：城市数={n_cities}，指标数={n_indicators}")

            df_one = run_performance_test(
                n_cities=n_cities,
                n_indicators=n_indicators,
                n_runs=n_runs,
                use_conf=use_conf,
                conf_mode=conf_mode,
                random_seed_base=random_seed_base
            )

            # 明细：去掉平均行，保留逐轮数据
            detail_df = df_one[df_one["轮次"] != "平均"].copy()
            all_detail_records.append(detail_df)

            # 平均：只取平均行
            avg_df = df_one[df_one["轮次"] == "平均"].copy()
            all_avg_records.append(avg_df)

    detail_result = pd.concat(all_detail_records, ignore_index=True)
    avg_result = pd.concat(all_avg_records, ignore_index=True)

    if save_csv:
        if output_dir is None:
            output_dir = CURRENT_DIR
        os.makedirs(output_dir, exist_ok=True)

        detail_path = os.path.join(output_dir, "scalability_test_detail.csv")
        avg_path = os.path.join(output_dir, "scalability_test_avg.csv")

        detail_result.to_csv(detail_path, index=False, encoding="utf-8-sig")
        avg_result.to_csv(avg_path, index=False, encoding="utf-8-sig")

        print(f"详细结果已保存：{detail_path}")
        print(f"平均结果已保存：{avg_path}")

    return detail_result, avg_result


if __name__ == "__main__":
    # 1. 单一规模测试（可直接对应论文表6-4）
    df_single = run_performance_test(
        n_cities=297,
        n_indicators=16,
        n_runs=3
    )
    print("\n========== 单一规模性能测试 ==========")
    print(df_single.to_string(index=False))

    # 2. 规模化性能测试
    detail_df, avg_df = run_scalability_test(
        city_sizes=[50, 100, 200, 297, 500],
        indicator_sizes=[8, 16, 24, 32],
        n_runs=3,
        save_csv=False
    )

    # print("\n========== 规模化性能测试-逐轮明细 ==========")
    # print(detail_df.to_string(index=False))

    # print("\n========== 规模化性能测试-平均结果 ==========")
    # print(avg_df.to_string(index=False))