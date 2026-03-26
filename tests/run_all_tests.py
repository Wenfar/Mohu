import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_runtime_env import get_runtime_env
from tests.test_functional import run_functional_tests
from tests.test_performance import run_performance_test, run_scalability_test
from tests.test_contract import get_decision_engine_contract
from tests.test_edge_cases import run_edge_case_tests
from plot_scalability import plot_scalability_from_excel


plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体（最常用）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号乱码


def main():
    output_dir = os.path.join(PROJECT_ROOT, "tests", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 各类测试 =====
    env_df = get_runtime_env()
    functional_df = run_functional_tests()

    # 单规模性能（论文表）
    performance_df = run_performance_test(
        n_cities=297,
        n_indicators=16,
        n_runs=3
    )

    # ⭐规模化性能（新增）
    scalability_detail_df, scalability_avg_df = run_scalability_test(
        city_sizes=[50, 100, 200, 297, 500],
        indicator_sizes=[8, 16, 24, 32],
        n_runs=3,
        save_csv=False  # 统一走 Excel，不单独存 CSV
    )

    contract_df = get_decision_engine_contract()
    edge_df = run_edge_case_tests()

    # ===== 2. 输出 Excel =====
    output_path = os.path.join(output_dir, "test_results.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        env_df.to_excel(writer, sheet_name="运行环境", index=False)
        functional_df.to_excel(writer, sheet_name="功能测试", index=False)
        performance_df.to_excel(writer, sheet_name="性能测试", index=False)

        # ⭐新增两个 Sheet
        scalability_avg_df.to_excel(writer, sheet_name="规模化性能-平均", index=False)
        scalability_detail_df.to_excel(writer, sheet_name="规模化性能-明细", index=False)

        contract_df.to_excel(writer, sheet_name="接口契约", index=False)
        edge_df.to_excel(writer, sheet_name="边缘情况", index=False)

    # ===== 3. 控制台输出 =====
    print("测试完成，结果已导出到：", output_path)
    

    plot_scalability_from_excel(output_path)

    # print("\n=== 运行环境 ===")
    # print(env_df.to_string(index=False))

    # print("\n=== 功能测试 ===")
    # print(functional_df.to_string(index=False))

    # print("\n=== 性能测试 ===")
    # print(performance_df.to_string(index=False))

    # print("\n=== 规模化性能（平均） ===")
    # print(scalability_avg_df.to_string(index=False))

    # print("\n=== 接口契约 ===")
    # print(contract_df.to_string(index=False))

    # print("\n=== 边缘情况 ===")
    # print(edge_df.to_string(index=False))


if __name__ == "__main__":
    main()
    