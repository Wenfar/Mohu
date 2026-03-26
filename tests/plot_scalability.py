import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_scalability_from_excel(excel_path, save_dir=None):
    """
    从 test_results.xlsx 读取“规模化性能-平均”并画图
    """

    if save_dir is None:
        save_dir = os.path.dirname(excel_path)

    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df = pd.read_excel(excel_path, sheet_name="规模化性能-平均")

    # =============================
    # 图1：城市规模 vs 耗时（固定指标数）
    # =============================
    for indicator in sorted(df["指标数"].unique()):
        subset = df[df["指标数"] == indicator]

        plt.figure()
        plt.plot(subset["城市数"], subset["端到端耗时(秒)"], marker='o')
        plt.xlabel("城市数量")
        plt.ylabel("端到端耗时（秒）")
        plt.title(f"城市规模对耗时影响（指标数={indicator}）")
        plt.grid()

        save_path = os.path.join(save_dir, f"城市规模_{indicator}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # =============================
    # 图2：指标规模 vs 耗时（固定城市数）
    # =============================
    for city in sorted(df["城市数"].unique()):
        subset = df[df["城市数"] == city]

        plt.figure()
        plt.plot(subset["指标数"], subset["端到端耗时(秒)"], marker='o')
        plt.xlabel("指标数量")
        plt.ylabel("端到端耗时（秒）")
        plt.title(f"指标规模对耗时影响（城市数={city}）")
        plt.grid()

        save_path = os.path.join(save_dir, f"指标规模_{city}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # =============================
    # 图3：阶段耗时占比（选最大规模）
    # =============================
    max_row = df.sort_values("端到端耗时(秒)", ascending=False).iloc[0]

    stages = [
        "Z-score标准化(秒)",
        "模糊回归迭代求解(秒)",
        "熵权法计算(秒)",
        "语义置信度网格搜索(秒)",
        "模糊综合评判(秒)"
    ]

    values = [max_row[s] for s in stages]

    plt.figure()
    plt.pie(values, labels=stages, autopct='%1.1f%%')
    plt.title(f"阶段耗时占比（城市={max_row['城市数']}，指标={max_row['指标数']}）")

    save_path = os.path.join(save_dir, "阶段耗时占比.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("所有图已生成，目录：", save_dir)