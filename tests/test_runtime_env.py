import platform
import importlib
import pandas as pd


def get_runtime_env():
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "xlsxwriter",
        "openpyxl"
    ]

    env_info = {
        "Python版本": platform.python_version(),
        "操作系统": f"{platform.system()} {platform.release()}",
        "处理器架构": platform.machine(),
    }

    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "未知")
        except Exception:
            version = "未安装"
        env_info[pkg] = version

    return pd.DataFrame(
        [{"项目": k, "版本/信息": v} for k, v in env_info.items()]
    )


if __name__ == "__main__":
    df = get_runtime_env()
    # print(df.to_string(index=False))