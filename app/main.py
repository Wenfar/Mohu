import streamlit as st
import sys
import os
import pandas as pd


# -----------------------------------------------------------------------------
# 1. 基础配置与路径设置（页面配置升级为宽屏沉浸式）
# -----------------------------------------------------------------------------
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_path not in sys.path:
    sys.path.append(project_path)

st.set_page_config(
    page_title="模糊决策数据分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': None,
        'About': '模糊决策数据分析平台 v2.1 | 基于熵权法+模糊综合评价'
    }
)

st.markdown("""
    
""", unsafe_allow_html=True)

with st.sidebar:
    # 顶部LOGO/标题区
    st.markdown('# 模糊决策数据分析平台', unsafe_allow_html=True)
    st.markdown('', unsafe_allow_html=True)
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    
    # 核心流程导航
    st.markdown("### 核心操作流程")
    page = st.radio(
        "",
        [
            "首页", 
            "数据导入", 
            "模型配置", 
            "综合评价"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('', unsafe_allow_html=True)
    
    # 功能说明区
    st.markdown("### 功能说明")
    st.caption("""
    ✅ 数据清洗与标准化  
    ✅ 多元模糊回归建模  
    ✅ 熵权法客观赋权  
    ✅ 模糊综合评价建模  
    ✅ 多维可视化分析  
    """)
    
    st.markdown('', unsafe_allow_html=True)
    # 版本信息
    st.caption("**版本**: v2.1 | 通用专业版")
    # st.caption("© 2025 数据分析研究中心")

# -----------------------------------------------------------------------------
# 4. 主页面路由（卡片化布局，提升阅读体验）
# -----------------------------------------------------------------------------

# --- 首页：向导页升级 ---
if page == "首页":
    st.title("模糊决策数据分析平台")
    st.markdown('', unsafe_allow_html=True)
    # st.markdown("### 平台简介")
    st.write("本平台是基于多元模糊回归算法构建的模糊决策数据分析平台，专为多维度指标综合评价打造，适用于区域发展、绩效考评、质量监测、资源分析等各类决策场景，全程自动化运算、可视化输出，降低专业门槛。")
    st.markdown('', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('', unsafe_allow_html=True)
        st.markdown("### 核心功能")
        st.markdown("""
        1. **标准化数据处理**：支持正向/负向/中性指标归一化、异常值清洗
        2. **客观权重计算**：熵权法赋权，剔除主观偏差
        3. **模糊综合评价**：适配不确定性数据，精准测算综合得分
        4. **可视化分析**：雷达图、排名柱状图、权重热力图一键生成
       
        """)
        st.info("💡 操作指引：按照左侧流程 **数据导入→模型配置→运行评价→结果分析** 逐步操作即可")
        st.markdown('', unsafe_allow_html=True)
    
    with col2:
        st.markdown('', unsafe_allow_html=True)
        st.markdown("### 数据类型模板")
        example_df = pd.DataFrame({
            '样本编号': ['样本1', '样本2', '样本3', '样本4'],
            '目标评分': [86.2, 81.5, 90.3, 88.7],
            '指标1(GDP)': [12500, 9800, 15600, 13200],
            '指标2(收入)': [45200, 39800, 52100, 48600],
            '指标3(就业率)': [96.2, 94.8, 97.5, 96.8]
        })
        st.dataframe(example_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ 数据格式要求：首列=样本名称，第二列=目标值，其余列为评价指标")
        st.markdown('', unsafe_allow_html=True)


elif page == "数据导入":
    st.title("📂 数据导入与预处理")
    st.markdown('', unsafe_allow_html=True)
    try:
        from pages.data_upload import render_page
        render_page()
    except ImportError:
        st.error("❌ 模块缺失：请确保 `pages/data_upload.py` 文件存在")
        st.code("""项目标准结构：
├── app.py（主程序）
└── pages/
    ├── data_upload.py    # 数据上传模块
    ├── model_config.py  # 模型配置模块
    ├── evaluation.py    # 评价运算模块
    ├── visualization.py # 可视化模块
    └── validation.py    # 模型验证模块
""")
    st.markdown('', unsafe_allow_html=True)

# --- 模型配置模块 ---
elif page == "模型配置":
    st.title("⚙️ 评价模型参数配置")
    st.markdown('', unsafe_allow_html=True)
    try:
        from pages.model_config import render_page
        render_page()
    except ImportError:
        st.error("❌ 模块缺失：请确保 `pages/model_config.py` 文件存在")
    st.markdown('', unsafe_allow_html=True)

# --- 运行评价模块 ---
elif page == "综合评价":
    st.title("🧮 模型运行结果")
    st.markdown('', unsafe_allow_html=True)
    try:
        from pages.evaluation import render_page
        render_page()
    except ImportError:
        st.error("❌ 模块缺失：请确保 `pages/evaluation.py` 文件存在")
    st.markdown('', unsafe_allow_html=True)





