import streamlit as st
import pandas as pd
import numpy as np

def render_page():
    # st.title("⚙️ 模型配置")
    
    # 检查是否已上传数据
    if 'df_processed' not in st.session_state or 'data_config' not in st.session_state:
        st.warning("⚠️ 请先在 **数据上传** 页面上传并处理数据")
        return
    
    df = st.session_state['df_processed']
    config = st.session_state['data_config']
    indicator_cols = config['indicator_cols']
    
    st.success(f"✅ 已加载数据：{config['n_samples']} 个样本，{config['n_indicators']} 个指标")
    
    # ==================== 指标类型配置 ====================
    st.subheader("1️⃣ 指标类型配置")
    
    st.info("""
    **指标类型说明**：
    - **正向指标** ✅：数值越大越好
    - **负向指标** ❌：数值越小越好
    系统会根据指标类型进行正向化处理。
    """)
    
    # 初始化指标类型配置
    if 'indicator_types' not in st.session_state:
        # 默认全部为正向
        st.session_state['indicator_types'] = {col: 1 for col in indicator_cols}
    
    # 快速设置
    st.markdown("**快速设置**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("全部设为正向 ✅", use_container_width=True):
            st.session_state['indicator_types'] = {col: 1 for col in indicator_cols}
            st.rerun()
    
    with quick_col2:
        if st.button("全部设为负向 ❌", use_container_width=True):
            st.session_state['indicator_types'] = {col: -1 for col in indicator_cols}
            st.rerun()
    
    st.markdown("---")
    
    # ===== 折叠配置区域 =====
    with st.expander("指标方向配置（点击展开）", expanded=False):

        st.markdown("**逐个配置**")

        indicator_config_data = []

        for i, col in enumerate(indicator_cols):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

            with col1:
                st.write(f"{i+1}. {col}")

            with col2:
                min_val = df[col].min()
                max_val = df[col].max()
                st.caption(f"{min_val:.2f} ~ {max_val:.2f}")

            with col3:
                current_type = st.session_state['indicator_types'][col]

                type_choice = st.radio(
                    f"类型_{i}",
                    options=[1, -1],
                    format_func=lambda x: "正向" if x == 1 else "负向",
                    index=0 if current_type == 1 else 1,
                    key=f"type_{col}",
                    label_visibility="collapsed",
                    horizontal=True
                )

                st.session_state['indicator_types'][col] = type_choice

            with col4:
                sample_vals = df[col].head(3).tolist()
                st.caption(f"{', '.join([f'{v:.2f}' for v in sample_vals])}")

            indicator_config_data.append({
                '指标': col,
                '类型': '正向' if type_choice == 1 else '负向',
                '最小值': min_val,
                '最大值': max_val,
                '平均值': df[col].mean()
            })
    
    # 显示配置摘要
    with st.expander("📊 查看配置摘要"):
        summary_df = pd.DataFrame(indicator_config_data)
        st.dataframe(summary_df, use_container_width=True)
        
        positive_count = sum(1 for t in st.session_state['indicator_types'].values() if t == 1)
        negative_count = len(indicator_cols) - positive_count
        
        col1, col2 = st.columns(2)
        col1.metric("正向指标", f"{positive_count} 个")
        col2.metric("负向指标", f"{negative_count} 个")
    
    # ==================== 模型参数配置 ====================
    st.subheader("2️⃣ 模型参数配置")
    
    tab1, tab2, tab3 = st.tabs(["🔧 基础参数", "🎛️ 高级参数", "📖 参数说明"])
    
    with tab1:
        st.markdown("**模糊回归参数**")
        
        reg_col1, reg_col2 = st.columns(2)
        
        with reg_col1:
            max_iter = st.number_input(
                "最大迭代次数",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="模糊回归的最大迭代次数，越大越精确但耗时更长"
            )
            
            tol = st.number_input(
                "收敛容差",
                min_value=1e-8,
                max_value=1e-3,
                value=1e-6,
                format="%.2e",
                help="迭代收敛的容差，越小越精确"
            )
        
        # with reg_col2:
        #     use_conf = st.checkbox(
        #         "启用语义置信度调节",
        #         value=True,
        #         help="根据回归结果与指标语义的一致性调节权重"
        #     )
            
        #     if use_conf:
        #         st.info("✅ 启用后，系统会自动降低与回归结构冲突的指标权重")
        
        st.markdown("---")
        st.markdown("**模糊综合评判参数**")
        
        fce_col1, fce_col2 = st.columns(2)
        
        with fce_col1:
            n_levels = st.selectbox(
                "评价等级数量",
                options=[3, 5, 7],
                index=1,
                help="综合评价的等级划分数量"
            )
        
        with fce_col2:
            # 根据等级数量生成标签
            if n_levels == 3:
                default_labels = ['低', '中', '高']
            elif n_levels == 5:
                default_labels = ['很低', '较低', '中等', '较高', '很高']
            else:
                default_labels = ['极低', '很低', '较低', '中等', '较高', '很高', '极高']
            
            grade_labels_str = st.text_input(
                "等级标签（用逗号分隔）",
                value=', '.join(default_labels),
                help="自定义每个等级的名称"
            )
            grade_labels = [label.strip() for label in grade_labels_str.split(',')]
    
    with tab2:
        st.markdown("**语义置信度调节参数（默认开启）**")

        use_conf = True
        st.success("✅ 系统默认启用语义置信度调节，并自动进行网格搜索调参")

        param_grid = {
            "conf_weight_alpha": [1.2, 1.5, 2.0],
            "conf_membership_alpha": [1.5, 2.0],
            "conf_reverse_threshold": [0.25, 0.3]
        }

    
    with tab3:
        st.markdown("""
        ### 参数说明
        
        #### 模糊回归参数
        
        1. **最大迭代次数**
           - 作用：控制模糊回归的迭代上限
           - 推荐值：50-100
           - 注意：样本量大时可适当增加
        
        2. **收敛容差**
           - 作用：判断迭代是否收敛的阈值
           - 推荐值：1e-6
           - 注意：过小可能导致不收敛
        
        3. **语义置信度调节**
           - 作用：根据回归系数与指标语义的一致性调节评价
           
        
        #### 评价等级参数
        
        1. **评价等级数量**
           - 3级：适合粗略分类
           - 5级：常用，区分度适中（推荐）
           - 7级：适合精细评价
        
        2. **等级标签**
           - 可自定义，如："差,中,良,优"
           - 必须与等级数量匹配
        
        #### 高级参数（Conf调节）
        
        这些参数控制语义置信度机制的强度：
        
        - **保守设置**：(1.2, 1.5, 0.35) - 温和调节
        - **推荐设置**：(1.5, 2.0, 0.30) - 平衡效果
        - **激进设置**：(2.5, 3.0, 0.25) - 强力调节
        """)
    
    # ==================== 保存配置 ====================
    st.markdown("---")
    
    if st.button("💾 保存配置并继续", type="primary", use_container_width=True):
        # 构建指标类型列表（按照indicator_cols的顺序）
        indicator_type_list = [st.session_state['indicator_types'][col] for col in indicator_cols]
        
        # 生成评价等级的三角隶属度参数
        levels = []
        step = 1.0 / n_levels
        for i in range(n_levels):
            if i == 0:
                levels.append((0.0, 0.0, step))
            elif i == n_levels - 1:
                levels.append((1.0 - step, 1.0, 1.0))
            else:
                levels.append((i * step, (i + 0.5) * step, (i + 1) * step))
        
        # 保存模型配置
        st.session_state['model_config'] = {
            'indicator_types': indicator_type_list,
            'regression': {
                'max_iter': max_iter,
                'tol': tol
            },
            'fce': {
                'levels': levels,
                'grade_labels': grade_labels,
                'param_grid': param_grid
            },
            'use_conf': use_conf
        }
        
        st.success("""
        ✅ **配置已保存！**
        
        请前往 **📈 综合评价** 页面运行模型。
        """)
        
        # 显示配置预览
        with st.expander("查看完整配置"):
            st.json(st.session_state['model_config'])