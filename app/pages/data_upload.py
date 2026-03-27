import streamlit as st
import pandas as pd
import numpy as np


def render_page():
    st.markdown("### 数据上传与预处理")

    st.markdown("""
    系统支持任意格式的Excel数据，但必须包含以下列：
    - **样本标识列**
    - **目标变量列**
    - **评价指标列**
    > 🚨 请确保数据格式正确，否则可能导致系统无法正常处理数据！
    """)

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传数据文件（Excel格式）",
        type=['xlsx', 'xls'],
        help="支持 .xlsx 和 .xls 格式"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            if df.empty:
                st.error("❌ 上传的文件为空，请检查 Excel 数据内容。")
                return

            st.success(f"✅ 成功上传数据！共 {len(df)} 行，{len(df.columns)} 列")

            st.session_state['df_uploaded'] = df

            #  数据预览 
            st.subheader("1️⃣ 数据预览")
            col1, col2, col3 = st.columns(3)
            col1.metric("总行数", len(df))
            col2.metric("总列数", len(df.columns))
            col3.metric("数据完整度", f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%")

            with st.expander("查看完整数据", expanded=False):
                st.dataframe(df, use_container_width=True)

            #  列角色分配  
            st.subheader("2️⃣ 设置列角色")

            st.info("""
            请为每列指定其在分析中的角色：
            - **样本ID列**：唯一标识每个评价对象
            - **目标变量列**：真实的评价结果，用于模糊回归与模型验证
            - **评价指标列**：用于综合评价的指标
            """)

            all_columns = df.columns.tolist()
            col1 = st.columns(1)[0]

            with col1:
                # 样本ID列
                sample_id_col = st.selectbox(
                    "样本ID列",
                    options=all_columns,
                    index=0,
                    help="用于标识每个评价对象的列"
                )

                # 时间列
                time_col_options = ['不使用时间列'] + all_columns
                time_col = st.selectbox(
                    "时间列",
                    options=time_col_options,
                    index=0,
                    help="如果数据包含多个时间段，请选择时间列"
                )
                if time_col == '不使用时间列':
                    time_col = None

                # 目标变量列（必选）
                target_col = st.selectbox(
                    "目标变量列（必选，用于模糊回归与语义置信度）",
                    options=all_columns,
                    index=0,
                    help="必须选择，否则系统无法进行模糊回归与语义置信度计算"
                )

            # 评价指标列（排除已选的列）
            excluded_cols = [sample_id_col]
            if time_col:
                excluded_cols.append(time_col)
            if target_col:
                excluded_cols.append(target_col)

            indicator_options = [col for col in all_columns if col not in excluded_cols]

            indicator_cols = st.multiselect(
                "评价指标列（多选）",
                options=indicator_options,
                default=indicator_options[:min(5, len(indicator_options))],
                help="选择用于综合评价的指标列"
            )

            # 验证选择
            if not indicator_cols:
                st.warning("⚠️ 请至少选择一个评价指标列")
                return

            # 显示选择摘要
            st.markdown("---")
            st.success(f"**已选择 {len(indicator_cols)} 个评价指标**")
            with st.expander("查看指标列表"):
                for i, col in enumerate(indicator_cols, 1):
                    st.write(f"{i}. {col}")
            st.markdown("---")

            # 数据质量检查 
            st.subheader("3️⃣ 数据质量检查")

            # 检查指标列的数据类型
            numeric_indicators = []
            non_numeric_indicators = []

            for col in indicator_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_indicators.append(col)
                else:
                    non_numeric_indicators.append(col)

            if non_numeric_indicators:
                st.error(f"❌ 以下指标列包含非数值数据，请检查: {', '.join(non_numeric_indicators)}")
                st.info("💡 建议：确保所有指标列都是数值类型（整数或小数）")
            else:
                st.success("✅ 所有指标列均为数值类型")

            # 检查目标变量列
            target_numeric = pd.to_numeric(df[target_col], errors='coerce')
            target_missing = target_numeric.isna().sum()
            if target_missing > 0:
                st.warning(f"⚠️ 目标变量列 `{target_col}` 存在 {target_missing} 个空值或非数值项，预处理时将一并处理")
            else:
                st.success(f"✅ 目标变量列 `{target_col}` 可正常用于模糊回归")

            # 检查缺失值
            missing_info = df[indicator_cols].isnull().sum()
            missing_cols = missing_info[missing_info > 0]

            if len(missing_cols) > 0:
                st.warning("⚠️ 以下指标列存在缺失值：")
                for col, count in missing_cols.items():
                    st.write(f"- `{col}`: {count} 个缺失值 ({count / len(df) * 100:.1f}%)")
            else:
                st.success("✅ 所有指标列数据完整，无缺失值")

            # 检查重复样本
            if time_col:
                dup_count = df.duplicated(subset=[sample_id_col, time_col]).sum()
            else:
                dup_count = df.duplicated(subset=[sample_id_col]).sum()

            if dup_count > 0:
                st.warning(f"⚠️ 存在 {dup_count} 个重复样本")
            else:
                st.success("✅ 无重复样本")

            #  数据预处理  
            st.subheader("4️⃣ 数据预处理")

            preprocess_col1, preprocess_col2 = st.columns(2)

            with preprocess_col1:
                # 时间筛选（如果有时间列）
                if time_col:
                    unique_times = sorted(df[time_col].dropna().unique())
                    selected_times = st.multiselect(
                        f"选择分析时间段（{time_col}）",
                        options=unique_times,
                        default=[unique_times[-1]] if unique_times else [],
                        help="可选择一个或多个时间段进行分析"
                    )
                else:
                    selected_times = None

            with preprocess_col2:
                # 缺失值处理
                missing_method = st.selectbox(
                    "缺失值处理方式",
                    options=["删除含缺失值的行", "均值填充", "中位数填充", "不处理"],
                    index=0,
                    help="选择如何处理缺失数据"
                )

            # 异常值处理
            outlier_method = st.selectbox(
                "异常值处理方式",
                options=["不处理", "3σ原则（删除）", "四分位数法（删除）", "Winsorize（截尾）"],
                index=0,
                help="选择如何处理极端异常值"
            )

            # 应用预处理按钮
            if st.button("🚀 应用预处理", type="primary", use_container_width=True):
                df_processed = df.copy()

                # 1. 时间筛选
                if time_col and selected_times:
                    df_processed = df_processed[df_processed[time_col].isin(selected_times)]
                    st.info(f"✓ 已筛选时间段: {', '.join(map(str, selected_times))}")

                # 2. 只保留需要的列
                cols_to_keep = [sample_id_col]
                if time_col:
                    cols_to_keep.append(time_col)
                if target_col:
                    cols_to_keep.append(target_col)
                cols_to_keep += indicator_cols

                df_processed = df_processed[cols_to_keep]

                # 3. 需要参与数值清洗的列：指标列 + 目标变量列
                numeric_process_cols = indicator_cols.copy()
                if target_col:
                    numeric_process_cols.append(target_col)

                # 4. 统一转为数值，非法内容转 NaN
                for col in numeric_process_cols:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

                # 5. 缺失值处理
                if missing_method == "删除含缺失值的行":
                    before_len = len(df_processed)
                    df_processed = df_processed.dropna(subset=numeric_process_cols)
                    st.info(f"✓ 已删除 {before_len - len(df_processed)} 行含缺失值的数据（含目标变量）")

                elif missing_method == "均值填充":
                    for col in numeric_process_cols:
                        if df_processed[col].isnull().any():
                            mean_val = df_processed[col].mean()
                            df_processed[col] = df_processed[col].fillna(mean_val)
                    st.info("✓ 已用均值填充缺失值（含目标变量）")

                elif missing_method == "中位数填充":
                    for col in numeric_process_cols:
                        if df_processed[col].isnull().any():
                            median_val = df_processed[col].median()
                            df_processed[col] = df_processed[col].fillna(median_val)
                    st.info("✓ 已用中位数填充缺失值（含目标变量）")

                # 6. 异常值处理（只处理指标列，不处理目标变量）
                if outlier_method == "3σ原则（删除）":
                    before_len = len(df_processed)
                    for col in indicator_cols:
                        mean = df_processed[col].mean()
                        std = df_processed[col].std()
                        df_processed = df_processed[
                            (df_processed[col] >= mean - 3 * std) &
                            (df_processed[col] <= mean + 3 * std)
                        ]
                    st.info(f"✓ 已删除 {before_len - len(df_processed)} 行异常值")

                elif outlier_method == "四分位数法（删除）":
                    before_len = len(df_processed)
                    for col in indicator_cols:
                        q1, q3 = df_processed[col].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        df_processed = df_processed[
                            (df_processed[col] >= q1 - 1.5 * iqr) &
                            (df_processed[col] <= q3 + 1.5 * iqr)
                        ]
                    st.info(f"✓ 已删除 {before_len - len(df_processed)} 行异常值")

                elif outlier_method == "Winsorize（截尾）":
                    for col in indicator_cols:
                        lower, upper = df_processed[col].quantile([0.05, 0.95])
                        df_processed[col] = df_processed[col].clip(lower, upper)
                    st.info("✓ 已对异常值进行截尾处理")

                # 7. 预处理后再校验一次
                if len(df_processed) == 0:
                    st.error("❌ 预处理后数据为空，请调整时间筛选、缺失值处理或异常值处理方式。")
                    return

                if target_col not in df_processed.columns:
                    st.error(f"❌ 预处理后目标变量列 `{target_col}` 丢失，请检查配置。")
                    return

                # 8. 保存处理后数据和配置到 session_state
                st.session_state['df_processed'] = df_processed
                st.session_state['data_config'] = {
                    'sample_id_col': sample_id_col,
                    'time_col': time_col,
                    'target_col': target_col,
                    'indicator_cols': indicator_cols,
                    'n_samples': len(df_processed),
                    'n_indicators': len(indicator_cols)
                }

                st.success(f"""
                ✅ **预处理完成！**

                - 样本数量: {len(df_processed)}
                - 评价指标: {len(indicator_cols)} 个
                - 数据完整度: {(1 - df_processed[indicator_cols].isnull().sum().sum() / (len(df_processed) * len(indicator_cols))) * 100:.1f}%
                """)

                #   处理后数据预览  
                st.subheader("5️⃣ 处理后数据预览")
                st.dataframe(df_processed.head(20), use_container_width=True)

                #   数据导出  
                st.subheader("6️⃣ 数据导出")

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    csv = df_processed.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载为 CSV",
                        data=csv,
                        file_name="processed_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with download_col2:
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_processed.to_excel(writer, index=False, sheet_name='处理后数据')

                    st.download_button(
                        label="📥 下载为 Excel",
                        data=output.getvalue(),
                        file_name="processed_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                #   下一步提示  
                st.info("""
                ✅ **数据准备完成！**

                请前往 **模型配置** 页面继续设置指标类型和模型参数。
                """)

        except Exception as e:
            st.error(f"❌ 数据处理失败: {str(e)}")
            st.info("💡 请检查文件格式是否正确，确保是有效的 Excel 文件")

    else:
        #   示例数据下载  
        st.info("💡 **还没有数据？** 可以下载示例数据进行测试")

        np.random.seed(42)
        cities = ['北京', '上海', '深圳', '广州', '杭州', '南京', '武汉', '成都']
        years = [2021, 2022, 2023]

        sample_data = []
        for year in years:
            for city in cities:
                sample_data.append({
                    '城市': city,
                    '年份': year,
                    '创新指数': np.random.uniform(70, 95),
                    'GDP(亿元)': np.random.uniform(1000, 4000),
                    'R&D投入强度(%)': np.random.uniform(2, 6),
                    '专利授权数(件)': np.random.randint(1000, 10000),
                    '高新技术企业数(家)': np.random.randint(500, 5000),
                    '科研人员数(万人)': np.random.uniform(5, 50),
                    '技术市场成交额(亿元)': np.random.uniform(50, 500)
                })

        sample_df = pd.DataFrame(sample_data)

        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='示例数据')

            instructions = pd.DataFrame({
                '列名': ['城市', '年份', '创新指数', 'GDP(亿元)', '...'],
                '说明': [
                    '样本ID列（必需）',
                    '时间列（可选）',
                    '目标变量（必选）',
                    '评价指标1',
                    '更多评价指标...'
                ],
                '数据类型': ['文本', '整数', '小数', '小数', '小数']
            })
            instructions.to_excel(writer, index=False, sheet_name='使用说明')

        st.download_button(
            label="📥 下载示例数据",
            data=output.getvalue(),
            file_name="示例_城市创新数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.markdown("""
        **示例数据说明**：
        - 包含 8 个城市，3 年数据（2021-2023）
        - 1 个样本ID列（城市）
        - 1 个时间列（年份）
        - 1 个目标变量（创新指数）
        - 5 个评价指标
        """)