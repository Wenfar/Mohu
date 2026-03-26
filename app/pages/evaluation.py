import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_path)

from models.decision_engine import DecisionEngine

def render_page():
    st.title("📈 综合评价")

    if 'df_processed' not in st.session_state:
        st.warning("⚠️ 请先在 **📊 数据上传** 页面上传数据")
        return

    if 'model_config' not in st.session_state:
        st.warning("⚠️ 请先在 **⚙️ 模型配置** 页面配置模型参数")
        return

    df = st.session_state['df_processed']
    data_config = st.session_state['data_config']
    model_config = st.session_state['model_config']

   
    # ==================== 1. 数据准备 ====================
    st.subheader("1️⃣ 数据准备")

    sample_names   = df[data_config['sample_id_col']].tolist()
    indicator_cols = data_config['indicator_cols']
    X = df[indicator_cols].values.astype(float)

    # if data_config['target_col']:
    #     y_true = df[data_config['target_col']].values.astype(float)
    #     has_target = True
    # else:
    #     y_true     = None
    #     has_target = False
    target_col = data_config.get('target_col')

    if not target_col:
        st.error("❌ 当前 data_config 中没有目标变量列。请返回“数据上传与预处理”页面，重新选择目标变量后点击“应用预处理”。")
        st.stop()

    if target_col not in df.columns:
        st.error(f"❌ 目标变量列 `{target_col}` 不在当前处理后数据中。请重新执行“应用预处理”。")
        st.write("当前数据列：", df.columns.tolist())
        st.stop()

    y_series = pd.to_numeric(df[target_col], errors='coerce')
    if y_series.isna().any():
        st.error(f"❌ 目标变量列 `{target_col}` 含有空值或非数值内容，无法进行模糊回归。")
        st.write(df[[target_col]].head(10))
        st.stop()

    y_true = y_series.values.astype(float)
    has_target = True
    
    
    st.info(f"""
    ✅ 数据准备完成
    - 样本名称：从 `{data_config['sample_id_col']}` 列提取
    - 指标数据：{X.shape[0]} 行 × {X.shape[1]} 列
    - 目标变量：{'已提取' if has_target else '未提供'}
    """)

    # ==================== 2. Z-score 标准化（与 notebook 保持一致）====================
    # 注意：此处标准化仅供模糊回归使用，用于校准结构方向（beta）。
    # EntropyFCEModel 内部会对原始 X 再做一次 Min-Max 正向化，两者不冲突。
    st.subheader("2️⃣ 数据标准化")

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-10)

    if has_target:
        y_mean, y_std = y_true.mean(), y_true.std()
        y_scaled = (y_true - y_mean) / (y_std + 1e-10)
    else:
        y_scaled = None

    st.success("✅ 已使用 Z-score 标准化（与模型训练 notebook 保持一致）")

    # ==================== 3. 模型训练与评价 ====================
    st.subheader("3️⃣ 模型训练与评价")

    if st.button("🚀 开始运行模型", type="primary", use_container_width=True):

        with st.spinner("正在构建决策引擎..."):
            try:
                # 与 notebook Cell 12 保持一致：不传 fce_config，
                # 由 DecisionEngine 使用默认 5 级参数，
                # 再通过 model_config 里的 grade_labels / levels 覆盖（如需）。
                engine = DecisionEngine(
                    regression_config={
                        "n":       X_scaled.shape[1],
                        "number":  X_scaled.shape[0],
                        "max_iter": model_config['regression']['max_iter'],
                        "tol":      model_config['regression']['tol']
                    },
                    indicator_type=model_config['indicator_types'],
                    fce_config={
                        "levels":       model_config['fce']['levels'],
                        "grade_labels": model_config['fce']['grade_labels'],
                        "param_grid":   model_config['fce']['param_grid']
                    }
                )
                st.success("✅ 决策引擎构建成功")
            except Exception as e:
                st.error(f"❌ 引擎构建失败: {str(e)}")
                return

        with st.spinner("正在运行模型..."):
            try:
                # 与 notebook Cell 13 保持一致：
                # X_eval = X_scaled（Z-score 后），X_reg = X_scaled，y_reg = y_scaled
                result = engine.full_decision(
                    X_eval=X_scaled,
                    X_reg=X_scaled,
                    y_reg=y_scaled if has_target else None,
                    city_names=sample_names,
                    use_conf=True,
                    conf_mode='discrete'
                )

                st.session_state['evaluation_result'] = result
                st.session_state['engine']   = engine
                st.session_state['X_scaled'] = X_scaled
                st.session_state['y_scaled'] = y_scaled
                st.success("✅ 模型运行完成！")
            except Exception as e:
                st.error(f"❌ 模型运行失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ==================== 4. 结果展示 ====================
    if 'evaluation_result' in st.session_state:
        result = st.session_state['evaluation_result']

        scores  = np.array([r["score"]      for r in result["evaluation"]])
        grades  = [r["grade"]               for r in result["evaluation"]]
        weights = result['weights']

        rank_order    = np.argsort(-scores)
        ranked_names  = [sample_names[i] for i in rank_order]
        ranked_scores = scores[rank_order]
        ranked_grades = [grades[i]        for i in rank_order]

        result_df = pd.DataFrame({
            data_config['sample_id_col']: sample_names,
            '综合得分': scores,
            '评价等级': grades,
            '排名': np.argsort(-scores) + 1
        })
        if has_target:
            result_df['真实值']   = y_true
            result_df['真实排名'] = np.argsort(-y_true) + 1
            result_df['排名差异'] = result_df['排名'] - result_df['真实排名']

        result_df_sorted = result_df.sort_values('排名')

        weight_df = pd.DataFrame({
            '指标': indicator_cols,
            '权重': weights,
            '权重(%)': weights * 100,
            '类型': ['正向 ✅' if t == 1 else '负向 ❌'
                    for t in model_config['indicator_types']]
        }).sort_values('权重', ascending=False)

        # ── 全局样式 ──────────────────────────────────────────────────
        st.markdown("""
        <style>
        :root {
            --c-bg:      #f0f2f6; --c-surface: #ffffff; --c-border:  #e0e3ec; 
            --c-text:    #1a1d27; --c-muted:   #6b7080;
            --c-accent:  #6c63ff; --c-accent2: #00d4aa; --c-warn:    #ff6b6b;
            
        }
        .ev-card { background:var(--c-surface); border:1px solid var(--c-border);
                   border-radius:16px; padding:24px 28px; margin-bottom:20px; }
        .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
        .kpi-box  { background:var(--c-surface); border:1px solid var(--c-border);
                    border-radius:12px; padding:18px 16px; text-align:center; }
        .kpi-box.accent  { border-color:var(--c-accent);  }
        .kpi-box.accent2 { border-color:var(--c-accent2); }
        .kpi-box.warn    { border-color:var(--c-warn);    }
        .kpi-label { font-size:11px; color:var(--c-muted); letter-spacing:.08em;
                     text-transform:uppercase; margin-bottom:6px; }
        .kpi-value { font-size:26px; font-weight:800; color:var(--c-text); line-height:1; }
        .kpi-sub   { font-size:11px; color:var(--c-muted); margin-top:4px;
                     white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .rank-row  { display:flex; align-items:center; gap:14px; padding:10px 0;
                     border-bottom:1px solid var(--c-border); }
        .rank-row:last-child { border-bottom:none; }
        .rank-num  { font-size:13px; font-weight:800; width:28px; text-align:center;
                     flex-shrink:0; color:var(--c-muted); }
        .rank-num.gold   { color:var(--c-gold); }
        .rank-num.silver { color:#c0c0c0; }
        .rank-num.bronze { color:#cd7f32; }
        .rank-name  { flex:1; font-size:14px; font-weight:600; color:var(--c-text);
                      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .rank-bar-wrap { width:140px; flex-shrink:0; }
        .rank-bar-bg   { background:var(--c-border); border-radius:99px; height:6px; overflow:hidden; }
        .rank-bar-fill { height:6px; border-radius:99px;
                         background:linear-gradient(90deg,var(--c-accent),var(--c-accent2)); }
        .rank-score { width:52px; text-align:right; font-size:13px; font-weight:700;
                      color:var(--c-text); flex-shrink:0; }
        .rank-grade { font-size:11px; padding:2px 8px; border-radius:99px;
                      background:var(--c-border); color:var(--c-muted); flex-shrink:0; }
        .wt-row  { display:flex; align-items:center; gap:10px; padding:8px 0;
                   border-bottom:1px solid var(--c-border); }
        .wt-row:last-child { border-bottom:none; }
        .wt-name { flex:1; font-size:13px; color:var(--c-text); font-weight:500;
                   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .wt-type { font-size:10px; padding:2px 6px; border-radius:99px; flex-shrink:0; }
        .wt-type.pos { background:#1a3a2a; color:#00d4aa; }
        .wt-type.neg { background:#3a1a1a; color:#ff6b6b; }
        .wt-bar-wrap { width:120px; flex-shrink:0; }
        .wt-bar-bg   { background:var(--c-border); border-radius:99px; height:5px; overflow:hidden; }
        .wt-bar-fill { height:5px; border-radius:99px; background:var(--c-accent); }
        .wt-pct { width:42px; text-align:right; font-size:12px; font-weight:700;
                  color:var(--c-text); flex-shrink:0; }
        .conf-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:10px; }
        .conf-chip { background:var(--c-bg); border:1px solid var(--c-border);
                     border-radius:10px; padding:10px 14px;
                     display:flex; flex-direction:column; gap:6px; }
        .conf-chip.high { border-color:var(--c-accent2); }
        .conf-chip.mid  { border-color:#f0a500; }
        .conf-chip.low  { border-color:var(--c-warn); }
        .conf-chip-name { font-size:12px; font-weight:600; color:var(--c-text);
                          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .conf-chip-bar-bg   { background:var(--c-border); border-radius:99px; height:4px; }
        .conf-chip-bar-fill { height:4px; border-radius:99px; }
        .conf-chip-bar-fill.high { background:var(--c-accent2); }
        .conf-chip-bar-fill.mid  { background:#f0a500; }
        .conf-chip-bar-fill.low  { background:var(--c-warn); }
        .conf-chip-label { font-size:11px; color:var(--c-muted); }
        .corr-row { display:flex; align-items:center; gap:16px; padding:10px 0;
                    border-bottom:1px solid var(--c-border); }
        .corr-row:last-child { border-bottom:none; }
        .corr-name { width:160px; font-size:13px; color:var(--c-muted); flex-shrink:0; }
        .corr-val  { font-size:22px; font-weight:800; color:var(--c-text); width:70px; flex-shrink:0; }
        .corr-bar-wrap { flex:1; }
        .corr-bar-bg   { background:var(--c-border); border-radius:99px; height:6px; overflow:hidden; }
        .corr-bar-fill { height:6px; border-radius:99px;
                         background:linear-gradient(90deg,var(--c-accent),var(--c-accent2)); }
        .corr-p { font-size:11px; color:var(--c-muted); width:80px; text-align:right; flex-shrink:0; }
        </style>
        """, unsafe_allow_html=True)

        # ── KPI 概览 ────────────────────────────────────────────────
        st.markdown("#### 📊 评价概览")
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-box accent">
            <div class="kpi-label">最高分</div>
            <div class="kpi-value">{scores.max():.4f}</div>
            <div class="kpi-sub">{sample_names[scores.argmax()]}</div>
          </div>
          <div class="kpi-box warn">
            <div class="kpi-label">最低分</div>
            <div class="kpi-value">{scores.min():.4f}</div>
            <div class="kpi-sub">{sample_names[scores.argmin()]}</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">平均分</div>
            <div class="kpi-value">{scores.mean():.4f}</div>
            <div class="kpi-sub">全样本</div>
          </div>
          <div class="kpi-box accent2">
            <div class="kpi-label">标准差</div>
            <div class="kpi-value">{scores.std():.4f}</div>
            <div class="kpi-sub">离散程度</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 排名榜单 + 完整数据 ─────────────────────────────────────
        tab_rank, tab_full = st.tabs(["🏆 排名榜单", "📋 完整数据"])
        with tab_rank:
            top_n = st.slider("显示前 N 名", min_value=5,
                              max_value=min(50, len(sample_names)),
                              value=min(20, len(sample_names)), step=5)
            score_max = ranked_scores[0] if ranked_scores[0] > 0 else 1.0
            rows_html = ""
            for i in range(top_n):
                rk      = i + 1
                num_cls = "gold" if rk==1 else "silver" if rk==2 else "bronze" if rk==3 else ""
                medal   = "🥇" if rk==1 else "🥈" if rk==2 else "🥉" if rk==3 else str(rk)
                bar_pct = ranked_scores[i] / score_max * 100
                rows_html += f"""
                <div class="rank-row">
                  <div class="rank-num {num_cls}">{medal}</div>
                  <div class="rank-name">{ranked_names[i]}</div>
                  <div class="rank-bar-wrap">
                    <div class="rank-bar-bg">
                      <div class="rank-bar-fill" style="width:{bar_pct:.1f}%"></div>
                    </div>
                  </div>
                  <div class="rank-score">{ranked_scores[i]:.4f}</div>
                  <div class="rank-grade">{ranked_grades[i]}</div>
                </div>"""
            st.markdown(f'<div class="ev-card">{rows_html}</div>', unsafe_allow_html=True)
        with tab_full:
            st.dataframe(
                result_df_sorted.style.background_gradient(subset=['综合得分'], cmap='RdYlGn'),
                use_container_width=True, height=420
            )

        # ── 指标权重 ────────────────────────────────────────────────
        st.markdown("#### ⚖️ 指标权重分布")
        wt_max = weight_df['权重(%)'].iloc[0]
        wt_rows_html = ""
        for _, row in weight_df.iterrows():
            bar_w    = row['权重(%)'] / wt_max * 100
            type_cls = "pos" if "正向" in row['类型'] else "neg"
            type_lbl = "正向"  if "正向" in row['类型'] else "负向"
            wt_rows_html += f"""
            <div class="wt-row">
              <div class="wt-name">{row['指标']}</div>
              <div class="wt-type {type_cls}">{type_lbl}</div>
              <div class="wt-bar-wrap">
                <div class="wt-bar-bg">
                  <div class="wt-bar-fill" style="width:{bar_w:.1f}%"></div>
                </div>
              </div>
              <div class="wt-pct">{row['权重(%)']:.2f}%</div>
            </div>"""
        st.markdown(f'<div class="ev-card">{wt_rows_html}</div>', unsafe_allow_html=True)

        

        # ── 语义置信度 ──────────────────────────────────────────────
        if model_config['use_conf'] and 'semantic_confidence' in result:
            conf = result['semantic_confidence']
    
            # ★ 关键修复：conf 为 None 时给出提示而不是报错
            if conf is None:
                if conf is None:
                    st.info("ℹ️ 当前已默认启用语义置信度调节，但由于未提供目标变量列，系统无法完成模糊回归，因此本次未生成语义置信度。")
            else:
                st.markdown("#### 🎯 语义置信度")
                chips_html     = ""
                low_conf_names = []
                for name, c in zip(indicator_cols, conf):
                    if c >= 0.8:
                        cls, label = "high", "高置信"
                    elif c >= 0.5:
                        cls, label = "mid",  "中置信"
                    else:
                        cls, label = "low",  "低置信"
                        low_conf_names.append(name)
                    chips_html += f"""
                    <div class="conf-chip {cls}">
                    <div class="conf-chip-name">{name}</div>
                    <div class="conf-chip-bar-bg">
                        <div class="conf-chip-bar-fill {cls}" style="width:{c*100:.0f}%"></div>
                    </div>
                    <div class="conf-chip-label">{label} · {c:.3f}</div>
                    </div>"""
                st.markdown(
                    f'<div class="ev-card"><div class="conf-grid">{chips_html}</div></div>',
                    unsafe_allow_html=True
                )


        # ── 导出 ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💾 结果导出")
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            csv = result_df_sorted.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 评价结果 (CSV)", csv,
                               "evaluation_result.csv", "text/csv",
                               use_container_width=True)
        with export_col2:
            weight_csv = weight_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 指标权重 (CSV)", weight_csv,
                               "indicator_weights.csv", "text/csv",
                               use_container_width=True)
        with export_col3:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df_sorted.to_excel(writer, sheet_name='评价结果',  index=False)
                weight_df.to_excel(       writer, sheet_name='指标权重',  index=False)
                if model_config['use_conf'] and 'semantic_confidence' in result:
                    pd.DataFrame({
                        '指标': indicator_cols,
                        '置信度': result['semantic_confidence']
                    }).to_excel(writer, sheet_name='语义置信度', index=False)
            st.download_button("📥 完整报告 (Excel)", output.getvalue(),
                               "complete_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        