import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import google.generativeai as genai
import plotly.express as px
import os
from textblob import TextBlob

# ==========================================
# 1. 页面级配置 & 全局 CSS 注入 
# ==========================================
st.set_page_config(page_title="Bubble Shooter 战情大屏", page_icon="📱", layout="wide")

st.markdown("""
<style>
    .report-header { font-size: 2.8em; font-weight: 800; border-bottom: 4px solid #1f77b4; padding-bottom: 10px; margin-bottom: 30px;}
    .section-title { font-size: 1.8em; font-weight: 700; color: #2c3e50; margin-top: 40px; margin-bottom: 20px; border-left: 5px solid #e74c3c; padding-left: 15px;}
    div[data-testid="metric-container"] { background-color: #ffffff; border-radius: 8px; padding: 20px; border-top: 4px solid #1f77b4; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心功能函数与数据清洗
# ==========================================
def init_gemini(api_key):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash') 
    except Exception as e:
        st.error(f"Gemini 初始化失败: {e}")
        return None

def analyze_text_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

# 轻量级业务归因引擎
def categorize_issue(text):
    text = str(text).lower()
    if re.search(r'ad|ads|commercial|video|广告', text): return '📺 广告体验'
    if re.search(r'crash|bug|freeze|loading|闪退|卡顿|黑屏', text): return '💥 性能与Bug'
    if re.search(r'hard|difficult|level|pass|难|卡关', text): return '🧗 关卡与难度'
    if re.search(r'money|pay|coin|gem|充值|钱', text): return '💰 商业化与付费'
    return '💬 常规反馈'

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['content'] = df['content'].fillna('')
        df['at'] = pd.to_datetime(df['at'])
        df['date'] = df['at'].dt.date
        df['week'] = df['at'].dt.isocalendar().week
        
        # 情感与业务双重打标
        df['star_rating'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
        df['nlp_sentiment_score'] = df['content'].apply(analyze_text_sentiment)
        df['true_sentiment'] = pd.cut(df['nlp_sentiment_score'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
        df['business_tag'] = df['content'].apply(categorize_issue)
        
        return df
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None

def get_zeus_style_insight(model, df):
    valid_reviews = df[df['content'].str.len() > 20]
    sample_size = min(80, len(valid_reviews))
    if sample_size == 0:
        return "⚠️ 当前筛选条件下无足够有效的文本评论供 AI 分析。"
        
    sample_reviews = valid_reviews.sample(sample_size)['content'].tolist()    
    text_to_analyze = "\n".join(sample_reviews)
    
    prompt = f"""
    作为高级游戏数据分析师，请阅读以下抽样的真实玩家评论数据，生成一份结构化商业诊断报告。
    你必须严格使用以下 Markdown 结构和表情符号进行输出，保持专业、客观，避免空话：

    ### 🤖 AI 深度洞察
    **📋 执行摘要**
    (用一段话概括当前选定数据范围内的核心口碑盘面与最致命危机)

    **💡 关键发现**
    (列举3-4条最突出的痛点或爽点，必须带有极强的游戏业务感，如“商业化变现”、“数值平衡”、“系统崩溃”等，并引用玩家原话作为佐证)

    **👥 核心用户画像**
    (描述最容易因为上述问题退坑的玩家特征)

    ### 🚀 战略改进建议
    * 🔧 **短期（1-3个月，针对研发与运营）**：(列出亟待修复的 Bug 或必须调整的策略)
    * 🎯 **中期（3-6个月，针对策划与发行）**：(系统玩法或商业化节奏的优化建议)
    * 🌟 **长期（6-12个月，针对大盘生态）**：(游戏核心机制或长期留存的升级方向)

    评论数据源:
    {text_to_analyze}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ **触发 API 频率限制 (429 Error)**。由于当前为免费配额，请等待约 30-60 秒后重试，或配置付费 API Key。"
        return f"AI 分析时出错: {e}"

@st.cache_data(show_spinner=False)
def generate_cached_report(_model, data_fingerprint, df):
    return get_zeus_style_insight(_model, df)


# ==========================================
# 3. 网页主程序与全局控制台
# ==========================================
# 自动读取环境变量或 secrets 中的 API Key
YOUR_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
gemini_model = init_gemini(YOUR_GEMINI_API_KEY)
DATA_FILENAME = "linkdesks.pop.bubblegames.bubbleshooter_all_reviews.csv"

if not os.path.exists(DATA_FILENAME):
    st.error(f"🚨 找不到数据文件：`{DATA_FILENAME}`。请确保它已经与本代码文件存放在同一个文件夹中！")
else:
    with st.spinner("正在加载底层业务数据..."):
        raw_df = load_and_clean_data(DATA_FILENAME)
        
    if raw_df is not None:
        # --- 全局侧边栏 (Sidebar) ---
        with st.sidebar:
            st.header("⚙️ 战情控制台")
            st.info("调整以下参数将实时刷新右侧大屏所有数据联动。")
            
            score_filter = st.multiselect("⭐ 星级过滤 (Score):", options=[5, 4, 3, 2, 1], default=[5, 4, 3, 2, 1])
            available_versions = raw_df['reviewCreatedVersion'].dropna().unique().tolist()
            version_filter = st.multiselect("📱 版本过滤 (Version):", options=available_versions, default=[])
            search_kw = st.text_input("🔑 关键词检索:", placeholder="如 'crash' 或 'level'")
            
            st.divider()
            if YOUR_GEMINI_API_KEY:
                st.success("🟢 AI 引擎已连接")
            else:
                st.warning("🔴 AI 引擎未配置")

        # --- 全局数据过滤逻辑 ---
        filtered_df = raw_df[raw_df['score'].isin(score_filter)]
        if version_filter:
            filtered_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(version_filter)]
        if search_kw:
            filtered_df = filtered_df[filtered_df['content'].str.contains(search_kw, case=False, na=False)]

        # ================= 大屏内容区 =================
        st.markdown('<div class="report-header">📱 Bubble Shooter 战情体检大屏</div>', unsafe_allow_html=True)
        
        if filtered_df.empty:
            st.warning("⚠️ 当前筛选条件下无数据，请放宽左侧侧边栏参数。")
        else:
            st.caption(f"🎯 当前匹配样本：**{len(filtered_df):,}** 条 | 📅 分析区间: {filtered_df['date'].min()} 至 {filtered_df['date'].max()}")

            # ---------------- 模块 1: 核心指标 ----------------
            avg_score = filtered_df['score'].mean()
            avg_sentiment = filtered_df['nlp_sentiment_score'].mean()
            pos_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Positive']) / len(filtered_df)) * 100
            neg_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Negative']) / len(filtered_df)) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总分析评价数", f"{len(filtered_df):,}")
            c2.metric("平均星级评分", f"{avg_score:.2f} / 5.0")
            c3.metric("NLP 情感均分", f"{avg_sentiment:.2f}", delta=">0为正向, <0为负向", delta_color="off")
            c4.metric("负面口碑占比", f"{neg_pct:.1f}%", delta=f"好评 {pos_pct:.1f}%", delta_color="inverse")

            st.divider() # 分割线

            # ---------------- 模块 2: 业务大盘与异常监测 (图表区) ----------------
            st.markdown('<div class="section-title">📊 业务大盘与异常监测</div>', unsafe_allow_html=True)
            
            # 周度趋势折线图
            weekly_trend = filtered_df.groupby('week').agg(
                avg_score=('score', 'mean'),
                review_count=('score', 'count')
            ).reset_index()
            
            if not weekly_trend.empty:
                fig_trend = px.line(weekly_trend, x='week', y='avg_score', text=weekly_trend['avg_score'].round(2),
                                    title="自然周平均星级走势 (结合图表寻找口碑拐点)", markers=True)
                fig_trend.update_traces(textposition="top center", line_color='#1f77b4')
                st.plotly_chart(fig_trend, use_container_width=True)

            # 树形图与箱线图并列
            col_tree, col_box = st.columns(2)
            with col_tree:
                pain_df = filtered_df[filtered_df['score'] <= 3]
                if not pain_df.empty:
                    tag_stats = pain_df.groupby('business_tag').size().reset_index(name='count')
                    fig_tree = px.treemap(
                        tag_stats, path=['business_tag'], values='count',
                        color='count', color_continuous_scale='Reds',
                        title="🗺️ 中差评痛点归因矩阵 (色块越深客诉越严重)"
                    )
                    fig_tree.update_layout(margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(fig_tree, use_container_width=True)
                else:
                    st.info("当前数据集中无足够的中差评数据生成归因矩阵。")

            with col_box:
                version_counts = filtered_df['reviewCreatedVersion'].value_counts()
                valid_versions = version_counts[version_counts > 5].index
                box_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(valid_versions)]
                
                if not box_df.empty:
                    fig_box = px.box(
                        box_df, x="reviewCreatedVersion", y="score", color="reviewCreatedVersion",
                        title="📦 主流版本质量分布 (箱体越长代表口碑分化越剧烈)",
                        points="outliers" 
                    )
                    fig_box.update_layout(margin=dict(t=40, l=10, r=10, b=10), showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("单版本数据量不足，无法绘制分布箱线图。")

            st.divider() # 分割线

            # ---------------- 模块 3: 动态 AI 商业诊断 ----------------
            st.markdown('<div class="section-title">🤖 AI 动态商业诊断</div>', unsafe_allow_html=True)
            
            data_fingerprint = f"{len(filtered_df)}_{filtered_df['at'].max()}"
            
            btn_c1, btn_c2, _ = st.columns([2, 2, 6])
            with btn_c1:
                generate_clicked = st.button("⚡ 立即生成/刷新 AI 报告", type="primary", use_container_width=True)
            with btn_c2:
                if st.button("🗑️ 清除云端缓存", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
                    
            if "ai_report_display" not in st.session_state:
                st.session_state.ai_report_display = None

            if generate_clicked:
                if not YOUR_GEMINI_API_KEY:
                    st.error("未配置 API Key，无法呼叫 AI 大模型。")
                else:
                    with st.spinner('AI 正在深度研判当前大盘数据... (若条件不变将秒读缓存)'):
                        report_content = generate_cached_report(gemini_model, data_fingerprint, filtered_df)
                        
                        if "出错" not in report_content and "429 Error" not in report_content:
                            st.session_state.ai_report_display = report_content
                            st.success("✅ 报告就绪！已部署至云端缓存。")
                        else:
                            st.error(report_content)

            if st.session_state.ai_report_display:
                st.markdown(
                    f"<div style='background-color: #f8f9fa; padding: 30px; border-radius: 12px; border: 1px solid #e9ecef; box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>"
                    f"{st.session_state.ai_report_display}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.info("👆 点击上方按钮，AI 将综合上方的业务图表盘面为您生成深度高管汇报纪要。")

            st.divider()

            # ---------------- 模块 4: 文本语义下钻 ----------------
            st.markdown('<div class="section-title">👁️ 原始文本语义聚类 (词云)</div>', unsafe_allow_html=True)
            c_neg, c_pos = st.columns(2)
            
            with c_neg:
                st.subheader("🔴 负面高频词聚类")
                neg_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Negative']['content'].tolist())
                if neg_words:
                    wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
                    fig_n, ax_n = plt.subplots(figsize=(8, 4))
                    ax_n.imshow(wordcloud_neg)
                    ax_n.axis('off')
                    st.pyplot(fig_n)
                else:
                    st.info("暂无足够的负面评价")

            with c_pos:
                st.subheader("🟢 正面高频词聚类")
                pos_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Positive']['content'].tolist())
                if pos_words:
                    wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
                    fig_p, ax_p = plt.subplots(figsize=(8, 4))
                    ax_p.imshow(wordcloud_pos)
                    ax_p.axis('off')
                    st.pyplot(fig_p)
                else:
                    st.info("暂无足够的正面评价")

            st.divider()

            # ---------------- 模块 5: 原始明细表 ----------------
            st.markdown('<div class="section-title">🔍 玩家原声定位明细</div>', unsafe_allow_html=True)
            display_df = filtered_df[['at', 'score', 'business_tag', 'content', 'reviewCreatedVersion']].sort_values(by='at', ascending=False)
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                height=500, 
                hide_index=True, 
                column_config={
                    "at": st.column_config.DatetimeColumn("评论时间", format="YYYY-MM-DD HH:mm"),
                    "score": st.column_config.NumberColumn("评分", format="%d ⭐"),
                    "business_tag": st.column_config.TextColumn("AI初步归因"),
                    "content": st.column_config.TextColumn("玩家原始评论", width="large"),
                    "reviewCreatedVersion": st.column_config.TextColumn("发生版本")
                }
            )



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import re
# import google.generativeai as genai
# import plotly.express as px
# import os
# from textblob import TextBlob

# # ==========================================
# # 1. 页面级配置 & 全局 CSS 注入 
# # ==========================================
# st.set_page_config(page_title="Bubble Shooter 智能体检报告", page_icon="📱", layout="wide")

# st.markdown("""
# <style>
#     .report-header { font-size: 2.5em; font-weight: 800; border-bottom: 3px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;}
#     .section-title { font-size: 1.5em; font-weight: 700; color: #1f77b4; margin-top: 20px; margin-bottom: 15px;}
#     div[data-testid="metric-container"] { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1f77b4;}
# </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 2. 核心功能函数与数据清洗
# # ==========================================
# def init_gemini(api_key):
#     if not api_key: return None
#     try:
#         genai.configure(api_key=api_key)
#         return genai.GenerativeModel('gemini-2.5-flash') 
#     except Exception as e:
#         st.error(f"Gemini 初始化失败: {e}")
#         return None

# def analyze_text_sentiment(text):
#     if not isinstance(text, str) or not text.strip():
#         return 0.0
#     return TextBlob(text).sentiment.polarity

# # 【新增融合】：轻量级业务归因引擎
# def categorize_issue(text):
#     text = str(text).lower()
#     if re.search(r'ad|ads|commercial|video|广告', text): return '📺 广告体验'
#     if re.search(r'crash|bug|freeze|loading|闪退|卡顿|黑屏', text): return '💥 性能与Bug'
#     if re.search(r'hard|difficult|level|pass|难|卡关', text): return '🧗 关卡与难度'
#     if re.search(r'money|pay|coin|gem|充值|钱', text): return '💰 商业化与付费'
#     return '💬 常规反馈'

# @st.cache_data
# def load_and_clean_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         df['content'] = df['content'].fillna('')
#         df['at'] = pd.to_datetime(df['at'])
#         df['date'] = df['at'].dt.date
#         df['week'] = df['at'].dt.isocalendar().week
        
#         # 情感与业务双重打标
#         df['star_rating'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
#         df['nlp_sentiment_score'] = df['content'].apply(analyze_text_sentiment)
#         df['true_sentiment'] = pd.cut(df['nlp_sentiment_score'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
#         df['business_tag'] = df['content'].apply(categorize_issue)
        
#         return df
#     except Exception as e:
#         st.error(f"数据解析失败: {e}")
#         return None

# def get_zeus_style_insight(model, df):
#     # 为防止超限，抽取最多 80 条有意义的评论
#     valid_reviews = df[df['content'].str.len() > 20]
#     sample_size = min(80, len(valid_reviews))
#     if sample_size == 0:
#         return "⚠️ 当前筛选条件下无足够有效的文本评论供 AI 分析。"
        
#     sample_reviews = valid_reviews.sample(sample_size)['content'].tolist()    
#     text_to_analyze = "\n".join(sample_reviews)
    
#     prompt = f"""
#     作为传音高级游戏数据分析师，请阅读以下抽样的真实玩家评论数据，生成一份结构化商业诊断报告。
#     你必须严格使用以下 Markdown 结构和表情符号进行输出，保持专业、客观，避免空话：

#     ### 🤖 AI 深度洞察
#     **📋 执行摘要**
#     (用一段话概括当前选定数据范围内的核心口碑盘面与最致命危机)

#     **💡 关键发现**
#     (列举3-4条最突出的痛点或爽点，必须带有极强的游戏业务感，如“商业化变现”、“数值平衡”、“系统崩溃”等，并引用玩家原话作为佐证)

#     **👥 核心用户画像**
#     (描述最容易因为上述问题退坑的玩家特征)

#     ### 🚀 战略改进建议
#     * 🔧 **短期（1-3个月，针对研发与运营）**：(列出亟待修复的 Bug 或必须调整的策略)
#     * 🎯 **中期（3-6个月，针对策划与发行）**：(系统玩法或商业化节奏的优化建议)
#     * 🌟 **长期（6-12个月，针对大盘生态）**：(游戏核心机制或长期留存的升级方向)

#     评论数据源:
#     {text_to_analyze}
#     """
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         if "429" in str(e):
#             return "⚠️ **触发 API 频率限制 (429 Error)**。由于当前为免费配额，请等待约 30-60 秒后重试，或配置付费 API Key。"
#         return f"AI 分析时出错: {e}"

# @st.cache_data(show_spinner=False)
# def generate_cached_report(_model, data_fingerprint, df):
#     return get_zeus_style_insight(_model, df)


# # ==========================================
# # 3. 网页主程序与全局控制台
# # ==========================================
# YOUR_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
# gemini_model = init_gemini(YOUR_GEMINI_API_KEY)
# DATA_FILENAME = "linkdesks.pop.bubblegames.bubbleshooter_all_reviews.csv"

# if not os.path.exists(DATA_FILENAME):
#     st.error(f"🚨 找不到数据文件：`{DATA_FILENAME}`。请确保它已经与本代码文件存放在同一个文件夹中！")
# else:
#     with st.spinner("正在加载底层业务数据..."):
#         raw_df = load_and_clean_data(DATA_FILENAME)
        
#     if raw_df is not None:
#         # --- 全局侧边栏 (Sidebar) 融合 ---
#         with st.sidebar:
#             st.header("⚙️ 全局数据控制台")
#             st.info("调整以下参数将实时刷新右侧所有大盘数据、图表及 AI 报告。")
            
#             score_filter = st.multiselect("⭐ 星级过滤 (Score):", options=[5, 4, 3, 2, 1], default=[5, 4, 3, 2, 1])
#             available_versions = raw_df['reviewCreatedVersion'].dropna().unique().tolist()
#             version_filter = st.multiselect("📱 版本过滤 (Version):", options=available_versions, default=[])
#             search_kw = st.text_input("🔑 关键词检索:", placeholder="如 'crash' 或 'level'")
            
#             st.divider()
#             if YOUR_GEMINI_API_KEY:
#                 st.success("🟢 AI 引擎已连接")
#             else:
#                 st.warning("🔴 AI 引擎未配置")

#         # --- 全局数据过滤逻辑 ---
#         filtered_df = raw_df[raw_df['score'].isin(score_filter)]
#         if version_filter:
#             filtered_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(version_filter)]
#         if search_kw:
#             filtered_df = filtered_df[filtered_df['content'].str.contains(search_kw, case=False, na=False)]

#         # --- 页面头部 ---
#         st.markdown('<div class="report-header">📱 Bubble Shooter 智能体检仪表盘</div>', unsafe_allow_html=True)
#         if not filtered_df.empty:
#             st.caption(f"当前筛选条件下共匹配 **{len(filtered_df):,}** 条数据 | 分析区间: {filtered_df['date'].min()} 至 {filtered_df['date'].max()}")
#         else:
#             st.caption("当前筛选条件下匹配到 0 条数据。")

#         # --- 核心空间折叠：Tabs 融合 ---
#         tab1, tab2, tab3 = st.tabs(["📊 大盘与业务归因", "🤖 AI 战略洞察", "🔍 原始 VOC 语义下钻"])

#         # ================= TAB 1: 大盘与归因 =================
#         with tab1:
#             if not filtered_df.empty:
#                 # 1. 基础指标 (融合原版计算方式)
#                 avg_score = filtered_df['score'].mean()
#                 avg_sentiment = filtered_df['nlp_sentiment_score'].mean()
#                 pos_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Positive']) / len(filtered_df)) * 100
#                 neg_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Negative']) / len(filtered_df)) * 100
                
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("分析评价数", f"{len(filtered_df):,}")
#                 c2.metric("平均星级评分", f"{avg_score:.2f} / 5.0")
#                 c3.metric("NLP 情感均分", f"{avg_sentiment:.2f}", delta=">0为正向, <0为负向", delta_color="off")
#                 c4.metric("负面口碑占比", f"{neg_pct:.1f}%", delta=f"好评 {pos_pct:.1f}%", delta_color="inverse")

#                 # 2. 周度趋势折线图 (保留你原版的优势组件)
#                 st.markdown('<div class="section-title">📈 评分走势与异常监测</div>', unsafe_allow_html=True)
#                 weekly_trend = filtered_df.groupby('week').agg(
#                     avg_score=('score', 'mean'),
#                     review_count=('score', 'count')
#                 ).reset_index()
#                 fig_trend = px.line(weekly_trend, x='week', y='avg_score', text=weekly_trend['avg_score'].round(2),
#                                     title="自然周平均星级走势 (结合图表寻找口碑拐点)", markers=True)
#                 fig_trend.update_traces(textposition="top center")
#                 st.plotly_chart(fig_trend, use_container_width=True)

#                 # 3. 核心痛点归因矩阵 (Treemap) & 版本质量监测 (Boxplot) 布局并列
#                 col_tree, col_box = st.columns(2)
                
#                 with col_tree:
#                     st.markdown('<div class="section-title">🗺️ 中差评痛点归因 (1-3星)</div>', unsafe_allow_html=True)
#                     pain_df = filtered_df[filtered_df['score'] <= 3]
#                     if not pain_df.empty:
#                         tag_stats = pain_df.groupby('business_tag').size().reset_index(name='count')
#                         fig_tree = px.treemap(
#                             tag_stats, path=['business_tag'], values='count',
#                             color='count', color_continuous_scale='Reds',
#                             title="色块越深表示该维度的客诉越严重"
#                         )
#                         fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10))
#                         st.plotly_chart(fig_tree, use_container_width=True)
#                     else:
#                         st.info("当前数据集中无足够的中差评数据。")

#                 with col_box:
#                     st.markdown('<div class="section-title">📦 主流版本质量分布</div>', unsafe_allow_html=True)
#                     version_counts = filtered_df['reviewCreatedVersion'].value_counts()
#                     valid_versions = version_counts[version_counts > 5].index
#                     box_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(valid_versions)]
                    
#                     if not box_df.empty:
#                         fig_box = px.box(
#                             box_df, x="reviewCreatedVersion", y="score", color="reviewCreatedVersion",
#                             title="箱体越长代表口碑分化越剧烈",
#                             points="outliers" # 仅显示异常极值点让画面更干净
#                         )
#                         fig_box.update_layout(margin=dict(t=30, l=10, r=10, b=10), showlegend=False)
#                         st.plotly_chart(fig_box, use_container_width=True)
#                     else:
#                         st.info("单版本数据量不足，无法绘制分布箱线图。")
#             else:
#                 st.warning("当前筛选条件下无数据，请放宽左侧侧边栏参数。")

#         # ================= TAB 2: AI 战略洞察 =================
#         with tab2:
#             st.markdown('<div class="section-title">🤖 动态 AI 商业诊断 (基于左侧筛选器)</div>', unsafe_allow_html=True)
            
#             # 使用筛选后的数据指纹，确保更改筛选条件时 AI 报告能随之更新
#             data_fingerprint = f"{len(filtered_df)}_{filtered_df['at'].max()}" if not filtered_df.empty else "empty"
            
#             btn_c1, btn_c2 = st.columns([2, 8])
#             with btn_c1:
#                 generate_clicked = st.button("⚡ 立即生成报告", type="primary", use_container_width=True)
#             with btn_c2:
#                 if st.button("🗑️ 清除云端缓存"):
#                     st.cache_data.clear()
#                     st.rerun()
                    
#             if "ai_report_display" not in st.session_state:
#                 st.session_state.ai_report_display = None

#             if generate_clicked:
#                 if not YOUR_GEMINI_API_KEY:
#                     st.error("未配置 API Key，无法呼叫 AI 大模型。")
#                 elif filtered_df.empty:
#                     st.warning("当前无数据可供分析。")
#                 else:
#                     with st.spinner('AI 正在深度研判当前筛选数据... (若条件不变将秒读缓存)'):
#                         report_content = generate_cached_report(gemini_model, data_fingerprint, filtered_df)
                        
#                         if "出错" not in report_content and "429 Error" not in report_content:
#                             st.session_state.ai_report_display = report_content
#                             st.success("✅ 报告就绪！已部署至云端缓存。")
#                         else:
#                             st.error(report_content)

#             if st.session_state.ai_report_display:
#                 st.markdown(
#                     f"<div style='background-color: rgba(128, 128, 128, 0.05); padding: 30px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);'>"
#                     f"{st.session_state.ai_report_display}"
#                     f"</div>", 
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.info("👆 点击上方按钮，基于左侧侧边栏的数据动态生成诊断报告。")

#         # ================= TAB 3: 原始 VOC 下钻 =================
#         with tab3:
#             if not filtered_df.empty:
#                 # 1. 词云图
#                 st.markdown('<div class="section-title">👁️ 原始文本语义聚类 (词云)</div>', unsafe_allow_html=True)
#                 c_neg, c_pos = st.columns(2)
                
#                 with c_neg:
#                     st.subheader("🔴 负面高频词")
#                     neg_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Negative']['content'].tolist())
#                     if neg_words:
#                         wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
#                         fig_n, ax_n = plt.subplots(figsize=(8, 4))
#                         ax_n.imshow(wordcloud_neg)
#                         ax_n.axis('off')
#                         st.pyplot(fig_n)
#                     else:
#                         st.info("暂无足够的负面评价")

#                 with c_pos:
#                     st.subheader("🟢 正面高频词")
#                     pos_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Positive']['content'].tolist())
#                     if pos_words:
#                         wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
#                         fig_p, ax_p = plt.subplots(figsize=(8, 4))
#                         ax_p.imshow(wordcloud_pos)
#                         ax_p.axis('off')
#                         st.pyplot(fig_p)
#                     else:
#                         st.info("暂无足够的正面评价")

#                 # 2. 原始明细表 (加入了 business_tag 字段)
#                 st.markdown('<div class="section-title">🔍 玩家原声定位明细</div>', unsafe_allow_html=True)
#                 display_df = filtered_df[['at', 'score', 'business_tag', 'content', 'reviewCreatedVersion']].sort_values(by='at', ascending=False)
                
#                 st.dataframe(
#                     display_df, 
#                     use_container_width=True,
#                     height=500, 
#                     hide_index=True, 
#                     column_config={
#                         "at": st.column_config.DatetimeColumn("评论时间", format="YYYY-MM-DD HH:mm"),
#                         "score": st.column_config.NumberColumn("评分", format="%d ⭐"),
#                         "business_tag": st.column_config.TextColumn("AI初步归因"),
#                         "content": st.column_config.TextColumn("玩家原始评论", width="large"),
#                         "reviewCreatedVersion": st.column_config.TextColumn("发生版本")
#                     }
#                 )
#             else:
#                 st.warning("当前筛选条件下无数据。")




# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # import re
# # import google.generativeai as genai
# # import plotly.express as px
# # import os
# # from textblob import TextBlob

# # # ==========================================
# # # 1. 页面级配置 & 全局 CSS 注入 
# # # ==========================================
# # st.set_page_config(page_title="Bubble Shooter 智能体检报告", page_icon="📱", layout="wide")

# # st.markdown("""
# # <style>
# #     .report-header { font-size: 2.5em; font-weight: 800; border-bottom: 3px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;}
# #     .section-title { font-size: 1.5em; font-weight: 700; color: #1f77b4; margin-top: 30px; margin-bottom: 15px;}
# #     div[data-testid="metric-container"] { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1f77b4;}
# # </style>
# # """, unsafe_allow_html=True)

# # # ==========================================
# # # 2. 核心功能函数与数据清洗
# # # ==========================================
# # def init_gemini(api_key):
# #     if not api_key: return None
# #     try:
# #         genai.configure(api_key=api_key)
# #         return genai.GenerativeModel('gemini-2.5-flash') 
# #     except Exception as e:
# #         st.error(f"Gemini 初始化失败: {e}")
# #         return None

# # def analyze_text_sentiment(text):
# #     if not isinstance(text, str) or not text.strip(): return 0.0
# #     return TextBlob(text).sentiment.polarity

# # # 新增：轻量级业务归因引擎
# # def categorize_issue(text):
# #     text = str(text).lower()
# #     if re.search(r'ad|ads|commercial|video|广告', text): return '📺 广告体验'
# #     if re.search(r'crash|bug|freeze|loading|闪退|卡顿|黑屏', text): return '💥 性能与Bug'
# #     if re.search(r'hard|difficult|level|pass|难|卡关', text): return '🧗 关卡与难度'
# #     if re.search(r'money|pay|coin|gem|充值|钱', text): return '💰 商业化与付费'
# #     return '💬 常规反馈'

# # @st.cache_data
# # def load_and_clean_data(file_path):
# #     try:
# #         df = pd.read_csv(file_path)
# #         df['content'] = df['content'].fillna('')
# #         df['at'] = pd.to_datetime(df['at'])
# #         df['date'] = df['at'].dt.date
# #         df['week'] = df['at'].dt.isocalendar().week
        
# #         # 情感与业务打标
# #         df['star_rating'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
# #         df['nlp_sentiment_score'] = df['content'].apply(analyze_text_sentiment)
# #         df['true_sentiment'] = pd.cut(df['nlp_sentiment_score'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
# #         df['business_tag'] = df['content'].apply(categorize_issue)
        
# #         return df
# #     except Exception as e:
# #         st.error(f"数据解析失败: {e}")
# #         return None

# # def get_zeus_style_insight(model, df):
# #     # 为防止超限，抽取最多 80 条有意义的评论
# #     sample_reviews = df[df['content'].str.len() > 20].sample(min(80, len(df)))['content'].tolist()
# #     text_to_analyze = "\n".join(sample_reviews)
    
# #     prompt = f"""
# #     作为传音高级游戏数据分析师，请阅读以下抽样的真实玩家评论数据，生成一份结构化商业诊断报告。
# #     你必须严格使用以下 Markdown 结构和表情符号进行输出，保持专业、客观，避免空话：

# #     ### 🤖 AI 深度洞察
# #     **📋 执行摘要**
# #     (用一段话概括当前选定数据范围内的核心口碑盘面与最致命危机)

# #     **💡 关键发现**
# #     (列举3-4条最突出的痛点或爽点，必须带有极强的游戏业务感，并引用玩家原话作为佐证)

# #     **👥 核心用户画像**
# #     (描述最容易因为上述问题退坑的玩家特征)

# #     ### 🚀 战略改进建议
# #     * 🔧 **短期（1-3个月，针对研发与运营）**：(列出亟待修复的 Bug 或必须调整的策略)
# #     * 🎯 **中期（3-6个月，针对策划与发行）**：(系统玩法或商业化节奏的优化建议)
# #     * 🌟 **长期（6-12个月，针对大盘生态）**：(游戏核心机制或长期留存的升级方向)

# #     评论数据源:
# #     {text_to_analyze}
# #     """
# #     try:
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         if "429" in str(e):
# #             return "⚠️ **触发 API 频率限制 (429 Error)**。由于当前为免费配额，请等待约 30-60 秒后重试，或配置付费 API Key。"
# #         return f"AI 分析时出错: {e}"

# # @st.cache_data(show_spinner=False)
# # def generate_cached_report(_model, data_fingerprint, df):
# #     return get_zeus_style_insight(_model, df)

# # # ==========================================
# # # 3. 网页主程序与全局控制台
# # # ==========================================
# # YOUR_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
# # gemini_model = init_gemini(YOUR_GEMINI_API_KEY)
# # DATA_FILENAME = "linkdesks.pop.bubblegames.bubbleshooter_all_reviews.csv"

# # if not os.path.exists(DATA_FILENAME):
# #     st.error(f"🚨 找不到数据文件：`{DATA_FILENAME}`。请确保它已经与本代码文件存放在同一个文件夹中！")
# # else:
# #     with st.spinner("正在加载底层业务数据..."):
# #         raw_df = load_and_clean_data(DATA_FILENAME)
        
# #     if raw_df is not None:
# #         # --- 全局侧边栏 (Sidebar) ---
# #         with st.sidebar:
# #             st.header("⚙️ 全局数据控制台")
# #             st.info("调整以下参数将实时刷新右侧所有大盘数据与图表。")
            
# #             score_filter = st.multiselect("⭐ 星级过滤 (Score):", options=[5, 4, 3, 2, 1], default=[5, 4, 3, 2, 1])
# #             available_versions = raw_df['reviewCreatedVersion'].dropna().unique().tolist()
# #             version_filter = st.multiselect("📱 版本过滤 (Version):", options=available_versions, default=[])
# #             search_kw = st.text_input("🔑 关键词检索:", placeholder="如 'crash' 或 'level'")
            
# #             st.divider()
# #             if YOUR_GEMINI_API_KEY:
# #                 st.success("🟢 AI 引擎已连接")
# #             else:
# #                 st.warning("🔴 AI 引擎未配置")

# #         # --- 全局数据过滤 ---
# #         filtered_df = raw_df[raw_df['score'].isin(score_filter)]
# #         if version_filter:
# #             filtered_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(version_filter)]
# #         if search_kw:
# #             filtered_df = filtered_df[filtered_df['content'].str.contains(search_kw, case=False, na=False)]

# #         # 头部标题
# #         st.markdown('<div class="report-header">📱 Bubble Shooter 智能体检仪表盘</div>', unsafe_allow_html=True)
# #         st.caption(f"当前筛选条件下共匹配 **{len(filtered_df):,}** 条数据 | 分析区间: {filtered_df['date'].min()} 至 {filtered_df['date'].max()}")

# #         # --- 核心空间折叠：Tabs ---
# #         tab1, tab2, tab3 = st.tabs(["📊 大盘与业务归因", "🤖 AI 战略洞察", "🔍 原始 VOC 语义下钻"])

# #         # ================= TAB 1: 大盘与归因 =================
# #         with tab1:
# #             if not filtered_df.empty:
# #                 # 1. 基础指标
# #                 avg_score = filtered_df['score'].mean()
# #                 avg_sentiment = filtered_df['nlp_sentiment_score'].mean()
# #                 pos_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Positive']) / len(filtered_df)) * 100
# #                 neg_pct = (len(filtered_df[filtered_df['true_sentiment'] == 'Negative']) / len(filtered_df)) * 100
                
# #                 c1, c2, c3, c4 = st.columns(4)
# #                 c1.metric("评价基数", f"{len(filtered_df):,}")
# #                 c2.metric("平均星级", f"{avg_score:.2f} ⭐")
# #                 c3.metric("NLP 情感均分", f"{avg_sentiment:.2f}", delta=">0为正向", delta_color="off")
# #                 c4.metric("负面情绪占比", f"{neg_pct:.1f}%", delta=f"正面 {pos_pct:.1f}%", delta_color="inverse")

# #                 # 2. 核心痛点归因矩阵 (Treemap)
# #                 st.markdown('<div class="section-title">🗺️ 中差评核心痛点归因矩阵 (1-3星)</div>', unsafe_allow_html=True)
# #                 pain_df = filtered_df[filtered_df['score'] <= 3]
# #                 if not pain_df.empty:
# #                     tag_stats = pain_df.groupby('business_tag').size().reset_index(name='count')
# #                     fig_tree = px.treemap(
# #                         tag_stats, path=['business_tag'], values='count',
# #                         color='count', color_continuous_scale='Reds',
# #                         title="面积与颜色越深，代表该业务模块的负面反馈越集中"
# #                     )
# #                     fig_tree.update_layout(margin=dict(t=40, l=10, r=10, b=10))
# #                     st.plotly_chart(fig_tree, use_container_width=True)
# #                 else:
# #                     st.info("当前筛选条件下无足够的中差评数据生成热力图。")

# #                 # 3. 版本质量分布监测 (Boxplot)
# #                 st.markdown('<div class="section-title">📦 主流版本质量分布监测</div>', unsafe_allow_html=True)
# #                 version_counts = filtered_df['reviewCreatedVersion'].value_counts()
# #                 # 仅展示评论数大于5的版本，避免视觉杂乱
# #                 valid_versions = version_counts[version_counts > 5].index
# #                 box_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(valid_versions)]
                
# #                 if not box_df.empty:
# #                     fig_box = px.box(
# #                         box_df, x="reviewCreatedVersion", y="score", color="reviewCreatedVersion",
# #                         title="箱体越长表示评分离散度越高（存在口碑分化），黑点代表异常极值",
# #                         points="all"
# #                     )
# #                     st.plotly_chart(fig_box, use_container_width=True)
# #                 else:
# #                     st.info("单版本数据量不足以绘制分布箱线图。")
# #             else:
# #                 st.warning("当前筛选条件下无数据，请调整左侧侧边栏参数。")

# #         # ================= TAB 2: AI 战略洞察 =================
# #         with tab2:
# #             st.markdown('<div class="section-title">🤖 高管汇报级战略建议 (云端缓存版)</div>', unsafe_allow_html=True)
            
# #             # 使用当前筛选数据的长度和最后一个时间戳作为缓存指纹
# #             data_fingerprint = f"{len(filtered_df)}_{filtered_df['at'].max()}" if not filtered_df.empty else "empty"
            
# #             btn_c1, btn_c2 = st.columns([2, 8])
# #             with btn_c1:
# #                 generate_clicked = st.button("⚡ 立即生成报告", type="primary", use_container_width=True)
# #             with btn_c2:
# #                 if st.button("🗑️ 清除云端缓存"):
# #                     st.cache_data.clear()
# #                     st.rerun()
                    
# #             if "ai_report_display" not in st.session_state:
# #                 st.session_state.ai_report_display = None

# #             if generate_clicked:
# #                 if not YOUR_GEMINI_API_KEY:
# #                     st.error("未配置 API Key，无法呼叫 AI 大模型。")
# #                 elif filtered_df.empty:
# #                     st.warning("当前无数据可供分析。")
# #                 else:
# #                     with st.spinner('AI 正在交叉验证数据，生成高管级汇报纪要... (若数据源未变将秒读缓存)'):
# #                         report_content = generate_cached_report(gemini_model, data_fingerprint, filtered_df)
                        
# #                         if "出错" not in report_content and "429 Error" not in report_content:
# #                             st.session_state.ai_report_display = report_content
# #                             st.success("✅ 报告就绪！已部署至云端缓存。")
# #                         else:
# #                             st.error(report_content)

# #             if st.session_state.ai_report_display:
# #                 st.markdown(
# #                     f"<div style='background-color: rgba(128, 128, 128, 0.05); padding: 30px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);'>"
# #                     f"{st.session_state.ai_report_display}"
# #                     f"</div>", 
# #                     unsafe_allow_html=True
# #                 )
# #             else:
# #                 st.info("👆 点击上方按钮，基于当前左侧侧边栏筛选出的数据生成定制化诊断报告。")

# #         # ================= TAB 3: 原始 VOC 下钻 =================
# #         with tab3:
# #             if not filtered_df.empty:
# #                 # 1. 词云图
# #                 st.markdown('<div class="section-title">👁️ 原始文本语义聚类 (词云)</div>', unsafe_allow_html=True)
# #                 c_neg, c_pos = st.columns(2)
                
# #                 with c_neg:
# #                     st.subheader("🔴 负面高频词")
# #                     neg_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Negative']['content'].tolist())
# #                     if neg_words:
# #                         wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
# #                         fig_n, ax_n = plt.subplots(figsize=(8, 4))
# #                         ax_n.imshow(wordcloud_neg)
# #                         ax_n.axis('off')
# #                         st.pyplot(fig_n)
# #                     else:
# #                         st.info("暂无足够的负面评价生成词云")

# #                 with c_pos:
# #                     st.subheader("🟢 正面高频词")
# #                     pos_words = " ".join(filtered_df[filtered_df['true_sentiment'] == 'Positive']['content'].tolist())
# #                     if pos_words:
# #                         wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
# #                         fig_p, ax_p = plt.subplots(figsize=(8, 4))
# #                         ax_p.imshow(wordcloud_pos)
# #                         ax_p.axis('off')
# #                         st.pyplot(fig_p)
# #                     else:
# #                         st.info("暂无足够的正面评价生成词云")

# #                 # 2. 原始明细表
# #                 st.markdown('<div class="section-title">🔍 玩家原声定位明细</div>', unsafe_allow_html=True)
# #                 display_df = filtered_df[['at', 'score', 'business_tag', 'content', 'reviewCreatedVersion']].sort_values(by='at', ascending=False)
                
# #                 st.dataframe(
# #                     display_df, 
# #                     use_container_width=True,
# #                     height=500, 
# #                     hide_index=True, 
# #                     column_config={
# #                         "at": st.column_config.DatetimeColumn("评论时间", format="YYYY-MM-DD HH:mm"),
# #                         "score": st.column_config.NumberColumn("评分", format="%d ⭐"),
# #                         "business_tag": st.column_config.TextColumn("业务归类"),
# #                         "content": st.column_config.TextColumn("玩家原始评论", width="large"),
# #                         "reviewCreatedVersion": st.column_config.TextColumn("发生版本")
# #                     }
# #                 )
# #             else:
# #                 st.warning("当前筛选条件下无数据。")
