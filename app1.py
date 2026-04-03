
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import re
# import json
# import google.generativeai as genai

# # ==========================================
# # 0. 页面全局配置
# # ==========================================
# st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# # 从 secrets 中安全读取 API Key，避免硬编码泄露
# try:
#     YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# except FileNotFoundError:
#     st.error("未找到 .streamlit/secrets.toml 文件，请配置 API Key。")
#     st.stop()

# # ==========================================
# # 1. 功能函数层
# # ==========================================

# @st.cache_resource
# def init_gemini(api_key):
#     try:
#         genai.configure(api_key=api_key)
#         # 建议使用稳定版模型进行结构化输出
#         model = genai.GenerativeModel('gemini-2.5-flash')
#         return model
#     except Exception as e:
#         st.error(f"Gemini 初始化失败: {e}")
#         return None

# @st.cache_data
# def load_and_clean_data(file_obj):
#     try:
#         df = pd.read_csv(file_obj)
#         df['content'] = df['content'].fillna('')
#         df['at'] = pd.to_datetime(df['at'])
#         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
#         return df
#     except Exception as e:
#         st.error(f"数据读取失败: {e}")
#         return None

# def generate_wordcloud(text, title):
#     if not text:
#         st.warning(f"{title} 没有足够的文本生成词云。")
#         return
#     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
#     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
#     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     ax.set_title(title, fontsize=20)
#     st.pyplot(fig)

# # 核心重构：强制 JSON 输出与容错解析
# @st.cache_data(show_spinner=False)
# def get_ai_summary_cached(reviews_list):
#     model = init_gemini(YOUR_GEMINI_API_KEY)
#     if not model or not reviews_list:
#         return None

#     text_to_analyze = "\n\n".join(reviews_list[:50]) # 取前50条避免超长
    
#     # 结构化 Prompt 注入
#     prompt = f"""
#     You are an expert Game Data Analyst. Analyze these negative user reviews (1-2 stars) for 'Bubble Shooter'.
#     Output the result STRICTLY as a JSON array of objects. Do not use markdown blocks, do not add explanation text.
#     Each object must have the following keys exactly:
#     - "title": A short, punchy summary of the complaint in Chinese (e.g., "广告频率过高").
#     - "severity": Must be exactly one of "High", "Medium", or "Low".
#     - "example": One direct quote from the reviews supporting this in its original language.
#     - "action": A one-sentence actionable suggestion for the dev team in Chinese.

#     Reviews to analyze:
#     {text_to_analyze}
#     """
    
#     try:
#         response = model.generate_content(prompt)
#         raw_text = response.text
        
#         # 容错处理：清洗大模型可能附带的 Markdown 标记 (```json ... ```)
#         json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
#         if json_match:
#             clean_json_str = json_match.group(0)
#             return json.loads(clean_json_str)
#         else:
#             return None # 格式严重错误
#     except Exception as e:
#         return {"error": str(e)}

# # ==========================================
# # 2. 交互界面层 (金字塔结构)
# # ==========================================

# st.title("🎯 玩家声音智能决策看板 (VOC)")
# st.markdown("---")

# st.sidebar.header("📁 数据导入")
# uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

# if uploaded_file is not None:
#     df = load_and_clean_data(uploaded_file)
    
#     if df is not None:
#         # 【置顶】模块 1: AI 智能归因 (执行摘要)
#         st.header("🚨 核心痛点归因与行动建议")
#         negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
#         with st.spinner("AI 正在深度解析负面反馈..."):
#             ai_insights = get_ai_summary_cached(negative_reviews)
            
#             if not ai_insights:
#                 st.error("AI 总结生成失败，请检查数据或重试。")
#             elif isinstance(ai_insights, dict) and "error" in ai_insights:
#                 st.error(f"AI 调用异常: {ai_insights['error']}")
#             else:
#                 # 动态渲染 JSON 为告警卡片
#                 cols = st.columns(len(ai_insights))
#                 for idx, insight in enumerate(ai_insights[:3]): # 最多展示前3个核心痛点
#                     with cols[idx]:
#                         # 根据严重程度动态分配颜色主题
#                         if insight.get("severity") == "High":
#                             st.error(f"🔥 {insight.get('title', '未知痛点')}")
#                         elif insight.get("severity") == "Medium":
#                             st.warning(f"⚡ {insight.get('title', '未知痛点')}")
#                         else:
#                             st.info(f"💡 {insight.get('title', '未知痛点')}")
                        
#                         st.markdown(f"**🗣️ 玩家原声:**\n> *\"{insight.get('example', '')}\"*")
#                         st.markdown(f"**🛠️ 建议动作:**\n{insight.get('action', '')}")

#         st.markdown("---")

#         # 【中间】模块 2: 宏观数据概览
#         st.header("📊 大盘健康度指标")
#         col1, col2, col3, col4 = st.columns(4)
#         avg_score = df['score'].mean()
#         neg_ratio = len(df[df['sentiment'] == 'Negative']) / len(df) * 100
        
#         col1.metric("平均评分", f"{avg_score:.2f} / 5", f"{avg_score - 4.0:.2f} 距健康基线")
#         col2.metric("总评论样本", f"{len(df):,}")
#         col3.metric("负面评论占比", f"{neg_ratio:.1f}%")
        
#         with col4:
#             st.bar_chart(df['score'].value_counts().sort_index(ascending=False), height=150)

#         st.markdown("---")

#         # 【底部】模块 3: 数据下钻探索 (折叠处理，降低认知负担)
#         st.header("🔍 数据下钻验证")
#         with st.expander("展开查看高频词云与原始评论流", expanded=False):
#             tab1, tab2, tab3 = st.tabs(["🔴 负面词云", "🟢 正面词云", "📋 原始明细表"])
            
#             with tab1:
#                 neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
#                 generate_wordcloud(neg_text, "Negative Trigger Words")
#             with tab2:
#                 pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
#                 generate_wordcloud(pos_text, "Positive Trigger Words")
#             with tab3:
#                 score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
#                 filtered_df = df[df['score'].isin(score_filter)]
#                 st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion']], use_container_width=True)

# else:
#     st.info("👋 请在侧边栏上传从 Google Play 爬取的 CSV 数据开始分析。")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import google.generativeai as genai
import io




# cd ~/Desktop/Bubble_VOC_Project
# streamlit run APP.py
# ==========================================
# 1. 配置区域 (需要你在本地配置)
# ==========================================
# st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# 请在此处填写你在 Google AI Studio 获取的 API Key
# 实际项目中应使用环境变量保存，这里为了 Demo 演示直接写出
YOUR_API_KEY = st.secrets["GEMINI_API_KEY"]

# ==========================================
# 2. 功能函数
# ==========================================

# 2.1 初始化 AI 模型 (Gemini)
def init_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Gemini 初始化失败，请检查 API Key: {e}")
        return None

# 2.2 数据清洗与处理
@st.cache_data # 缓存数据，避免每次刷新网页都重新运行
def load_and_clean_data(file_obj):
    # 读取你上传的 CSV 文件
    try:
        df = pd.read_csv(file_obj)
        df['content'] = df['content'].fillna('')
        df['at'] = pd.to_datetime(df['at'])
        # 简单的情感打标：4-5星为正向，1-2星为负向
        df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
        return df
    except Exception as e:
        st.error(f"数据读取失败: {e}")
        return None

# 2.3 生成词云图
def generate_wordcloud(text, title):
    if not text:
        st.warning(f"{title} 没有足够的文本生成词云。")
        return
    
    # 简单的英文停用词
    stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
    
    # 清洗文本
    clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    
    # 使用 Matplotlib 显示
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=20)
    st.pyplot(fig)

# 2.4 调用 AI (Gemini) 进行评论总结
def get_ai_summary(model, reviews_list):
    if not model:
        return "Gemini 未初始化。"
    if not reviews_list:
        return "没有足够的评论供 AI 分析。"

    # 将差评论合并为一个长文本（限制长度避免 token 超限，Demo 只取前 50 条）
    text_to_analyze = "\n\n".join(reviews_list[:50])
    
    prompt = f"""
    You are an expert Game Data Analyst and VOC (Voice of Customer) specialist. 
    Below is a list of negative user reviews (1-2 stars) for our mobile game 'Bubble Shooter'.
    
    Analyze these reviews and provide a summary for the development team in Chinese (中文).
    Your output must include:
    1. A bulleted list of the Top 3 main complaints from players.
    2. For each complaint, provide one concrete example review from the text.
    
    User Reviews:
    \"\"\"
    {text_to_analyze}
    \"\"\"
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析时出错: {e}"

# ==========================================
# 3. 网页主界面 (Main App)
# ==========================================
st.title("🏹 VOC玩家声音智能分析系统")
st.markdown("---")
YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# 初始化 Gemini 模型
gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# 3.1 侧边栏：文件上传
st.sidebar.header("数据导入")
uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

if uploaded_file is not None:
    # 加载数据
    with st.spinner('正在处理数据...'):
        df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # ==========================================
        # 模块 1: 核心评级指标 (Descriptive)
        # ==========================================
        st.header("1. 玩家评分概览")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        # 指标卡
        with col1:
            avg_score = df['score'].mean()
            st.metric(label="平均得分", value=f"{avg_score:.2f} / 5")
        with col2:
            total_reviews = len(df)
            st.metric(label="总评论数", value=total_reviews)
            
        # 评分占比图
        with col3:
            score_counts = df['score'].value_counts().sort_index(ascending=False)
            st.bar_chart(score_counts)
        
        st.markdown("---")
        
        # ==========================================
        # 模块 2: 核心文本洞察 (Descriptive)
        # ==========================================
        st.header("2. 评论文本视觉探索")
        
        tab1, tab2 = st.tabs(["🔴 负面评论词云 (1-2星)", "🟢 正面评论词云 (4-5星)"])
        
        with tab1:
            neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
            generate_wordcloud(neg_text, "Negative Reviews Keywords")
            
        with tab2:
            pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
            generate_wordcloud(pos_text, "Positive Reviews Keywords")
            
        st.markdown("---")

        # ==========================================
        # 模块 3: AI 智能差评归因 (Diagnostic - AI 落地)
        # ==========================================
        st.header("3. ✨ AI 智能差评归因 (今日 Top 3 痛点)")
        
        # 准备数据供 AI 分析
        negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
        # 增加一个按钮触发 AI 分析，避免打开网页就自动调用扣费
        if st.button("运行 AI 智能总结"):
            if YOUR_GEMINI_API_KEY == "你的_GEMINI_API_KEY_粘贴在这里":
                st.warning("请先在 App.py 代码中填写你的 Gemini API Key。")
            else:
                with st.spinner('Gemini 正在冥想并分析几百条差评...'):
                    ai_summary = get_ai_summary(gemini_model, negative_reviews)
                    
                    # 显示 AI 分析结果
                    st.info("AI 总结报告如下：")
                    st.markdown(ai_summary)
                    
        # ==========================================
        # 模块 4: 数据详情与筛选 (Exploratory)
        # ==========================================
        st.markdown("---")
        st.header("4. 数据详情预览")
        # 允许玩家筛选评分
        score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
        filtered_df = df[df['score'].isin(score_filter)]
        st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion', 'sourceCountry']].head(100))

else:
    # 未上传文件时的欢迎界面
    st.info("👋 请在侧边栏上传从 Google Play 爬取的 Bubble Shooter CSV 评论数据开始分析。")
    st.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png") # 这里放置一张 Streamlit 介绍图
