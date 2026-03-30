
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
