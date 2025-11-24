import streamlit as st
import time
import plotly.graph_objects as go
from src.pipeline.naive_rag import naive_rag
from src.pipeline.re_rag import re_rag
from src.pipeline.dqr_rag import dqr_rag
from src.utils.metrics import compute_mrr, llm_judge, estimate_cost

# Page config
st.set_page_config(
    page_title="RAG Evaluation Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS with stunning visuals - ENHANCED VISIBILITY
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with animated gradient - darker for better contrast */
    .main {
        background: linear-gradient(135deg, #4c51bf 0%, #5a3d7a 50%, #c44fc4 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
        padding: 2rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Premium glass cards - enhanced visibility */
    [data-testid="column"] {
        background: white !important;
        backdrop-filter: blur(20px);
        padding: 2.5rem !important;
        border-radius: 24px !important;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.25), 0 0 0 2px rgba(255, 255, 255, 0.4) !important;
        margin: 1rem !important;
        border: 2px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="column"] * {
        color: #0f172a !important;
    }
    
    [data-testid="column"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 35px 90px rgba(0, 0, 0, 0.3), 0 0 0 2px rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Typography - enhanced readability */
    h1 {
        color: white !important;
        text-align: center;
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        text-shadow: 0 6px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 0.75rem !important;
    }
    
    h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        margin-bottom: 1.75rem !important;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* FORCE SIDEBAR HEADERS TO PURE WHITE (STRONGEST OVERRIDE) */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] div[role="heading"],
    section[data-testid="stSidebar"] .css-1wvskj5,
    section[data-testid="stSidebar"] .css-17eq0hr {
        color: #ffffff !important;
    }

    /* Pipeline headers with color badges - bolder */
    h3:has(+ .answer-box) {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border-left: 6px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Specific color coding for each pipeline */
    [data-testid="column"]:nth-child(1) h3 {
        border-left-color: #667eea !important;
    }
    
    [data-testid="column"]:nth-child(2) h3 {
        border-left-color: #10b981 !important;
    }
    
    [data-testid="column"]:nth-child(3) h3 {
        border-left-color: #8b5cf6 !important;
    }
    
    /* Subtitle - more visible */
    .subtitle {
        color: white !important;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 3rem;
        text-shadow: 0 3px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Answer boxes with premium styling - larger text */
    .answer-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border: 3px solid transparent !important;
        background-clip: padding-box;
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1.5rem 0 !important;
        font-size: 1.15rem !important;
        line-height: 1.9 !important;
        color: #0f172a !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        position: relative;
        overflow: hidden;
        font-weight: 500;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 20px 20px 0 0;
    }
    
    .answer-box p, .answer-box div, .answer-box span {
        color: #0f172a !important;
        font-size: 1.15rem !important;
    }
    
    /* Premium metrics - larger and clearer */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        padding: 1.75rem !important;
        border-radius: 20px !important;
        border: 2px solid #e2e8f0 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .stMetric label {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #10b981 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    /* Input field with glow effect - larger */
    .stTextInput label {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput input {
        background: white !important;
        color: #0f172a !important;
        border: 3px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: white !important;
        box-shadow: 0 12px 40px rgba(255, 255, 255, 0.4), 0 0 0 5px rgba(255, 255, 255, 0.15) !important;
        outline: none;
    }
    
    /* Premium expanders - more visible */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%) !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        transition: all 0.3s ease;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        transform: translateX(6px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .streamlit-expanderContent {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 0 0 16px 16px !important;
        padding: 1.5rem !important;
        margin-top: -2px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .streamlit-expanderContent p, 
    .streamlit-expanderContent div {
        color: #0f172a !important;
        line-height: 1.8;
        font-size: 1.05rem !important;
    }
    
    /* Stunning tabs - larger and clearer */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        padding: 12px;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1.125rem 2.25rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #4c51bf !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
        font-weight: 800 !important;
    }
    
    
    /* Force sidebar headings to pure white */
   section[data-testid="stSidebar"] h3 {
       color: #ffffff !important;
   }
    
    /* Info/Success boxes - more prominent */
    .stInfo, .stSuccess {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 197, 253, 0.15)) !important;
        border: 3px solid #3b82f6 !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(134, 239, 172, 0.15)) !important;
        border-color: #22c55e !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.15);
    }
    
    .stInfo *, .stSuccess * {
        color: #0f172a !important;
        font-weight: 600;
        font-size: 1.05rem !important;
    }
    
    /* Captions - more visible */
    .caption, [data-testid="stCaptionContainer"], small {
        color: #475569 !important;
        font-size: 0.95rem !important;
        font-weight: 600;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: white !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.6);
    }
    
    /* Welcome card animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .welcome-card {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Pipeline cards hover effect */
    .pipeline-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .pipeline-card:hover {
        transform: translateY(-8px) scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function
def judge_to_numeric(judge_score):
    if isinstance(judge_score, (int, float)):
        return float(judge_score)
    judge_str = str(judge_score).lower()
    if 'correct' in judge_str and 'partially' not in judge_str:
        return 10.0
    elif 'partially' in judge_str:
        return 7.0
    elif 'incorrect' in judge_str:
        return 3.0
    return 5.0

# Header with premium styling
st.markdown("<h1>🚀 RAG Evaluation Suite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Compare three advanced RAG pipelines in real-time with comprehensive analytics</p>", unsafe_allow_html=True)

# Sidebar with enhanced design - WHITE HEADINGS
with st.sidebar:
    st.markdown("<h3 style='color: white !important; font-size: 1.4rem;'>📊 Pipeline Overview</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 1.05rem; line-height: 1.8; font-weight: 500;'>
    
    **🔵 Naive RAG**  
    Basic retrieval + generation
    
    **🟢 Re-RAG**  
    Enhanced with reranking
    
    **🟣 DQR-RAG**  
    Dynamic query rewriting
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3 style='color: white !important; font-size: 1.4rem;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    show_debug = st.checkbox("Show debug info", False)
    show_chunks = st.checkbox("Show retrieved chunks", True)
    
    st.markdown("---")
    st.markdown("<h3 style='color: white !important; font-size: 1.4rem;'>📈 Metrics Guide</h3>", unsafe_allow_html=True)
    with st.expander("📍 MRR@5"):
        st.markdown("<p style='font-size: 1.05rem;'>Mean Reciprocal Rank - measures retrieval quality</p>", unsafe_allow_html=True)
    with st.expander("⭐ Answer Quality"):
        st.markdown("<p style='font-size: 1.05rem;'>LLM-based evaluation of answer relevance</p>", unsafe_allow_html=True)
    with st.expander("⚡ Latency"):
        st.markdown("<p style='font-size: 1.05rem;'>Total processing time in seconds</p>", unsafe_allow_html=True)
    with st.expander("💰 Cost"):
        st.markdown("<p style='font-size: 1.05rem;'>Estimated API cost in USD</p>", unsafe_allow_html=True)

# Query input
query = st.text_input(
    "🔍 Enter your query:",
    placeholder="Ask me anything about your documents...",
)

if query:
    with st.spinner('🧠 Processing query across all pipelines...'):
        start_all = time.time()

        tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Comparison", "🔬 Analysis"])
        
        # Call pipelines
        res_naive = naive_rag(query)
        naive_chunks = res_naive.get("chunks") or res_naive.get("retrieved_chunks") or []
        naive_answer = res_naive.get("answer", "No answer returned")
        naive_latency = res_naive.get("latency_sec") or res_naive.get("latency") or 0.0
        naive_cost = estimate_cost(res_naive)

        res_rerank = re_rag(query)
        rerank_chunks = res_rerank.get("chunks") or res_rerank.get("retrieved_chunks") or []
        rerank_answer = res_rerank.get("answer", "No answer returned")
        rerank_latency = res_rerank.get("latency") or 0.0
        rerank_cost = estimate_cost(res_rerank)

        res_dqr = dqr_rag(query)
        dqr_chunks = res_dqr.get("retrieved_chunks") or res_dqr.get("chunks") or []
        dqr_answer = res_dqr.get("answer", "No answer returned")
        dqr_latency = res_dqr.get("latency") or 0.0
        dqr_cost = estimate_cost(res_dqr)

        # Compute metrics
        mrr_naive = compute_mrr(naive_chunks)
        mrr_rerank = compute_mrr(rerank_chunks)
        mrr_dqr = compute_mrr(dqr_chunks)

        judge_naive = llm_judge(query, naive_answer)
        judge_rerank = llm_judge(query, rerank_answer)
        judge_dqr = llm_judge(query, dqr_answer)
        
        judge_naive_num = judge_to_numeric(judge_naive)
        judge_rerank_num = judge_to_numeric(judge_rerank)
        judge_dqr_num = judge_to_numeric(judge_dqr)

        # Tab 1: Results
        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("<h3>🔵 Naive RAG</h3>", unsafe_allow_html=True)
                st.markdown("**✨ Answer:**")
                st.markdown(f"""<div class='answer-box'>{naive_answer}</div>""", unsafe_allow_html=True)
                
                if show_chunks:
                    st.markdown("**📚 Retrieved Chunks:**")
                    for i, c in enumerate(naive_chunks[:5], 1):
                        chunk_text = c.get('text', '') if isinstance(c, dict) else str(c)
                        filename = c.get('filename', 'N/A') if isinstance(c, dict) else 'N/A'
                        distance = c.get('distance', 0.0) if isinstance(c, dict) else 0.0
                        
                        with st.expander(f"📄 Chunk {i} - {filename[:25]}"):
                            st.write(chunk_text[:300] + ('...' if len(chunk_text) > 300 else ''))
                            st.caption(f"📁 Source: {filename}")
                            st.caption(f"📏 Distance: {distance:.4f}")

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("MRR@5", f"{mrr_naive:.3f}")
                    st.metric("Latency", f"{naive_latency:.2f}s")
                with col_b:
                    st.metric("Quality", str(judge_naive))
                    st.metric("Cost", f"${naive_cost:.5f}")

            with col2:
                st.markdown("<h3>🟢 Re-RAG</h3>", unsafe_allow_html=True)
                st.markdown("**✨ Answer:**")
                st.markdown(f"""<div class='answer-box'>{rerank_answer}</div>""", unsafe_allow_html=True)
                
                if show_chunks:
                    st.markdown("**📚 Retrieved Chunks:**")
                    for i, c in enumerate(rerank_chunks[:5], 1):
                        chunk_text = c.get('text', '') if isinstance(c, dict) else str(c)
                        filename = c.get('filename', 'N/A') if isinstance(c, dict) else 'N/A'
                        distance = c.get('distance', 0.0) if isinstance(c, dict) else 0.0
                        
                        with st.expander(f"📄 Chunk {i} - {filename[:25]}"):
                            st.write(chunk_text[:300] + ('...' if len(chunk_text) > 300 else ''))
                            st.caption(f"📁 Source: {filename}")
                            st.caption(f"📏 Distance: {distance:.4f}")

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("MRR@5", f"{mrr_rerank:.3f}", delta=f"{mrr_rerank-mrr_naive:.3f}")
                    st.metric("Latency", f"{rerank_latency:.2f}s")
                with col_b:
                    st.metric("Quality", str(judge_rerank))
                    st.metric("Cost", f"${rerank_cost:.5f}")

            with col3:
                st.markdown("<h3>🟣 DQR-RAG</h3>", unsafe_allow_html=True)
                rewritten_query = res_dqr.get("rewritten_query", "N/A")
                st.info(f"✨ **Rewritten Query:** {rewritten_query}")
                
                st.markdown("**✨ Answer:**")
                st.markdown(f"""<div class='answer-box'>{dqr_answer}</div>""", unsafe_allow_html=True)
                
                if show_chunks:
                    st.markdown("**📚 Retrieved Chunks:**")
                    for i, c in enumerate(dqr_chunks[:5], 1):
                        chunk_text = c.get('text', '') if isinstance(c, dict) else str(c)
                        filename = c.get('filename', 'N/A') if isinstance(c, dict) else 'N/A'
                        distance = c.get('distance', 0.0) if isinstance(c, dict) else 0.0
                        
                        with st.expander(f"📄 Chunk {i} - {filename[:25]}"):
                            st.write(chunk_text[:300] + ('...' if len(chunk_text) > 300 else ''))
                            st.caption(f"📁 Source: {filename}")
                            st.caption(f"📏 Distance: {distance:.4f}")

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("MRR@5", f"{mrr_dqr:.3f}", delta=f"{mrr_dqr-mrr_naive:.3f}")
                    st.metric("Latency", f"{dqr_latency:.2f}s")
                with col_b:
                    st.metric("Quality", str(judge_dqr))
                    st.metric("Cost", f"${dqr_cost:.5f}")

        # Tab 2: Comparison
        with tab2:
            st.markdown("### 📊 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # MRR
                fig_mrr = go.Figure(data=[
                    go.Bar(
                        x=['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                        y=[mrr_naive, mrr_rerank, mrr_dqr],
                        marker=dict(
                            color=['#667eea', '#10b981', '#8b5cf6'],
                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                        ),
                        text=[f"{mrr_naive:.3f}", f"{mrr_rerank:.3f}", f"{mrr_dqr:.3f}"],
                        textposition='outside',
                        textfont=dict(size=16, color='#1e293b', family='Inter', weight=700)
                    )
                ])
                fig_mrr.update_layout(
                    title={'text': '📍 MRR@5 Comparison', 'font': {'size': 22, 'color': '#1e293b', 'family': 'Inter'}},
                    yaxis_title='Score',
                    height=380,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1e293b', size=13, family='Inter'),
                    showlegend=False,
                    margin=dict(t=80, b=60, l=60, r=40)
                )
                st.plotly_chart(fig_mrr, width='stretch')
                
                # Latency
                fig_latency = go.Figure(data=[
                    go.Bar(
                        x=['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                        y=[naive_latency, rerank_latency, dqr_latency],
                        marker=dict(
                            color=['#667eea', '#10b981', '#8b5cf6'],
                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                        ),
                        text=[f"{naive_latency:.2f}s", f"{rerank_latency:.2f}s", f"{dqr_latency:.2f}s"],
                        textposition='outside',
                        textfont=dict(size=16, color='#1e293b', family='Inter', weight=700)
                    )
                ])
                fig_latency.update_layout(
                    title={'text': '⚡ Latency Comparison', 'font': {'size': 22, 'color': '#1e293b', 'family': 'Inter'}},
                    yaxis_title='Seconds',
                    height=380,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1e293b', size=13, family='Inter'),
                    showlegend=False,
                    margin=dict(t=80, b=60, l=60, r=40)
                )
                st.plotly_chart(fig_latency, width='stretch')
            
            with col2:
                # Quality
                fig_quality = go.Figure(data=[
                    go.Bar(
                        x=['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                        y=[judge_naive_num, judge_rerank_num, judge_dqr_num],
                        marker=dict(
                            color=['#667eea', '#10b981', '#8b5cf6'],
                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                        ),
                        text=[str(judge_naive), str(judge_rerank), str(judge_dqr)],
                        textposition='outside',
                        textfont=dict(size=16, color='#1e293b', family='Inter', weight=700)
                    )
                ])
                fig_quality.update_layout(
                    title={'text': '⭐ Answer Quality', 'font': {'size': 22, 'color': '#1e293b', 'family': 'Inter'}},
                    yaxis_title='Score',
                    yaxis_range=[0, 12],
                    height=380,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1e293b', size=13, family='Inter'),
                    showlegend=False,
                    margin=dict(t=80, b=60, l=60, r=40)
                )
                st.plotly_chart(fig_quality, width='stretch')
                
                # Cost
                fig_cost = go.Figure(data=[
                    go.Bar(
                        x=['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                        y=[naive_cost, rerank_cost, dqr_cost],
                        marker=dict(
                            color=['#667eea', '#10b981', '#8b5cf6'],
                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                        ),
                        text=[f"${naive_cost:.5f}", f"${rerank_cost:.5f}", f"${dqr_cost:.5f}"],
                        textposition='outside',
                        textfont=dict(size=16, color='#1e293b', family='Inter', weight=700)
                    )
                ])
                fig_cost.update_layout(
                    title={'text': '💰 Cost Comparison', 'font': {'size': 22, 'color': '#1e293b', 'family': 'Inter'}},
                    yaxis_title='USD',
                    height=380,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1e293b', size=13, family='Inter'),
                    showlegend=False,
                    margin=dict(t=80, b=60, l=60, r=40)
                )
                st.plotly_chart(fig_cost, width='stretch')

        # Tab 3: Analysis
        with tab3:
            st.markdown("### 🔬 Detailed Analysis")
            
            scores = {
                'Naive RAG': mrr_naive + judge_naive_num/10,
                'Re-RAG': mrr_rerank + judge_rerank_num/10,
                'DQR-RAG': mrr_dqr + judge_dqr_num/10
            }
            winner = max(scores, key=scores.get)
            
            st.success(f"🏆 **Best Overall Performance:** {winner} (Score: {scores[winner]:.3f})")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Top Performers")
                st.markdown("<div style='font-size: 1.1rem; line-height: 2; font-weight: 500;'>", unsafe_allow_html=True)
                
                best_mrr = max(mrr_naive, mrr_rerank, mrr_dqr)
                best_mrr_name = ['Naive RAG', 'Re-RAG', 'DQR-RAG'][[mrr_naive, mrr_rerank, mrr_dqr].index(best_mrr)]
                st.markdown(f"**📍 Best MRR:** {best_mrr_name} ({best_mrr:.3f})")
                
                best_quality = max(judge_naive_num, judge_rerank_num, judge_dqr_num)
                best_quality_name = ['Naive RAG', 'Re-RAG', 'DQR-RAG'][[judge_naive_num, judge_rerank_num, judge_dqr_num].index(best_quality)]
                st.markdown(f"**⭐ Best Quality:** {best_quality_name} ({best_quality}/10)")
                
                fastest = min(naive_latency, rerank_latency, dqr_latency)
                fastest_name = ['Naive RAG', 'Re-RAG', 'DQR-RAG'][[naive_latency, rerank_latency, dqr_latency].index(fastest)]
                st.markdown(f"**⚡ Fastest:** {fastest_name} ({fastest:.2f}s)")
                
                cheapest = min(naive_cost, rerank_cost, dqr_cost)
                cheapest_name = ['Naive RAG', 'Re-RAG', 'DQR-RAG'][[naive_cost, rerank_cost, dqr_cost].index(cheapest)]
                st.markdown(f"**💰 Most Cost-Effective:** {cheapest_name} (${cheapest:.5f})")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 💡 Recommendations")
                st.markdown("<div style='font-size: 1.1rem; line-height: 2; font-weight: 500;'>", unsafe_allow_html=True)
                
                accuracy_best = max(['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                                   key=lambda x: [mrr_naive, mrr_rerank, mrr_dqr][['Naive RAG', 'Re-RAG', 'DQR-RAG'].index(x)])
                st.markdown(f"**For accuracy:** Use **{accuracy_best}**")
                
                speed_best = min(['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                                key=lambda x: [naive_latency, rerank_latency, dqr_latency][['Naive RAG', 'Re-RAG', 'DQR-RAG'].index(x)])
                st.markdown(f"**For speed:** Use **{speed_best}**")
                
                quality_best = max(['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                                  key=lambda x: [judge_naive_num, judge_rerank_num, judge_dqr_num][['Naive RAG', 'Re-RAG', 'DQR-RAG'].index(x)])
                st.markdown(f"**For quality:** Use **{quality_best}**")
                
                cost_best = min(['Naive RAG', 'Re-RAG', 'DQR-RAG'], 
                               key=lambda x: [naive_cost, rerank_cost, dqr_cost][['Naive RAG', 'Re-RAG', 'DQR-RAG'].index(x)])
                st.markdown(f"**For cost-efficiency:** Use **{cost_best}**")
                st.markdown("</div>", unsafe_allow_html=True)
            
            if show_debug:
                st.markdown("---")
                st.markdown("#### 🐛 Debug Information")
                with st.expander("View Raw Responses"):
                    st.json({
                        "naive_rag": res_naive,
                        "re_rag": res_rerank,
                        "dqr_rag": res_dqr,
                        "scores": {
                            "naive": {"raw": judge_naive, "numeric": judge_naive_num},
                            "rerank": {"raw": judge_rerank, "numeric": judge_rerank_num},
                            "dqr": {"raw": judge_dqr, "numeric": judge_dqr_num}
                        }
                    })

    # Summary metrics with premium styling
    total_time = time.time() - start_all
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Total Time", f"{total_time:.2f}s")
    with col2:
        st.metric("🔄 Pipelines", "3")
    with col3:
        st.metric("📦 Total Chunks", f"{len(naive_chunks) + len(rerank_chunks) + len(dqr_chunks)}")
    with col4:
        st.metric("💰 Total Cost", f"${naive_cost + rerank_cost + dqr_cost:.5f}")

else:
    # Premium welcome screen
    st.markdown("""
    <div class='welcome-card' style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); padding: 4rem 2rem; border-radius: 32px; text-align: center; box-shadow: 0 25px 70px rgba(0,0,0,0.2), 0 0 0 2px rgba(255,255,255,0.4); border: 2px solid rgba(255,255,255,0.5);'>
        <h2 style='color: #0f172a; margin-bottom: 1.5rem; font-size: 2.75rem; font-weight: 800; letter-spacing: -0.02em;'>👋 Welcome to RAG Evaluation Suite</h2>
        <p style='color: #475569; font-size: 1.3rem; margin-bottom: 3rem; font-weight: 600;'>Enter a query above to compare three different RAG pipelines side-by-side</p>
        
        <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 3rem;'>
            <div class='pipeline-card' style='background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%); padding: 2.5rem; border-radius: 24px; max-width: 260px; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2); border: 3px solid rgba(102, 126, 234, 0.3);'>
                <div style='font-size: 3.5rem; margin-bottom: 1.25rem;'>🔵</div>
                <h3 style='color: #667eea; font-size: 1.75rem; margin-bottom: 1rem; font-weight: 800;'>Naive RAG</h3>
                <p style='color: #475569; font-size: 1.1rem; line-height: 1.7; font-weight: 500;'>Simple retrieval and generation baseline</p>
            </div>
            
            <div class='pipeline-card' style='background: linear-gradient(135deg, #f8fafc 0%, #d1fae5 100%); padding: 2.5rem; border-radius: 24px; max-width: 260px; box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2); border: 3px solid rgba(16, 185, 129, 0.3);'>
                <div style='font-size: 3.5rem; margin-bottom: 1.25rem;'>🟢</div>
                <h3 style='color: #10b981; font-size: 1.75rem; margin-bottom: 1rem; font-weight: 800;'>Re-RAG</h3>
                <p style='color: #475569; font-size: 1.1rem; line-height: 1.7; font-weight: 500;'>Enhanced with intelligent reranking</p>
            </div>
            
            <div class='pipeline-card' style='background: linear-gradient(135deg, #f8fafc 0%, #ede9fe 100%); padding: 2.5rem; border-radius: 24px; max-width: 260px; box-shadow: 0 10px 30px rgba(139, 92, 246, 0.2); border: 3px solid rgba(139, 92, 246, 0.3);'>
                <div style='font-size: 3.5rem; margin-bottom: 1.25rem;'>🟣</div>
                <h3 style='color: #8b5cf6; font-size: 1.75rem; margin-bottom: 1rem; font-weight: 800;'>DQR-RAG</h3>
                <p style='color: #475569; font-size: 1.1rem; line-height: 1.7; font-weight: 500;'>Dynamic query rewriting for better results</p>
            </div>
        </div>
        
        <div style='margin-top: 3rem; padding: 2rem; background: rgba(59, 130, 246, 0.12); border-radius: 20px; border: 3px solid rgba(59, 130, 246, 0.3);'>
            <p style='color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0;'>💡 Pro Tip: Toggle "Show retrieved chunks" in the sidebar to see detailed source information</p>
        </div>
    </div>
    """, unsafe_allow_html=True)