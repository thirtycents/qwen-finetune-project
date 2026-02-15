"""
Qwen3-0.6B Function Calling 项目仪表板

多页面 Streamlit 应用，包含：
1. 概览页 - 项目介绍
2. 训练监控 - 实时训练曲线
3. 知识库 - 文档浏览
4. 推理测试 - 模型对话
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import glob
import requests
import time

# ---- Configuration ----
st.set_page_config(
    page_title="Qwen-0.6-ft Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "qwen3-0.6b-fc-lora")
LOG_FILE = os.path.join(OUTPUT_DIR, "trainer_log.jsonl")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")

# ---- Helper Functions ----
def load_log_data(log_file):
    data = []
    if not os.path.exists(log_file):
        return pd.DataFrame()
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def render_overview():
    st.header("📋 Project Overview")
    if os.path.exists(README_FILE):
        with open(README_FILE, "r") as f:
            st.markdown(f.read())
    else:
        st.error("README.md not found!")

def render_training_monitor():
    st.header("📈 Training Monitor")
    
    if not os.path.exists(OUTPUT_DIR):
        st.warning(f"Output directory not found: {OUTPUT_DIR}")
        st.info("Start training with `bash scripts/train.sh`")
        return

    # Auto-refresh mechanism
    auto_refresh = st.checkbox("🔄 Auto Refresh (Every 5s)", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    df = load_log_data(LOG_FILE)
    
    if df.empty:
        st.warning("No training logs found yet. Training might be starting...")
        return

    # Filter for loss logs (exclude eval logs which have 'eval_loss')
    train_df = df[df['loss'].notna()].copy()
    
    if not train_df.empty:
        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        latest = train_df.iloc[-1]
        
        col1.metric("Current Step", int(latest.get('current_steps', latest.get('step', 0))))
        col2.metric("Current Epoch", f"{latest.get('epoch', 0):.2f}")
        col3.metric("Training Loss", f"{latest.get('loss', 0):.4f}")
        col4.metric("Learning Rate", f"{latest.get('learning_rate', 0):.2e}")

        # 创建多子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Training Loss", "Learning Rate", "Gradient Norm", "Training Speed"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Loss curve
        x_col = "current_steps" if "current_steps" in train_df.columns else "step"
        fig.add_trace(
            go.Scatter(x=train_df[x_col], y=train_df["loss"], mode="lines", name="Loss", line=dict(color="red")),
            row=1, col=1
        )
        
        # Learning rate
        if "learning_rate" in train_df.columns:
            fig.add_trace(
                go.Scatter(x=train_df[x_col], y=train_df["learning_rate"], mode="lines", name="LR", line=dict(color="blue")),
                row=1, col=2
            )
        
        # Gradient norm
        if "grad_norm" in train_df.columns:
            fig.add_trace(
                go.Scatter(x=train_df[x_col], y=train_df["grad_norm"], mode="lines", name="Grad Norm", line=dict(color="green")),
                row=2, col=1
            )
        
        # Training speed
        if "train_samples_per_second" in train_df.columns:
            fig.add_trace(
                go.Scatter(x=train_df[x_col], y=train_df["train_samples_per_second"], mode="lines", name="Samples/s", line=dict(color="orange")),
                row=2, col=2
            )
        
        fig.update_layout(height=700, showlegend=False)
        fig.update_xaxes(title_text="Training Steps")
        st.plotly_chart(fig, use_container_width=True)
    
    # Eval Logs
    eval_df = df[df['eval_loss'].notna()].copy()
    if not eval_df.empty:
        st.subheader("Evaluation Metrics")
        fig_eval = px.line(eval_df, x="current_steps" if "current_steps" in eval_df.columns else "step", 
                           y="eval_loss", title="Evaluation Loss", markers=True)
        st.plotly_chart(fig_eval, use_container_width=True)

    # Raw Data Expander
    with st.expander("Raw Log Data"):
        st.dataframe(df)

def render_knowledge_base():
    st.header("📚 Knowledge Base")
    
    if not os.path.exists(DOCS_DIR):
        st.error(f"Docs directory not found: {DOCS_DIR}")
        return

    docs_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    docs_map = {os.path.basename(f): f for f in docs_files}
    
    if not docs_map:
        st.warning("No documentation files found.")
        return

    selected_doc = st.sidebar.selectbox("Select Document", list(docs_map.keys()))
    
    if selected_doc:
        with open(docs_map[selected_doc], "r") as f:
            st.markdown(f.read())

def render_inference_playground():
    st.header("🎮 Inference Playground")
    st.info("Ensure `bash scripts/serve.sh` is running!")

    # Sidebar settings
    with st.sidebar:
        st.subheader("Model Settings")
        api_base = st.text_input("API Base URL", "http://localhost:8000/v1/chat/completions")
        model_name = st.text_input("Model Name", "qwen3-0.6b-fc-merged")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 64, 2048, 512)

    # System prompt
    system_prompt = "You are a helpful assistant with access to tools. When you need to call a function, output a JSON object with 'name' and 'arguments' fields."
    st.info(f"**System Prompt**: {system_prompt}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask Qwen something... (e.g., 'What is the weather in Beijing?')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            try:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                with st.spinner("Model thinking..."):
                    response = requests.post(api_base, json=payload, timeout=30)
                    response.raise_for_status()
                    
                    result = response.json()
                    assistant_message = result["choices"][0]["message"]["content"]
                
                st.markdown(assistant_message)
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to vLLM service. Ensure `bash scripts/serve.sh` is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. Check model service status.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def render_evaluation_benchmark():
    st.header(t("eval_header"))
    st.info("ℹ️ Run `python eval/evaluate.py` and `python eval/benchmark.py` to generate real data. Below is a simulation.")

    tab1, tab2, tab3 = st.tabs([t("offline_metrics"), t("online_perf"), t("engineering")])

    with tab1:
        st.subheader("Offline Evaluation (Quality)")
        # Mock Data for illustration
        categories = ['Parse Rate', 'Schema Hit', 'Func Accuracy', 'Param F1', 'Exec Rate', 'BFCL Score']
        
        # Comparison Data
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[0.05, 0.03, 0.02, 0.01, 0.01, 0.1],
            theta=categories,
            fill='toself',
            name='Base Model (Qwen3-0.6B)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.95, 0.88, 0.90, 0.85, 0.80, 0.75],
            theta=categories,
            fill='toself',
            name='SFT (LoRA)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.97, 0.92, 0.93, 0.90, 0.88, 0.82],
            theta=categories,
            fill='toself',
            name='SFT + GRPO (Final)'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        {t("metrics_def")}
        - **Parse Rate**: Success rate of JSON parsing.
        - **Schema Hit**: Adherence to tool definitions.
        - **Exec Rate**: Success rate of simulated execution.
        """)

    with tab2:
        st.subheader("Online Performance (vLLM)")
        
        # Mock Benchmark Data
        concurrency = [1, 2, 4, 8, 16, 32]
        ttft_p50 = [15, 18, 25, 45, 80, 150] # ms
        tps = [45, 85, 160, 300, 550, 900]   # tokens/s
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lat = px.line(x=concurrency, y=ttft_p50, markers=True, 
                              title="Latency vs Concurrency", labels={'x': 'Concurrency', 'y': 'TTFT P50 (ms)'})
            st.plotly_chart(fig_lat, use_container_width=True)
            
        with col2:
            fig_tps = px.bar(x=concurrency, y=tps, 
                             title="Throughput vs Concurrency", labels={'x': 'Concurrency', 'y': 'Tokens/s'})
            st.plotly_chart(fig_tps, use_container_width=True)

        st.success(t("rec_concurrency"))

    with tab3:
        st.subheader("Engineering Delivery Status")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Docker Image", "Ready", "vllm-inference:latest")
        col2.metric("K8s Deployment", "Configured", "replicas: 1")
        col3.metric("CI/CD Pipeline", "Active", "GitHub Actions")
        
        st.markdown("### Deployment Checklist")
        st.checkbox("Docker Build", value=True, disabled=True)
        st.checkbox("Kubernetes Manifests", value=True, disabled=True)
        st.checkbox("Prometheus Monitoring", value=True, disabled=True)
        st.checkbox("Grafana Dashboards", value=True, disabled=True)

# ---- Main Layout ----
st.sidebar.title("🚀 Qwen-0.6-ft Control")
page = st.sidebar.radio("Navigation", ["Overview", "Training Monitor", "Evaluation & Benchmark", "Knowledge Base", "Inference Playground"])

if page == "Overview":
    render_overview()
elif page == "Training Monitor":
    render_training_monitor()
elif page == "Evaluation & Benchmark":
    render_evaluation_benchmark()
elif page == "Knowledge Base":
    render_knowledge_base()
elif page == "Inference Playground":
    render_inference_playground()
