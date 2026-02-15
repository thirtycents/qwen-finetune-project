# 📊 Qwen-0.6-ft Dashboard

## Overview
A visualization frontend for monitoring training, exploring documentation, and testing the model.

## Features
- **Project Overview**: View project README and details.
- **Training Monitor**: Real-time visualization of training loss and metrics from `trainer_log.jsonl`.
- **Knowledge Base**: Browse and read project documentation (`docs/`).
- **Inference Playground**: Chat interface to test the trained model via vLLM.

## How to Run
```bash
bash scripts/start_dashboard.sh
```

## Requirements
- Python 3.10+
- Streamlit
- Plotly
- Pandas
