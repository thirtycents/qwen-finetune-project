# HANDOFF CONTEXT

## USER REQUESTS (AS-IS)
- "你检索下最新文档 应该是有适配的" (Check latest docs for adapter - confirmed PyTorch Nightly for RTX 5070 Ti)
- "可视化页面要有中英文切换2种语言" (Dashboard needs bilingual support)
- "请你把这个项目跑起来" (Run the project - failed at training due to OOM)
- "将还需要完成的 未完成的 已经完成的 以及上下文总结成文档,用于后续开发" (Summarize for future dev)
- "还需要把你遇到的问题全部记录下来，写一个遇到的问题与解决方法的md，用来在面试时说" (Document issues/solutions for interview prep)

## GOAL
Resolve CUDA OOM error during training on 12GB GPU, verify bilingual dashboard, and use the created interview doc for preparation.

## WORK COMPLETED
- **Environment Setup**: Updated `scripts/setup_env.sh` to install PyTorch Nightly (CUDA 12.8) for RTX 5070 Ti support.
- **Dashboard I18n**: Implemented English/Chinese language switching in `dashboard/app.py`.
- **Data Preparation**: Successfully ran `scripts/prepare_data.py`.
- **Training Attempt**: Launched training via `scripts/train.sh` (failed with OOM).
- **Documentation**: Created `docs/面试_常见问题与解决方案.md` covering Hardware, OOM, i18n, and Shell scripting issues.

## CURRENT STATE
- **Training**: FAILED with `torch.OutOfMemoryError` on 12GB GPU.
    - Error: `CUDA out of memory. Tried to allocate 9.27 GiB...`
- **Dashboard**: Code implemented but untested (dependent on model).
- **Git**: Uncommitted changes in `scripts/`, `configs/`, `dashboard/`, and `docs/`.

## PENDING TASKS
- **Fix OOM** (Priority 1):
    - Edit `configs/qwen3_lora_sft.yaml`: Set `per_device_train_batch_size` to `1`.
    - Edit `configs/qwen3_lora_sft.yaml`: Set `gradient_accumulation_steps` to `4` (or 8).
    - Run: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training.
- **Fix Dashboard Script**: Edit `scripts/start_dashboard.sh` to verify the conda hook fix works.
- **Verify Dashboard**: Run `streamlit run dashboard/app.py` after obtaining a model.

## KEY FILES
- `docs/面试_常见问题与解决方案.md`: **NEW** - Critical for user's interview prep.
- `training_v3.log`: Contains the OOM error trace.
- `scripts/setup_env.sh`: Custom PyTorch Nightly installation logic.
- `dashboard/app.py`: Main dashboard logic with i18n.
- `scripts/train.sh`: Training entry point.
- `configs/qwen3_lora_sft.yaml`: Training configuration (needs tuning).

## IMPORTANT DECISIONS
- Used PyTorch Nightly (`--pre`) to support Blackwell architecture (RTX 5070 Ti).
- Implemented a simple dictionary-based translation system for the dashboard.
- Created a dedicated "Interview Prep" document separating technical troubleshooting from project code.

## EXPLICIT CONSTRAINTS
- User hardware: RTX 5070 Ti (12GB VRAM).
- Dashboard must support Chinese and English.

## CONTEXT FOR CONTINUATION
- The user is preparing for interviews, so the `docs/面试_...` file is as important as the code.
- The training failure is purely a configuration issue (Batch Size > VRAM). Tuning the config yaml will likely fix it.
- **Next Step**: Tune `configs/qwen3_lora_sft.yaml` and restart `scripts/train.sh`.
