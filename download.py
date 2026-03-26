from huggingface_hub import snapshot_download, login

login(token='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')
model_id = "meta-llama/Llama-3.1-8B-Instruct"
local_dir = "/research/d7/fyp25/pyli3/chat/meta-llama/Llama-3.1-8B-Instruct"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
