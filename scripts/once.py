from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-1.5B",
    local_dir="../models/base/qwen2-2B",
    local_dir_use_symlinks=False
)