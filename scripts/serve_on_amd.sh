#!/usr/bin/env bash
# Run on an AMD Developer Cloud GPU droplet (AMD Instinct MI300X, ROCm).
# Serves Gemma (port 8000) and Whisper (port 8001) with vLLM's OpenAI-compatible API.
set -euo pipefail
: "${HF_TOKEN:?Set HF_TOKEN (Gemma is a gated model on Hugging Face)}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY to protect the endpoint}"
IMAGE=rocm/vllm:latest
COMMON="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
  -e HF_TOKEN=$HF_TOKEN -v $HOME/.cache/huggingface:/root/.cache/huggingface"

docker run -d --name gemma $COMMON -p 8000:8000 -e HIP_VISIBLE_DEVICES=0 $IMAGE \
  vllm serve google/gemma-3-12b-it --port 8000 --api-key "$VLLM_API_KEY" --max-model-len 16384

docker run -d --name whisper $COMMON -p 8001:8001 -e HIP_VISIBLE_DEVICES=1 $IMAGE \
  vllm serve openai/whisper-large-v3 --port 8001 --api-key "$VLLM_API_KEY"

echo "Gemma:   http://<this-ip>:8000/v1   Whisper: http://<this-ip>:8001/v1"
echo "Check:   rocm-smi   and   curl -H \"Authorization: Bearer \$VLLM_API_KEY\" localhost:8000/v1/models"
