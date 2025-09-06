# lamas
LLM collaboration with Multi-Agent Systems

### Environment installation

```shell
uv venv lamas --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> For Hugging Face cluster users, add `export UV_LINK_MODE=copy` to your `.bashrc` to suppress cache warnings from `uv`

Next, install vLLM and FlashAttention:

```shell
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```
