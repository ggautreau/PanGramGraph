# PanGramGraph as a Hugging Face Space (Docker SDK, CPU): the page and the model's server.
# Built from space_build/ (python3 space/assemble.py), which holds this file, the code and the data.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 HOST=0.0.0.0 PORT=7860
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# the model's weights and code in the image, so a start does not wait for a download
RUN python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('macwiatrak/bacformer-causal-complete-genomes', trust_remote_code=True)"
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "start.py"]
