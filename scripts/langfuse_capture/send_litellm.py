"""Two real LiteLLM Bedrock Converse calls against fake_bedrock.py, logged by LiteLLM's `langfuse` callback
(or its `langfuse_otel` integration with --otel)."""
import os
import sys
import time

port, otel = sys.argv[1], "--otel" in sys.argv
os.environ.update(LANGFUSE_PUBLIC_KEY="pk-lf-cctest", LANGFUSE_SECRET_KEY="sk-lf-cctest", LANGFUSE_HOST="http://127.0.0.1:43000",
                  AWS_ACCESS_KEY_ID="x", AWS_SECRET_ACCESS_KEY="x", AWS_REGION_NAME="us-east-1")
import litellm  # noqa: E402

if otel:
    litellm.callbacks = ["langfuse_otel"]
else:
    litellm.success_callback = ["langfuse"]
for _ in range(2):
    r = litellm.completion(model="bedrock/converse/us.anthropic.claude-sonnet-4-6",
                           messages=[{"role": "user", "content": "hi"}], api_base=f"http://127.0.0.1:{port}",
                           metadata={"generation_name": "litellm-otel" if otel else "litellm-legacy"})
    print("litellm usage:", dict(r.usage))
time.sleep(8)  # both integrations send in the background
