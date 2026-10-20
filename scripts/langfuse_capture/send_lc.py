"""Two real ChatBedrockConverse calls against the fake Bedrock, traced by Langfuse's LangChain CallbackHandler."""
import os, sys
os.environ.update(LANGFUSE_PUBLIC_KEY="pk-lf-cctest", LANGFUSE_SECRET_KEY="sk-lf-cctest", LANGFUSE_BASE_URL="http://127.0.0.1:43000",
                  LANGFUSE_HOST="http://127.0.0.1:43000", LANGFUSE_TRACING_ENVIRONMENT="langchain",
                  AWS_ACCESS_KEY_ID="x", AWS_SECRET_ACCESS_KEY="x")
from langchain_aws import ChatBedrockConverse
from langfuse import get_client
from langfuse.langchain import CallbackHandler
llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1",
                          endpoint_url=f"http://127.0.0.1:{sys.argv[1]}")
h = CallbackHandler()
for _ in range(2):
    m = llm.invoke("hi", config={"callbacks": [h]})
    print("langchain usage_metadata:", m.usage_metadata)
get_client().flush()
