"""Two real Strands Agent calls through BedrockModel (non-streaming) against the fake Bedrock,
traced to local Langfuse over OTLP."""
import base64, os, sys
tag = sys.argv[1]
auth = base64.b64encode(b"pk-lf-cctest:sk-lf-cctest").decode()
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://127.0.0.1:43000/api/public/otel/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = f"Authorization=Basic {auth}"
os.environ["AWS_ACCESS_KEY_ID"] = "x"; os.environ["AWS_SECRET_ACCESS_KEY"] = "x"
os.environ["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"] = f"http://127.0.0.1:{sys.argv[2]}"
from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
tel = StrandsTelemetry().setup_otlp_exporter()
model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-6", region_name="us-east-1", streaming=False)
agent = Agent(model=model, name=f"strands-{tag}", trace_attributes={"langfuse.environment": tag})
agent("hello"); agent("hello again")
tel.tracer_provider.force_flush()
print("strands traces sent")
