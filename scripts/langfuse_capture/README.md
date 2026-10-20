# What Langfuse stores for cached Bedrock calls

These scripts produced `tests/fixtures/langfuse_observations_2026-10.json` and the findings on
[cachecanary.com/langfuse](https://cachecanary.com/langfuse/) (October 20, 2026). Each library makes two real calls
to `fake_bedrock.py`, a local stand-in for Bedrock Converse that returns the token counts of a real Bedrock capture
([strands-agents/sdk-python#3546](https://github.com/strands-agents/sdk-python/issues/3546)): the first call writes
5,848 tokens to the cache, the second reads them, each with 10 new input tokens, 4 output tokens and a total of 5,862.
The traces go to a real local Langfuse, and the right answer for every pair is a 50% hit rate.

Versions used: Langfuse 4.64.0 (official docker compose), strands-agents 1.61.1 and 1.53.0, litellm 1.106.0 with
langfuse 4.18.0, langchain-aws 1.9.0 with langfuse 5.0.3. No AWS account or credentials are needed.

1. Start Langfuse from its [docker-compose.yml](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml)
   with the web port on 43000 and a project created at start-up (`LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-cctest`,
   `LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-cctest`, plus the other `LANGFUSE_INIT_*` values).
2. `python fake_bedrock.py 45999 &`
3. One virtual environment per library, then:
   - `python send_strands.py new 45999` (strands-agents[otel] 1.61.1) and `python send_strands.py old 45999` (1.53.0)
   - `python send_litellm.py 45999` and `python send_litellm.py 45999 --otel`
   - `python send_lc.py 45999`
4. Read the records back with `/api/public/v2/observations?type=GENERATION&fields=core,basic,model,usage`, or run
   `cachecanary langfuse --host http://127.0.0.1:43000 --since 1h` with the same keys.
