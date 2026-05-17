# Compliance Guardrail Experiments on tau2-bench

This repository contains a tau2-bench based evaluation setup for studying whether a dedicated guardrail layer can reduce policy-violating tool calls in customer-service agents.

The important idea is simple:

```text
User simulator → Agent → Guardrail middleware → Tool execution
```

The agent still decides what tool to call. The guardrail middleware sees that tool call before it reaches the real tool. If the call violates a mapped policy rule, the guard blocks it and returns feedback to the agent instead of executing the side effect.

The project currently supports experiments across:

- `airline`
- `telecom`
- `retail`

It also contains a policy-tool mapper pipeline that maps policy passages to the tools they govern.

---

## Start Here: Web Viewer and Experiment Runner

The main way to work with the project is the custom viewer in:

```text
compliant_agents_fresh/viewer/
```

It is not the old `web/leaderboard` app. The viewer is the project-specific UI for:

- browsing all completed simulation runs
- opening individual task conversations
- comparing guarded vs unguarded experiments
- analyzing reward, policy violation rate, latency, and guard blocks
- scheduling full tau2 experiments from the browser
- scheduling policy-tool-mapper runs from the browser
- cancelling queued or running experiments
- exporting plots to `;`-separated CSV

Run it from the repository root:

```bash
cd compliant_agents_fresh
python3 viewer/server.py
```

Then open:

```text
http://localhost:8765
```

Use a different port if needed:

```bash
cd compliant_agents_fresh
VIEWER_PORT=8766 python3 viewer/server.py
```

The viewer reads simulation results from:

```text
data/simulations/
```

and policy-mapper results from:

```text
policy_tool_mapper/output/
```

---

## Install and Configure

Install dependencies:

```bash
cd compliant_agents_fresh
uv sync
```

Create or update the environment file:

```text
compliant_agents_fresh/.env
```

Typical keys used by experiments:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

The exact keys needed depend on the models you run.

---

## Running Full Agent Experiments

The core command is:

```bash
uv run tau2 run \
  --domain <domain> \
  --agent llm_agent \
  --guardrail-config <guardrail_config> \
  --agent-llm <agent_model> \
  --user-llm <user_model> \
  --num-trials <n> \
  --max-concurrency <n>
```

Results are written to:

```text
data/simulations/<run_id>/results.json
```

### Baseline Without Guard

Use `guardrail_configs/null.json`.

```bash
uv run tau2 run \
  --domain airline \
  --agent llm_agent \
  --guardrail-config guardrail_configs/null.json \
  --agent-llm gpt-4.1-mini \
  --user-llm gpt-5.1 \
  --num-trials 3 \
  --max-concurrency 3
```

### With Guard

Use the domain guard config.

```bash
uv run tau2 run \
  --domain airline \
  --agent llm_agent \
  --guardrail-config guardrail_configs/airline_llm_guard.json \
  --agent-llm gpt-4.1-mini \
  --user-llm gpt-5.1 \
  --guard-llm gpt-4.1-mini \
  --num-trials 3 \
  --max-concurrency 3
```

`--guard-llm` overrides the LLM used by all LLM-based guards in the guardrail config. Everything else in the config stays unchanged.

### Domain-Specific Guard Configs

Current main configs:

```text
guardrail_configs/airline_llm_guard.json
guardrail_configs/telecom_llm_guard.json
guardrail_configs/retail_llm_guard.json
guardrail_configs/null.json
```

`null.json` is the no-guard baseline.


### Run Specific Task IDs

```bash
uv run tau2 run \
  --domain airline \
  --agent llm_agent \
  --guardrail-config guardrail_configs/airline_llm_guard.json \
  --agent-llm gpt-4.1-mini \
  --user-llm gpt-5.1 \
  --guard-llm gpt-4.1-mini \
  --num-trials 1 \
  --max-concurrency 1 \
  --task-ids 0 1 4 5 9 11 12
```


---

## Reading Simulation Results

The recommended path is the web viewer:

```bash
cd compliant_agents_fresh
python3 viewer/server.py
```

Open:

```text
http://localhost:8765
```

Useful pages in the viewer:

- **Simulation Runs**: all result folders with reward, policy violation rate, latency, guard model, agent model, user model, and sample count.
- **Experiment Analyzer**: select guarded and unguarded runs per model and compare policy violation rate, reward, latency, and guard block counts.
- **Task Conversation View**: inspect the actual dialogue and tool calls for one task/trial.
- **Experiment Scheduler**: create queued tau2 runs from the browser instead of hand-writing CLI commands.
- **Mapper Scheduler / Analyzer**: run and compare policy-tool-mapper experiments.

The analyzer’s guard-block graph is count-based:

- **Total guard blocks**: all tool calls blocked by guards.
- **Correctly blocked**: blocked calls that matched an explicit `unauthorized_action` compliance assertion in the task JSON.
- **Unmatched block chats**: links to conversations where the guard blocked a call that did not match an explicit task compliance assertion.

---

## Policy-Tool Mapper

The policy-tool mapper creates mappings from tools to the policy passages that constrain them.

There are two modes:

### LLM Mode

```text
chunker → profiler → mapper → sweeper
```

The LLM directly maps policy statements to tools.

### Retrieval Mode

```text
chunker → profiler → BM25 + embedding search → cross-encoder top-k reranker → LLM judge → sweeper
```

This mode first retrieves candidate policy passages, reranks them, and then asks an LLM judge to verify the candidates.

---

## Show Already Completed Policy-Mapper Results

This is the most important command when you want the full overview of existing mapper results.

It does not rerun mapping or evaluation. It only reads existing files from:

```text
policy_tool_mapper/output/
```

Airline:

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain airline \
  --compare
```

Telecom:

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain telecom \
  --compare
```

Retail:

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain retail \
  --compare
```

---

## Policy-Mapper Inputs

Policy and tool input files live in:

```text
policy_tool_mapper/input/
```

Expected files:

```text
policy_tool_mapper/input/airlinePolicy.md
policy_tool_mapper/input/airlineTools.json
policy_tool_mapper/input/telecomPolicy.md
policy_tool_mapper/input/telecomTools.json
policy_tool_mapper/input/retailPolicy.md
policy_tool_mapper/input/retailTools.json
```

Ground truth lives in:

```text
policy_tool_mapper/ground_truth/
```

Important files:

```text
policy_tool_mapper/ground_truth/airline-ground-truth.json
policy_tool_mapper/ground_truth/telecom-ground-truth.json
policy_tool_mapper/ground_truth/retail-ground-truth.json
```

Outputs are written to:

```text
policy_tool_mapper/output/
```

---

## Run the Full Policy-Mapper Pipeline

Run from `compliant_agents_fresh/`.

### LLM Mapping

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain airline \
  --models gpt-4.1-mini gpt-4.1 gpt-5.1 gpt-5.4 \
  --mode llm
```

For telecom:

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain telecom \
  --models gpt-4.1-mini gpt-4.1 gpt-5.1 gpt-5.4 \
  --mode llm
```

For retail:

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain retail \
  --models gpt-4.1-mini gpt-4.1 gpt-5.1 gpt-5.4 \
  --mode llm
```

### Retrieval Mapping

```bash
uv run python policy_tool_mapper/run_pipeline.py \
  --domain airline \
  --models gpt-4.1-mini gpt-4.1 gpt-5.1 gpt-5.4 \
  --mode retrieval \
  --ce-top-k 20
```

The default retrieval setup uses:

```text
Embedding model: text-embedding-3-large
Cross-encoder:   cross-encoder/ms-marco-MiniLM-L-6-v2
CE top-k:        20
```


The comparison table includes:

- model
- mode: `llm` or `retrieval`
- confidence slice: `high` or `all`
- macro precision
- macro recall
- macro F1
- micro precision
- micro recall
- micro F1

The files that feed the comparison look like:

```text
policy_tool_mapper/output/airline-eval-gpt-4.1-mini-high.json
policy_tool_mapper/output/airline-eval-gpt-4.1-mini-all.json
policy_tool_mapper/output/airline-eval-gpt-4.1-mini-retrieval-high.json
policy_tool_mapper/output/airline-eval-gpt-4.1-mini-retrieval-all.json
```

You can also inspect and compare these results in the web viewer under the mapper analyzer.

---

## Run One Mapper Call Manually

Use `policy-map` directly when debugging one mapping run.

```bash
uv run policy-map \
  --policy policy_tool_mapper/input/airlinePolicy.md \
  --openapi policy_tool_mapper/input/airlineTools.json \
  --output policy_tool_mapper/output/airline-mappings-debug.json \
  --model gpt-4.1-mini \
  --mode llm
```

Retrieval mode:

```bash
uv run policy-map \
  --policy policy_tool_mapper/input/airlinePolicy.md \
  --openapi policy_tool_mapper/input/airlineTools.json \
  --output policy_tool_mapper/output/airline-mappings-debug-retrieval.json \
  --model gpt-4.1-mini \
  --mode retrieval \
  --ce-top-k 20
```

Evaluate one mapping file:

```bash
uv run policy-map-eval \
  --predicted policy_tool_mapper/output/airline-mappings-debug.json \
  --ground-truth policy_tool_mapper/ground_truth/airline-ground-truth.json \
  --output policy_tool_mapper/output/airline-eval-debug.json
```

Evaluate high-confidence mappings only:

```bash
uv run policy-map-eval \
  --predicted policy_tool_mapper/output/airline-mappings-debug.json \
  --ground-truth policy_tool_mapper/ground_truth/airline-ground-truth.json \
  --output policy_tool_mapper/output/airline-eval-debug-high.json \
  --confidence-high-only
```

---

## Key Data Files

Tasks:

```text
data/tau2/domains/airline/tasks.json
data/tau2/domains/telecom/tasks.json
data/tau2/domains/retail/tasks.json
```

Task splits:

```text
data/tau2/domains/airline/split_tasks.json
data/tau2/domains/telecom/split_tasks.json
data/tau2/domains/retail/split_tasks.json
```

Policies:

```text
data/tau2/domains/airline/policy.md
data/tau2/domains/telecom/main_policy.md
data/tau2/domains/retail/policy.md
```

Tools:

```text
src/tau2/domains/airline/tools.py
src/tau2/domains/telecom/tools.py
src/tau2/domains/retail/tools.py
```

Databases:

```text
data/tau2/domains/airline/db.json
data/tau2/domains/retail/db.json
data/tau2/domains/telecom/db.toml
data/tau2/domains/telecom/user_db.toml
```

Guard configs:

```text
guardrail_configs/
```

Simulation output:

```text
data/simulations/
```

Policy-mapper output:

```text
policy_tool_mapper/output/
```

---

## Suggested Experiment Pattern

For each domain and model, run one baseline and one guarded experiment:

```bash
# Without guard
uv run tau2 run \
  --domain telecom \
  --agent llm_agent \
  --guardrail-config guardrail_configs/null.json \
  --agent-llm gpt-4.1-mini \
  --user-llm gpt-5.1 \
  --num-trials 3 \
  --max-concurrency 3

# With guard
uv run tau2 run \
  --domain telecom \
  --agent llm_agent \
  --guardrail-config guardrail_configs/telecom_llm_guard.json \
  --agent-llm gpt-4.1-mini \
  --user-llm gpt-5.1 \
  --guard-llm gpt-4.1-mini \
  --num-trials 3 \
  --max-concurrency 3
```

Then open the viewer and compare the pair in the **Experiment Analyzer**.

The main plots to report are:

- policy violation rate
- reward
- latency change
- guard block counts

For the policy mapper, run both:

- `--mode llm`
- `--mode retrieval`

Then use:

```bash
uv run python policy_tool_mapper/run_pipeline.py --domain <domain> --compare
```

or the mapper analyzer in the viewer.

---

## Common Notes

### Rate limits

If you hit provider rate limits:

- lower `--max-concurrency`
- use fewer models in one mapper batch
- run fewer trials
- resume later with `--auto-resume`

For Anthropic especially, input-token-per-minute limits can be hit even when request count looks low.

### Guard false positives

The viewer’s unmatched block list is a conservative diagnostic, not a perfect semantic false-positive oracle.

It counts a guard block as “correctly blocked” only if the blocked tool call matches an explicit `unauthorized_action` compliance assertion in the task JSON. A block that does not match such an assertion is shown as unmatched so the chat can be inspected manually.

---

## Repository Map

```text
compliant_agents_fresh/
├── data/tau2/domains/              # Domain policies, tasks, DBs, splits
├── data/simulations/               # Completed tau2 experiment results
├── guardrail_configs/              # Guardrail JSON configs
├── policy_tool_mapper/             # Policy-to-tool mapping pipeline
│   ├── input/                      # Mapper policies and tool JSON
│   ├── ground_truth/               # Human ground-truth mappings
│   └── output/                     # Mapper outputs and eval files
├── src/tau2/guardrails/            # Guard middleware and guard implementations
├── src/tau2/domains/               # Domain tool implementations
└── viewer/                         # Custom web viewer, scheduler, analyzers
```

---

## Paper-Facing Metrics

For guardrail architecture experiments:

- primary safety metric: policy violation rate
- utility metric: reward
- cost metric: average task latency
- diagnostic metric: guard block counts and unmatched block chats

For policy-tool mapping:

- primary balanced metric: F1
- safety-oriented metric: recall
- precision-oriented metric: precision
- report both macro and micro scores
- compare `llm` vs `retrieval`
- compare `high` vs `all` confidence slices
