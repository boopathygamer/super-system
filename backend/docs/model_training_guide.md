# Model Training Guide — Do I Need to Train a Model?

## Short Answer: No Training Required

Super System is an **API-based agentic orchestration platform**. It does **not** require any model training, fine-tuning, or GPU infrastructure. It connects to pre-trained large language models (LLMs) via their cloud APIs and uses them as intelligent engines for code generation, reasoning, and analysis.

You only need a valid **API key** from at least one supported provider to get started.

---

## How the System Uses Models

Super System integrates with three major LLM providers through their official APIs:

| Provider | Model | Environment Variable | SDK |
|----------|-------|---------------------|-----|
| Google Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` | `google-generativeai` |
| Anthropic Claude | `claude-sonnet-4-20250514` | `CLAUDE_API_KEY` | `anthropic` |
| OpenAI ChatGPT | `gpt-4o` | `OPENAI_API_KEY` | `openai` |

Every interaction with a model is a **stateless API call**. The system sends a prompt, receives a response, and processes it. No weights are downloaded, no gradients are computed, and no training loops are executed.

### Quick Setup

1. Get an API key from any supported provider.
2. Set the environment variable (or add it to a `.env` file):
   ```bash
   # Pick one (or more) of the following:
   GEMINI_API_KEY=your-key-here
   CLAUDE_API_KEY=your-key-here
   OPENAI_API_KEY=your-key-here
   ```
3. Run the system:
   ```bash
   python main.py --chat          # Interactive chat
   python main.py --providers     # List available providers
   python main.py                 # Start API server
   ```

---

## Components That May Sound Like Training (But Are Not)

### 1. Code Evolution Engine (`brain/evolution.py`)

**What it does:** Generates multiple code implementations via LLM API calls, then **benchmarks them locally** by executing each candidate in a sandbox and measuring speed, memory usage, and correctness.

**Why it is not training:**
- The LLM generates code candidates via standard API calls (no fine-tuning).
- Selection is based on **benchmark execution results**, not gradient descent.
- The LLM does not learn from the results — it is stateless across calls.

Think of it as: *"LLM as code generator + evolutionary selection via benchmarks."*

### 2. AESCE — Auto-Evolution & Synthesized Consciousness Engine (`brain/aesce.py`)

**What it does:** Identifies recurring failures from memory, maps them to specific source files, uses the LLM to generate improved versions of those files, tests the mutations in an isolated Shadow Matrix, and stages successful improvements.

**Why it is not training:**
- The LLM is asked to **rewrite source code files** — it is used as a code generator.
- No model weights are updated. The LLM remains unchanged.
- What "evolves" is the **system's own Python source code**, not a neural network.

### 3. Self-Mutator (`brain/self_mutator.py`)

**What it does:** Reads a target Python file from disk, sends it to the LLM with context about past failures, and asks for an improved version. Generates multiple variants for testing.

**Why it is not training:**
- Pure text generation via API. No backpropagation or weight updates.
- The mutations target **source code files**, not model parameters.

### 4. Reward Model (`brain/reward_model.py`)

**What it does:** Scores generated code across six dimensions (static analysis, property checks, scenario tests, critic evaluation, code quality, security). Uses domain-specific weight matrices and EMA normalization.

**Why it is not training:**
- The reward model **evaluates code quality** using rule-based and heuristic scoring.
- It does not train any neural network. Its weights are predefined scoring multipliers.
- It is used by the evolution engine to **rank candidates**, not to update model parameters.

### 5. Memory Manager (`brain/memory.py`)

**What it does:** Stores and retrieves conversation history, learned principles, and past failures using ChromaDB (a vector database).

**Why it is not training:**
- Vector storage for semantic search — not model training.
- The LLM itself does not incorporate these memories into its weights.

---

## What Would Require Training (and Why Super System Doesn't)

| Approach | Requires Training? | Super System's Approach |
|----------|-------------------|------------------------|
| Fine-tuning an LLM on custom data | ✅ Yes (GPU, datasets, hours) | ❌ Not used — relies on pre-trained APIs |
| Training a model from scratch | ✅ Yes (massive compute) | ❌ Not used |
| LoRA / QLoRA adapter training | ✅ Yes (GPU, adapter weights) | ❌ Not used |
| Reinforcement Learning from Human Feedback | ✅ Yes (reward model + PPO) | ❌ Not used — reward model only scores, no policy updates |
| Prompt engineering + API calls | ❌ No | ✅ This is what Super System does |
| Evolutionary code selection | ❌ No | ✅ This is what Super System does |
| RAG (Retrieval-Augmented Generation) | ❌ No | ✅ Memory Manager provides context retrieval |

---

## FAQ

### Q: Do I need a GPU to run Super System?
**No.** All model inference happens on the provider's cloud servers via API. Your local machine only runs the orchestration logic, tool execution, and benchmarking.

### Q: Can I use a local/self-hosted model instead of APIs?
The system is designed for cloud APIs, but the `ProviderRegistry` architecture supports adding custom providers. You would implement the `ModelProvider` interface in `core/model_providers.py` to point at a local model server (e.g., Ollama, vLLM, or a Hugging Face endpoint).

### Q: Does the Code Evolution Engine train the model to write better code?
**No.** It generates multiple candidates via the LLM API, executes them in a sandbox, and selects the fastest/most correct one. The LLM itself does not improve — the *selection process* finds the best output from the model's existing capabilities.

### Q: Does AESCE modify the model?
**No.** AESCE modifies **the system's own Python source code** (e.g., `router.py`, `verifier.py`). The LLM is used as a tool to generate improved code. The model's weights are never touched.

### Q: What about the Reward Model — isn't that part of RLHF training?
The reward model in Super System is a **scoring function**, not a training component. In true RLHF, a reward model guides policy gradient updates to the LLM's weights. Here, the reward model only **ranks candidate code** — no policy updates or gradient descent is involved.

### Q: Will the system get smarter over time without training?
Yes, through two mechanisms:
1. **Memory accumulation** — The Memory Manager stores learned principles and past failures, providing richer context for future prompts.
2. **Source code self-mutation** — AESCE can rewrite the system's own code to fix recurring issues, making the *orchestration logic* smarter without touching the LLM.
