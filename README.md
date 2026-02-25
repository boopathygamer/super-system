# 🧠 Universal AI Agent — Super Agent

A powerful, multi-model AI agent system with specialized personas, 10 domain experts, advanced reasoning engines, and a rich set of autonomous capabilities — all accessible via a CLI or a REST API.

---

## Table of Contents

1. [What This Project Can Do](#what-this-project-can-do)
2. [Architecture Overview](#architecture-overview)
3. [Setup & Installation](#setup--installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [REST API Reference](#rest-api-reference)
7. [Domain Experts](#domain-experts)
8. [Reasoning Strategies](#reasoning-strategies)
9. [Communication Personas](#communication-personas)
10. [Specialized Agent Profiles](#specialized-agent-profiles)
11. [Built-in Tools](#built-in-tools)
12. [Safety & Ethics Layer](#safety--ethics-layer)
13. [Configuration](#configuration)

---

## What This Project Can Do

### Multi-Model LLM Backend
Connect to any of the three leading AI providers — or switch between them at runtime:

| Provider | Model (default) | Environment Variable |
|---|---|---|
| 🟢 Google Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| 🟣 Anthropic Claude | `claude-sonnet-4-20250514` | `CLAUDE_API_KEY` |
| 🔵 OpenAI ChatGPT | `gpt-4o` | `OPENAI_API_KEY` |

The system **auto-detects** which API keys are available and picks the best provider, with automatic **failover** if a provider is unavailable.

---

### Core Agent Capabilities

| Feature | Description |
|---|---|
| **Interactive Chat** | Full conversational chat with persistent session memory |
| **Thinking Loop** | Multi-iteration self-reflection for higher-confidence answers |
| **Tool Use** | 12 built-in tools (web search, code execution, file ops, etc.) |
| **Memory** | Hybrid vector + BM25 memory that learns from past conversations |
| **Session Management** | Persistent JSONL sessions with automatic compaction |
| **Background Processes** | Async task execution with polling |
| **Skills Registry** | Dynamically loadable skill modules |
| **Streaming (SSE)** | Server-Sent Events for real-time response streaming |
| **Workspace Injection** | Agent-aware context bootstrapping from workspace files |

---

### Specialized Autonomous Modes

| CLI Flag | Mode | What it does |
|---|---|---|
| `--evolve` | **Code Evolution Engine** | Generates multiple implementations, benchmarks them in a sandbox, and returns the fastest/most-correct solution via RLHF-style genetic selection |
| `--nightwatch` | **Night Watch Daemon** | Proactive background daemon that runs nightly security and system audits |
| `--audit` | **Threat Hunter** | Red-team security audit of any source file — finds injection vectors, hardcoded secrets, race conditions, and proposes remediation patches |
| `--transpile` | **Reverse-Engineering Transpiler** | Reads legacy code from a directory and rewrites it in a target language (e.g., Python → Rust) |
| `--board-meeting` | **Devil's Advocate** | Assembles a virtual Board of Directors to stress-test a business plan, surface hidden risks, and produce a Risk Matrix |
| `--tutor` | **Socratic Auto-Tutor** | Interactive tutoring session on any topic using the Socratic method |
| `--syndicate` | **Content Factory** | Transforms a single text file into a multi-channel content package (blog, tweet thread, LinkedIn post, etc.) |
| `--organize` | **Digital Estate Archivist** | Scans a messy directory and intelligently reorganizes and renames files |
| `--contract-audit` | **Toxic Clause Hunter** | Reviews legal contracts (PDF/DOCX) and flags predatory or one-sided clauses |
| `--deploy-swarm` | **Active Defense Swarm** | Deploys honeypots and tarpits to detect and slow down intrusion attempts |
| `--deep-research` | **Deep Web Intelligence Bot** | Compiles a research dossier on any topic using live web search + scraping |

---

### 10 Domain Experts

The agent automatically classifies every request into one of ten knowledge domains and injects the matching expert context:

`💻 Code` · `✍️ Writing` · `🔢 Math` · `📊 Business` · `🎨 Creative` · `📚 Education` · `💪 Health` · `⚖️ Legal` · `📈 Data` · `🌟 Lifestyle`

---

### 5 Adaptive Communication Personas

The agent detects the user's communication style and adapts automatically:

`🌱 Beginner (Friendly Guide)` · `💼 Professional` · `🎓 Student` · `🎨 Creative` · `👔 Executive`

---

### 4 Advanced Reasoning Strategies

| Strategy | Best for |
|---|---|
| **Chain-of-Thought** | Step-by-step logical problems, math, debugging |
| **Tree-of-Thought** | Open-ended problems with multiple valid approaches |
| **Analogy Reasoning** | Explaining unfamiliar concepts via familiar ones |
| **Socratic Method** | Educational / tutoring scenarios |

---

## Architecture Overview

```
super-agent/
└── backend/
    ├── main.py                  # CLI entry point
    ├── api/
    │   ├── server.py            # FastAPI REST server
    │   └── models.py            # Pydantic request/response models
    ├── agents/
    │   ├── controller.py        # Master orchestrator (10 subsystems)
    │   ├── compiler.py          # Task → TaskSpec
    │   ├── generator.py         # Hypothesis generation
    │   ├── experts/             # Domain router + 10 expert definitions
    │   ├── persona/             # 5 adaptive communication personas
    │   ├── profiles/            # Specialized agent modes
    │   │   ├── threat_hunter.py
    │   │   ├── devils_advocate.py
    │   │   ├── socratic_tutor.py
    │   │   ├── contract_hunter.py
    │   │   └── deep_researcher.py
    │   ├── proactive/           # Background / autonomous agents
    │   │   ├── night_watch.py
    │   │   ├── archivist.py
    │   │   └── swarm_defense.py
    │   ├── safety/              # Content filter, PII guard, ethics engine
    │   ├── sessions/            # JSONL-backed session persistence
    │   ├── skills/              # Dynamically loadable skill modules
    │   └── tools/               # 12 built-in tools
    ├── brain/
    │   ├── memory.py            # Hybrid vector + BM25 memory
    │   ├── thinking_loop.py     # Self-reflection iterations
    │   ├── advanced_reasoning.py # 4 reasoning strategies
    │   ├── evolution.py         # Code Evolution Engine (RLHF)
    │   ├── transpiler.py        # Reverse-engineering transpiler
    │   ├── content_factory.py   # Content syndication pipeline
    │   └── ...
    ├── core/
    │   ├── model_providers.py   # Gemini / Claude / ChatGPT adapters
    │   ├── model_router.py      # Failover routing
    │   └── streaming.py         # SSE streaming
    └── config/
        └── settings.py          # All configuration
```

**Agent Controller — 10 Subsystems:**

1. **Tool Policy Engine** — Per-agent access control profiles (`minimal | coding | assistant | full`)
2. **Loop Detector** — Circuit-breaker guardrail to prevent infinite tool-call loops
3. **Session Manager** — JSONL-backed persistent conversations with agent-to-agent messaging
4. **Process Manager** — Background async task execution with polling
5. **Workspace + Skills** — Context injection from workspace files and dynamic skill modules
6. **Streaming** — SSE token streaming with sentence-level coalescing
7. **Domain Router** — Auto-classifies requests into 10 expert domains
8. **Persona Engine** — Adapts communication style to detected user type
9. **Advanced Reasoner** — Selects optimal reasoning strategy per request
10. **Safety Layer** — Input/output content filtering, PII redaction, ethics engine

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- At least one API key: `GEMINI_API_KEY`, `CLAUDE_API_KEY`, or `OPENAI_API_KEY`

### Install

```bash
cd backend
pip install -r requirements.txt
```

### Configure

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose a provider or leave "auto" to let the system decide
LLM_PROVIDER=auto

GEMINI_API_KEY=your-gemini-key-here
CLAUDE_API_KEY=your-claude-key-here
OPENAI_API_KEY=your-openai-key-here
```

---

## Quick Start

```bash
cd backend

# Start the API server (auto-detect provider)
python main.py

# List available/configured providers
python main.py --providers

# Interactive chat in the terminal
python main.py --chat

# Chat with a specific provider
python main.py --chat --provider claude
```

---

## CLI Reference

```
python main.py [OPTIONS]
```

### General Options

| Flag | Description |
|---|---|
| `--chat` | Start interactive chat in the terminal |
| `--provider auto\|gemini\|claude\|chatgpt` | LLM provider (default: `auto`) |
| `--api-key KEY` | Override API key for the chosen provider |
| `--providers` | List all configured/available providers |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Set log verbosity |

### Interactive Chat Commands

Once inside `--chat`, type these commands:

| Command | Action |
|---|---|
| `/think <message>` | Force the thinking loop (multi-iteration reasoning) |
| `/stats` | Show memory and session statistics |
| `/reset` | Clear conversation history |
| `/provider` | Show active provider name, model, and stats |
| `/models` | List all available providers |
| `/switch gemini\|claude\|chatgpt` | Switch provider mid-conversation |
| `quit` / `exit` | Exit the chat |

### Specialized Modes

```bash
# Code Evolution Engine
python main.py --evolve "write a function to find the longest palindrome in a string"

# Security Audit (Threat Hunter)
python main.py --audit path/to/file.py

# Legacy Code Transpiler (e.g., Python → Rust)
python main.py --transpile path/to/legacy_dir/ --target-lang rust

# Night Watch Daemon (immediate audit for demo)
python main.py --nightwatch

# Devil's Advocate — Board Meeting Risk Matrix
python main.py --board-meeting path/to/business_plan.pdf

# Socratic Auto-Tutor
python main.py --tutor "Quantum Entanglement"

# Content Factory — Syndicate a blog post
python main.py --syndicate path/to/draft.txt

# Digital Estate Archivist — Organize a directory
python main.py --organize ~/Downloads

# Toxic Clause Hunter — Contract Review
python main.py --contract-audit path/to/contract.pdf

# Active Defense Swarm
python main.py --deploy-swarm

# Deep Web Research Dossier
python main.py --deep-research "transformer architecture in LLMs"
```

---

## REST API Reference

Start the server with `python main.py`, then access it at `http://127.0.0.1:8000`.

Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

### Endpoints

#### `GET /health`
System health check.

```json
{
  "status": "ready",
  "model_loaded": true,
  "vision_ready": false,
  "memory_entries": 42,
  "tools_available": 12
}
```

#### `POST /chat`
Conversational chat with optional thinking loop.

**Request:**
```json
{
  "message": "Explain the CAP theorem",
  "use_thinking": true
}
```

**Response:**
```json
{
  "answer": "...",
  "confidence": 0.92,
  "iterations": 3,
  "mode": "thinking",
  "tools_used": [],
  "thinking_steps": ["Step 1: ...", "Step 2: ..."],
  "duration_ms": 1240
}
```

#### `POST /chat/stream`
Same as `/chat` but returns a **Server-Sent Events** stream for real-time token display.

#### `POST /agent/task`
Submit a complex task for the full agent pipeline (tool use, thinking, etc.).

**Request:**
```json
{
  "task": "Search the web for the latest Python 3.13 release notes and summarize them",
  "use_thinking": true,
  "max_tool_calls": 5
}
```

#### `GET /memory/stats`
View memory / bug-diary statistics.

#### `GET /memory/failures`
List the last 20 failure records from the bug diary.

#### `GET /sessions`
List active sessions.

#### `GET /sessions/{session_id}/history`
Get conversation transcript for a session.

#### `GET /processes`
List background processes.

#### `GET /processes/{process_id}`
Poll a background process for status.

#### `GET /agent/stats`
Comprehensive agent statistics (tools, sessions, loop detector, skills, etc.).

---

## Domain Experts

Every request is automatically classified into one of these domains. The matching expert injects specialized context, reasoning hints, and a tailored response format:

| Domain | Expert | Specialization |
|---|---|---|
| `code` | 💻 Code & Engineering | Python, JS, Rust, Go, DevOps, architecture |
| `writing` | ✍️ Writing & Communication | Essays, emails, copywriting, storytelling |
| `math` | 🔢 Math & Science | Algebra, calculus, physics, chemistry |
| `business` | 📊 Business & Strategy | Marketing, finance, SWOT, startup advice |
| `creative` | 🎨 Creative & Design | UX/UI, branding, color theory, ideation |
| `education` | 📚 Education & Learning | Tutoring, study plans, Socratic method |
| `health` | 💪 Health & Wellness | Nutrition, fitness, sleep, mental wellness* |
| `legal` | ⚖️ Legal Information | Contracts, IP, tenant rights, regulations* |
| `data` | 📈 Data & Analytics | Statistics, SQL, visualization, BI |
| `lifestyle` | 🌟 Lifestyle & Productivity | Travel, cooking, productivity, finance basics |

*These domains include automatic disclaimers reminding users to consult a licensed professional.

---

## Reasoning Strategies

The agent auto-selects one of four strategies per request:

| Strategy | When used | How it works |
|---|---|---|
| **Chain-of-Thought** | Math, debugging, factual analysis | Sequential numbered steps to the solution |
| **Tree-of-Thought** | Architecture, design decisions | Branches multiple approaches, evaluates each, picks best |
| **Analogy Reasoning** | Explaining new concepts | Maps the unfamiliar to a familiar analogy, then extrapolates |
| **Socratic Method** | Educational contexts | Guides through a sequence of questions to reach understanding |

---

## Communication Personas

The persona engine detects the user's background from language cues and adapts tone automatically:

| Persona | Trigger signals | Style |
|---|---|---|
| 🌱 **Beginner** | "what is", "I don't understand", "explain" | Simple language, analogies, step-by-step |
| 💼 **Professional** | Technical vocabulary, formal phrasing | Concise, direct, actionable |
| 🎓 **Student** | "homework", "class", "assignment" | Educational, encouraging, Socratic |
| 🎨 **Creative** | "brainstorm", "ideas", "creative" | Inspirational, divergent, open-ended |
| 👔 **Executive** | "ROI", "strategy", "board", "metrics" | Bottom-line, data-driven, bullet-point |

---

## Specialized Agent Profiles

### 🕵️ Threat Hunter
Runs an automated red-team security audit of any source file. Detects:
- SQL / command / XSS injection vectors
- Hardcoded secrets and PII leakage paths
- Race conditions and logic bypasses
- Insecure deserialization and unvalidated input

Outputs a **Security Audit Report** with severity ratings, proof-of-concept exploits, and remediation code.

### 👔 Devil's Advocate (Board Meeting)
Simulates a virtual Board of Directors reviewing a business plan. Surfaces:
- Hidden financial risks
- Market assumptions that may be flawed
- Competitor threats
- Regulatory exposure

Produces a structured **Risk Matrix**.

### 🎓 Socratic Auto-Tutor
Conducts an interactive tutoring session on any topic using the Socratic method — asking guiding questions instead of simply providing answers, building genuine understanding.

### ⚖️ Toxic Clause Hunter
Parses PDF or DOCX legal contracts and identifies:
- Unilateral modification clauses
- Liability caps and indemnification traps
- Non-compete / non-solicitation overreach
- Auto-renewal and cancellation penalty terms

### 🌙 Night Watch Daemon
Background daemon that runs nightly system and security audits, monitoring for anomalies and generating reports.

### 📁 Digital Estate Archivist
Scans a messy directory, uses the LLM to understand file contents and context, then reorganizes and renames files into a logical structure.

### 🧬 Code Evolution Engine
1. Generates multiple candidate implementations of a given function/algorithm
2. Sandboxes and benchmarks each for speed and memory
3. Uses RLHF-style genetic selection to return the optimal solution

### 🔄 Reverse-Engineering Transpiler
Reads an entire directory of legacy source code (any language), understands the logic and structure, and rewrites it in the target language while preserving behavior.

### 🌐 Deep Web Intelligence Bot
Uses DuckDuckGo search and BeautifulSoup scraping to compile a comprehensive research dossier on any topic from live web sources.

### 🛡️ Active Defense Swarm
Deploys automated honeypots and tarpits to detect, slow down, and log potential intrusion attempts.

---

## Built-in Tools

The agent can invoke these tools autonomously when solving tasks:

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo web search |
| `code_executor` | Sandboxed Python code execution |
| `file_ops` | Read, write, list files |
| `calculator` | Safe math expression evaluator |
| `data_analyzer` | Statistical analysis of datasets |
| `doc_reader` | Extract text from PDF and DOCX files |
| `image_analyzer` | Vision analysis of images |
| `knowledge` | Query the internal knowledge base |
| `writer` | Structured document generation |
| `task_planner` | Break complex goals into sub-tasks |
| `web_tester` | Playwright-based browser automation |
| `session_tools` | Spawn and communicate with sub-agent sessions |

### Tool Policy Profiles

Tool access is controlled by a profile assigned per agent:

| Profile | Tools allowed |
|---|---|
| `minimal` | `calculator`, `knowledge` |
| `coding` | + `code_executor`, `file_ops` |
| `assistant` | + `web_search`, `writer`, `task_planner` (default) |
| `full` | All tools including `web_tester`, `image_analyzer` |

---

## Safety & Ethics Layer

Every request and response passes through a three-stage safety system:

1. **Content Filter** — Blocks harmful, violent, or policy-violating input/output
2. **Ethics Engine** — Evaluates actions against an ethics policy before tool execution
3. **PII Guard** — Automatically redacts personally identifiable information (emails, phone numbers, SSNs, etc.) from all responses

Blocked requests receive a polite refusal message and are logged with an audit trail in the session store.

---

## Configuration

All settings live in `backend/config/settings.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `auto` | Active provider |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `CLAUDE_API_KEY` | — | Anthropic Claude API key |
| `OPENAI_API_KEY` | — | OpenAI ChatGPT API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model name |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `LLM_API_HOST` | `127.0.0.1` | API server bind address |
| `LLM_API_PORT` | `8000` | API server port |
| `LLM_CORS_ORIGINS` | `localhost:3000,…` | Allowed CORS origins |
| `LLM_BASE_DIR` | `C:\llm` | Base data directory |

---

## Running Tests

```bash
cd backend
pytest tests/
```
