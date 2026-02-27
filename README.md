# 🚀 SuperChain — Universal AI Agent

An advanced self-evolving AI agent system with expert-level reasoning, self-healing code generation, and autonomous learning capabilities.

## Architecture

```
super-agent/
├── backend/
│   ├── agents/           # Agent profiles, tools, and control logic
│   │   ├── profiles/     # Domain-expert agent implementations
│   │   ├── tools/        # Calculator, code executor, web tools
│   │   ├── sandbox/      # Shadow Matrix isolated execution
│   │   └── loop_detector.py  # Anti-loop guardrails
│   ├── brain/            # Core intelligence modules
│   │   ├── thinking_loop.py       # Synthesize → Verify → Learn loop
│   │   ├── memory.py              # Bug Diary & failure tracking
│   │   ├── long_term_memory.py    # Episodic + Procedural + Knowledge Graph
│   │   ├── hypothesis.py          # Multi-hypothesis reasoning
│   │   ├── verifier.py            # Multi-layer verification stack
│   │   ├── metacognition.py       # Confidence gating & self-awareness
│   │   ├── solver/                # Self-healing code solver pipeline
│   │   ├── predictive_engine.py   # Speculative pre-computation
│   │   ├── token_compressor.py    # Token budget optimizer
│   │   ├── async_pipeline.py      # Adaptive concurrency
│   │   ├── confidence_oracle.py   # Bayesian confidence calibration
│   │   ├── cross_pollination.py   # Inter-domain knowledge transfer
│   │   ├── adversarial_tester.py  # Red team autopilot
│   │   ├── cognitive_router.py    # Dynamic model routing
│   │   ├── reasoning_replay.py    # Rewindable thought chains
│   │   ├── zk_proofs.py           # Zero-knowledge execution proofs
│   │   └── temporal_memory.py     # Tiered memory with decay
│   ├── api/              # FastAPI endpoints
│   ├── config/           # Settings and configuration
│   └── tests/            # Comprehensive test suite
```

## Key Features

- **Self-Thinking Loop** — Synthesize → Verify → Learn with continuous self-improvement
- **Multi-Hypothesis Reasoning** — Bayesian-weighted parallel hypothesis exploration
- **Self-Healing Code** — Auto-detects bugs, generates fixes, evolves solutions
- **Expert Tutor** — 8 teaching techniques including gamified learning and flowcharts
- **Long-Term Memory** — Episodic, procedural, and knowledge graph persistence
- **Adversarial Self-Testing** — Red team autopilot for robustness
- **Zero-Knowledge Proofs** — Cryptographic verification of computations
- **Temporal Memory** — Tiered decay with resurrection for optimal recall

## Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

## Security

- No `eval()` — all math uses safe AST-based evaluation
- SHA-256 hashing throughout (no MD5)
- Sandboxed code execution via Shadow Matrix
- Loop detection with circuit breakers

## License

MIT License — see [LICENSE](LICENSE)
