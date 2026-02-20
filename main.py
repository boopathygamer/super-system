"""
Custom LLM System — Main Entry Point
─────────────────────────────────────
Universal AI Agent — Multi-Model Provider System

Supports:
  - Gemini (Google)    — set GEMINI_API_KEY
  - Claude (Anthropic) — set CLAUDE_API_KEY
  - ChatGPT (OpenAI)   — set OPENAI_API_KEY

Usage:
    python main.py                          # Start API server (auto-detect provider)
    python main.py --chat                   # Interactive chat (auto-detect)
    python main.py --chat --provider gemini # Chat with Gemini
    python main.py --chat --provider claude # Chat with Claude
    python main.py --chat --provider chatgpt # Chat with ChatGPT
    python main.py --providers              # List available providers
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s │ %(name)-25s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_provider_registry(provider: str = "auto", api_key: str = None):
    """
    Create and configure the provider registry.

    Args:
        provider: Which provider to use (auto/gemini/claude/chatgpt/local)
        api_key: Optional API key (auto-assigned to matching provider)

    Returns:
        Configured ProviderRegistry
    """
    from core.model_providers import ProviderRegistry

    # If user passed --api-key, figure out which provider it's for
    gemini_key = claude_key = openai_key = None
    if api_key:
        if provider == "gemini":
            gemini_key = api_key
        elif provider == "claude":
            claude_key = api_key
        elif provider == "chatgpt":
            openai_key = api_key
        else:
            # Auto-detect: try to guess from key format
            if api_key.startswith("AIza"):
                gemini_key = api_key
            elif api_key.startswith("sk-ant-"):
                claude_key = api_key
            elif api_key.startswith("sk-"):
                openai_key = api_key
            else:
                # Default to gemini if can't guess
                gemini_key = api_key

    registry = ProviderRegistry.auto_detect(
        preferred=provider,
        gemini_key=gemini_key,
        claude_key=claude_key,
        openai_key=openai_key,
    )

    return registry


def show_banner(registry=None):
    """Show the startup banner with provider info."""
    provider_info = ""
    if registry and registry.active:
        p = registry.active
        provider_info = f"   Provider: {p.name.upper()} ({p.model})"
    else:
        provider_info = "   Provider: NONE DETECTED"

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🧠  Universal AI Agent  v2.0                               ║
    ║   ──────────────────────────                                 ║
    ║   Multi-Model • Multi-Domain • Multi-Persona                 ║
    ║                                                              ║
    ║   Models:                                                    ║
    ║     🟢 Gemini (Google)     — GEMINI_API_KEY                  ║
    ║     🟣 Claude (Anthropic)  — CLAUDE_API_KEY                  ║
    ║     🔵 ChatGPT (OpenAI)   — OPENAI_API_KEY                  ║
    ║                                                              ║
    ║   {provider_info:<59}║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def start_server(provider: str = "auto", api_key: str = None):
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import api_config

    registry = create_provider_registry(provider, api_key)
    show_banner(registry)

    # Store registry for the API server to pick up
    import builtins
    builtins._llm_registry = registry

    uvicorn.run(
        "api.server:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        workers=api_config.workers,
        log_level="info",
    )


def interactive_chat(provider: str = "auto", api_key: str = None):
    """Run interactive chat in the terminal with any model provider."""
    from agents.controller import AgentController

    registry = create_provider_registry(provider, api_key)
    show_banner(registry)

    print(registry.status_display())

    # Get the generate function from the registry
    generate_fn = registry.generate_fn()

    print("\n🤖 Initializing agent...")
    agent = AgentController(generate_fn=generate_fn)

    active = registry.active
    model_name = f"{active.name}/{active.model}" if active else "none"

    print(f"\n{'═' * 60}")
    print(f"  💬 Interactive Chat — Model: {model_name}")
    print(f"  Commands:")
    print(f"    /think     — Force thinking loop")
    print(f"    /stats     — Show memory stats")
    print(f"    /reset     — Clear conversation")
    print(f"    /provider  — Show active provider")
    print(f"    /switch X  — Switch to provider X (gemini/claude/chatgpt)")
    print(f"    /models    — List all available providers")
    print(f"{'═' * 60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "/quit"):
                print("\nGoodbye! 👋")
                break

            # ── Special commands ──
            if user_input == "/stats":
                stats = agent.get_stats()
                print(f"\n📊 Stats: {stats}\n")
                continue

            if user_input == "/reset":
                agent.reset_conversation()
                print("🔄 Conversation cleared.\n")
                continue

            if user_input == "/provider":
                if registry.active:
                    p = registry.active
                    stats = p.get_stats()
                    print(f"\n🔌 Active: {p.name} ({p.model})")
                    print(f"   Calls: {stats['calls']} | Errors: {stats['errors']}")
                    if stats['calls'] > 0:
                        print(f"   Avg latency: {stats['avg_latency_ms']:.0f}ms")
                    print()
                continue

            if user_input == "/models":
                print(f"\n{registry.status_display()}\n")
                for info in registry.list_providers():
                    status = "🟢 ACTIVE" if info["active"] else "⚪ ready"
                    print(f"  {status}  {info['name']:<10} {info['model']}")
                print()
                continue

            if user_input.startswith("/switch"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /switch gemini|claude|chatgpt\n")
                    continue
                target = parts[1].lower()
                if registry.set_active(target):
                    # Update the generate function
                    generate_fn = registry.generate_fn()
                    agent._generate = generate_fn
                    p = registry.active
                    print(f"\n🔄 Switched to: {p.name} ({p.model})\n")
                else:
                    print(f"\n❌ Provider '{target}' not available.")
                    print(f"   Available: {[p['name'] for p in registry.list_providers()]}\n")
                continue

            # ── Normal message processing ──
            use_thinking = user_input.startswith("/think")
            if use_thinking:
                user_input = user_input[6:].strip()

            print("\n🧠 Thinking...")
            result = agent.process(
                user_input=user_input,
                use_thinking_loop=use_thinking,
            )

            print(f"\nAssistant: {result.answer}")
            print(
                f"\n  ⎡ confidence={result.confidence:.3f} | "
                f"mode={result.mode} | "
                f"provider={registry.active_name} | "
                f"{result.duration_ms:.0f}ms ⎤\n"
            )

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def list_providers():
    """Show all available model providers."""
    registry = create_provider_registry()
    print(f"\n{registry.status_display()}")
    print()
    for info in registry.list_providers():
        status = "🟢 ACTIVE" if info["active"] else "⚪ ready"
        print(f"  {status}  {info['name']:<10} → {info['model']}")
    print()

    # Show how to configure
    from config.settings import provider_config
    if not provider_config.has_any_api_key:
        print("  💡 No API keys detected! Set environment variables:")
        print("     set GEMINI_API_KEY=your-key    (for Google Gemini)")
        print("     set CLAUDE_API_KEY=your-key    (for Anthropic Claude)")
        print("     set OPENAI_API_KEY=your-key    (for OpenAI ChatGPT)")
        print()
        print("  Or create a .env file in the project root:")
        print("     GEMINI_API_KEY=your-key-here")
        print("     CLAUDE_API_KEY=your-key-here")
        print("     OPENAI_API_KEY=your-key-here")
        print()




def main():
    parser = argparse.ArgumentParser(
        description="Universal AI Agent — Multi-Model Provider System"
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--provider", type=str, default="auto",
        choices=["auto", "gemini", "claude", "chatgpt"],
        help="LLM provider to use (default: auto-detect)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the chosen provider"
    )
    parser.add_argument(
        "--providers", action="store_true",
        help="List all available model providers"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.providers:
        list_providers()
    elif args.chat:
        interactive_chat(provider=args.provider, api_key=args.api_key)
    else:
        start_server(provider=args.provider, api_key=args.api_key)


if __name__ == "__main__":
    main()
