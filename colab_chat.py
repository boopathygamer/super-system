import gradio as gr
from core.model_providers import ProviderRegistry
from agents.controller import AgentController

# 1. Initialize the AI System (Auto-detects Mistral if no keys are set)
registry = ProviderRegistry.auto_detect(preferred="auto")
generate_fn = registry.generate_fn()
agent = AgentController(generate_fn=generate_fn)

def process_chat(message, history):
    """
    Gradio requires a specific function signature for chat interfaces.
    'message' is the new user input.
    'history' is a list of [user_message, bot_message] pairs.
    """
    if not message.strip():
        return ""
        
    try:
        # Ask the Universal AI Agent (ignoring thinking loop for simple UI)
        result = agent.process(user_input=message, use_thinking_loop=False)
        
        # Append some helpful metrics to the response
        active = registry.active
        model_name = f"{active.name}/{active.model}" if active else "local/Mistral-7B"
        footer = f"\n\n*(Provider: {model_name} | Latency: {result.duration_ms:.0f}ms | Confidence: {result.confidence:.2f})*"
        
        return result.answer + footer
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 2. Build the Gradio Chat Interface
demo = gr.ChatInterface(
    fn=process_chat,
    title="Universal AI Agent 🤖",
    description="Running inside Google Colab. Chat with your models directly from this UI!",
    theme="soft"
)

# 3. Launch the UI (This will render directly in the Colab output block)
if __name__ == "__main__":
    print("Launching Web UI...")
    demo.launch(share=True, debug=True)
