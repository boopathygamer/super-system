"""
FastAPI Server — Main API for the Custom LLM System.
─────────────────────────────────────────────────────
Endpoints:
  POST /chat           — Conversational chat with memory
  POST /vision/analyze — Upload image + question → expert analysis
  POST /agent/task     — Submit complex task for agent
  GET  /memory/stats   — View bug diary statistics
  GET  /health         — System health check
"""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

# ── Security constants ──
_MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
_ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
_ALLOWED_ORIGINS = os.getenv(
    "LLM_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000",
).split(",")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    api_config, model_config, UPLOADS_DIR,
)
from api.models import (
    ChatRequest, ChatResponse,
    VisionRequest, VisionResponse,
    AgentTaskRequest, AgentTaskResponse,
    MemoryStatsResponse, HealthResponse,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Custom LLM System",
    description=(
        "Mistral 7B Enhanced with Vision, Self-Thinking Brain & Agents. "
        "Features: multimodal image analysis, self-improving from mistakes, "
        "multi-hypothesis reasoning, and professional agent assistants."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ──────────────────────────────────────────────
# Global State (initialized on startup)
# ──────────────────────────────────────────────

class AppState:
    model = None
    tokenizer = None
    engine = None
    vision_pipeline = None
    agent_controller = None
    is_ready = False

state = AppState()


@app.on_event("startup")
async def startup():
    """Initialize all components on server startup."""
    logger.info("=" * 60)
    logger.info("Starting Custom LLM System...")
    logger.info("=" * 60)

    try:
        # 1. Load model
        from core.model_loader import load_model
        from core.tokenizer import MistralTokenizer
        from core.inference import InferenceEngine

        logger.info("Loading Mistral 7B...")
        state.model = load_model()
        state.tokenizer = MistralTokenizer()
        state.engine = InferenceEngine(state.model, state.tokenizer)
        logger.info("✅ Model loaded")

        # 2. Initialize vision pipeline
        try:
            from vision.pipeline import VisionPipeline
            state.vision_pipeline = VisionPipeline(
                state.model, state.tokenizer, state.engine
            )
            # Inject into agent tool
            from agents.tools.image_analyzer import set_vision_pipeline
            set_vision_pipeline(state.vision_pipeline)
            logger.info("✅ Vision pipeline ready")
        except Exception as e:
            logger.warning(f"⚠️ Vision pipeline failed to initialize: {e}")

        # 3. Initialize agent
        from agents.controller import AgentController
        generate_fn = lambda prompt: state.engine.generate(prompt)
        state.agent_controller = AgentController(generate_fn=generate_fn)
        logger.info("✅ Agent controller ready")

        state.is_ready = True
        logger.info("=" * 60)
        logger.info("🚀 System ready! All components initialized.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    return HealthResponse(
        status="ready" if state.is_ready else "loading",
        model_loaded=state.model is not None,
        vision_ready=state.vision_pipeline is not None,
        memory_entries=(
            len(state.agent_controller.memory.failures)
            if state.agent_controller else 0
        ),
        tools_available=(
            len(state.agent_controller.tools.list_tools())
            if state.agent_controller else 0
        ),
    )


# ──────────────────────────────────────────────
# Chat Endpoint
# ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational chat with memory and self-thinking."""
    if not state.is_ready:
        raise HTTPException(503, "System not ready")

    start = time.time()

    try:
        if request.use_thinking:
            result = state.agent_controller.process(
                user_input=request.message,
                use_thinking_loop=True,
            )
            return ChatResponse(
                answer=result.answer,
                confidence=result.confidence,
                iterations=result.iterations,
                mode=result.mode,
                tools_used=[t.get("tool", "") for t in result.tools_used],
                thinking_steps=[
                    f"Step {s.iteration}: {s.action_taken} (conf={s.verification.confidence:.3f})"
                    for s in (result.thinking_trace.steps if result.thinking_trace else [])
                ],
                duration_ms=(time.time() - start) * 1000,
            )
        else:
            answer = state.agent_controller.chat(request.message)
            return ChatResponse(
                answer=answer,
                confidence=0.8,
                duration_ms=(time.time() - start) * 1000,
            )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, "Generation failed. Please try again.")


# ──────────────────────────────────────────────
# Vision Endpoint
# ──────────────────────────────────────────────

@app.post("/vision/analyze", response_model=VisionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form("Describe this image in detail."),
    mode: str = Form("general"),
    chain_of_thought: bool = Form(True),
):
    """Upload and analyze an image with expert-level detail."""
    if not state.is_ready:
        raise HTTPException(503, "System not ready")
    if state.vision_pipeline is None:
        raise HTTPException(503, "Vision pipeline not available")

    start = time.time()

    # ── Validate file extension ──
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in _ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported image format. Allowed: {', '.join(sorted(_ALLOWED_IMAGE_EXTENSIONS))}",
        )

    # ── Validate file size ──
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_SIZE:
        raise HTTPException(
            400,
            f"File too large ({len(contents) // 1024 // 1024}MB). Max: {_MAX_UPLOAD_SIZE // 1024 // 1024}MB",
        )

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOADS_DIR / f"{file_id}{file_ext}"

    try:
        with open(save_path, "wb") as f:
            f.write(contents)

        # Analyze
        analysis = state.vision_pipeline.analyze(
            image=str(save_path),
            question=question,
            mode=mode,
            chain_of_thought=chain_of_thought,
        )

        return VisionResponse(
            analysis=analysis,
            mode=mode,
            confidence=0.8,
            duration_ms=(time.time() - start) * 1000,
        )

    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Vision error: {e}", exc_info=True)
        raise HTTPException(500, "Image analysis failed. Please try again.")
    finally:
        # Clean up uploaded file on error
        if save_path.exists() and 'analysis' not in dir():
            try:
                save_path.unlink()
            except OSError:
                pass


# ──────────────────────────────────────────────
# Agent Endpoint
# ──────────────────────────────────────────────

@app.post("/agent/task", response_model=AgentTaskResponse)
async def agent_task(request: AgentTaskRequest):
    """Submit a complex task for the agent to solve."""
    if not state.is_ready:
        raise HTTPException(503, "System not ready")

    start = time.time()

    try:
        result = state.agent_controller.process(
            user_input=request.task,
            use_thinking_loop=request.use_thinking,
            max_tool_calls=request.max_tool_calls,
        )

        thinking_dict = None
        if result.thinking_trace:
            thinking_dict = {
                "iterations": result.thinking_trace.iterations,
                "final_confidence": result.thinking_trace.final_confidence,
                "mode": result.thinking_trace.mode.value,
                "steps": [
                    {
                        "iteration": s.iteration,
                        "action": s.action_taken,
                        "confidence": s.verification.confidence if s.verification else 0,
                    }
                    for s in result.thinking_trace.steps
                ],
            }

        return AgentTaskResponse(
            answer=result.answer,
            confidence=result.confidence,
            iterations=result.iterations,
            mode=result.mode,
            tools_used=result.tools_used,
            thinking_trace=thinking_dict,
            duration_ms=(time.time() - start) * 1000,
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(500, "Agent task failed. Please try again.")


# ──────────────────────────────────────────────
# Memory Endpoint
# ──────────────────────────────────────────────

@app.get("/memory/stats", response_model=MemoryStatsResponse)
async def memory_stats():
    """View bug diary and memory statistics."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    stats = state.agent_controller.memory.get_stats()
    return MemoryStatsResponse(**stats)


@app.get("/memory/failures")
async def list_failures():
    """List all failure records from the bug diary."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    from dataclasses import asdict
    failures = state.agent_controller.memory.failures
    return {
        "count": len(failures),
        "failures": [asdict(f) for f in failures[-20:]],  # Last 20
    }


# ──────────────────────────────────────────────
# Streaming Endpoint (SSE)
# ──────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming chat.
    Returns text chunks, tool events, and thinking steps.
    """
    if not state.is_ready:
        raise HTTPException(503, "System not ready")

    from fastapi.responses import StreamingResponse
    from core.streaming import StreamProcessor, StreamConfig

    processor = StreamProcessor(StreamConfig(
        chunk_size=50,
        break_on="sentence",
    ))

    def event_stream():
        try:
            # Build prompt — chat() returns a string, chunk it for streaming
            answer = state.agent_controller.chat(request.message)

            # Stream answer in sentence-sized chunks (not char-by-char)
            chunk_size = processor.config.chunk_size if hasattr(processor, 'config') else 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                events = processor.process_token(chunk)
                for evt in events:
                    yield evt.to_sse()

            # Final events
            for evt in processor.finish():
                yield evt.to_sse()

        except Exception as e:
            import json
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': 'Streaming failed'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ──────────────────────────────────────────────
# Session Endpoints
# ──────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    return {
        "sessions": state.agent_controller.session_manager.list_sessions(
            active_only=True,
        ),
    }


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str, limit: int = 50):
    """Get session transcript history."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    messages = state.agent_controller.session_manager.get_history(
        session_id=session_id,
        limit=limit,
    )
    return {"session_id": session_id, "messages": messages}


# ──────────────────────────────────────────────
# Process Endpoints
# ──────────────────────────────────────────────

@app.get("/processes")
async def list_processes():
    """List background processes."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    return {"processes": state.agent_controller.list_processes()}


@app.get("/processes/{process_id}")
async def poll_process(process_id: str):
    """Poll a background process for status."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    return state.agent_controller.poll_process(process_id)


# ──────────────────────────────────────────────
# Agent Stats Endpoint
# ──────────────────────────────────────────────

@app.get("/agent/stats")
async def agent_stats():
    """Get comprehensive agent statistics."""
    if not state.agent_controller:
        raise HTTPException(503, "Agent not initialized")

    return state.agent_controller.get_stats()

