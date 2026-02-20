"""
Code Executor Tool — Run Python code in a sandboxed subprocess.
Hardened: AST-validated eval, code blocklist, length limits.
"""

import ast
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from agents.tools.registry import registry, RiskLevel
from config.settings import agent_config

logger = logging.getLogger(__name__)

# ── Security constants ──
MAX_CODE_LENGTH = 10_000  # chars
MAX_EXPRESSION_LENGTH = 1_000

DANGEROUS_PATTERNS = [
    # Original blocklist
    "import os", "import sys", "import subprocess", "import shutil",
    "__import__", "eval(", "exec(", "compile(",
    "open(", "os.system", "os.popen", "os.remove", "os.unlink",
    "shutil.rmtree", "subprocess.", "ctypes.",
    "importlib.", "builtins", "globals(", "locals(",
    "getattr(", "setattr(", "delattr(",
    # Network attack & exploitation tools
    "import scapy", "from scapy", "import nmap", "from nmap",
    "import paramiko", "from paramiko",
    "import socket", "from socket",
    "import requests", "from requests",
    # Data exfiltration patterns
    "import smtplib", "from smtplib",
    "import ftplib", "from ftplib",
    "import telnetlib", "from telnetlib",
    # Keylogging / screen capture
    "import pynput", "from pynput",
    "import keyboard", "from keyboard",
    "import pyautogui", "from pyautogui",
    # Credential theft
    "import win32crypt", "from win32crypt",
    "import sqlite3",  # often used to steal browser data
    "import winreg", "from winreg",
]

# AST node types allowed in safe expression evaluation
_SAFE_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.Call, ast.Constant, ast.Num, ast.Str,
    ast.Name, ast.Load, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Subscript, ast.Index, ast.Slice,
    ast.Attribute,
)

# Names allowed in safe expression context
_SAFE_NAMES = frozenset({
    "abs", "len", "max", "min", "sum", "round", "int", "float",
    "str", "bool", "list", "dict", "tuple", "set", "range",
    "sorted", "reversed", "enumerate", "zip", "map", "filter",
    "True", "False", "None",
    "math", "pi", "e", "sqrt", "log", "sin", "cos", "tan",
    "ceil", "floor", "pow",
})


def _validate_expression_ast(expression: str) -> bool:
    """Validate that an expression only contains safe AST nodes."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            return False
        # Block dangerous attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False
        # Block dangerous names
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                return False
    return True


def _check_code_safety(code: str) -> str | None:
    """Check code for dangerous patterns. Returns error message or None."""
    if len(code) > MAX_CODE_LENGTH:
        return f"Code too long ({len(code)} chars, max {MAX_CODE_LENGTH})"

    code_lower = code.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in code_lower:
            return f"Blocked dangerous pattern: {pattern}"

    return None


@registry.register(
    name="execute_python",
    description="Execute Python code in a sandboxed subprocess. Returns stdout, stderr, and exit code.",
    risk_level=RiskLevel.HIGH,
    parameters={"code": "Python code to execute", "timeout": "Timeout in seconds (default 30)"},
)
def execute_python(code: str, timeout: int = None) -> dict:
    """Execute Python code safely in a subprocess with safety checks."""
    # ── Safety checks ──
    safety_error = _check_code_safety(code)
    if safety_error:
        return {"stdout": "", "stderr": safety_error, "exit_code": -1, "success": False}

    timeout = timeout or agent_config.sandbox_timeout

    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        return {
            "stdout": result.stdout[:50_000],  # Cap output size
            "stderr": result.stderr[:10_000],
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "exit_code": -1,
            "success": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {type(e).__name__}",
            "exit_code": -1,
            "success": False,
        }
    finally:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass


@registry.register(
    name="evaluate_expression",
    description="Safely evaluate a Python math expression and return the result.",
    risk_level=RiskLevel.LOW,
    parameters={"expression": "Python math expression to evaluate"},
)
def evaluate_expression(expression: str) -> dict:
    """Evaluate a simple Python expression with AST validation."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"success": False, "result": None, "error": "Expression too long"}

    # ── AST validation — block everything except safe nodes ──
    if not _validate_expression_ast(expression):
        return {
            "success": False,
            "result": None,
            "error": "Expression contains unsafe operations",
        }

    # Restricted builtins — no access to __import__, open, etc.
    import math
    safe_builtins = {
        "abs": abs, "len": len, "max": max, "min": min,
        "sum": sum, "round": round, "int": int, "float": float,
        "str": str, "bool": bool, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "range": range,
        "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter,
        "True": True, "False": False, "None": None,
        "math": math, "pi": math.pi, "e": math.e,
        "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "ceil": math.ceil, "floor": math.floor,
        "pow": pow,
    }

    try:
        # Compile with restricted builtins — NO __builtins__ access
        code = compile(expression, "<expression>", "eval")
        result = eval(code, {"__builtins__": {}}, safe_builtins)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "result": None, "error": f"{type(e).__name__}: {e}"}
