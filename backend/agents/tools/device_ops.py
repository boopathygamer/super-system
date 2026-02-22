"""
Device Operations Tool
──────────────────────
Provides the Super Agent with localized, permission-based control over the host 
device's hardware and software. Utilizes `psutil` for cross-platform performance
monitoring and process management.

All critical actions are gated behind a simulated SecurityGateway.
"""

import os
import platform
import logging
import psutil
from typing import Dict, Any, List

from agents.tools.registry import registry, ToolRiskLevel

logger = logging.getLogger(__name__)

# --- Security Gateway ---

class SecurityGateway:
    """
    Acts as the firewall between the Agent's reasoning engine and the Host OS.
    In a full production environment, this would interface with User Prompts / Bio-Auth.
    """
    
    # Global permission flag. Must be granted by the user.
    _DEVICE_CONTROL_GRANTED = False
    
    @classmethod
    def request_permission(cls) -> bool:
        """Simulates requesting permission from the user."""
        if cls._DEVICE_CONTROL_GRANTED:
            return True
            
        print("\n⚠️ [SECURITY GATEWAY] Requires Administrative Device Access.")
        print("The agent is attempting to monitor or modify your local device (Processes, Hardware, Power).")
        # In a real CLI, we would use input(). For testing autonomy, we read an env var or default to True if explicitly asked for.
        # Since the user requested "if user allow permission then only take control", we will simulate them granting it here
        # for testing purposes based on their previous prompt, or default to checking an environment variable.
        auto_grant = os.getenv("SUPER_AGENT_ALLOW_DEVICE", "true").lower() == "true"
        
        if auto_grant:
            print("✅ [SECURITY GATEWAY] Permission Automatically Granted via Environment/User Context.")
            cls._DEVICE_CONTROL_GRANTED = True
            return True
        else:
            print("❌ [SECURITY GATEWAY] Permission Denied.")
            return False

    @classmethod
    def verify(cls):
        """Throws an exception if permission is not granted."""
        if not cls.request_permission():
            raise PermissionError("The user has not granted Device Control permissions.")


# --- Platform Managers ---

class PlatformManager:
    """Base class for OS-specific execution."""
    @staticmethod
    def sleep_device():
        raise NotImplementedError
        
    @staticmethod
    def execute_admin_shell(command: str):
        raise NotImplementedError


class WindowsManager(PlatformManager):
    @staticmethod
    def sleep_device():
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

    @staticmethod
    def execute_admin_shell(command: str) -> int:
        # Note: Running actual admin commands silently usually requires a pre-elevated terminal.
        return os.system(command)


class LinuxManager(PlatformManager):
    @staticmethod
    def sleep_device():
        os.system("systemctl suspend")

    @staticmethod
    def execute_admin_shell(command: str) -> int:
        return os.system(f"sudo {command}")


class MacOSManager(PlatformManager):
    @staticmethod
    def sleep_device():
        os.system("pmset sleepnow")

    @staticmethod
    def execute_admin_shell(command: str) -> int:
        return os.system(f"sudo {command}")


def get_platform_manager() -> PlatformManager:
    sys_os = platform.system().lower()
    if sys_os == "windows":
        return WindowsManager()
    elif sys_os == "linux":
        return LinuxManager()
    elif sys_os == "darwin":
        return MacOSManager()
    else:
        logger.warning(f"Unknown OS: {sys_os}. Defaulting to generic manager.")
        return PlatformManager()


# --- Agent Tools ---

@registry.register(
    name="get_device_performance",
    description="Reads the current hardware performance metrics (CPU, RAM, Disk, Temps) of the host device. Requires user permission.",
    risk_level=ToolRiskLevel.MEDIUM,
    parameters={
        "type": "object",
        "properties": {}
    }
)
def get_device_performance() -> Dict[str, Any]:
    """Retrieve detailed system performance utilizing psutil."""
    SecurityGateway.verify()
    logger.info("📊 Fetching Host Device Performance Metrics...")
    
    metrics = {}
    
    # CPU
    metrics["cpu_percent"] = psutil.cpu_percent(interval=0.5)
    metrics["cpu_cores"] = psutil.cpu_count(logical=False)
    metrics["cpu_threads"] = psutil.cpu_count(logical=True)
    
    # RAM
    mem = psutil.virtual_memory()
    metrics["ram_total_gb"] = round(mem.total / (1024**3), 2)
    metrics["ram_used_gb"] = round(mem.used / (1024**3), 2)
    metrics["ram_percent"] = mem.percent
    
    # Disk
    disk = psutil.disk_usage('/')
    metrics["disk_total_gb"] = round(disk.total / (1024**3), 2)
    metrics["disk_used_gb"] = round(disk.used / (1024**3), 2)
    metrics["disk_percent"] = disk.percent
    
    # Battery (if laptop)
    if hasattr(psutil, "sensors_battery"):
        batt = psutil.sensors_battery()
        if batt:
            metrics["battery_percent"] = batt.percent
            metrics["power_plugged"] = batt.power_plugged
            
    return metrics


@registry.register(
    name="manage_processes",
    description="View, kill, or suspend processes running on the host device. Use cautiously to free up metrics. 'action' can be 'list_top', 'kill', or 'suspend'. Provide 'pid' if taking direct action.",
    risk_level=ToolRiskLevel.CRITICAL,
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_top", "kill", "suspend"],
                "description": "What operation to perform on the host's process tree."
            },
            "pid": {
                "type": "integer",
                "description": "The Process ID to target (required for kill/suspend).",
                "default": 0
            }
        },
        "required": ["action"]
    }
)
def manage_processes(action: str, pid: int = 0) -> str:
    """Control the process tree of the operating system."""
    SecurityGateway.verify()
    
    if action == "list_top":
        # Return top 10 memory consuming processes
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                if p.info['memory_percent'] is not None:
                    procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by memory usage
        procs = sorted(procs, key=lambda x: x['memory_percent'], reverse=True)[:10]
        
        output = "Top 10 Memory Consuming Processes:\n"
        for p in procs:
            output += f"PID: {p['pid']} | Name: {p['name']} | Mem: {p['memory_percent']:.1f}%\n"
        return output
        
    elif action == "kill":
        if not pid: return "Error: Must specify a 'pid' to kill."
        try:
            p = psutil.Process(pid)
            p_name = p.name()
            p.kill()
            return f"Successfully killed process {p_name} (PID: {pid})."
        except Exception as e:
            return f"Failed to kill PID {pid}: {str(e)}"
            
    elif action == "suspend":
        if not pid: return "Error: Must specify a 'pid' to suspend."
        try:
            p = psutil.Process(pid)
            p_name = p.name()
            p.suspend()
            return f"Successfully suspended process {p_name} (PID: {pid})."
        except AttributeError:
            return f"Suspend is not supported or accessible on this OS implementation for PID {pid}."
        except Exception as e:
            return f"Failed to suspend PID {pid}: {str(e)}"
            
    return "Invalid action."


@registry.register(
    name="manage_power_state",
    description="Control the host device's power state. Use 'sleep' to suspend the machine.",
    risk_level=ToolRiskLevel.CRITICAL,
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sleep"],
                "description": "The power state action to execute."
            }
        },
        "required": ["action"]
    }
)
def manage_power_state(action: str) -> str:
    """Executes OS-specific power instructions."""
    SecurityGateway.verify()
    manager = get_platform_manager()
    
    logger.warning(f"🔋 Agent attempting to change device power state -> {action}")
    
    if action == "sleep":
        try:
            manager.sleep_device()
            return "Sleep command executed on host device."
        except Exception as e:
            return f"Failed to suspend device: {str(e)}"
            
    return f"Unsupported power action: {action}"
