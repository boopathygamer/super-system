"""
The Shadow Matrix
─────────────────
An isolated hyper-execution environment used exclusively by the 
Synthesized Consciousness Engine to benchmark self-mutated brain code 
against regression tests before committing them to the master branch.
"""

import logging
from typing import List
from agents.tools.code_executor import CodeExecutor

logger = logging.getLogger(__name__)

class ShadowMatrix:
    def __init__(self):
        self.executor = CodeExecutor()

    def run_gauntlet(self, mutated_code: str, target_file_name: str, regression_tests: List[str]) -> tuple[bool, str]:
        """
        Runs the mutated codebase against past failure tests in memory.
        Returns (passed_all, matrix_report).
        """
        logger.info(f"🕶️ Entering The Shadow Matrix: Benchmarking mutated `{target_file_name}`...")

        if not regression_tests:
            logger.warning("No regression tests mapped for this mutation. Blind acceptance is too dangerous.")
            return False, "Aborted: Zero regression tests available."

        successful_tests = 0
        total_tests = len(regression_tests)
        report = ""

        # Run each regression test
        for i, test in enumerate(regression_tests):
            logger.info(f"   -> Shadow Matrix Test {i+1}/{total_tests}...")
            
            # Combine the mutated backend code with the test block
            matrix_simulation = f"{mutated_code}\n\n# --- REGRESSION TEST ---\n{test}\n"
            
            try:
                result = self.executor.execute(matrix_simulation, timeout=10)
                
                if result.error:
                    logger.warning(f"   ❌ Matrix Test {i+1} Failed: {result.error}")
                    report += f"Test {i+1} Failed: {result.error}\n"
                else:
                    logger.info(f"   ✅ Matrix Test {i+1} Passed.")
                    successful_tests += 1
                    report += f"Test {i+1} Passed.\n"
                    
            except Exception as e:
                return False, f"Matrix catastrophic crash: {e}"

        win_rate = successful_tests / total_tests
        passed = win_rate == 1.0  # Must be flawless to overwrite production code

        if passed:
            logger.info("🟢 Shadow Matrix Protocol Completed: Flawless Victory.")
            return True, "All tests passed cleanly in isolated benchmarking."
        else:
            logger.warning(f"🔴 Shadow Matrix Protocol Failed. Mutated code broke regression constraints (Win Rate: {win_rate:.0%})")
            return False, report
