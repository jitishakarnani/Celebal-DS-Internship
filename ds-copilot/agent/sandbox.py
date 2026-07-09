import subprocess
import sys
import tempfile
import os


class CodeSandbox:
    """
    Executes LLM-generated Python code safely in an isolated subprocess.
    Captures stdout, stderr, and any generated chart image.
    """

    def __init__(self, data_path: str, timeout: int = 20):
        self.data_path = data_path
        self.timeout = timeout

    def run(self, code: str) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generated_code.py")
            output_chart_path = os.path.join(tmpdir, "output_chart.png")

            ext = os.path.splitext(self.data_path)[1].lower()
            if ext == ".csv":
                load_line = f'df = pd.read_csv(r"{self.data_path}")'
            elif ext in [".xlsx", ".xls"]:
                load_line = f'df = pd.read_excel(r"{self.data_path}")'
            elif ext == ".json":
                load_line = f'df = pd.read_json(r"{self.data_path}")'
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            wrapped_code = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

{load_line}

{code}

plt.savefig(r"{output_chart_path}", bbox_inches='tight')
"""
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapped_code)

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
                success = result.returncode == 0

                chart_bytes = None
                if success and os.path.exists(output_chart_path):
                    with open(output_chart_path, "rb") as img:
                        chart_bytes = img.read()

                return {
                    "success": success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "chart_bytes": chart_bytes,
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Execution timed out (possible infinite loop).",
                    "chart_bytes": None,
                }