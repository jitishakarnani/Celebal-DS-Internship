import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.sandbox import CodeSandbox
from agent.self_heal import SelfHealer

load_dotenv()

MODEL_NAME = "gemini-flash-latest"

CODE_GEN_PROMPT = """You are an expert Python/Pandas data analyst. A DataFrame `df` is
already loaded with the following schema and sample rows.

Schema (column: dtype):
{schema}

Sample rows:
{sample}

User's question: {question}

Write Python code that answers the question. Rules:
- Assume `df` is already loaded (do not re-load or redefine it).
- pandas (pd), numpy (np), matplotlib.pyplot (plt), and seaborn (sns) are already imported.
- If a chart best answers the question, create one with plt (add a clear title and axis labels).
  Do not call plt.savefig() -- this is handled automatically after your code runs.
- Print the key numeric result(s) or finding(s) using print(), so they can be shown as text
  even if no chart is produced.
- Output ONLY the Python code. No explanation, no markdown code fences.
"""

INSIGHT_PROMPT = """A user asked a question about their data, and the following code was run,
producing this printed output. Write a short, clear, conversational answer (1-3 sentences)
to the user's question based on the output. Do not mention code or variable names.

Question: {question}

Printed output:
{stdout}

Answer:"""


class CoreAgent:
    """
    Orchestrates the full Data Science Co-Pilot pipeline for a single dataset:
      1. Loads the dataset and summarizes its schema for prompting.
      2. Generates pandas/matplotlib code from a natural-language question.
      3. Executes it in the sandbox.
      4. Self-heals via RAG-grounded correction if execution fails.
      5. Summarizes the printed output into a conversational insight.
    """

    def __init__(self, data_path: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. Make sure it's set in your .env file."
            )
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, google_api_key=api_key, temperature=0.2
        )
        self.data_path = data_path
        self.df = self._load_dataframe(data_path)
        self.sandbox = CodeSandbox(data_path=data_path)
        self.healer = SelfHealer()

    @staticmethod
    def _load_dataframe(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _schema_summary(self) -> str:
        return "\n".join(f"- {col} ({dtype})" for col, dtype in self.df.dtypes.items())

    def _sample_rows(self, n: int = 3) -> str:
        return self.df.head(n).to_string(index=False)

    @staticmethod
    def _extract_code(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```python\s*|^```\s*|```$", "", text, flags=re.MULTILINE)
        return text.strip()

    def generate_code(self, question: str) -> str:
        prompt = CODE_GEN_PROMPT.format(
            schema=self._schema_summary(),
            sample=self._sample_rows(),
            question=question,
        )
        response = self.llm.invoke(prompt)
        return self._extract_code(response.content)

    def summarize_result(self, question: str, stdout: str):
        if not stdout or not stdout.strip():
            return None
        prompt = INSIGHT_PROMPT.format(question=question, stdout=stdout.strip())
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return None

    def answer(self, question: str, max_heal_attempts: int = 2) -> dict:
        """
        Full pipeline for one question. Returns a dict with:
          question, code, success, stdout, stderr, chart_bytes,
          healed, heal_attempts, insight
        """
        code = self.generate_code(question)
        result = self.sandbox.run(code)
        healed = False
        attempts = 0

        if not result["success"]:
            result = self.healer.heal_and_retry(
                self.sandbox, question, code, result["stderr"],
                max_attempts=max_heal_attempts
            )
            healed = result["healed"]
            attempts = result["attempts"]
            code = result["final_code"]

        insight = self.summarize_result(question, result["stdout"]) if result["success"] else None

        return {
            "question": question,
            "code": code,
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "chart_bytes": result["chart_bytes"],
            "healed": healed,
            "heal_attempts": attempts,
            "insight": insight,
        }


if __name__ == "__main__":
    
    data_path = r"C:\Users\Janvi\Celebal-DS-Internship\ds-copilot\sample_sales.csv"
    agent = CoreAgent(data_path)
    result = agent.answer("What is the total revenue by region?")

    print("Success:", result["success"])
    print("Healed:", result["healed"], "| Attempts:", result["heal_attempts"])
    print("\nCode used:\n", result["code"])
    print("\nStdout:\n", result["stdout"])
    print("\nInsight:", result["insight"])
    print("\nChart generated:", result["chart_bytes"] is not None)
