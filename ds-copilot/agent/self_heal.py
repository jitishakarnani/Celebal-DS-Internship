import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.sandbox import CodeSandbox
from rag.retriever import KnowledgeRetriever

load_dotenv()

MODEL_NAME = "gemini-flash-latest"

SELF_HEAL_PROMPT = """You are a Python/Pandas debugging expert. The following code was
generated to answer a user's question about a DataFrame `df`, but it failed when executed.

User's question: {question}

Code that failed:
```python
{code}
```

Error produced:
{error}

Relevant debugging knowledge:
{hints}

Fix the code so it correctly answers the user's question and runs without error.
Rules:
- Assume `df` is already loaded (do not re-load or redefine it).
- Do not include import statements or plt.savefig() — these are handled automatically.
- Output ONLY the corrected Python code, no explanation, no markdown code fences.
"""


class SelfHealer:
    """
    Takes a failed piece of sandbox-generated code + its error, retrieves
    relevant pandas/debugging knowledge via RAG, and asks the LLM to produce
    a corrected version. Can optionally drive a full retry loop against a
    CodeSandbox instance.
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. Make sure it's set in your .env file "
                "and that .env is loaded (python-dotenv)."
            )
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, google_api_key=api_key, temperature=0.2
        )
        self.retriever = KnowledgeRetriever()

    @staticmethod
    def _content_to_text(content) -> str:
        """Normalizes LLM response content to a plain string, since some
        Gemini responses return a list of parts instead of a plain string."""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return content

    @staticmethod
    def _extract_code(text) -> str:
        """Strips markdown code fences if the LLM added them anyway."""
        text = SelfHealer._content_to_text(text)
        text = text.strip()
        text = re.sub(r"^```python\s*|^```\s*|```$", "", text, flags=re.MULTILINE)
        return text.strip()

    def generate_fix(self, question: str, failed_code: str, error_message: str, k: int = 2) -> str:
        """Returns corrected code as a string (not executed here)."""
        hints = self.retriever.retrieve(error_message, k=k)
        hint_text = "\n\n".join(hints) if hints else "No specific hints found in knowledge base."

        prompt = SELF_HEAL_PROMPT.format(
            question=question, code=failed_code, error=error_message, hints=hint_text
        )
        response = self.llm.invoke(prompt)
        return self._extract_code(response.content)

    def heal_and_retry(self, sandbox: CodeSandbox, question: str, code: str,
                        error_message: str, max_attempts: int = 2) -> dict:
        """
        Repeatedly generates a fix and re-runs it in the sandbox, up to
        max_attempts times. Returns the sandbox result dict, plus:
          - 'final_code': the code that ultimately ran (fixed or original)
          - 'attempts': number of self-heal attempts made
          - 'healed': True if a fix was needed and succeeded
        """
        current_code = code
        current_error = error_message

        for attempt in range(1, max_attempts + 1):
            fixed_code = self.generate_fix(question, current_code, current_error)
            result = sandbox.run(fixed_code)

            if result["success"]:
                result["final_code"] = fixed_code
                result["attempts"] = attempt
                result["healed"] = True
                return result

            current_code = fixed_code
            current_error = result["stderr"]

        # All attempts exhausted — return the last (failed) result
        result["final_code"] = current_code
        result["attempts"] = max_attempts
        result["healed"] = False
        return result


if __name__ == "__main__":
    data_path = r"C:\Users\Janvi\Celebal-DS-Internship\ds-copilot\sample_sales.csv"

    sandbox = CodeSandbox(data_path=data_path)
    healer = SelfHealer()

    broken_code = """
region_revenue = df.groupby('regionn')['revenue'].sum()  # typo: 'regionn'
region_revenue.plot(kind='bar', title='Revenue by Region')
"""
    result = sandbox.run(broken_code)
    print("Initial run success:", result["success"])
    print("Error:", result["stderr"][:200])

    if not result["success"]:
        healed = healer.heal_and_retry(
            sandbox, question="Show revenue by region",
            code=broken_code, error_message=result["stderr"]
        )
        print("\nHealed:", healed["healed"], "| Attempts:", healed["attempts"])
        print("Final code:\n", healed["final_code"])
        print("Success:", healed["success"])