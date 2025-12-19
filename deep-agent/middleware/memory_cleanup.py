"""
Memory Cleanup Middleware - LLM-based memory trimming.

Keeps only the best N memories per .txt file using LLM selection.
"""

from datetime import datetime
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI


# =============================================================================
# TRIM PROMPT
# =============================================================================

TRIM_SYSTEM_PROMPT = """Trim this memory file to keep only the {max_memories} MOST VALUABLE bullet points.

File: {file_key}

Current content:
{current_content}

Rules:
- Each bullet point (line starting with "-") is 1 memory
- Keep the {max_memories} most specific, actionable, and diverse memories
- Maintain the markdown format (## headers, - bullet points)
- Remove duplicates and vague items
- Return ONLY the trimmed content, no code blocks or explanations
"""


class MemoryCleanupMiddleware(AgentMiddleware):
    """LLM-based memory trimmer that keeps only the best N memories per .txt file."""

    def __init__(self, store_instance, max_memories_per_file: int = 30, cleanup_model: str = "gpt-4o-mini"):
        self.max_memories = max_memories_per_file
        self.store = store_instance
        self.llm = ChatOpenAI(model=cleanup_model, temperature=0)

    def after_agent(self, state, runtime):
        """Trim all .txt memory files after each agent run."""
        try:
            # Find all .txt files in /memories/
            all_items = list(self.store.search(("filesystem",)))
            txt_files = [item for item in all_items if item.key.startswith("/memories/") and item.key.endswith(".txt")]

            for txt_file in txt_files:
                self._trim_file(txt_file)
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")

        return None

    def _trim_file(self, file_item):
        """Trim a single .txt file using LLM."""
        try:
            # Get current content
            content_lines = file_item.value.get("content", [])
            current_content = "\n".join(content_lines) if isinstance(content_lines, list) else str(content_lines)

            # Count memories (each bullet point = 1 memory)
            memory_count = current_content.count("\n- ")

            # Skip if already small enough
            if memory_count <= self.max_memories:
                return

            # Format the prompt with runtime values
            prompt = TRIM_SYSTEM_PROMPT.format(
                max_memories=self.max_memories,
                file_key=file_item.key,
                current_content=current_content
            )

            response = self.llm.invoke(prompt)
            trimmed = response.content.strip()

            # Remove markdown code blocks if present (we do not want this)
            if "```" in trimmed:
                trimmed = trimmed.replace("```markdown", "").replace("```", "").strip()

            # Save trimmed version
            self.store.put(
                ("filesystem",),
                file_item.key,
                {
                    "content": trimmed.split("\n"),
                    "created_at": file_item.value.get("created_at", datetime.now().isoformat()),
                    "modified_at": datetime.now().isoformat(),
                }
            )

            print(f"🧹 Trimmed {file_item.key}: {memory_count} → {self.max_memories} memories")

        except Exception as e:
            print(f"⚠️ Failed to trim {file_item.key}: {e}")
