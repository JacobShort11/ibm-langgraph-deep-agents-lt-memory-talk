PROMPT = """Trim this memory file to keep only the {max_memories} MOST VALUABLE bullet points.

File: {file_key}

Current content:
{current_content}

Rules:
- Each bullet point (line starting with "-") is 1 memory
- Keep the {max_memories} most specific, actionable, and diverse memories
- Maintain the markdown format (## headers, - bullet points)
- Remove duplicates and vague items
- Return ONLY the trimmed content, no code blocks or explanations"""