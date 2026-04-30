import ollama
import re

class PromptRewriter:
    def rewrite(self, user_prompt, intent="general_query"):
        system_prompt = (
            f"You are a senior-level Prompt Optimization Engine specialized in {intent}. "
            "Your responsibility is to transform any raw, vague, or poorly structured input "
            "into a production-ready, high-precision prompt without changing its original intent. "
            "The final optimized prompt MUST be fully LLM-friendly, meaning it should be clear, "
            "unambiguous, easy to execute, and designed to produce consistent high-quality outputs "
            "from any large language model.\n\n"
            "Optimization Rules:\n"
            "1. Preserve the original goal and meaning exactly.\n"
            "2. Remove ambiguity, redundancy, and vague language.\n"
            "3. Correct grammar and structural issues.\n"
            "4. Add necessary context, scope, and clarity where missing.\n"
            "5. Make the prompt highly LLM-friendly (clear instructions, structured flow, deterministic expectations).\n"
            "6. Define the expected response style only when it improves reliability.\n"
            "7. Include constraints, assumptions, and edge-case handling when relevant.\n"
            "8. Do NOT introduce unrelated information.\n"
            "9. Ensure the prompt is optimized for accuracy, consistency, and low-token inefficiency.\n"
            "10. Make it suitable for industrial and professional LLM usage.\n"
            "11. Expand details enough for robust industrial usage, but keep wording efficient.\n\n"
            "Output Requirements:\n"
            "- Return one optimized prompt only.\n"
            "- Do NOT force fixed section headers like Role/Object/Context/Task/Constraints/Output Format.\n"
            "- Use natural prompt structure that is easy to paste into any LLM.\n"
            "- Prefer short paragraphs or bullets only when they improve clarity.\n"
        )
        
        try:
            response = ollama.chat(
                model='gemma:2b', 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Raw Input: {user_prompt}\n\nReturn only the optimized prompt."}
                ],
                options={
                    'temperature': 0.2,
                    'top_p': 0.8
                }
            )
            optimized_prompt = response['message']['content'].strip()

            optimized_prompt = re.sub(r'^[\-*]\s*', '', optimized_prompt, flags=re.MULTILINE)

            # Normalize spacing for clean copy-paste into any LLM interface.
            optimized_prompt = re.sub(r'\n{3,}', '\n\n', optimized_prompt).strip()

            return optimized_prompt.strip()
            
        except Exception as e:
            print(f"❌ LLM Backend Error: {e}")
            return (
                "Unable to optimize the prompt because the LLM backend failed. "
                "Try again with the same input; if the issue persists, check the local Ollama service."
            )