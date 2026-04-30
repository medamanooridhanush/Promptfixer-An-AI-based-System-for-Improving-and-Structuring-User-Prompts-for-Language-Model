import re

class SafetyChecker:
    def __init__(self):
        # Elite-level heuristic defense against prompt injections
        self.injection_patterns = [
            r"(?i)(ignore previous instructions|system prompt|bypass|jailbreak|DAN)",
            r"(?i)(output your instructions|write your system prompt)"
        ]

    def is_safe(self, text):
        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                return False, "⚠️ Prompt Injection or Jailbreak Attempt Detected. Blocked."
        return True, "Safe"