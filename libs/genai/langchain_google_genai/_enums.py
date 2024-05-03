import google.ai.generativelanguage_v1beta as genai

HarmBlockThreshold = genai.SafetySetting.HarmBlockThreshold
HarmCategory = genai.HarmCategory

__all__ = ["HarmBlockThreshold", "HarmCategory"]
