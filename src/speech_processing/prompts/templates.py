from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.base import BasePromptTemplate


class ICBHI2017PromptTemplate(BasePromptTemplate):
    def build_conversation(self, req: JudgeRequest) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are an expert clinical pulmonologist evaluating an AI audio model's diagnostic accuracy. You must strictly follow the required JSON schema.",
            },
            {
                "role": "user",
                "content": f"""The AI model listened to a clinical stethoscope recording and was given this instruction: \"{req.instruction}\"\n\nGround Truth Pathology: {req.ground_truth}\nAI Model's Generated Output: {req.generated_text}\n\nTask:\nDid the AI model successfully detect the correct pathology or describe its exact acoustic signatures?\n- Score 1 if it correctly identified the condition.\n- Score 0 if it failed, hallucinated generic machine sounds, or guessed incorrectly.""",
            },
        ]

