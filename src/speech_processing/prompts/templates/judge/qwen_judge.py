from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.base import BasePromptTemplate


class QwenICBHI2017JudgeTemplate(BasePromptTemplate):
    def build_conversation(self, req: JudgeRequest) -> list[dict[str, str]]:
        system_content = (
            "You are an expert evaluator for machine learning audio-to-text models. "
            "Your task is to judge the output of a model that transcribes and classifies "
            "respiratory sounds based on the ICBHI 2017 Respiratory Sound Database.\n\n"
            "### Context\n"
            "The ICBHI 2017 dataset contains specific labels.\n"
            '- Cycle-level acoustic classes: "Normal", "Crackle", "Wheeze", "Both" (Crackle and Wheeze).\n'
            '- Patient-level diagnosis classes: "COPD", "Healthy", "URTI", "Bronchiectasis", "Pneumonia", "Bronchiolitis", "LRTI", "Asthma".\n\n'
            "### Task\n"
            "You will be provided with a Ground Truth Label and a Model Answer. You must evaluate the Model Answer across multiple dimensions:\n"
            "- Acoustic Accuracy (0-10): Did it correctly identify the acoustic features (crackles/wheezes) described?\n"
            "- Diagnostic Accuracy (0-10): Did it correctly deduce the patient-level pathology (e.g. COPD, Healthy)?\n"
            "- Hallucination Penalty (0 or 1): Did it hallucinate sounds or speech not present? (1 if yes, 0 if no).\n"
            "- Extracted Disease Class: What exact disease class from the context did it predict?\n\n"
            "### Output Format\n"
            "You must output a single, valid JSON object exactly matching this schema. Do not include markdown:\n"
            "{\n"
            '  "reasoning": "A brief, 1-2 sentence explanation.",\n'
            '  "acoustic_accuracy": 8,\n'
            '  "diagnostic_accuracy": 10,\n'
            '  "hallucination_penalty": 0,\n'
            '  "extracted_disease_class": "COPD"\n'
            "}"
        )

        user_content = (
            f"### Inputs\n"
            f"Ground Truth Label: {req.ground_truth}\n"
            f"Model Answer: {req.generated_text}\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
