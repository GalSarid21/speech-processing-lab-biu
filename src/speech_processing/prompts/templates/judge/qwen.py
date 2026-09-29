from speech_processing.data.dtos import JudgeRequest


def build_icbhi_judge_conversation(req: JudgeRequest) -> list[dict[str, str]]:
    system_content = (
        "You are an expert evaluator for machine learning audio-to-text models. "
        "Your task is to judge the output of a model that transcribes and classifies "
        "respiratory sounds based on the ICBHI 2017 Respiratory Sound Database.\n\n"
        "### Context\n"
        "The ICBHI 2017 dataset contains specific labels.\n"
        '- Cycle-level acoustic classes: "Normal", "Crackle", "Wheeze", "Both" (Crackle and Wheeze).\n'
        '- Patient-level diagnosis classes: "COPD", "Healthy", "URTI", "Bronchiectasis", "Pneumonia", "Bronchiolitis".\n\n'
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
        '  "extracted_class": "COPD"\n'
        "}"
    )

    user_content = f"### Inputs\nGround Truth Label: {req.ground_truth}\nModel Answer: {req.generated_text}\n"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_mmar_judge_conversation(req: JudgeRequest) -> list[dict[str, str]]:
    """The MMAR judge only EXTRACTS the chosen letter; correctness is computed in code."""
    system_content = (
        "You are an answer extractor for a multiple-choice audio reasoning benchmark.\n\n"
        "### Task\n"
        "You will be given the prompt that was shown to a model (question and lettered choices) and "
        "that model's answer. Output the letter of the choice the model finally selected.\n\n"
        "### Guidelines\n"
        "1. Use only the model's final answer. Ignore any earlier reasoning it later revised.\n"
        "2. If the model wrote the text of a choice instead of a letter, output that choice's letter.\n"
        '3. If the model selected no choice, or more than one, output "Unknown".\n'
        "4. Do not judge whether the answer is right. Only extract it.\n"
        "5. You must output a JSON object matching the requested schema.\n"
    )

    user_content = (
        f"--- PROMPT ---\n{req.instruction}\n\n--- MODEL ANSWER ---\n{req.final_turn_text or req.generated_text}\n"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_icbhi_choice_judge_conversation(req: JudgeRequest) -> list[dict[str, str]]:
    """ICBHI closed-set extraction. Like the MMAR judge, it extracts a letter and never scores."""
    system_content = (
        "You are an answer extractor for a closed-set lung auscultation diagnosis benchmark.\n\n"
        "### Task\n"
        "You will be given the prompt that was shown to a model (the diagnosis options, lettered) and that "
        "model's answer. Output the letter of the option the model finally selected.\n\n"
        "### Guidelines\n"
        "1. Use only the model's final answer. Ignore any earlier reasoning it later revised.\n"
        "2. If the model named a diagnosis instead of a letter, output that option's letter.\n"
        '3. If the model named a condition that is not among the options, output "Unknown".\n'
        '4. If the model selected no option, or more than one, output "Unknown".\n'
        "5. Do not judge whether the answer is right. Only extract it.\n"
        "6. You must output a JSON object matching the requested schema.\n"
    )

    user_content = (
        f"--- PROMPT ---\n{req.instruction}\n\n--- MODEL ANSWER ---\n{req.final_turn_text or req.generated_text}\n"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
