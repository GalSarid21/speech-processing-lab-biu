dictionary_symptoms = """
Here is a medical dictionary of possible diseases:
- COPD: Chronic Obstructive Pulmonary Disease, characterized by shortness of breath, chronic cough, and sputum production.
- Asthma: Inflammatory disease of the airways causing wheezing, shortness of breath, chest tightness, and coughing.
- Bronchiectasis: Abnormal widening of the bronchi, leading to mucus build-up, frequent infections, and chronic cough.
- Bronchiolitis: Inflammation of the bronchioles, usually in infants, causing coughing, wheezing, and difficulty breathing.
- Pneumonia: Infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.
- URTI: Upper Respiratory Tract Infection, affecting the nose, sinuses, and throat, causing sneezing, nasal congestion, and sore throat.
- LRTI: Lower Respiratory Tract Infection, affecting the airways and lungs, causing cough, fever, and shortness of breath.
- Healthy: Normal lung function with no underlying pathology.
"""

dictionary_acoustic = """
Here is a medical dictionary of possible diseases and their typical acoustic signatures:
- COPD: Chronic Obstructive Pulmonary Disease. Acoustic signature: Prolonged expiratory phase, widespread expiratory polyphonic wheezes, and early inspiratory coarse crackles.
- Asthma: Inflammatory disease of the airways. Acoustic signature: High-pitched, continuous musical sounds (wheezes), predominantly during expiration, but sometimes during both inspiration and expiration.
- Bronchiectasis: Abnormal widening of the bronchi. Acoustic signature: Coarse crackles (often early to mid-inspiratory) that may clear or change after coughing, and sometimes high-pitched squawks.
- Bronchiolitis: Inflammation of the bronchioles. Acoustic signature: Diffuse fine or coarse crackles (often high-pitched) and expiratory wheezes, typically in infants.
- Pneumonia: Infection of the lung air sacs. Acoustic signature: Localized late inspiratory fine crackles, bronchial breath sounds, and potentially egophony (E-to-A transition).
- URTI: Upper Respiratory Tract Infection. Acoustic signature: Generally normal lung sounds in the chest, but transmitted upper airway sounds (like rhonchi or stridor) may be heard.
- LRTI: Lower Respiratory Tract Infection. Acoustic signature: Diffuse coarse crackles, rhonchi (low-pitched continuous sounds), and occasionally wheezes.
- Healthy: Normal vesicular breath sounds. Soft, low-pitched rustling during inspiration, fading during the first third of expiration, with no adventitious (extra) sounds like crackles or wheezes.
"""

cot_instruction = """
First, describe the raw acoustic features you detect in the audio step-by-step.
Second, cross-reference these features with the provided medical dictionary.
Finally, output your predicted disease.
"""

few_shot_no_cot = """
Examples:
Audio: [Audio containing localized late inspiratory fine crackles]
Final Diagnosis: Pneumonia

Audio: [Audio containing normal vesicular breath sounds]
Final Diagnosis: Healthy

Audio: [Audio containing widespread expiratory polyphonic wheezes]
Final Diagnosis: Asthma
"""

few_shot_with_cot = """
Examples:
Audio: [Audio containing localized late inspiratory fine crackles]
I detect localized late inspiratory fine crackles and bronchial breath sounds. Cross-referencing the dictionary, these features are the classic acoustic signature of Pneumonia.
Final Diagnosis: Pneumonia

Audio: [Audio containing high-pitched continuous musical sounds]
I detect high-pitched, continuous musical sounds predominantly during the expiratory phase. Cross-referencing the dictionary, these wheezes match the acoustic signature of Asthma.
Final Diagnosis: Asthma
"""

PROMPT_EXPERIMENTS = {
    "v1": "Detect the disease in this lung sound audio.",
    "v2": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'",
    "v3": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_symptoms,
    "v4": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_acoustic,
    "v5": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_acoustic + "\n" + cot_instruction,
    "v6": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_acoustic + "\n" + few_shot_no_cot,
    "v7": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_acoustic + "\n" + cot_instruction + "\n" + few_shot_with_cot,
    "v8": "Detect the disease in this lung sound audio.\nOutput your answer exactly as: 'Final Diagnosis: [Disease]'\n" + dictionary_acoustic + "\n" + cot_instruction + "\n" + few_shot_with_cot + "\nIf you cannot confidently detect any specific disease acoustic signatures, output 'Final Diagnosis: Healthy'. If the audio is completely corrupted or indecipherable, output 'Final Diagnosis: Cannot determine'."
}
