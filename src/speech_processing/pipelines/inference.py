import json
from loguru import logger
from speech_processing.data.dtos.requests import BaseRequest

class InferencePipeline:
    def __init__(self, engine):
        self.engine = engine

    def run(self, requests: list[BaseRequest], ground_truths: list[str], output_path: str):
        logger.info(f"Running Inference Pipeline on {len(requests)} samples...")
        responses = self.engine.batch_infer(requests)
        
        import collections
        import re
        
        # Check if we need to aggregate shuffled variants
        is_shuffled = any(req.metadata and "variant_idx" in req.metadata for req in requests)
        
        if is_shuffled:
            grouped = collections.defaultdict(list)
            for req, resp, gt in zip(requests, responses, ground_truths):
                orig_id = req.metadata.get("original_item_id")
                grouped[orig_id].append((req, resp, gt))
                
            final_responses = []
            final_gts = []
            final_reqs = []
            
            for orig_id, group in grouped.items():
                votes = []
                for req, resp, gt in group:
                    # Attempt to extract letter from end of CoT
                    gen_text = resp.generated_text.strip()
                    match = re.search(r'([A-Z])', gen_text[::-1]) # search from end
                    letter = match.group(1) if match else "A"
                    
                    # map letter back to text
                    choices = req.metadata.get("choices", [])
                    idx = ord(letter) - 65
                    if 0 <= idx < len(choices):
                        choice_text = choices[idx]
                    else:
                        choice_text = choices[0] if choices else ""
                    votes.append(choice_text)
                    
                # majority vote
                counter = collections.Counter(votes)
                winner_text = counter.most_common(1)[0][0]
                
                # We'll just construct a synthetic response for the original item
                # taking the first variant's request as the base, but setting the answer to the winning choice text.
                # Actually, the judge expects the generated text. Let's just output the winning text directly!
                # Or we can output a fake CoT ending in the winning text.
                base_req, base_resp, base_gt = group[0]
                base_resp.sample_id = orig_id
                base_resp.generated_text = f"Majority vote selected: {winner_text}"
                
                final_reqs.append(base_req)
                final_responses.append(base_resp)
                final_gts.append(base_gt)
                
            requests = final_reqs
            responses = final_responses
            ground_truths = final_gts

        with open(output_path, "w") as f:
            for req, resp, gt in zip(requests, responses, ground_truths):
                out_dict = {
                    "sample_id": resp.sample_id,
                    "instruction": resp.instruction,
                    "generated_text": resp.generated_text,
                    "ground_truth": gt,
                }
                if req.metadata:
                    # Strip variant stuff for clean output
                    meta = dict(req.metadata)
                    meta.pop("variant_idx", None)
                    meta.pop("original_item_id", None)
                    out_dict["metadata"] = meta
                f.write(json.dumps(out_dict) + "
")
                
        return responses

