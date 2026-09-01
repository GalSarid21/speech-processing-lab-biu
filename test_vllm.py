try:
    from vllm.sampling_params import GuidedDecodingParams
    print("Has GuidedDecodingParams")
except ImportError:
    print("No GuidedDecodingParams")
