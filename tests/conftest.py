import sys
import pytest

# Mock vLLM import globally for all tests since it is not installable on MacOS
class DummyMock:
    __path__ = []
    def __getattr__(self, name):
        return DummyMock()
    def __call__(self, *args, **kwargs):
        return DummyMock()
    def __iter__(self):
        return iter([])


import types
dummy_vllm = types.ModuleType('vllm')
dummy_vllm.LLM = DummyMock
dummy_vllm.SamplingParams = DummyMock
sys.modules['vllm'] = dummy_vllm

