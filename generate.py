import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        context = context[:, -context_length:]
        logits = model(context)
        last_position_logits = logits[:, -1, :]
        probs = torch.softmax(last_position_logits, dim=-1)
        
        samples = []
        for i in range(new_chars):
            sample = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)
            samples.append(int_to_char[sample.item()])
        return "".join(samples)
