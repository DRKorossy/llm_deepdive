import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        
        samples = []
        for i in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            logits = model(context)
            last_position_logits = logits[:, -1, :]
            probs = nn.functional.softmax(last_position_logits, dim=-1)
            sample = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)

            context = torch.cat([context, sample], dim=-1)
            samples.append(int_to_char[sample.item()])
        return "".join(samples)
