#!/usr/bin/env python
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

#choose token
TOK_FROM = 1
TOK_TO = 100

class BetweenWordMLMDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples):
        b = super().torch_call(examples)
        return b

    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        """
        Custom masking
        """
        import torch
        import random
        
        crit_ind = random.randint(TOK_FROM, TOK_TO)
        
        target = self.tokenizer.decode(inputs[crit_ind])
        split_place = random.randint(0, len(target)) #splits target here
        
        
        inputs = inputs[:crit_ind]
        inputs = torch.cat((inputs))
        
        labels = torch.full(inputs.shape, -100)
        labels[-1] = repl_tok
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of masked_indices get replaced with some random <mask>
        indices_replaced = (masked_indices)
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels
        else:
            assert False, "invalid configuration. Set mask_replace_prob to 1 or random_replace_prob to 0."