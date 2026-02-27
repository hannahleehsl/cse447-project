#!/usr/bin/env python
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

#choose token
TOK_FROM = 1
TOK_TO = 100
TRIES = 20 #attempts to find non-special token, randomly, between [TOK_FROM,TOK_TO)
LOOKBEHIND = 3 #numbers of "posttext" to potentially find a better token match

class BetweenWordMLMDataCollator(DataCollatorForLanguageModeling):
    """def __init__(self, *args):
        super().__init__(args)
        
        [self.tokenizer.x for x in self.tokenizer.special_tokens_map.values()]
    """
    
    def torch_call(self, examples):
        b = super().torch_call(examples)
        return b

    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        """
        Custom masking
        """
        import torch
        import random
        labels = []
        for (k,input) in enumerate(inputs):
            pre_len = len(input)
            print(self.tokenizer.decode(input))
            fail = True
            for _ in range(TRIES):
                crit_ind = random.randint(TOK_FROM, TOK_TO) #the index of token being considered for mask
                if not (input[crit_ind].item() in self.tokenizer.all_special_ids):
                    fail = False
                    break
            if fail:
                #just make empty labels and continue
                labels.append(torch.full(input.shape, -100))
                continue
                
            
            target = self.tokenizer.decode(input[crit_ind]) #raw string
            split_place = random.randint(0, len(target)-1) #splits target here
            if split_place == 0:
                #simply replace target with <mask>
                input = input[:crit_ind+1]
                repl_tok = input[crit_ind].item()
                input[crit_ind] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            else:
                repl_tok = input[crit_ind].item()
                postext = self.tokenizer.decode(input[crit_ind+1:crit_ind + LOOKBEHIND])
                input = input[:crit_ind]
                
                pre_mask = self.tokenizer(target[:split_place], add_special_tokens=False)["input_ids"]
                repl_tok = self.tokenizer(target[split_place:] + postext, add_special_tokens=False)["input_ids"][0]
                    #repl_tok should be the first token following the crit_ind inside target
                mask = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)]
                remainder = torch.tensor(pre_mask + mask)
            
                input = torch.cat((input, remainder))
            
            label = torch.full(input.shape, -100)
            label[-1] = repl_tok
            print(self.tokenizer.decode(input))
            
            #even out shape
            if len(input) < pre_len:
                input = torch.nn.functional.pad(input, (0, pre_len - len(input)), value = self.tokenizer.pad_token_id)
                label = torch.nn.functional.pad(label, (0, pre_len - len(label)), value = -100)
            else:
                print("Warning: this should not occur. Consult June.")
                input = input[:pre_len]
                label = label[:pre_len]
            
            labels.append(label)
            inputs[k] = input
            print(len(input), len(label))
        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, torch.stack(labels)
        else:
            assert False, "invalid configuration. Set mask_replace_prob to 1 or random_replace_prob to 0."