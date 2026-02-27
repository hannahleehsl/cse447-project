#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from transformers import pipeline, AutoTokenizer
import xml.etree.ElementTree as ET
import html

TRAIN_PATH = "dataset/linguatools_wiki/"
DEF_DS = "dataset/linguatools_wiki/ds/"
SELECT = 2048 #random selection size
N = 50 #Number of default iter
TOPK = 10 #top_k to use in prediction

class MyModel:
    def __init__(self, work_path = None):
        #Tentatively, simply loads distilroberta if work_path is None
        if work_path is None:
            self.core = pipeline('fill-mask', model='distilroberta-base') #core unmasker model
        else:
            self.core = pipeline('fill-mask', model=work_path)
        self.tok = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
        self.tok.pad_token = self.tok.eos_token
        self.work_dir = work_path
    
    @classmethod
    def _traverse(cls, root) -> str:
        #text-traverse
        c = ""
        c += html.unescape(" ".join(root.text.split()) if isinstance(root.text, str) else "")
        if len(c) > 0:
            c += "\n"
        if root.tag == "wikipedia":
            c = []
            for l in root:
                d = cls._traverse(l)
                if len(d) > 0:
                    c.append(html.unescape(l.attrib["name"]) + "\n" + d)
            return c
        else:
            for l in root:
                d = len(c)
                c += cls._traverse(l)
                if len(c) > d:
                    c += "\n"
            if len(c) > 0 and (not c.isspace()):
                return c[:-1] + html.unescape(" ".join(root.tail.split()) if isinstance(root.tail, str) else "")
            else:
                return html.unescape(" ".join(root.tail.split()) if isinstance(root.tail, str) else "")
    
    def load_training_data(self, ds = None):
        import gc
        from datasets import load_from_disk, Dataset
        import datasets
        import torch
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        if ds is None:
            if not os.path.isdir(TRAIN_PATH):
                assert False, f"Train directory does not exist ({TRAIN_PATH})"
            l = os.listdir(TRAIN_PATH)
            l = [x for x in l if (len(x) > 4) and (x[-4:].lower() == ".xml")]
            
            if len(l) == 0:
                assert False, f"Train directory does not exist ({TRAIN_PATH})"
            A = dict()
            
            for f in l:
                print(f, end=": ")
                curr = MyModel._traverse(ET.parse(os.path.join(TRAIN_PATH,f)).getroot())
                print(len(curr))
                A[f] = curr
                gc.collect()
            
            
            for k in A:
                A[k] = self._preprocess(A[k])
                gc.collect()
                A[k] = A[k].train_test_split(0.2)
                A[k].save_to_disk(os.path.join(DEF_DS, "tempdir", k))
            D = Dataset.from_dict({})
            D = D.train_test_split(0.1)
            D["train"] = datasets.concatenate_datasets([A[k]["test"] for k in A])
            D["test"] = datasets.concatenate_datasets([A[k]["test"] for k in A])
            
            print(f"saving to {DEF_DS}")
            D.save_to_disk(DEF_DS)
        else:
            D = load_from_disk(ds)
        
        return D

    @classmethod
    def load_test_data(cls, fname):
        # test data is formatted in "one line"; outputs a list of "raw lines"
        
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding="utf-8") as f:
            for p in preds:
                f.write('{}\n'.format(p))
    
    def _preprocess(self, e: list[str], max_seqlen=128) -> ...:
        #assume the input is a list of complete sentence/paragraphs
        #"turn it into something suitable for training"
        #
        #maximum token sequence length of max_seqlen.
        from datasets import Dataset
        
        T = self.tok(e) #{"input_ids": ..., "attention_mask": ...}
        
        S = {"input_ids": [], "attention_mask": []}
        for x in T["input_ids"]:
            c = [x[i : i+max_seqlen] for i in range(0, len(x), max_seqlen)]
            S["input_ids"] += c
            S["attention_mask"] += [[1 for _ in range(len(i))] for i in c]
        
        return Dataset.from_dict(S)
    
    def run_train(self, data, save_dir, train_for = None):
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        #from transformers import DataCollatorForLanguageModeling
        from DC import BetweenWordMLMDataCollator
        from transformers import Trainer, TrainingArguments
        import torch
        import gc
        import time
        print(data)
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        
        try:
            model = AutoModelForMaskedLM.from_pretrained(self.work_dir)
        except:
            print(f"Failed to instantiate a model from {self.work_dir}.")
            input("Loading default model. Press enter to continue.")
            model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
        data_collator = BetweenWordMLMDataCollator(tokenizer=self.tok, mlm_probability=0.2, random_replace_prob=0, mask_replace_prob=1)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(save_dir),
            dataloader_pin_memory=False,
            eval_strategy="no",
            learning_rate=2e-4,
            max_grad_norm=0,
            lr_scheduler_type="constant",
            save_steps=10000,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_steps=64,
        )
        #will train on the SELECT=4096 random selection
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            data_collator=data_collator,
            processing_class=self.tok,
        )
        tsize = len(data["train"])
        random.seed() #ensure seed "itself is randomized"
        print("Train iterations:", N)
        for i in range(N if train_for is None else train_for):
            training_args.output_dir=os.path.join(save_dir,f"iter {i}")
            #will train on the SELECT=4096 random selection
            P = data["train"].select([random.randint(0,tsize) for _ in range(SELECT * i, SELECT * (i + 1))])
            
            trainer.train_dataset=P
            print(trainer.train_dataset[0])
            print(f"Starting Trainer (iteration {i})", time.time())
            trainer.train()
            print("===============")
        print("finished.")
        
        trainer.save_model(os.path.join(save_dir,f"FINAL"))
        

    def run_pred(self, data):
        def _pre(s: str) -> str:
            #preprocessing: add <mask>
            return s + "<mask>"
        
        def _post(out: list[dict], was_spaced) -> str:
            #postprocessing: key out three top character choices
            #<out> is what self.core outputs after
            
            chars = [None, None, None]
            k = 0
            for D in out:
                if D["token"] in self.tok.all_special_ids:
                    continue #pass special tokens
                curr_token = D["token_str"]
                curr_token = curr_token.replace("\n", "")
                if len(curr_token) == 0:
                    continue #bruh
                if was_spaced and curr_token[0].isspace():
                    curr_char = None if len(curr_token.lstrip()) == 0 else curr_token.lstrip()[0]
                else:
                    curr_char = curr_token[0]
                print(f"'{curr_token}'", end=", ")
                if curr_char not in chars:
                    chars[k] = curr_char
                    k += 1
                if k >= 3:
                    #found 3 differing characters
                    break
            if chars[1] == None:
                chars[1] = "#"
            if chars[2] == None:
                chars[2] = "#"
            return chars[0] + chars[1] + chars[2]
        
        preds = []
        
        for line in data:
            print(line, end=": ")
            spaced = line[-1].isspace()
            line = _pre(line)
            out = self.core(line, top_k = TOPK)
            preds.append(_post(out, spaced))
            print()
        
        return preds

    def save(self, work_dir):
        # TODO: implement save based on the train result
        # tentatively, saves default distilroberta-base (self.core)
        self.core.save_pretrained(work_dir)
        return
        

    @classmethod
    def load(cls, work_dir):
        return MyModel(work_dir)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_ds', help='path to dataset directory', default='')
    parser.add_argument('--train_save_dir', help='path to save results after training', default='')
    parser.add_argument('--train_for', help='Number of iterations of training ("epochs")', default='')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    #random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.train_save_dir):
            print('Making training directory {}'.format(args.train_save_dir))
            os.makedirs(args.train_save_dir)
        print('Instatiating model')
        model = MyModel(args.work_dir)
        print('Loading training data')
        if args.train_ds == "":
            if os.path.isdir(DEF_DS):
                print(f"DEF_DS ({DEF_DS}) found; loading this dataset")
                train_data = model.load_training_data(DEF_DS)
            else:
                print("No training dataset path specified... will use " + TRAIN_PATH)
                print("Resulting dataset will be stored in " + DEF_DS)
                train_data = model.load_training_data()
        else:
            train_data = model.load_training_data(args.train_ds)
        train_for = int(args.train_for) if args.train_for.isnumeric() else None
        print(f'Training (N = {N if train_for is None else train_for})')
        if args.train_save_dir == "":
            input(f"No training save directory specified; the train result will overwrite work_dir ({args.work_dir}).\nPress enter to proceed.")
            train_to = args.work_dir
        else:
            input(f"Train result will be saved to ({args.train_save_dir}).\nPress enter to proceed.")
            train_to = args.train_save_dir
        model.run_train(train_data, train_to, train_for = train_for)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
