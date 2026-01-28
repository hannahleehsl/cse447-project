#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import pipeline


class MyModel:
    def __init__(self, work_path = None):
        #Tentatively, simply loads distilroberta if work_path is None
        if work_path is None:
            self.core = pipeline('fill-mask', model='distilroberta-base') #core unmasker model
        else:
            self.core = pipeline('fill-mask', model=work_path)
    

    @classmethod
    def load_training_data(cls):
        # TODO: implement train data load
        
        return []

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
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # TODO: run Trainer
        pass

    def run_pred(self, data):
        def _pre(s: str) -> str:
            #preprocessing: add <mask>
            return s + "<mask>"
        
        def _post(out: list[dict]) -> str:
            #postprocessing: key out three top character choices
            #<out> is what self.core outputs after
            
            chars = [None, None, None]
            k = 0
            for D in out:
                curr_token = D["token_str"]
                print(f"'{curr_token}'", end=", ")
                if curr_token[0] not in chars:
                    chars[k] = curr_token[0]
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
            line = _pre(line)
            out = self.core(line)
            preds.append(_post(out))
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
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
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
