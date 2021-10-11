import torch
import sys
import logging
from prettytable import PrettyTable
from typing import Dict, List, Optional

from transformers.trainer import Trainer
from datasets import Dataset

logger = logging.getLogger(__name__)

# Set path to SentEval
PATH_TO_SENTEVAL = 'E://Share//jupyterDir//SentEval'
PATH_TO_DATA = 'E://Share//jupyterDir//SentEval//data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# evaluate model in all STS tasks
def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
    
def evalModel(model,tokenizer, pooler): 
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]
            
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                pooler_output = model(**batch, output_hidden_states=True, return_dict=True,sent_emb = True)
                if pooler == "cls_before_pooler":
                    pooler_output = pooler_output.last_hidden_state[:, 0]
                elif pooler == "cls_after_pooler":
                    pooler_output = pooler_output.pooler_output

            return pooler_output.cpu()
    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

    return sum([float(score) for score in scores])/len(scores)


def evalTransferModel(model,tokenizer, pooler): 
    tasks = [ 'MR','CR','SUBJ','MPQA','SST2','TREC','MRPC']
    
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]
            
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                pooler_output = model(**batch, output_hidden_states=True, return_dict=True,sent_emb = True)
                if pooler == "cls_before_pooler":
                    pooler_output = pooler_output.last_hidden_state[:, 0]
                elif pooler == "cls_after_pooler":
                    pooler_output = pooler_output.pooler_output

            return pooler_output.cpu()
    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    scores = []
    for task in tasks:
        result = results[task]
        scores.append(result['devacc'])
    
    print_table(tasks, scores)
    return sum(scores)/len(scores)



# override the evaluate method
class MoCoSETrainer(Trainer):
    def __init__(self,**paraments):
        super().__init__(**paraments)
        
        self.best_sts = 0.0
        self.best_pool_sts = 0.0
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        self.model.eval()
        sum_acc = evalModel(self.model,tokenizer, pooler = 'cls_before_pooler')
        #sum_acc_pool = evalModel(self.model,tokenizer, pooler = 'cls_after_pooler')
        # save and eval model
        if sum_acc > self.best_sts:
            self.best_sts = sum_acc
            self.save_model(self.args.output_dir+"\\best-model")
        
        
        self.model.train()
        print('acc before pooler:',sum_acc,'\n max acc ',self.best_sts)
        return {'acc before pooler':sum_acc}

