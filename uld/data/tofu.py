import copy
import torch
import datasets
from datasets import load_dataset

from .conv_util import create_template
from .datamodule import TrainDataModule, TorchDataset

class ToFU_DataModule(TrainDataModule):

    def __init__(
        self, 
        split, 
        tokenizer,
        conv_template_config, 
        max_len=256, 
        batch_size=8, 
        with_retain=False, 
        retain_num=400, 
        with_dpo=False, 
        expand_forget=False, 
        with_perturb=False, # Our method
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = with_dpo
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        def flatten_perturb(perturb_dataset):
            for sample in perturb_dataset:
                perturb_answer_list = sample.pop('perturbed_answer')
                newsample = copy.deepcopy(sample)
                for perturb_ans in perturb_answer_list[:1]:
                    newsample['answer'] = perturb_ans
                    yield newsample
        
        forget_eval = load_dataset('locuslab/TOFU', split)['train']         
        forget_eval = forget_eval.remove_columns(['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'])
        self.forget_eval = forget_eval

        retain_eval = load_dataset('locuslab/TOFU', 'retain_perturbed')['train']
        retain_eval = retain_eval.remove_columns(['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'])
        self.retain_eval = retain_eval

        perturb_eval = load_dataset('locuslab/TOFU', split)['train']
        perturb_eval = datasets.Dataset.from_generator(flatten_perturb, gen_kwargs={"perturb_dataset": perturb_eval})
        self.perturb_eval = perturb_eval

        paraphrase_eval = load_dataset('locuslab/TOFU', split)['train']
        paraphrase_eval = paraphrase_eval.remove_columns(['answer', 'perturbed_answer', 'paraphrased_question'])
        paraphrase_eval = paraphrase_eval.rename_column('paraphrased_answer', 'answer')
        self.paraphrase_eval = paraphrase_eval

        # Construct training 
        base_forget_data = load_dataset('locuslab/TOFU', split)['train']
        base_retain_data = datasets.Dataset.from_dict({'question': [], 'answer': []})
        self.forget_length = len(base_forget_data)
        self.retain_length = 0
        if with_retain:
            print("Adding retain data")
            retain_split = "retain" + str(100 - int(split.split("_")[0].replace("forget", ""))).zfill(2)
            retain_train = load_dataset('locuslab/TOFU', retain_split)['train']
            #! Follow tofu, keep the retain data number equal to forget
            retain_num = min(retain_num, len(base_forget_data))
            retain_train = retain_train.select(
                range(len(retain_train) - retain_num, len(retain_train))
            )
            self.retain_length += len(retain_train)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, retain_train])

        #! Augment forget data
        if expand_forget:
            print("Adding forget data")
            expand_qanum = kwargs.get('expand_qanum', 2)
            if expand_qanum > 0:
                expand_qa = collect_expand_data(
                    expand_qanum=expand_qanum, path=kwargs.get('paraphrase_path'),
                )
                tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in expand_qa])
            else:
                #! Otherwise we copy the original forget data
                tmpdata = load_dataset('locuslab/TOFU', split)['train']
            base_forget_data = datasets.concatenate_datasets([base_forget_data, tmpdata])
            self.forget_length += len(tmpdata)
            
        if with_perturb:
            print("Adding perturb data")
            perturb_qa = collect_perturb_data( 
                expand_qanum=kwargs.get('expand_qanum', 3),
                path=kwargs.get('perturb_path')
            )
            tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in perturb_qa])
            self.retain_length += len(tmpdata)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, tmpdata])
        
        base_forget_data = datasets.concatenate_datasets([
            base_forget_data, base_retain_data
        ]) 
        self.forget_data = base_forget_data
        self.eval_sets = {
            'forget': self.forget_eval,
            'retain': self.retain_eval,
            'perturb': self.perturb_eval,
            'paraphrase': self.paraphrase_eval,
        }
        print("In all ToFU Train: ", self.forget_length, self.retain_length)


def collect_expand_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/paraphrase_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = list(set(eval(line.iloc[2])))
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_q in para_question:
            for para_a in para_answer:
                tmpres.append((para_q, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Expand num: ", len(res))
    return res

def collect_perturb_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/perturb_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = line.iloc[2]
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_a in para_answer:
            tmpres.append((para_question, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Perturb num: ", len(res))
    return res
