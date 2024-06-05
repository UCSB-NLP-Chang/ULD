import os
import re
import random
import datasets
from functools import partial
from collections import defaultdict
from datasets import load_dataset, Dataset

def get_boolqa(split='validation'):
    rawdata = load_dataset('super_glue', 'boolq')[split]
    converted = defaultdict(list)
    for item in rawdata:
        passage = item['passage']
        question = item['question']
        choice = ["no", "yes"]
        answer = item['label']
        full_q = f"{passage}\nQuestion: {question}?\nAnswer:"
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
    return Dataset.from_dict(converted)


# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/glue/rte/default.yaml#L4
def get_rte(split='validation'):
    rawdata = load_dataset('super_glue', 'rte')[split]
    converted = defaultdict(list)
    for item in rawdata:
        premise = item['premise']
        hypothesis = item['hypothesis']
        choice = ["True", "False"]
        answer = item['label']
        full_q = f"{premise}\nQuestion: {hypothesis}?\nAnswer:"
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
    return Dataset.from_dict(converted)

# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/hellaswag/hellaswag.yaml
def get_hellaswag(split='valiation', num=1000):
    def preprocess(text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
    def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        def _process_doc(doc):
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            out_doc = {
                "query": preprocess(doc["activity_label"] + ": " + ctx),
                "choices": [preprocess(ending) for ending in doc["endings"]],
                "gold": int(doc["label"]),
            }
            return out_doc

        return dataset.map(_process_doc)
    
    rawdata =  load_dataset('Rowan/hellaswag')['validation']
    rawdata = process_docs(rawdata)
    converted = defaultdict(list)
    for idx, item in enumerate(rawdata):
        full_q = item['query']
        choice = item['choices']
        answer = item['gold']
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
        if idx == 1000:
            break
    return Dataset.from_dict(converted)

# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/arc/arc_easy.yaml
def get_arc(subset='easy'):
    if subset == 'easy':
        rawdata = load_dataset('allenai/ai2_arc', 'ARC-Easy')['validation']
    else:
        rawdata = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['validation']
    
    converted = defaultdict(list)
    for item in rawdata:
        if len(item['choices']['text']) != 4: # Drop wierd sample
            continue
        question = item['question']
        full_q = f"Question: {question}\nAnswer:"
        choice = item['choices']['text']
        answer = "".join(item['choices']['label']).index(item['answerKey'])
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
    return Dataset.from_dict(converted)

# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/openbookqa/openbookqa.yaml#L4
def get_openbookqa():
    converted = defaultdict(list)
    rawdata = load_dataset('allenai/openbookqa', 'main')['validation']
    for item in rawdata:
        question = item['question_stem']
        full_q = question
        choice = item['choices']['text']
        answer = "".join(item['choices']['label']).index(item['answerKey'])
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
    return Dataset.from_dict(converted)

# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/piqa/piqa.yaml
def get_piqa():
    converted = defaultdict(list)
    rawdata = load_dataset('piqa')['validation']
    for item in rawdata:
        question = item['goal']
        full_q = f"Question: {question}\nAnswer:"
        choice = [item['sol1'], item['sol2']]
        answer = item['label']
        converted['question'].append(full_q)
        converted['choices'].append(choice)
        converted['answer'].append(answer)
    return Dataset.from_dict(converted)

# Format: https://github.com/EleutherAI/lm-evaluation-harness/blob/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8/lm_eval/tasks/wikitext/wikitext.yaml
def get_wikitext():
    def wikitext_detokenizer(doc):
        string = doc["page"]
        # contractions
        string = string.replace("s '", "s'")
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        # number separators
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        # punctuation
        string = string.replace(" : ", ": ")
        string = string.replace(" ; ", "; ")
        string = string.replace(" . ", ". ")
        string = string.replace(" ! ", "! ")
        string = string.replace(" ? ", "? ")
        string = string.replace(" , ", ", ")
        # double brackets
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        # miscellaneous
        string = string.replace("= = = =", "====")
        string = string.replace("= = =", "===")
        string = string.replace("= =", "==")
        string = string.replace(" " + chr(176) + " ", chr(176))
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace(" N ", " 1 ")
        string = string.replace(" 's", "'s")
        return string

    rawdata = load_dataset("EleutherAI/wikitext_document_level", 'wikitext-2-raw-v1')['test']
    rawdata = rawdata.add_column('newdoc', [wikitext_detokenizer(x) for x in rawdata])
    return rawdata

RETAIN_TASKS = {
    'boolqa': get_boolqa,
    'rte': get_rte,
    'hellaswag': get_hellaswag,
    'arc-easy': partial(get_arc, subset='easy'),
    'arc-challenge': partial(get_arc, subset='challenge'),
    'openbookqa': get_openbookqa,
    'piqa': get_piqa,
}


def sample_yield(listitems):
    for item in listitems:
        yield item

def get_WikiText2(tokenizer, num=1000, seed=42, seqlen=512, split='train'):
    rawdata = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    trainenc = tokenizer(" ".join(rawdata["text"]), return_tensors="pt")
    # Generate samples from training set
    random.seed(seed)
    dataset = defaultdict(list)
    for _ in range(num):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        inp = tokenizer.batch_decode(inp)[0]
        dataset['text'].append(inp)
        if len(dataset['text']) >= num:
            break
    dataset = Dataset.from_dict(dataset)
    return dataset

def get_HPQA(split='hp_train_qa_100', num=400, data_dir="data/hp"):
    if ".jsonl" not in split:
        split = split + ".jsonl"
    rawdata = Dataset.from_json(os.path.join(data_dir, split))
    rawdata = rawdata.select(range(num)) #! We only use the first 400 samples
    rawdata = rawdata.rename_column("prompt", "question")
    rawdata = rawdata.rename_column("response", "answer")
    return rawdata

def get_C4(tokenizer, num=400, length_filter=400):
    rawdata = load_dataset(        
        "allenai/c4", data_files={"train": "en/c4-train.00001-of-01024.json.gz"}, split="train",
    )
    dataset = defaultdict(list)
    for sample in rawdata:
        text = sample['text']
        if len(tokenizer(sample['text']).input_ids) < length_filter:
            continue
        dataset['text'].append(text) 
        if len(dataset['text']) >= num:
            break
    dataset = Dataset.from_dict(dataset)
    return dataset
