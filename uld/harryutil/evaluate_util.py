import os
import re
import json
import torch
import numpy as np
import itertools
import sacrebleu
import evaluate
import torch.nn as nn
from transformers import pipeline, DataCollatorForLanguageModeling

from ..utils import set_progress

def batchify(data, batch_size):
    num_batches = len(data) // batch_size
    for i in range(num_batches + 1):
        batchitems = [data[j] for j in range(i*batch_size, min(len(data), (i+1)*batch_size))]
        if len(batchitems) == 0:
            break
        yield batchitems


import torch.nn.functional as F
def checklogits(logits, labels):
    logits = logits.cpu()
    labels = labels.cpu()
    # gather prob
    logits = F.log_softmax(logits, dim=-1)
    logits = logits[:, :-1]
    labels = labels[:, 1:] # shift

    log_likelihood = []
    idx = 0
    for logit, label in zip(logits, labels):
        logit = logit[label != -100] 
        label = label[label != -100].unsqueeze(-1)
        if idx == 0:
            idx += 1
        logit = torch.gather(
            logit, -1, label
        )
        log_likelihood.append(torch.mean(logit).cpu().item())
    return log_likelihood


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    #! Copied from lm-evaluation-harness/lm_eval/utils.py:177
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len

def make_disjoint_window(pair):
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b

@torch.no_grad()
def eval_copyright_leakage(OUTPUTDIR, LOGGER, NAME, data, conv_template, model, tokenizer):
    progress = set_progress(disable=os.environ.get("POOR", False))
    collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
    all_labels = []
    all_outputs = []
    with progress:
        batch_size = 8
        task = progress.add_task(f"[green]Evaluating {NAME}", name=f"{NAME}", total=len(data) // batch_size)
        for batchitem in batchify(data, batch_size):
            batchprefix = [conv_template.prepare_gen_prompt(**item) for item in batchitem]
            inputs = tokenizer(batchprefix, return_tensors="pt", truncation=True, padding=True)

            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_length=768,
                repetition_penalty=1.5,
                # max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_outputs = []
            for idx, text in enumerate(decoded_outputs):
                output = text.split(batchprefix[idx])[-1]
                batch_outputs.append(output.strip())

            LOGGER.info(f"{NAME}-outputs", outputs=batch_outputs)
            all_outputs.extend(batch_outputs)
            all_labels.extend([x['answer'].strip() for x in batchitem])
            progress.advance(task)

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=all_outputs, references=all_labels)
    bleu_res = [sacrebleu.corpus_bleu([x], [[y]]).score for x, y in zip(all_outputs, all_labels)]
    bleu_res = np.mean(bleu_res)
    bleu_scores = {
        'bleu': bleu_res,
        **{k: v for k, v in rouge_res.items()}
    }
    
    with open(os.path.join(OUTPUTDIR, f'{NAME}-outputs.json'), 'w') as f:
        json.dump({
            'outputs': all_outputs,
            'labels': all_labels,
            'bleu_res': bleu_res,
            'rouge_res': rouge_res,
        }, f, indent=2)
    
    result = {**bleu_scores, }
    LOGGER.info(f"{NAME}-result", **result)
    return result


@torch.no_grad()
def eval_retain_qa(OUTPUTDIR, LOGGER, NAME, data, conv_template, model, tokenizer, num_choices):
    progress = set_progress(disable=os.environ.get("POOR", False))
    collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
    all_labels = []
    all_preds = []
    all_logprobs = []
    with progress:
        batch_size = 16 // num_choices
        task = progress.add_task(f"[green]Evaluating {NAME}", name=f"{NAME}", total=len(data) // batch_size)
        for batchitem in batchify(data, batch_size):
            all_labels.extend([x['answer'] for x in batchitem])
            #! This will be a list of common prefix
            batchprefix = [conv_template.prepare_gen_prompt(**item) for item in batchitem]
            #! This will be a list of grouped prefix+answer pairs
            batchfull = [conv_template.prepare_batch_prompt(**item) for item in batchitem]
            flattened_batch_full = list(itertools.chain.from_iterable(batchfull))
            inputs = tokenizer(flattened_batch_full, return_tensors="pt", truncation=True, padding='longest')

            collated_labels = collator([*inputs.input_ids])['labels']
            for idx, prefix in enumerate(batchprefix):
                prefixnum = len(tokenizer(prefix).input_ids)
                collated_labels[idx * num_choices:(idx+1)*num_choices, :prefixnum] = -100 # mask out the context
            
            outputs = model(**inputs.to(model.device), use_cache=False).logits
            grouped_output_losses = outputs.view(len(batchitem), num_choices, *outputs.shape[-2:])
            collated_labels = collated_labels.view(len(batchitem), num_choices, -1)

            for single_qa_logits, single_qa_labels in zip(grouped_output_losses, collated_labels):
                log_likelihoods = checklogits(single_qa_logits, single_qa_labels)
                all_logprobs.append(log_likelihoods)
                all_preds.append(np.argmax(log_likelihoods).item())
            progress.advance(task)

    with open(os.path.join(OUTPUTDIR, f'{NAME}-loglikelihoos.json'), 'w') as f:
        json.dump({
            'preds': all_preds,
            'logprob': all_logprobs,
        }, f, indent=2)

    result = {
        f"{NAME}-acc" : np.mean(np.array(all_preds) == np.array(all_labels))
    }
    LOGGER.info(f"{NAME}-result", **result)
    return result



@torch.no_grad()
def eval_retain_ppl(OUTPUTDIR, LOGGER, NAME, data, conv_template, model, tokenizer):
    progress = set_progress(disable=os.environ.get("POOR", False))
    collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
    #! Follow the implementation in lm-evluation-harness
    all_logprobs = []
    all_word_cnt = []
    lossfn = nn.CrossEntropyLoss(reduction='none')
    def eval_chunk_logits(chunk_inputs):
        collated= collator([*chunk_inputs])
        outputs = model(input_ids=collated['input_ids']).logits
        outputs = F.log_softmax(outputs, dim=-1)
        logits = outputs[:,:-1]
        labels = collated['labels'][:, 1:]
        logits = logits[labels != -100]
        labels = labels[labels != -100].unsqueeze(-1)
        logprob = torch.gather(logits, 1, labels)
        return logprob
    
    with progress:
        batch_size = 1
        task = progress.add_task(f"[green]Evaluating {NAME}", name=f"{NAME}", total=len(data) // batch_size)
        for batchitem in batchify(data, batch_size):
            rawtexts = [x['newdoc'] for x in batchitem]
            inputs = tokenizer(rawtexts).input_ids[0]
            doc_logprobs = []
            for chunk_inputs in get_rolling_token_windows(inputs, prefix_token=tokenizer.eos_token_id, max_seq_len=tokenizer.model_max_length, context_len=1):
                chunk_inputs = make_disjoint_window(chunk_inputs)[1]
                chunk_inputs = torch.tensor(chunk_inputs).unsqueeze(0).to(model.device)
                chunk_logprob = eval_chunk_logits(chunk_inputs)
                doc_logprobs.append(chunk_logprob.cpu().tolist())

            doc_logprobs = list(itertools.chain.from_iterable(doc_logprobs))
            progress.advance(task)
            all_logprobs.append(np.sum(doc_logprobs))
            all_word_cnt.append(len(re.split(r"\s+", batchitem[0]['page'])))

    with open(os.path.join(OUTPUTDIR, f'{NAME}-loglikelihood.json'), 'w') as f:
        json.dump({
            'logprob': all_logprobs,
        }, f, indent=2)

    result = {
        f"{NAME}-word-ppl" : np.exp(- np.sum(all_logprobs) / np.sum(all_word_cnt)).item()
    }
    LOGGER.info(f"{NAME}-result", **result)
    return result    