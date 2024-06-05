import os
import csv
import json
import torch
import numpy as np

from .data_module import TextDatasetQA, custom_data_collator, get_batch_loss
from .evaluate_util import eval_rouge_recall
from .utils import get_model_utility, get_forget_quality, get_forget_quality_func
from ..utils import set_progress

def prepare_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=custom_data_collator 
    )
    return loader


def prepare_dataset(dataname, tokenizer, conv_template, split, question_key, answer_key, max_num=-1):
    dataset = TextDatasetQA(
        data_path=dataname,
        tokenizer=tokenizer,
        conv_template=conv_template, 
        split=split,
        question_key=question_key,
        answer_key=answer_key,
        max_num=max_num
    )
    return dataset


def tofu_eval(OUTPUTDIR, LOGGER, configs, model, tokenizer, right_pad_tokenizer, conv_template, only_forget_quality=False):
    progress = set_progress(disable=os.getenv("POOR", False)) 
    with progress:
        if not only_forget_quality:
            eval_tasks = [
                configs.dataset.split, 
                "retain_perturbed", 
                "real_authors_perturbed", "world_facts_perturbed", 
            ]
        else:
            eval_tasks = [
                configs.dataset.split, "retain_perturbed", 
            ]

        eval_task = progress.add_task(
            "evalbar",
            name="[green][Main Evaluate]",
            total=len(eval_tasks),
        )

        for eval_split in eval_tasks:
            task_name = eval_split if eval_split != configs.dataset.split else "eval_log_forget"
            question_key = "question"
            answer_key = "answer"
            base_answer_key = "answer" if eval_split in ["real_authors_perturbed", "world_facts_perturbed"] else "paraphrased_answer"
            perturbed_answer_key = "perturbed_answer"
            batch_size = configs.dataset.eval.batch_size
            print("Batch size", batch_size)
            MAX_NUM = 300 #! Tofu official implementation only use first 300

            eval_logs = {}
            #! evaluate generations
            gen_outputs = []
            ground_truths = []
            input_strings = []
            eval_dataset = prepare_dataset(
                configs.dataset.name, tokenizer, conv_template, eval_split, question_key, answer_key, max_num=MAX_NUM
            )
            eval_dataloader = prepare_loader(
                eval_dataset, batch_size,
            )
            with torch.no_grad():
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-generation]", name=f"{task_name}", total=len(eval_dataloader),
                )
                def batch_generator(tfdataset, batch_size): #! we only need question and answer for eval_dataset
                    for i in range(0, len(tfdataset), batch_size):
                        batchdata = tfdataset.select(range(i, min(i + batch_size, len(tfdataset))))
                        keys = ["question", "answer"]
                        out = ()
                        for k in keys:
                            out += ([item[k] for item in batchdata],)
                        yield out

                for batch in batch_generator(eval_dataset.data, batch_size):
                    questions, answers = batch
                    gen_inputs = [
                        conv_template.prepare_gen_prompt(question, answer) for question, answer in zip(questions, answers)
                    ]
                    inputs = tokenizer(
                        gen_inputs, add_special_tokens=True, return_tensors="pt", padding=True, 
                    ).to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_length=configs.dataset.eval.generation.max_length,
                        max_new_tokens=configs.dataset.eval.generation.max_new_tokens, 
                        do_sample=False, 
                        use_cache=True, 
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    out_strs = tokenizer.batch_decode(
                        outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

                    # print(gen_inputs, "\n", out_strs)
                    gen_outputs.extend(out_strs)
                    input_strings.extend(gen_inputs)
                    ground_truths.extend(answers)

                    # LOGGER.info("Generation", input=gen_inputs, output=out_strs)
                    progress.advance(gen_task) #? update progress

            torch.cuda.empty_cache()
            rougeL = eval_rouge_recall(gen_outputs, ground_truths)
            eval_logs.update(rougeL)
            eval_logs['generated_text'] = list(zip(input_strings, gen_outputs, ground_truths))

            LOGGER.info("GenerationResult", generationout=np.mean(rougeL['rougeL_recall']))

            #! evaluate next-token probs
            eval_dataset = prepare_dataset(
                configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, answer_key, max_num=MAX_NUM
            )
            eval_dataloader = prepare_loader(
                eval_dataset, batch_size,
            )
            with torch.no_grad():
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-nexttoken]", name=f"{task_name}", total=len(eval_dataloader),
                )
                for batch in eval_dataloader:
                    input_ids, labels, attention_mask = batch
                    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
                    for k, v in batch.items():
                        batch[k] = v.to(model.device)
                    outputs = model(**batch) #! forward to get logits
                    gt_loss = get_batch_loss(outputs.logits, batch['labels'])
                    num_token_gt = (batch['labels'] != -100).sum(-1)
                    eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss / num_token_gt).cpu().numpy().tolist()
                    eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
                    eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()
                    progress.advance(gen_task) #? update progress

            #! evaluate ratio

            base_eval_dataloader = prepare_loader(
                prepare_dataset(
                    configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, base_answer_key, max_num=MAX_NUM
                ),
                max(1, batch_size // 4),
            )
            perturb_dataloader = prepare_loader(
                prepare_dataset(
                    configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, perturbed_answer_key, max_num=MAX_NUM
                ),
                max(1, batch_size // 4),
            )

            with torch.no_grad():
                tmp_logs = {}
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-perturb_ratio]", name=f"{task_name}", total=len(eval_dataloader),
                )
                for batch, perturb_batch in zip(base_eval_dataloader, perturb_dataloader):
                    input_ids, labels, attention_mask = batch
                    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
                    perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
                    if len(perturb_input_ids.shape) > 2:
                        bsz, seq_len = perturb_input_ids.shape[0:2]
                    else:
                        bsz = perturb_input_ids.shape[0]
                        seq_len = 1
                    perturb_batch = {
                        "input_ids": perturb_input_ids.view(bsz*seq_len, -1), 
                        "labels": perturb_labels.view(bsz*seq_len, -1), 
                        "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)
                    }

                    #send to device
                    for k, v in batch.items():
                        batch[k] = v.to(model.device)
                    for k, v in perturb_batch.items():
                        perturb_batch[k] = v.to(model.device)

                    outputs = model(**batch, use_cache=False)
                    outputs.logits = outputs.logits.cpu()
                    torch.cuda.empty_cache()
                    perturb_outputs = model(**perturb_batch, use_cache=False)
                    perturb_outputs.logits = perturb_outputs.logits.cpu()
                    torch.cuda.empty_cache()

                    gt_loss = get_batch_loss(outputs.logits, batch['labels'].cpu())
                    perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels'].cpu()).view(bsz, seq_len)
                    num_token_gt = (batch['labels']!=-100).sum(-1).cpu()
                    num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1).cpu()
                    mean_perturb_loss = perturb_loss.mean(dim=1)
                    ratio = (mean_perturb_loss - gt_loss).mean()

                    tmp_logs['average_perturb_loss'] = tmp_logs.get('average_perturb_loss', []) + (perturb_loss/num_token_perturb).tolist()
                    tmp_logs['avg_paraphrased_loss'] = tmp_logs.get('avg_paraphrased_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
                    tmp_logs['paraphrased_loss'] = tmp_logs.get('paraphrased_loss', []) + gt_loss.tolist()
                    tmp_logs['perturb_loss'] = tmp_logs.get('perturb_loss', []) + perturb_loss.tolist()
                    tmp_logs['num_token_paraphrased'] = tmp_logs.get('num_token_paraphrased', []) + num_token_gt.tolist()
                    tmp_logs['num_token_perturb'] = tmp_logs.get('num_token_perturb', []) + num_token_perturb.tolist()
                    del outputs, perturb_outputs, batch, perturb_batch
                    torch.cuda.empty_cache()
                    progress.advance(gen_task) #? update progress

                eval_logs.update(tmp_logs)
                if eval_split == configs.dataset.split:
                    retain_result = json.load(open(configs.dataset.eval.retain_result, 'r'))['eval_log_forget.json']
                    forget_quality = get_forget_quality_func(eval_logs, retain_result)
                    avg_true_prob = np.exp(-1 * np.array(eval_logs['avg_gt_loss']))
                    avg_false_prob = np.exp(-1 * np.array(eval_logs['average_perturb_loss']))
                    avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
                    avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
                    gt_probs = np.exp(-1 * np.array(eval_logs['avg_gt_loss']))
                    LOGGER.info("ForgetResult", 
                                forget_quality=forget_quality['Forget Quality'], forget_proba=avg_gt_prob,
                    )
                    tmp_logs['forget truth ratio'] = forget_quality['Forget Truth Ratio']

                #! Save intermediate results
                eval_logs.update(tmp_logs)
                save_name = os.path.join(OUTPUTDIR, f"{task_name}.json")
                with open(save_name, "w") as f:
                    json.dump(eval_logs, f, indent=2)
                progress.advance(eval_task) #? update progress
 
        #! Final result
        aggregated_logs = {}
        for eval_split in eval_tasks:
            task_name = eval_split if eval_split != configs.dataset.split else "eval_log_forget"
            eval_logs = json.load(open(os.path.join(OUTPUTDIR, f"{task_name}.json"), 'r'))
            aggregated_logs[f"{task_name}.json"] = eval_logs

        model_utility = get_model_utility(aggregated_logs)
        retain_result = json.load(open(configs.dataset.eval.retain_result, 'r'))
        forget_quality = get_forget_quality(aggregated_logs, retain_result)
        forget_quality.pop('Forget Truth Ratio')
        aaggregate_stat = {**model_utility, **forget_quality}

        #! Save final result 
        with open(os.path.join(OUTPUTDIR, "aggregate_stat.csv"), 'w') as csvfile:
            field_names = list(aaggregate_stat.keys())
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerow(aaggregate_stat)
