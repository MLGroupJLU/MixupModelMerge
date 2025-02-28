import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems, stream_jsonl

from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp
from utils.load_config import cache_dir
import difflib
# import re
# from process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
# from process_results.math.olympiad.utils import proof_rearrangement_process_results
# from process_results.math.AMPS_Hard.utils import amps_hard_process_results


finetuned_model_backbone_mapping_dict = {
    "WizardLM-7B-V1.0": "llama-7b-hf",
    "WizardLM-7B-V1.0-recovered": "llama-7b-hf", 
    "WizardLM-13B-V1.2": "Llama-2-13b-hf",
    "WizardLM-70B-V1.0": "Llama-2-70b-hf",
    "WizardMath-7B-V1.0": "Llama-2-7b-hf",
    "WizardMath-13B-V1.0": "Llama-2-13b-hf",
    "WizardMath-70B-V1.0": "Llama-2-70b-hf",
    "WizardCoder-Python-7B-V1.0": "CodeLlama-7b-Python-hf",
    "WizardCoder-Python-13B-V1.0": "CodeLlama-13b-Python-hf",
    "WizardCoder-Python-34B-V1.0": "CodeLlama-34b-Python-hf",
    "llama-2-13b-code-alpaca": "Llama-2-13b-hf"
}


def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    # 处理 None 或空字符串
    if not llm_answer:
        print("Error: llm_answer is empty or None")
        return 0  # 或者返回合适的默认值

    # 处理 split 为空的情况
    lines = list(filter(None, llm_answer.split('\n')))
    if not lines:
        print("Error: llm_answer split result is empty")
        return 0  # 或者返回默认值

    llm_answer = lines[-1]  # 取最后一行

    if debug and ground_truth not in llm_answer:
        a = ground_truth
        b = llm_answer
        m = difflib.SequenceMatcher(a=a, b=b)
        pad = 10

        for tag, i1, i2, j1, j2 in m.get_opcodes():
            length = min(len(llm_answer), len(ground_truth))
            mi1, mi2, mj1, mj2 = max(0, i1 - pad), min(length, i2 + pad), max(0, j1 - pad), min(length, j2 + pad)
            if tag == 'replace':
                print("<changed>", a[i1:i2], b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
            if tag == 'delete':
                print("<deleted>", a[i1:i2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
            if tag == 'insert':
                print("<inserted>", b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])

    return int(ground_truth in llm_answer)  # 正确返回 int 类型


def test_typos(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None):
    # 读取parquet格式数据
    df = pd.read_parquet(test_data_path)
    instructions = df['turns'].tolist()
    answers = df['ground_truth'].tolist()
    
    instructions = instructions[start_index:end_index]
    answers = answers[start_index:end_index]
    batch_instructions = batch_data(instructions, batch_size=10)

    stop_tokens = ["The answer is:"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    problem_prompt_typos = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Text with errors:\n{text_with_errors}\n\n"
        "### Response: the corrected paragraph is:"
    )   
    
    res_completions = []
    for idx, prompt in enumerate(batch_instructions):
        if isinstance(prompt, list):
            for id_per, p in enumerate(prompt):
                p = p[0]  # 将array转换为字符串
                prompt_text = p.split('\n\n')[0]  # 提取prompt部分
                text_with_errors = '\n\n'.join(p.split('\n\n')[1:])  # 提取含错误的文本部分
                p = problem_prompt_typos.format(instruction=prompt_text, text_with_errors=text_with_errors)
                prompt[id_per] = p
        else:
            assert isinstance(prompt, list), "Prompt must be a list"
        
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text.strip()
            res_completions.append(generated_text)
    
    match_result = []       
    for llm_answer, answer in zip(res_completions, answers):
        print(f"Debug: llm_answer = {repr(llm_answer)}")
        score = typos_process_results(answer, llm_answer)
        match_result.append(score)
    accuracy = sum(match_result) / len(match_result)
        
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"Language test data length is {len(res_completions)}, accuracy is {accuracy}")
    logger.info(args)

    del llm
    torch.cuda.empty_cache()


def test_instruction_following(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None, model_id="gpt-3.5-turbo"):
    # 读取parquet格式数据
    df = pd.read_parquet(test_data_path)
    instructions = df['turns'].tolist()
    tasks = df['task'].tolist()
    
    instructions = instructions[start_index:end_index]
    tasks = tasks[start_index:end_index]
    
    batch_instructions = batch_data(instructions, batch_size=50)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_instructions):
        if isinstance(prompt, list):
            for id_per, p in enumerate(prompt):
                p = p[0]  # 将array转换为字符串
                prompt[id_per] = p
        else:
            prompt = [prompt]
            
        completions = llm.generate(prompt, sampling_params)
        
        for output in completions:
            generated_text = output.outputs[0].text
            # 处理每个生成的文本，将不必要的部分去掉
            # 第一步：去掉不需要的部分 "The answer is:"
            processed_text = generated_text.replace("The answer is:", "")
            # 第四步：去掉开头和结尾的标点符号
            processed_text = processed_text.strip(".,!?;: ")            
            # 第二步：去掉开头和结尾的换行符
            processed_text = processed_text.strip("\n")           
            # 第三步：去掉开头和结尾的空格
            processed_text = processed_text.strip()
            res_completions.append(processed_text)

    task_outputs = {"summarize": [], "story_generation": [], "paraphrase": [], "simplify": []}

    for i, (response, task) in enumerate(zip(res_completions, tasks)):
        output = {
            "question_id": df.iloc[i]['question_id'],
            "answer_id": "cgnejS7SLoRNiyfaXju8mj",  # placeholder for answer_id
            "model_id": model_id,
            "choices": [{"index": 0, "turns": [response]}],
            "tstamp": 1732707009.808021  # placeholder for timestamp
        }
        if task in task_outputs:
            task_outputs[task].append(output)
        else:
            logger.warning(f"Unexpected task type: {task}")

    # 根据task类型保存到不同的路径
    for task, output_list in task_outputs.items():
        if output_list:
            task_folder = os.path.join(save_gen_results_folder, task, "model_answer")
            if not os.path.exists(task_folder):
                os.makedirs(task_folder)
            output_file = os.path.join(task_folder, f"{model_id}.jsonl")
            write_jsonl(output_file, output_list)

    del llm
    torch.cuda.empty_cache()
    
    
def test_coding(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None, model_id=None):
    # 读取parquet格式数据
    df = pd.read_parquet(test_data_path)
    instructions = df['turns'].tolist()
    tasks = df['task'].tolist()
    
    instructions = instructions[start_index:end_index]
    tasks = tasks[start_index:end_index]
    
    batch_instructions = batch_data(instructions, batch_size=50)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_instructions):
        if isinstance(prompt, list):
            for id_per, p in enumerate(prompt):
                p = p[0]  # 将array转换为字符串
                prompt[id_per] = p
        else:
            prompt = [prompt]
            
        completions = llm.generate(prompt, sampling_params)
        
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    task_outputs = {"coding_completion": [], "LCB_generation": []}

    for i, (response, task) in enumerate(zip(res_completions, tasks)):
        output = {
            "question_id": df.iloc[i]['question_id'],
            "answer_id": "cgnejS7SLoRNiyfaXju8mj",  # placeholder for answer_id
            "model_id": model_id,
            "choices": [{"index": 0, "turns": [response]}],
            "tstamp": 1732707009.808021  # placeholder for timestamp
        }
        if task in task_outputs:
            task_outputs[task].append(output)
        else:
            logger.warning(f"Unexpected task type: {task}")

    # 根据task类型保存到不同的路径
    for task, output_list in task_outputs.items():
        if output_list:
            task_folder = os.path.join(save_gen_results_folder, task, "model_answer")
            if not os.path.exists(task_folder):
                os.makedirs(task_folder)
            output_file = os.path.join(task_folder, f"{model_id}.jsonl")
            write_jsonl(output_file, output_list)

    del llm
    torch.cuda.empty_cache()

def recover_from_pretrained_model(finetuned_model_name, pretrained_model_name, args, logger: logging.Logger, recovered_model_save_path: str, recover_manner: str):
    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name), device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
    except:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir, device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model_name, cache_dir=cache_dir, device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetuned_model_name, cache_dir=cache_dir)

    # set the pad_token of pretrained and finetuned tokenizer
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=pretrained_model,
        tokenizer=pretrained_tokenizer,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
    )

    logger.info(f"recovering {args.finetuned_model_name}...")
    pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
    finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
    recovered_params = {}
    with torch.no_grad():
        if recover_manner == "add":
            for param_name in finetuned_param_dict.keys():
                recovered_params[param_name] = finetuned_param_dict[param_name] + pretrained_param_dict[param_name]
        else:
            raise NotImplementedError(f"not implemented error for recover_manner {recover_manner}!")
    # copy the recovered parameters to the original model
    for param_name, param_value in finetuned_model.named_parameters():
        param_value.data.copy_(recovered_params[param_name])
    logger.info(f"saving recovered {finetuned_model_name} model at {recovered_model_save_path}...")
    os.makedirs(recovered_model_save_path, exist_ok=True)
    finetuned_model.save_pretrained(save_directory=recovered_model_save_path)
    finetuned_tokenizer.save_pretrained(save_directory=recovered_model_save_path)


def create_llm(finetuned_model_name, pretrained_model_name, args, logger: logging.Logger, tensor_parallel_size=1, just_inference=False, save_model_path=None):
    if just_inference:
        if os.path.exists(os.path.join(cache_dir, finetuned_model_name)):
            llm = LLM(model=os.path.join(cache_dir, finetuned_model_name), tensor_parallel_size=tensor_parallel_size)
        else:
            assert os.path.exists(finetuned_model_name)
            llm = LLM(model=finetuned_model_name, tensor_parallel_size=tensor_parallel_size)
        assert save_model_path is None
    else:
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name), device_map="cpu")
            pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
            finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
            finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
        except:
            pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir, device_map="cpu")
            pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
            finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model_name, cache_dir=cache_dir, device_map="cpu")
            finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetuned_model_name, cache_dir=cache_dir)

        # set the pad_token of pretrained and finetuned tokenizer
        # note that WizardMath-70B-V1.0 adds two tokens {"<pad>": 32000, "[PAD]": 32001} with (32002, 8192) token embedding size
        # therefore, for WizardMath-70B-V1.0, we add one distinct pad_token "<pad>[PAD]" to reshape the token embedding size to (32001, 8192)
        if "WizardMath-70B-V1.0" in finetuned_model_name:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>[PAD]"),
                model=pretrained_model,
                tokenizer=pretrained_tokenizer,
            )
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>[PAD]"),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )
        else:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=pretrained_model,
                tokenizer=pretrained_tokenizer,
            )
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )

        # set random seed to guarantee reproducibility
        set_random_seed(seed=0)
        masked_param_dict = mask_model_weights(finetuned_model=finetuned_model, pretrained_model=pretrained_model,
                                               exclude_param_names_regex=[], weight_format=args.weight_format,
                                               weight_mask_rate=args.weight_mask_rate,
                                               use_weight_rescale=args.use_weight_rescale, mask_strategy=args.mask_strategy)
        # copy the masked parameters to the original model
        for param_name, param_value in finetuned_model.named_parameters():
            if param_name in masked_param_dict:
                param_value.data.copy_(masked_param_dict[param_name])

        logger.info(f"saving model at {save_model_path}...")
        os.makedirs(save_model_path, exist_ok=True)
        finetuned_model.save_pretrained(save_directory=save_model_path)
        finetuned_tokenizer.save_pretrained(save_directory=save_model_path)
        logger.info(f"model is saved")
        llm = LLM(model=save_model_path, tensor_parallel_size=tensor_parallel_size)

    return llm


def test_alpaca_eval(llm, finetuned_model_name, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                     save_model_path=None, save_gen_results_folder=None):
    try:
        eval_set = datasets.load_dataset(path=os.path.join(cache_dir, "alpaca_eval"), name="alpaca_eval")["eval"]
        # eval_set = datasets.load_dataset(path="/home/wangxu/MergeLM/alpaca_eval", name="alpaca_eval")["eval"]
    except:
        eval_set = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval", cache_dir=cache_dir)["eval"]

    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"sampling params is {sampling_params}")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    generator_name = save_model_path if save_model_path is not None else finetuned_model_name
    logger.info(f"generator name is {generator_name}")

    for idx, (prompt, reference_output) in enumerate(zip(instructions, reference_outputs)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"

        generated_outputs = []
        prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            generated_outputs.append({
                "instruction": reference_output["instruction"],
                "output": generated_text,
                "generator": generator_name,
                "dataset": reference_output["dataset"]
            })

        write_jsonl(output_file, generated_outputs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.json")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_gsm8k(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None):
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"gsm8k test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start_index:end_index]
    gsm8k_answers = gsm8k_answers[start_index:end_index]
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=60)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            results.append(float(y_pred) == float(prompt_answer))
        else:
            results.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"gsm8k test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_hendrycks_math(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"MATH test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=50)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"MATH test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_human_eval(llm, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None):
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code['completion']
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            code['completion'] = completion
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_mbpp(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None):
    problems = read_mbpp(test_data_path)
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long, we choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)

    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(len(problems))]
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                logger.info("completion matches # Test examples")
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            outputs[task_id - 11].append(completion)

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference LLMs")
    parser.add_argument("--finetuned_model_name", type=str, default="WizardMath-13B-V1.0", help="name of the finetuned language model",
                        choices=["WizardLM-7B-V1.0", "WizardLM-13B-V1.2", "WizardLM-70B-V1.0",
                                 "WizardMath-7B-V1.0", "WizardMath-13B-V1.0", "WizardMath-70B-V1.0",
                                 "WizardCoder-Python-7B-V1.0", "WizardCoder-Python-13B-V1.0", "WizardCoder-Python-34B-V1.0",
                                 "llama-2-13b-code-alpaca"])
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="dataset to be used", choices=["alpaca_eval", "gsm8k", "MATH", "human_eval", "mbpp"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.0, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
    parser.add_argument("--wizardcoder_use_llama2_as_backbone", action="store_true", default=False, help="whether to use llama-2 as the backbone for WizardCoder")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    if args.weight_mask_rate == 0.0:
        save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}"
        save_model_path = None
        just_inference = True
    else:
        save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}_rescale_{args.use_weight_rescale}"
        if args.mask_strategy == "magnitude":
            save_model_name = f"{save_model_name}_strategy_{args.mask_strategy}"
        if args.weight_format == "finetuned_weight":
            save_model_name = f"{save_model_name}_weight_format_{args.weight_format}"
        if args.wizardcoder_use_llama2_as_backbone:
            assert args.finetuned_model_name in ["WizardCoder-Python-7B-V1.0", "WizardCoder-Python-13B-V1.0"]
            if args.finetuned_model_name == "WizardCoder-Python-7B-V1.0":
                finetuned_model_backbone_mapping_dict["WizardCoder-Python-7B-V1.0"] = "Llama-2-7b-hf"
            else:
                finetuned_model_backbone_mapping_dict["WizardCoder-Python-13B-V1.0"] = "Llama-2-13b-hf"
            save_model_name = f"{save_model_name}_llama_2_as_backbone"
        save_model_path = f"./save_models/{args.dataset_name}/{save_model_name}"
        just_inference = False
    if args.dataset_name == "alpaca_eval":
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["human_eval", "mbpp"]:
        save_gen_results_folder = f"./save_gen_codes_results/{args.dataset_name}/{save_model_name}"
    else:
        save_gen_results_folder = None
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./save_logs/{args.dataset_name}/{save_model_name}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./save_logs/{args.dataset_name}/{save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    if args.finetuned_model_name == "WizardLM-7B-V1.0":
        # add the pretrained llama-7b-hf weights to recover WizardLM-7B-V1.0
        recovered_model_save_path = os.path.join(cache_dir, f"{args.finetuned_model_name}-recovered")
        if not os.path.exists(recovered_model_save_path):
            recover_from_pretrained_model(finetuned_model_name=args.finetuned_model_name,
                                          pretrained_model_name=finetuned_model_backbone_mapping_dict[args.finetuned_model_name],
                                          args=args, logger=logger, recovered_model_save_path=recovered_model_save_path,
                                          recover_manner="add")
        args.finetuned_model_name = f"{args.finetuned_model_name}-recovered"

    llm = create_llm(finetuned_model_name=args.finetuned_model_name,
                     pretrained_model_name=finetuned_model_backbone_mapping_dict[args.finetuned_model_name],
                     args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                     just_inference=just_inference, save_model_path=save_model_path)

    if args.dataset_name == "alpaca_eval":
        test_alpaca_eval(llm=llm, finetuned_model_name=args.finetuned_model_name,
                         args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_model_path=save_model_path, save_gen_results_folder=save_gen_results_folder)

    elif args.dataset_name == "gsm8k":
        args.test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger,
                   start_index=args.start_index, end_index=args.end_index, save_model_path=save_model_path)

    elif args.dataset_name == "MATH":
        args.test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, save_model_path=save_model_path)
    elif args.dataset_name == "human_eval":
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_model_path=save_model_path, save_gen_results_folder=save_gen_results_folder)
    else:
        assert args.dataset_name == "mbpp"
        args.test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger,
                  start_index=args.start_index, end_index=args.end_index,
                  save_model_path=save_model_path, save_gen_results_folder=save_gen_results_folder)
    logger.info(f"inference of {args.finetuned_model_name} is completed")

    sys.exit()