import promptbench as pb

from promptbench.models import LLMModel
from promptbench.prompt_attack import Attack

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"

import logging
import datetime
import torch
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PromptBench Attack Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--attack_method', type=str, required=True, help='Attack method to use')
    args = parser.parse_args()

    # 创建logs文件夹(如果不存在)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 生成日志文件名,包含时间戳
    log_filename = f"logs/prompt_attack_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create model
    logger.info("Creating model...")
    llm = LLMModel(model=args.model_path, parallel_size=2)
    
    # create dataset
    logger.info("Loading dataset...")
    dataset = pb.DatasetLoader.load_dataset("sst2")

    # try part of the dataset
    # dataset = dataset[:2]
    logger.info(f"Using {len(dataset)} examples from dataset")

    # create prompt
    prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'.\nText:{content}\nPlease respond with 'positive' or 'negative'."
    logger.info(f"sst2 test prompt is {prompt}")

    # define the projection function required by the output process
    def proj_func(pred):
        mapping = {
            "positive": 1,
            "negative": 0
        }
        return mapping.get(pred, -1)

    # define the evaluation function required by the attack
    # if the prompt does not require any dataset, for example, "write a poem", you still need to include the dataset parameter
    def eval_func(prompt, dataset, model):
        preds = []
        labels = []
        for d in tqdm(dataset, desc="Processing SST-2 Dataset"):
            input_text = pb.InputProcess.basic_format(prompt, d)
            raw_output = model(input_text)

            output = pb.OutputProcess.cls(raw_output, proj_func)
            preds.append(output)

            labels.append(d["label"])
        
        accuracy = pb.Eval.compute_cls_accuracy(preds, labels)
        
        logger.info(f"Current prompt is: {prompt}")
        logger.info(f"Current accuracy is: {accuracy}")
        
        return accuracy

            
    # define the unmodifiable words in the prompt
    # for example, the labels "positive" and "negative" are unmodifiable, and "content" is modifiable because it is a placeholder
    # if your labels are enclosed with '', you need to add \' to the unmodifiable words (due to one feature of textattack)
    unmodifiable_words = ["positive\'", "negative\'", "content"]
    logger.info(f"Unmodifiable words: {unmodifiable_words}")

    # print all supported attacks
    attacks = Attack.attack_list() #"textbugger", "deepwordbug", "textfooler", "bertattack", "checklist", "stresstest", "semantic"
    logger.info(f"Supported attacks: {attacks}")

    # create attack, specify the model, dataset, prompt, evaluation function, and unmodifiable words
    logger.info("Creating attack...")
    attack = Attack(llm, args.attack_method, dataset, prompt, eval_func, unmodifiable_words, verbose=True)

    # print attack result
    result = attack.attack()
    logger.info(f"Attack result: {result}")

    # 清理资源
    if 'llm' in locals():
        del llm
    torch.cuda.empty_cache()