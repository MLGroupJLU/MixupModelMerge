import argparse
import sys
import logging
import os
import time
from vllm import LLM, SamplingParams

from inference_llms_instruct_math_code import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp, test_typos, test_zebra_puzzle, test_data, test_connection, test_math_live, test_instruction_following, test_coding

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for direct inference merged LLMs")
    parser.add_argument('--load_model_path', type=str, default="/root/autodl-tmp/save_merge_models/math_code_0.62/math/ties_merging", help="path to load the model")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="numbers of gpus to use")
    parser.add_argument("--evaluate_task", type=str, default="instruction_following", choices=["instruction_following", "coding","typos"], help="task to be evaluated")
    parser.add_argument("--model_id", type=str, default="Merged_model", help="ID of the model to use")
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()
    
    #--------------------------------logger------------------ -----------#
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = "./logs"
    os.makedirs(save_merge_log_path, exist_ok=True)

    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
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
    #--------------------------------------------------------------------#

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    llm = LLM(model=args.load_model_path, tensor_parallel_size=args.tensor_parallel_size)

    elif args.evaluate_task == "typos": #language数据集,可以
        logger.info(f"evaluating model on language task...")
        # save_gen_results_folder = "./save_gen_language_results/direct_inference"
        test_data_path = "language/data/test-00000-of-00001.parquet"
        test_typos(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                     start_index=0, end_index=50,
                     save_model_path=None)
        
    elif args.evaluate_task == "instruction_following": #livebench
        logger.info(f"evaluating model on instruction_following task...")
        save_gen_results_folder = "./save_gen_instruction_following_results/instruction_following"
        test_data_path = "instruction_following/data/test-00000-of-00001.parquet"
        test_instruction_following(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                        start_index=0, end_index=sys.maxsize,
                        save_model_path=None, model_id=args.model_id, save_gen_results_folder=save_gen_results_folder)
        
    elif args.evaluate_task == "coding": #livebench
        logger.info(f"evaluating model on coding task...")
        save_gen_results_folder = "./save_gen_coding_following_results/direct_inference"
        test_data_path = "coding/data/test-00000-of-00001.parquet"
        test_coding(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                        start_index=0, end_index=sys.maxsize,
                        save_model_path=None, model_id=args.model_id, save_gen_results_folder=save_gen_results_folder)

    logger.info(f"inference completed")

    sys.exit()