import os
import json
import glob
import argparse
import subprocess

import numpy as np

dict_dataset_path = {
    'arxiv': "<path_to>/tokenized_data/redpajama_v1/arxiv/quip_llama2_7b_e8p_2bit",
    "c4": "<path_to>/tokenized_data/c4/quip_llama2_7b_e8p_2bit",
    "github": "<path_to>/tokenized_data/redpajama_v1/github/quip_llama2_7b_e8p_2bit",
    "stackexchange": "<path_to>/tokenized_data/redpajama_v1/stackexchange/quip_llama2_7b_e8p_2bit",
    "wikipedia": "<path_to>/tokenized_data/redpajama_v1/wikipedia/quip_llama2_7b_e8p_2bit",
    "refinedweb1": "<path_to>/tokenized_data/falcon-refinedweb/quip_llama2_7b_e8p_2bit",
    "refinedweb2": "<path_to>/tokenized_data/falcon-refinedweb/post_processed/quip_llama2_7b_e8p_2bit/",
    "starcoder": "<path_to>/tokenized_data/starcoderdata",
}

def get_bin_len_files(folder_path):
    bin_len_files = glob.glob(os.path.join(folder_path, '**', '*.bin.len'), recursive=True)
    return bin_len_files

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def verify_dataset_path(total_dataset_files, dict_dataset_path, folder_name="quip_llama2_7b_e8p_2bit", file_extension="bin.len"):
    def count_files(folder, file_extension):
        command = f"ls {folder} | grep '{file_extension}' | wc -l"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()

    for dataset_file in total_dataset_files:
        assert os.path.exists(dataset_file)
    print("##### pass existence assertion #####")    

    matching_folders = []
    for dataset in dict_dataset_path:
        dataset_path = dict_dataset_path[dataset]
        if folder_name in dataset_path:
            matching_folders.append(dataset_path)
        else:
            for root, dirs, files in os.walk(dataset_path):
                if folder_name in dirs:
                    matching_folders.append(os.path.join(root, folder_name))

    total_file_count = 0
    for folder in matching_folders:
        file_count = count_files(folder, file_extension)
        total_file_count += int(file_count)

    assert len(total_dataset_files) == total_file_count
    print("##### pass file counts #####")    

def main(args):
    ##### 1. get a list of all "*.bin.len" files #####
    if os.path.exists(os.path.join(args.output_dir, "total_dataset_files.json")):
        with open(os.path.join(args.output_dir, "total_dataset_files.json"), 'r') as file:
            total_dataset_files = json.load(file)
        print("##### total_dataset_files.json loaded #####")
    else:
        total_dataset_files = []

        for dataset in dict_dataset_path:
            print(f"dataset={dataset}")
            dataset_path = dict_dataset_path[dataset]
            dataset_files = get_bin_len_files(dataset_path)
            total_dataset_files += dataset_files

        with open(os.path.join(args.output_dir, "total_dataset_files.json"), 'w') as file:
            json.dump(total_dataset_files, file)
        print("##### total_dataset_files.json saved #####")

    if args.verify:
        verify_dataset_path(total_dataset_files, dict_dataset_path)
        print("##### pass verification #####")

    ##### 2. get categorical distribution #####
    if os.path.exists(os.path.join(args.output_dir, 'list_of_number_of_tokens_in_the_file.npy')) and os.path.exists(os.path.join(args.output_dir, 'categorical_dist.npy')):
        list_of_number_of_tokens_in_the_file = np.load(os.path.join(args.output_dir, 'list_of_number_of_tokens_in_the_file.npy'))
        categorical_dist = np.load(os.path.join(args.output_dir, 'categorical_dist.npy'))
        print("##### list_of_number_of_tokens_in_the_file.json loaded #####")
        print("##### categorical_dist.json loaded #####")
    else:
        list_of_number_of_tokens_in_the_file = []
        for i, dataset_file in enumerate(total_dataset_files):
            if i % 500 == 0:
                print(f"##### current json loading [{i}/{len(total_dataset_files)}]#####")
            data = load_json(dataset_file)
            list_of_number_of_tokens_in_the_file.append(data['number_of_tokens_in_the_file'])
            
        list_of_number_of_tokens_in_the_file = np.array(list_of_number_of_tokens_in_the_file)
        categorical_dist = list_of_number_of_tokens_in_the_file / np.sum(list_of_number_of_tokens_in_the_file)

        np.save(os.path.join(args.output_dir, 'list_of_number_of_tokens_in_the_file.npy'), list_of_number_of_tokens_in_the_file)
        np.save(os.path.join(args.output_dir, 'categorical_dist.npy'), categorical_dist)

        print("##### list_of_number_of_tokens_in_the_file.json saved #####")
        print("##### categorical_dist.json saved #####")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action="store_true")
    parser.add_argument('--output_dir', type=str, default="/scratch/yk2516/repos/PAC_Bayes/token-bounds-llms/data/llm360")

    args = parser.parse_args()

    main(args)

