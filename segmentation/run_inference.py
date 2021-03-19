import os
import json
import argparse

from trainers.inference import InferenceAgent
from trainers.meta_inference import MetaInferenceAgent


AGENTS = {
    "inference": InferenceAgent,
    "meta_inference": MetaInferenceAgent
}


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to run inference.")
    parser.add_argument("-d", "--data_path",
                        type=str,
                        required=True,
                        help="Path to directory containing datasets.")
    parser.add_argument("--data_map",
                        type=str,
                        default=None,
                        help="Path to .json file with train/val/test split for experiment.")
    parser.add_argument("-c", "--model",
                        type=str,
                        required=True,
                        help="Path to .json model config file.")
    parser.add_argument("--trainer",
                        type=str,
                        required=True,
                        help="Path to .json trainer config file.")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to load model weights from checkpoint file.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json index->class name file.")
    parser.add_argument("-s", "--set_type",
                        type=str,
                        default="val",
                        help="One of the train/val/test sets to perform inference.")
    parser.add_argument("--out_dir",
                        type=str,
                        default=None,
                        help="Path to output directory to store inference results. " + \
                             "Defaults to `.../data_path/inference/`")
    parser.add_argument('-n', '--name',
                        type=str,
                        default=None,
                        help='Experiment name, used as directory name in inf_dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_arguments()

    set_type = args.set_type.lower()
    assert set_type in {"train", "val", "test"}, "Only train/val/test sets permitted."

    # Classes is dict of {class_name --> class_id}
    with open(args.classes, 'r') as f:
        classes = json.load(f)

    # Contains config parameters for model
    with open(args.model, 'r') as f:
        model_config = json.load(f)

    # Contains config parameters for trainer
    with open(args.trainer, 'r') as f:
        trainer_config = json.load(f)

    agent_name = trainer_config.get("inference_agent", "inference")
    agent = AGENTS[agent_name]
    inference_agent = agent.create_inference_agent(
        args.data_path,
        args.data_map,
        args.out_dir,
        args.name,
        trainer_config,
        model_config,
        classes,
        args.checkpoint
    )

    # Run inference
    inference_agent.infer(set_type)

    print(f"Inference complete for set {set_type}!")
