import argparse
from src.pipelines.train import TrainPipeline
from src.utils.utils import Config, timeit


@timeit
def main(task):
    conf = Config().load_config()
    if task == "train":
        pipeline = TrainPipeline(config=conf)
    else:
        raise ValueError(f"Unknown task '{task}'. Supported tasks: 'train', 'inference'")
    
    print(pipeline)
    pipeline.execute()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline tasks")
    parser.add_argument(
        "--task", type=str, required=True, help="Task to run: 'train' or  'inference'"
    )
    args = parser.parse_args()
    print(args)
    main(args.task)