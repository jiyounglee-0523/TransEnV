import argparse
from typing import Type, Union
from dataclasses import dataclass, field



@dataclass
class GenerationConfig:
    temperature: float = field(default=0.8, metadata={"help": "temperature value for generation"})
    top_p: float = field(default=0.95, metadata={"help": "top_p value for generation"})
    batch_size: int = field(default=30, metadata={"help": "batch size for generation"})
    max_tokens: int = field(default=2000)
    rerun: str = field(default=None)
    one_transform: bool = field(default=False, metadata={"action": "store_true"})



@dataclass
class ModelConfig:
    model_name: str = field(default="google/gemma-2-27b-it", metadata={"help": "Model to Use", "choices": ['google/gemma-2-27b-it', 'gpt-4o-mini']})
    port_num: int = field(default=8000, metadata={"help": "vLLM port number"})
    tokenizer: str = field(default='google/gemma-2-27b-it')



@dataclass
class DatasetConfig:
    dataset_name: str = field(metadata={"help": "Dataset name", "choices": ['mmlu', 'gsm8k', 'arc', 'hellaswag', 'truthfulqa', 'winogrande']})
    sampling: bool = field(default=False, metadata={"action": "store_true"})

    

@dataclass
class TaskConfig:
    task_name: str = field(default="english_dialect", metadata={"choices": ['english_dialect', 'cefr', 'L1']})
    dialect: str = field(default="Urban African American Vernacular English", metadata={"help": "English Dialect"})
    l1: str = field(default='Arabic', metadata={"choices": ['Arabic', 'French', 'German', 'Italian', 'Japanese', 'Mandarin', 'Portuguese', 'Russian', 'Spanish', 'Turkish']})
    cefr_level: str = field(default="A", metadata={"help": "CEFR Level"})



@dataclass
class SaveConfig:
    save_path: str = field(default="./")
    file_name: str = field(default="tmp.pk", metadata={"help": "save file name"})
    data_path: str = field(default="./")




def add_arguments_from_dataclass(parser: argparse.ArgumentParser, config_class: Type[Union[GenerationConfig, ModelConfig, DatasetConfig, TaskConfig, SaveConfig]]) -> None:
    """
    Dynamically adds arguments from a dataclass to an argparse.ArgumentParser.
    """
    for field_name, field_spec in config_class.__dataclass_fields__.items():
        field_type = field_spec.type
        default = field_spec.default
        help_text = field_spec.metadata.get("help", "")
        if field_type == bool:
            parser.add_argument(
                f"--{field_name}",
                default=default,
                action="store_true" if default is False else "store_false",
                help=help_text,
            )
        else:
            parser.add_argument(f"--{field_name}", type=field_type, default=default, help=help_text)


def filter_args_for_dataclass(args: argparse.Namespace, config_class: Type) -> dict:
    """
    Filters arguments specific to a given dataclass from argparse.Namespace.
    """
    return {key: value for key, value in vars(args).items() if key in config_class.__dataclass_fields__}



def parse_args() -> tuple[GenerationConfig, ModelConfig, DatasetConfig, TaskConfig, SaveConfig]:
    """
    Parses command-line arguments and maps them to multiple dataclass instances.
    """
    parser = argparse.ArgumentParser(description="Combined configuration parser")

    # Add arguments for each configuration class
    add_arguments_from_dataclass(parser, GenerationConfig)
    add_arguments_from_dataclass(parser, ModelConfig)
    add_arguments_from_dataclass(parser, DatasetConfig)
    add_arguments_from_dataclass(parser, TaskConfig)
    add_arguments_from_dataclass(parser, SaveConfig)

    args = parser.parse_args()

    # Instantiate each configuration dataclass from the parsed arguments
    generation_config = GenerationConfig(**filter_args_for_dataclass(args, GenerationConfig))
    model_config = ModelConfig(**filter_args_for_dataclass(args, ModelConfig))
    dataset_config = DatasetConfig(**filter_args_for_dataclass(args, DatasetConfig))
    task_config = TaskConfig(**filter_args_for_dataclass(args, TaskConfig))
    save_config = SaveConfig(**filter_args_for_dataclass(args, SaveConfig))

    return generation_config, model_config, dataset_config, task_config, save_config


