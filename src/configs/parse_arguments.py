import argparse
from typing import Type, Union

from configs import *



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
