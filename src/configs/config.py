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
