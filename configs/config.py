from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Model IDs
    base_model_id: str = "Salesforce/blip2-flan-t5-xl"
    t5_model_id: str = "google/flan-t5-xl"
    prompt_model_id: str = "Qwen/Qwen-1_8B"
    image_model_id: str = "dreamlike-art/dreamlike-photoreal-2.0"
    qa_model_id: str = "llava-hf/llava-1.5-7b-hf"
    trained_model_id: str = "./model/finetuned-blip2-flan-t5-xl"

    # Paths
    generated_dir: Path = field(default_factory=lambda: Path("./data/generated"))
    given_dir: Path = field(default_factory=lambda: Path("./data/given"))
    real_dir: Path = field(default_factory=lambda: Path("./data/real"))
    output_model_dir: Path = field(default_factory=lambda: Path("./model/finetuned-blip2-flan-t5-xl"))
    submission_save_path: str = "./test_inference_final.csv"

    # Dataset generation
    num_prompt_generations: int = 1000
    use_real_data: bool = True
    num_real_images: int = 3000
    balance_answer_dist: bool = False
    categories: list = field(default_factory=lambda: [
        "- Nature (e.g. landscape, animal, weather, plants)\n",
        "- Travel (e.g. tourist spots, local streets, vehicles, airports)\n",
        "- Casual (e.g. daily life, work, family, kids, friends, school, sports)\n",
        "- Food (e.g. meals, cafes, snacks, fruits, drinks)\n",
    ])

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: list = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    logging_steps: int = 10
    save_steps: int = 200
    learning_rate: float = 5e-5
    save_total_limit: int = 3
    input_max_length: int = 384
    target_max_length: int = 128
