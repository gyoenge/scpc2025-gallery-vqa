from configs.config import Config
from dataset.generate.prompts import generate_prompts
from dataset.generate.images import generate_images
from dataset.generate.qa_pairs import generate_qa_pairs
from dataset.generate.real_qa import download_coco_images, generate_real_qa_pairs


def main():
    cfg = Config()
    cfg.generated_dir.mkdir(parents=True, exist_ok=True)

    prompt_csv = cfg.generated_dir / "scene_prompt.csv"
    if not prompt_csv.exists():
        print("Step 1/3: Generating scene prompts...")
        generate_prompts(cfg)
    else:
        print(f"Step 1/3: Skipping (found {prompt_csv})")

    qa_csv = cfg.generated_dir / "question_answer.csv"
    if not qa_csv.exists():
        print("Step 2/3: Generating images...")
        generate_images(cfg)
        print("Step 3/3: Generating QA pairs...")
        generate_qa_pairs(cfg)
    else:
        print(f"Steps 2-3/3: Skipping (found {qa_csv})")

    if cfg.use_real_data:
        real_qa_csv = cfg.real_dir / "real_question_answer.csv"
        if not real_qa_csv.exists():
            print("Step 4/4: Downloading real images and generating QA pairs...")
            download_coco_images(cfg)
            generate_real_qa_pairs(cfg)
        else:
            print(f"Step 4/4: Skipping (found {real_qa_csv})")


if __name__ == "__main__":
    main()
