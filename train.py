from configs.config import Config
from dataset.loader import build_dataset
from model.build import load_blip2_for_training
from model.trainer import make_trainer


def main():
    cfg = Config()

    print("Loading model...")
    model, processor = load_blip2_for_training(cfg)

    print("Building dataset...")
    dataset = build_dataset(cfg, processor)

    print("Training...")
    trainer = make_trainer(model, dataset, cfg, processor)
    trainer.train()

    print("Saving...")
    trainer.save_model()
    processor.tokenizer.save_pretrained(str(cfg.output_model_dir))
    print(f"Model saved to {cfg.output_model_dir}")


if __name__ == "__main__":
    main()
