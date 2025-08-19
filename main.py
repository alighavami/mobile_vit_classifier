import argparse
from train import train_loop
from eval import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileViT Plant Disease Classifier")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model-path", type=str, default="outputs/checkpoints/mobilevit_final.pth")

    args = parser.parse_args()
    if args.command == "train":
        train_loop(epochs=args.epochs)
    elif args.command == "eval":
        evaluate(args.model_path)
    else:
        parser.print_help()