import os
import argparse
import ijson
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

def create_sample(data_dir, out_path, budgets_mb):
    budgets_bytes = {k: v * 1024 * 1024 for k, v in budgets_mb.items()}
    collected = {k: 0 for k in budgets_bytes}
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    file_mapping = {
        'acts': 'train_acts.json',
        'high': 'train_high_courts.json',
        'supreme': 'train_supreme_court.json',
        'tribunal': 'train_tribunals.json'
    }
    
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for category, filename in file_mapping.items():
            if category not in budgets_bytes:
                continue
                
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found. Skipping.")
                continue
                
            print(f"Sampling from {filename}...")
            with open(filepath, 'rb') as f_in:
                for item in ijson.items(f_in, 'item'):
                    text = item.get('text', '').replace('\n', ' ').strip()
                    if not text:
                        continue
                        
                    f_out.write(text + '\n')
                    collected[category] += len(text.encode('utf-8'))
                    
                    if collected[category] >= budgets_bytes[category]:
                        print(f"Reached byte budget for {category}.")
                        break
                        
    print("Sampling complete. Collected bytes per source:")
    print({k: f"{v / (1024 * 1024):.2f} MB" for k, v in collected.items()})

def train_variants(sample_path, out_dir):
    vocab_sizes = [16000, 24000, 32000, 48000]

    special_tokens = [
        "<|endoftext|>", "<|doc_sep|>", "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ] + [f"<|unused{i}|>" for i in range(10)]

    os.makedirs(out_dir, exist_ok=True)

    def build(size, split_digits):
        tok = Tokenizer(models.BPE())

        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
        if split_digits:
            tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Digits(individual_digits=True),
                byte_level,
            ])
        else:
            tok.pre_tokenizer = byte_level
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=size,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train([sample_path], trainer)
        return tok

    for size in vocab_sizes:
        print(f"Training Vocab Size: {size}")

        build(size, split_digits=False).save(
            os.path.join(out_dir, f"legal_{size}.json"))

        build(size, split_digits=True).save(
            os.path.join(out_dir, f"legal_ds_{size}.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="training_data/raw/Phase_1_Pretraining/Train")
    parser.add_argument("--sample_out", default="training_data/tokenizer_sample.txt")
    parser.add_argument("--model_out", default="checkpoints/tokenizers")
    args = parser.parse_args()
    
    budgets = {'acts': 100, 'supreme': 150, 'high': 50, 'tribunal': 50}
    
    if not os.path.exists(args.sample_out):
        create_sample(args.data_dir, args.sample_out, budgets)
        
    train_variants(args.sample_out, args.model_out)