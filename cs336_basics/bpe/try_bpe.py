from train_bpe import train_bpe
def main():
	input_str = "./data/TinyStoriesV2-GPT4-train.txt"
	# Call the core train_bpe function, passing through additional kwargs (e.g., num_processes)
	vocab, merges = train_bpe(
		input_path=input_str,
		vocab_size=10000,
		special_tokens=["<|endoftext|>"],
		num_processes=4
	)
	print("Vocab:")
	print(vocab)
	print("Merges:")
	print(merges)

if __name__ == "__main__":
	main()