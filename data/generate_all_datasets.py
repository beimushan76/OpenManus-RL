import argparse
from generate_sft_verl import main as generate_sft
from generate_train_agentgym_all import main as generate_rl


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for SFT and GRPO/PPO training")
    parser.add_argument('--sft_output_dir', required=True,
                        help='Directory for the SFT dataset')
    parser.add_argument('--rl_output_dir', required=True,
                        help='Base directory for RL datasets (one subdir per env)')
    parser.add_argument('--sft_valid_ratio', type=float, default=0.1,
                        help='Validation ratio for the SFT dataset')
    args = parser.parse_args()

    print('Generating SFT dataset...')
    generate_sft(args.sft_output_dir, args.sft_valid_ratio)

    print('\nGenerating RL dataset...')
    generate_rl(args.rl_output_dir)


if __name__ == '__main__':
    main()
