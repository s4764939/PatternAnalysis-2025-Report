import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def plot_training_log(csv_file_path, output_path=None):
    """
    Reads a training log from a CSV file and plots the key metrics.

    Args:
        csv_file_path (str): The path to the CSV log file.
        output_path (str, optional): Path to save the plot image. 
                                     If None, the plot is displayed interactively.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        return

    required_columns = ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 
                        'val_precision', 'val_recall', 'val_f1', 'lr']
    
    # Check if all required columns exist in the DataFrame
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: CSV is missing required columns: {', '.join(missing)}", file=sys.stderr)
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training & Validation Metrics', fontsize=20, y=0.98)

    # Plot 1: Training and Validation Loss
    # A decreasing loss shows the model is learning. A diverging validation
    # loss can indicate overfitting.
    axs[0, 0].plot(df['epoch'], df['train_loss'], label='Training Loss', marker='.', linestyle='-')
    axs[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='.', linestyle='--')
    axs[0, 0].set_title('Model Loss', fontsize=14)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot 2: Validation Accuracy
    axs[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='.', color='g')
    axs[0, 1].set_title('Validation Accuracy', fontsize=14)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Plot 3: Validation Classification Metrics
    # Precision, Recall, and F1-Score provide insight into the classifier's
    # performance, which is especially useful for imbalanced datasets.
    axs[1, 0].plot(df['epoch'], df['val_precision'], label='Precision', marker='.')
    axs[1, 0].plot(df['epoch'], df['val_recall'], label='Recall', marker='.')
    axs[1, 0].plot(df['epoch'], df['val_f1'], label='F1-Score', marker='.', linestyle='--')
    axs[1, 0].set_title('Validation Classification Metrics', fontsize=14)
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()

    # Plot 4: Learning Rate Schedule
    axs[1, 1].plot(df['epoch'], df['lr'], label='Learning Rate', marker='.', color='r')
    axs[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Plot successfully saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
    else:
        plt.show()

def main():
    """Main function to parse arguments and run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot training and validation metrics from a CSV log file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'csv_file', 
        type=str, 
        help="Path to the input CSV log file."
    )
    parser.add_argument(
        '--save', 
        type=str, 
        dest='output_path',
        help="Path to save the output plot image (e.g., 'metrics.png').\n"
             "If not provided, the plot will be displayed interactively."
    )
    
    args = parser.parse_args()
    plot_training_log(args.csv_file, args.output_path)

if __name__ == '__main__':
    main()