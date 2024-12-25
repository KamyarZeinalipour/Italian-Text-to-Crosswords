# Italian Crossword Clue Generator

This project provides a Python script to generate Italian crossword clues using language models from Hugging Face. The script supports two models:

- **Llama3-8B-ITA-Text-to-Cross**: `Kamyar-zeinalipour/Llama3-8B-ITA-Text-to-Cross`
- **Mistral-7B-ITA-Text-to-Cross**: `Kamyar-zeinalipour/Mistral-7B-ITA-Text-to-Cross`

The user can choose between different "clue types" to generate diverse styles of crossword clues.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Input File Format](#input-file-format)
- [Output File](#output-file)
- [Available Clue Types](#available-clue-types)
- [Notes](#notes)

---

## Prerequisites

- **Python 3.7 or higher**
- **CUDA-compatible GPU (optional but recommended)**
  - The script uses GPU acceleration if available. If no GPU is present, you can adjust the code to run on CPU (note that performance may be significantly slower).

---

## Installation

1. **Clone the Repository or Copy the Script**

   You can clone the repository or copy the script `generate_crossword_clues.py` to your local machine.

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   Install the necessary Python packages using `pip`:

   ```bash
   pip install transformers pandas argparse
   ```

   - **transformers**: For loading language models and tokenizers.
   - **pandas**: For handling CSV files.
   - **argparse**: For parsing command-line arguments (usually included in the Python standard library).

4. **Ensure You Have Access to the Models**

   - The script downloads models from Hugging Face Hub if they are not already present.
   - Ensure you have an internet connection, or download the models ahead of time.

---

## Usage

Run the script from the command line with the required arguments.

```bash
python generate_crossword_clues.py --input-file INPUT_FILE --output-file OUTPUT_FILE --model-name MODEL_NAME --clue-type CLUE_TYPE
```

### Command-Line Arguments

- `--input-file`: **(Required)** Path to the input CSV file containing the text and keywords.
- `--output-file`: Path to save the output CSV file. Defaults to `output.csv` if not specified.
- `--model-name`: **(Required)** Name or path of the model to use. Options:
  - `"Kamyar-zeinalipour/Llama3-8B-ITA-Text-to-Cross"`
  - `"Kamyar-zeinalipour/Mistral-7B-ITA-Text-to-Cross"`
- `--clue-type`: **(Required)** Type of clue to generate. Options:
  - `bare_noun_phrase`
  - `definite_article_phrase`
  - `clitic`
  - `copular_sentence`
- `--temperature`: Temperature for text generation. Defaults to `0.1`.

### Examples

**Example 1: Using the Llama3 Model with the 'bare_noun_phrase' Clue Type**

```bash
python generate_crossword_clues.py \
    --input-file input.csv \
    --output-file output.csv \
    --model-name "Kamyar-zeinalipour/Llama3-8B-ITA-Text-to-Cross" \
    --clue-type bare_noun_phrase
```

**Example 2: Using the Mistral Model with the 'copular_sentence' Clue Type**

```bash
python generate_crossword_clues.py \
    --input-file input.csv \
    --output-file output.csv \
    --model-name "Kamyar-zeinalipour/Mistral-7B-ITA-Text-to-Cross" \
    --clue-type copular_sentence
```

**Example 3: Specifying a Different Temperature**

```bash
python generate_crossword_clues.py \
    --input-file input.csv \
    --output-file output.csv \
    --model-name "Kamyar-zeinalipour/Llama3-8B-ITA-Text-to-Cross" \
    --clue-type clitic \
    --temperature 0.7
```

---

## Input File Format

The **input CSV file** must contain the following columns:

- `text`: The Italian text related to the keyword.
- `keyword`: The keyword for which the crossword clue is to be generated.

**Example `input.csv`:**

| text                                              | keyword  |
|---------------------------------------------------|----------|
| "Il colosseo è un antico anfiteatro a Roma."      | Colosseo |
| "La Torre di Pisa è famosa per la sua inclinazione." | Torre di Pisa |
| "Leonardo da Vinci è noto per la Gioconda."         | Leonardo da Vinci |

---

## Output File

The **output CSV file** will contain:

- `text`: The original text from the input.
- `keyword`: The keyword from the input.
- `Generated Italian Crossword Clue`: The generated crossword clue.
- `Error` (optional): Any error messages encountered during processing.

**Example `output.csv`:**

| text                                              | keyword            | Generated Italian Crossword Clue                                   | Error |
|---------------------------------------------------|--------------------|--------------------------------------------------------------------|-------|
| "Il colosseo è un antico anfiteatro a Roma."      | Colosseo           | "antico anfiteatro romano"                                         |       |
| "La Torre di Pisa è famosa per la sua inclinazione." | Torre di Pisa      | "monumento celebre per la pendenza"                                |       |
| "Leonardo da Vinci è noto per la Gioconda."         | Leonardo da Vinci  | "genio rinascimentale autore di capolavori"                        |       |

---

## Available Clue Types

Select the clue type using the `--clue-type` argument. The following clue types are available:

1. **bare_noun_phrase**
   - Generates clues with the syntax of a noun phrase without a determiner.
   - **Example**: "genio rinascimentale autore di capolavori"

2. **definite_article_phrase**
   - Generates clues starting with a definite article followed by a noun and possibly adjectives.
   - **Example**: "Il genio del Rinascimento autore della Gioconda"

3. **clitic**
   - Uses clitics, possessives, or demonstrative pronouns if the keyword is not the subject.
   - **Example**: "Celebre per la sua pendenza"

4. **copular_sentence**
   - Generates clues in the form of copular sentences without the subject.
   - **Example**: "È un antico anfiteatro romano"

---

## Notes

- **Model Download**: If the specified model is not present locally, it will be downloaded from Hugging Face Hub. Ensure you have a stable internet connection.
  
- **GPU Usage**: The script is optimized for running with a CUDA-compatible GPU. If you do not have a GPU, you can modify the script:
  - Remove `.cuda()` calls to run on CPU.
  - Be aware that running on CPU may significantly increase processing time.

- **Adjusting Generation Parameters**: You can fine-tune the output by adjusting parameters in the `get_code_completion` function within the script:
  - `max_new_tokens`: Maximum number of tokens to generate.
  - `temperature`: Controls randomness. Lower values make the output more deterministic.
  - `top_k`, `top_p`: Controls the sampling strategy.

- **Error Handling**: If an error occurs during processing (e.g., model inference fails), the script will log the error in the output CSV under the `Error` column.

- **Input Data Quality**: The quality of the generated clues heavily depends on the input text and keyword. Ensure that the input data is clean and properly formatted.

- **Extending Clue Types**: You can add more clue types by modifying the `prompts` dictionary in the script. Each clue type must have a unique key and a corresponding prompt template.

---

## Contact and Support

If you encounter any issues or have questions, feel free to reach out to the maintainers or contributors of the project.

---

## License

This project is licensed under the [MIT License](LICENSE.txt).

---

## Acknowledgments

- **Hugging Face Transformers**: For providing the tools to work with state-of-the-art language models.

---

Happy puzzling!
