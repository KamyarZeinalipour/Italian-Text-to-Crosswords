from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import re

# Define the prompts with names
prompts = {
    'bare_noun_phrase': '''
Sei un esperto di cruciverba.
Genera un indizio conciso e intelligente in italiano per cruciverba educativi basato su una Parola Chiave specificata e la sua relazione con un Testo assegnato. Per eseguire correttamente questo compito, segui le linee guida seguenti:
PAROLA CHIAVE: {keyword}
TESTO: {text}

Osserva i seguenti passaggi:
1. Sostituisci ogni pronome nel testo con frasi complete che esprimano i loro referenti.
2. Dividi il testo in piccole frasi indipendenti che potrebbero essere comprese fuori contesto.
3. Individua le frasi concise che contengono la Parola Chiave e caratterizzano meglio la parola chiave. Cerca di selezionare frasi da diverse parti del Testo.
4. Genera un indizio breve e intelligente per il cruciverba in italiano dalle frasi selezionate. Assicurati che la parola chiave rimanga assente dall'indizio. Ogni indizio deve avere la sintassi di una frase nominale senza determinante: il nodo radice di ogni indizio deve essere un nome comune o proprio e può essere seguito da una proposizione relativa o altri complementi o aggiunti. Genera indizi da tutte le parti del testo e usa tutte le informazioni fornite per generare l'indizio.
5. Assicurati che l'indizio funzioni come una descrizione o definizione della parola chiave piuttosto che una domanda, concentrandoti sui dettagli della parola chiave.
6. Assicurati che le informazioni dell'indizio possano essere ricondotte al testo. Assicurati che l'indizio sia pertinente e sufficiente per identificare la parola chiave. Assicurati che la parola chiave non appaia nell'indizio. Assicurati che nessuna parte della parola chiave sia presente nell'indizio.
''',
    'definite_article_phrase': '''
Sei un esperto di cruciverba.
Genera un indizio conciso e intelligente in italiano per cruciverba educativi basato su una Parola Chiave specificata e la sua relazione con un Testo assegnato. Per eseguire correttamente questo compito, segui le linee guida seguenti:
PAROLA CHIAVE: {keyword}
TESTO: {text}

Osserva i seguenti passaggi:
1. Sostituisci ogni pronome nel testo con frasi complete che esprimano i loro referenti.
2. Dividi il testo in piccole frasi indipendenti che potrebbero essere comprese fuori contesto.
3. Individua le frasi concise che contengono la Parola Chiave e caratterizzano meglio la parola chiave. Cerca di selezionare frasi da diverse parti del Testo.
4. Genera un indizio breve e intelligente per il cruciverba in italiano dalle frasi selezionate. Assicurati che la parola chiave rimanga assente dall'indizio. Ogni indizio deve avere la sintassi di una frase con articolo determinativo (seguito da un nome e possibilmente aggettivi). Può essere seguito da una proposizione relativa o altri complementi o aggiunti. Genera indizi da tutte le parti del testo e usa tutte le informazioni fornite per generare l'indizio.
5. Assicurati che l'indizio funzioni come una descrizione o definizione della parola chiave piuttosto che una domanda, concentrandoti sui dettagli della parola chiave.
6. Assicurati che le informazioni dell'indizio possano essere ricondotte al testo. Assicurati che l'indizio sia pertinente e sufficiente per identificare la parola chiave. Assicurati che la parola chiave non appaia nell'indizio. Assicurati che nessuna parte della parola chiave sia presente nell'indizio.
''',
    'clitic': '''
Sei un esperto di cruciverba.
Genera un indizio conciso e intelligente in italiano per cruciverba educativi basato su una Parola Chiave specificata e la sua relazione con un Testo assegnato. Per eseguire correttamente questo compito, segui le linee guida seguenti:
PAROLA CHIAVE: {keyword}
TESTO: {text}

Osserva i seguenti passaggi:
1. Sostituisci ogni pronome nel testo con frasi complete che esprimano i loro referenti.
2. Dividi il testo in piccole frasi indipendenti che potrebbero essere comprese fuori contesto.
3. Individua le frasi concise che contengono la Parola Chiave e caratterizzano meglio la parola chiave. Cerca di selezionare frasi da diverse parti del Testo.
4. Genera un indizio breve e intelligente per il cruciverba in italiano dalle frasi selezionate. Assicurati che la parola chiave rimanga assente dall'indizio. Se la Parola Chiave non è il soggetto della frase, assicurati che sia sostituita con un adeguato clitico, possessivo o pronome dimostrativo. Genera indizi da tutte le parti del testo e usa tutte le informazioni fornite per generare l'indizio.
5. Assicurati che l'indizio funzioni come una descrizione o definizione della parola chiave piuttosto che una domanda, concentrandoti sui dettagli della parola chiave.
6. Assicurati che le informazioni dell'indizio possano essere ricondotte al testo. Assicurati che l'indizio sia pertinente e sufficiente per identificare la parola chiave. Assicurati che la parola chiave non appaia nell'indizio. Assicurati che nessuna parte della parola chiave sia presente nell'indizio.
''',
    'copular_sentence': '''
Sei un esperto di cruciverba.
Genera un indizio conciso e intelligente in italiano per cruciverba educativi basato su una Parola Chiave specificata e la sua relazione con un Testo assegnato. Per eseguire correttamente questo compito, segui le linee guida seguenti:
PAROLA CHIAVE: {keyword}
TESTO: {text}

Osserva i seguenti passaggi:
1. Sostituisci ogni pronome nel testo con frasi complete che esprimano i loro referenti.
2. Dividi il testo in piccole frasi indipendenti che potrebbero essere comprese fuori contesto.
3. Individua le frasi concise che contengono la Parola Chiave e caratterizzano meglio la parola chiave. Cerca di selezionare frasi da diverse parti del Testo.
4. Genera un indizio breve e intelligente per il cruciverba in italiano dalle frasi selezionate. Assicurati che la parola chiave rimanga assente dall'indizio. Ogni indizio deve essere una frase copulare, in cui la parola chiave costituisce il soggetto. La sintassi di ogni indizio deve corrispondere quindi a una frase copulare senza il soggetto. Ad esempio: "è <indizio>". Genera indizi da tutte le parti del testo e usa tutte le informazioni fornite per generare l'indizio.
5. Assicurati che l'indizio funzioni come una descrizione o definizione della parola chiave piuttosto che una domanda, concentrandoti sui dettagli della parola chiave.
6. Assicurati che le informazioni dell'indizio possano essere ricondotte al testo. Assicurati che l'indizio sia pertinente e sufficiente per identificare la parola chiave. Assicurati che la parola chiave non appaia nell'indizio. Assicurati che nessuna parte della parola chiave sia presente nell'indizio.
'''
}

def get_model_type(model_name):
    """Infer the model type ('llama3' or 'mistral') based on the model name."""
    if 'Llama3' in model_name or 'llama3' in model_name.lower():
        return 'llama3'
    elif 'Mistral' in model_name or 'mistral' in model_name.lower():
        return 'mistral'
    else:
        raise ValueError(f"Cannot determine model type from model_name: {model_name}")

def format_row(row, model_type, selected_prompt):
    """Format the input row into the prompt format required by the model."""
    # Format the user message by inserting text and keyword into the selected prompt
    user_message = selected_prompt.format(keyword=row['keyword'], text=row['text'])

    if model_type == 'llama3':
        # Formatting for Llama3
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"Sei un assistente inestimabile che crea indizi per cruciverba in italiano basati sul testo italiano fornito e una parola chiave.\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    elif model_type == 'mistral':
        # Formatting for Mistral
        formatted_prompt = f'<s>[INST] {user_message} [/INST]'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return formatted_prompt

def extract_text(text, model_type):
    """Extract the assistant's response from the generated text based on model type."""
    if model_type == 'llama3':
        # Extraction logic for Llama3 format
        try:
            if text.count('<|end_header_id|>\n\n') > 1:
                response_part = text.split('<|end_header_id|>\n\n')[2]
                assistant_response = response_part.split('<|end_of_text|>')[0]
                assistant_response = assistant_response.replace('<|eot_id|><|start_header_id|>assistant', '')
                return assistant_response.strip()
        except IndexError:
            pass  # If the expected format isn't found
    elif model_type == 'mistral':
        # Extraction logic for Mistral format
        try:
            closing_inst = '[/INST]'
            idx = text.find(closing_inst)
            if idx != -1:
                assistant_response = text[idx + len(closing_inst):]
                return assistant_response.strip()
        except IndexError:
            pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return None  # If extraction fails

def get_first_three_clues(generated_text):
    """
    Extracts up to the first three clues from the generated_text string.

    Parameters:
    generated_text (str): The string containing all the clues.

    Returns:
    str: A string containing up to the first three clues separated by new lines.
    """
    # Split the generated text into lines
    lines = generated_text.strip().split('\n')
    
    # Take up to the first three non-empty lines
    clues = [line.strip() for line in lines if line.strip()]
    first_three_clues = clues[:3]
    return '\n'.join(first_three_clues)

def get_code_completion(prompt, model, tokenizer, temperature):
    """Generates completion for the given prompt using the model."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def main(args):
    # Load the model and tokenizer
    model_name = args.model_name
    selected_prompt = prompts[args.clue_type]
    print(f"Loading the model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model {model_name} loaded successfully.")
    
    # Infer the model type from the model name
    model_type = get_model_type(model_name)
    print(f"Inferred model type: {model_type}")

    # Read the input CSV
    df = pd.read_csv(args.input_file)

    # List to store the outputs
    outputs = []

    for index, row in df.iterrows():
        # Create the prompt using the formatting function
        prompt = format_row(row, model_type, selected_prompt)

        try:
            # Generate the response
            response = get_code_completion(prompt, model, tokenizer, args.temperature)
            generated_text = extract_text(response, model_type)
            generated_text = get_first_three_clues(generated_text)

            # Display progress
            print(f"Processing index {index}:")
            print(f"Input Text: \n{row['text']}")
            print(f"Input Keyword: {row['keyword']}")
            print(f"Generated Clue: \n{generated_text}\n")

            # Append the result
            outputs.append({
                'text': row['text'],
                'keyword': row['keyword'],
                'Generated Italian Crossword Clue': generated_text
            })

        except Exception as e:
            print(f"Error processing index {index}: {e}")
            outputs.append({
                'text': row['text'],
                'keyword': row['keyword'],
                'Generated Italian Crossword Clue': None,
                'Error': str(e)
            })

    # Save the outputs to a CSV file
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Italian crossword clues using a language model.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output-file', type=str, default='output.csv', help='Path to save the output CSV file.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for text generation.')
    parser.add_argument('--model-name', type=str, required=True, help='Name or path of the model to use.')
    parser.add_argument('--clue-type', type=str, required=True, choices=list(prompts.keys()), help='Type of the clue to generate.')

    args = parser.parse_args()
    main(args)
