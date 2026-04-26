import ollama
import sys

def generate_response(prompt, model="gemma4:31b-cloud"):
    """
    Volá lokální Ollama endpoint a vrací vygenerovaný text.
    """
    print(f"Dotazuji se modelu '{model}'...")
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Chyba při komunikaci s Ollamou: {e}"

def main():
    # Výchozí prompt nebo prompt z argumentů
    prompt = "Napiš krátký úvod do historie umělé inteligence."
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    response = generate_response(prompt)
    
    print("\n" + "="*20)
    print("ODPOVĚĎ:")
    print("="*20)
    print(response)
    print("="*20)

if __name__ == "__main__":
    main()
