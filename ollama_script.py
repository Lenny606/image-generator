import ollama
import sys

def generate_response(prompt, model="gemma4:31b-cloud", context=None):
    """
    Volá lokální Ollama endpoint. Přijímá 'context' pro udržení paměti konverzace.
    Vrací (odpověď, nový_kontext).
    """
    try:
        response = ollama.generate(
            model=model, 
            prompt=prompt, 
            context=context,
            system="Jsi kreativní asistent, který odpovídá stručně a k věci."
        )
        return response['response'], response.get('context')
    except Exception as e:
        print(f"Chyba při komunikaci s Ollamou: {e}")
        return None, None

def main():
    model = "gemma4:31b-cloud"
    context = None
    
    print(f"--- Interaktivní režim s kontextem (model: {model}) ---")
    print("Napište 'exit' pro ukončení.\n")

    while True:
        user_input = input("Vy: ")
        if user_input.lower() in ['exit', 'quit', 'konec']:
            break
            
        response_text, context = generate_response(user_input, model=model, context=context)
        
        if response_text:
            print(f"\nOllama: {response_text}\n")
        else:
            print("Nepodařilo se získat odpověď.")

if __name__ == "__main__":
    main()
