from openai import OpenAI


def test_model(model_to_test: str, client: OpenAI, test_sentence: str = "Morning sir !") -> None:
    print(f"Testing model {model_to_test}...")

    try:
        response = client.chat.completions.create(
            model=model_to_test,
            messages=[
                {"role": "user", "content": test_sentence}
            ],
            max_tokens=50,
        )
        print(f'Réponse de {model_to_test} : {response.choices[0].message.content}')
    except Exception as e:
        print("Erreur lors de la connexion :", e)