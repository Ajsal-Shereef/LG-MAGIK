import json
import requests

# OpenRouter API key
API_KEY = "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"

def query_openrouter(prompt: str) -> str:
    """
    Sends a prompt to the OpenRouter API and returns the assistant's response.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "x-ai/grok-4-fast:free",
        "messages": [
            {"role": "system", "content": "You are an assistant that helps an RL agent adapt to a new target task by modifying scene descriptions."},
            {"role": "user", "content": prompt}
        ],
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    prompt = (
        "The agent was trained on a source task (e.g., pick the green ball) but now needs to perform a target task (e.g., pick the red ball).\n\n"
        "Rules:\n"
        "- Extract the source object and target object from the given task specification.\n"
        "- Whenever the target object appears in the scene, replace it with the source object (keeping position, distance, and all other details unchanged) "
        "and return the output with \"imagine\" as true and \"description\" as the changed input description.\n"
        "- If the target object does not appear in the scene, return the output with \"imagine\" as false without any description.\n\n"
        "Return only valid JSON, with no extra text, no explanations, no code, no markdown.\n"
        "Format:\n"
        "{\n"
        "  \"imagine\": true/false,\n"
        "  \"description\": \"...\"\n"
        "}\n\n"
        "Source task: Pick orange ball\n"
        "Target task: Pick blue ball\n"
        "Input description: Agent sees to the bottom left, a yellow ball which is 2 units apart at (2,5) and to the bottom right, a blue ball which is 2 units apart at (5,6)."
    )

    reply = json.loads(query_openrouter(prompt))
    print("Response:", reply)