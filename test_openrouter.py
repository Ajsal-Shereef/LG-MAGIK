import os
import json
from dotenv import load_dotenv
from architectures.common_utils import query_openrouter

# Load the .env file
load_dotenv(dotenv_path="config/.env")

# Access the API key
api_key = os.getenv('API_KEY')

if __name__ == "__main__":
    # Example input for testing
    prompt = (
        "Target task: Pick only green ball.\n"
        "Input description: Agent sees a green ball at (2,5) and a red ball at (4,5)."
    )

    # System prompt: reasoning-based imagination (agent only knows pick red ball)
    system_prompt = (
        "You are an assistant that rewrites scene descriptions so an RL agent (which only knows a single skill) "
        "can solve new target tasks by imagining alternative scenes. Use common-sense and logical reasoning "
        "to produce the minimal scene-change that will allow the agent to use its known skill.\n\n"
        "Goal:\n"
        "- Given a Target task and an Input description of the current scene, produce a rewritten scene description "
        "that, when imagined by the agent, makes the target task solvable using the agent's single known skill.\n"
        "- The model should reason about which object(s) to change or remove, but must output only the final JSON (no explanations).\n\n"
        "Imagination rules:\n"
        "1. If the Target task is \"pick the green ball\" and the scene contains only a green ball (no red ball), "
        "replace the green ball with a red ball at the same coordinates.\n"
        "2. If the Target task is \"pick only the green ball\" and the scene contains both a red ball and a green ball, "
        "remove the red ball, and replace the green ball with a red ball at the same coordinates.\n"
        "   The final imagined scene should therefore contain only the single (imagined) red ball at the green ball's location.\n"
        "3. Preserve all spatial details (positions, coordinates, distances) when making replacements or removals.\n"
        "4. Do not introduce any new objects or change any other objects unless required by rules 1–2.\n"
        "5. If the transformation above cannot be applied because the scene contains no relevant balls, return {\"imagine\": false}.\n\n"
        "Output format (strict JSON only — no extra text, no markdown):\n"
        "{\n"
        "  \"imagine\": true|false,\n"
        "  \"description\": \"<rewritten scene description>\"\n"
        "}\n\n"
        "Few-shot examples:\n\n"
        "Example A:\n"
        "Target task: Pick the green ball.\n"
        "Input description: Agent sees a green ball at (2,5).\n"
        "Output:\n"
        "{\n"
        "  \"imagine\": true,\n"
        "  \"description\": \"Agent sees a red ball at (2,5).\"\n"
        "}\n\n"
        "Example B:\n"
        "Target task: Pick only the green ball.\n"
        "Input description: Agent sees a red ball at (3,4) and a green ball at (6,7).\n"
        "Output:\n"
        "{\n"
        "  \"imagine\": true,\n"
        "  \"description\": \"Agent sees a red ball at (6,7).\"\n"
        "}\n\n"
        "Example C:\n"
        "Target task: Pick the green ball.\n"
        "Input description: Agent sees an empty room.\n"
        "Output:\n"
        "{\n"
        "  \"imagine\": false,\n"
        "  \"description\": \"Agent sees an empty room.\"\n"
        "}\n\n"
        "Example D:\n"
        "Target task: Pick the green ball.\n"
        "Input description: Agent sees a green ball at (6,7) and a locked gate at (10,7).\n"
        "Output:\n"
        "{\n"
        "  \"imagine\": true,\n"
        "  \"description\": \"Agent sees a red ball at (6,7) and a locked gate at (10,7).\"\n"
        "}\n"
    )

    reply = json.loads(query_openrouter(system_prompt, prompt, api_key))
    print("Response:", reply)
