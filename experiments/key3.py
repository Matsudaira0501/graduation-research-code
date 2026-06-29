from pathlib import Path   
from dotenv import load_dotenv
from keyphrase_extractors import GenerationBasedExtractor
from keyphrase_extractors.generation_based import ResponseSchema
from langrila import Agent
from langrila.openai import OpenAIClient

input_text = "これはテスト用のテキストです。"

env_filepath = Path("../../.env_api")
load_dotenv(dotenv_path=env_filepath)

azure_openai_client = OpenAIClient(
    api_key_env_name="AZURE_API_KEY",
    api_type="azure",
    azure_api_version="2024-08-01-preview",
    azure_endpoint_env_name="AZURE_ENDPOINT",
    azure_deployment_env_name="AZURE_DEPLOYMENT",
)

agent = Agent(
    client=azure_openai_client,
    model="gpt-4o",
    temperature=0.0,
    response_format=ResponseSchema,
)

extractor = GenerationBasedExtractor(agent=agent, logger=None)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("GPT-4o")

results = keyphrases.keyphrases
if results and isinstance(results[0], list):
    results = results[0]

for rank, kp in enumerate(results, start=1):
    # score を小数3桁に丸めて表示
    print(f"{rank:2d}. {kp.phrase}\t(score={kp.score:.3f})")

print("-" * 80)