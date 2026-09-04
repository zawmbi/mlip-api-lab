import json
import os
from typing import Any, Dict
from litellm import completion
from pydantic import BaseModel



# loading api key from .env
from dotenv import load_dotenv

load_dotenv()



# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")


class Itinerary(BaseModel):
    destination: str
    price_range: str
    ideal_visit_times: list[str]
    top_attractions: list[str]


def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.

    response = completion(
        model=MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        messages=[{"role": "user", "content":
            f"Give a travel itinerary for {destination} as JSON with keys: "
            "destination (string), price_range (string), "
            "ideal_visit_times (list of strings), top_attractions (list of strings)."
        }],
        response_format={"type": "json_object"},
    )

    data = Itinerary(**json.loads(response.choices[0].message.content)).model_dump()


    return data
