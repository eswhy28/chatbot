import gradio as gr
from fastapi import FastAPI, Body
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple
import httpx
import os
from dotenv import load_dotenv

from swarmauri.llms.concrete.GroqModel import GroqModel
from swarmauri.messages.concrete.SystemMessage import SystemMessage
from swarmauri.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load API key from environment variable
API_KEY = os.getenv('API_KEY')
if API_KEY is None:
    raise ValueError("API_KEY not found in environment variables.")

try:
    llm = GroqModel(api_key=API_KEY)
    allowed_models = llm.allowed_models
except Exception as e:
    raise RuntimeError(f"Failed to initialize GroqModel: {e}")

conversation = MaxSystemContextConversation()

class PredictInput(BaseModel):
    message: str
    history: List[Tuple[str, str]]
    system_context: str
    model_name: str

    model_config = ConfigDict(protected_namespaces=())

def load_model(selected_model):
    try:
        return GroqModel(api_key=API_KEY, name=selected_model)
    except Exception as e:
        return None

@app.post("/run/predict")
async def predict(data: PredictInput = Body(...)):
    llm = load_model(data.model_name)
    if llm is None:
        return {"result": "Failed to load model."}

    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    agent.conversation.system_context = SystemMessage(content=data.system_context)

    try:
        result = agent.exec(data.message)
        return {"result": str(result)}
    except Exception as e:
        return {"result": f"Error occurred while executing the command: {str(e)}"}

async def gradio_predict(message, history, system_context, model_name):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/run/predict",
            json={
                "message": message,
                "history": history,
                "system_context": system_context,
                "model_name": model_name
            },
            timeout=60.0  # Increase timeout to 60 seconds
        )
        result = response.json().get("result", "No result returned")
    return result

demo = gr.ChatInterface(
    fn=gradio_predict,
    additional_inputs=[
        gr.Textbox(label="System Context", lines=2),
        gr.Dropdown(label="Model Name", choices=allowed_models, value=allowed_models[0])
    ],
    title="A System Context Conversation",
    description="Interact with the agent using a selected model and system context.",
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
