import os 
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from google import genai 
from src.graph.graph_builder import build_graph
from src.memory.store import init_db
MAX_UPLOAD_BYTES = 20*1024*1024
@asynccontextmanager
async def lifespan(app:FastAPI):
    init_db()
    yield
app = FastAPI(title="AI data scientist agent",lifespan=lifespan)
graph = build_graph()
def get_client()->genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=500,detail="Server is missing its API key ")
    return genai.Client(api_key=key)
@app.get("/health")
def health():
    return {"status":"ok"}
@app.post("/run")
def run(file:UploadFile =File(...),objective:str = Form(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400,detail="Please upload  a .csv file")
    fd,path = tempfile.mkstemp(suffix=".csv")
    try:
        size = 0
        with os.fdopen(fd,"wb") as tmp:
            while chunk:=file.file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=443,detail="File too large (20 MB max)")
                tmp.write(chunk)
        initial_state = {
            "csv_path": path,
            "user_objective": objective,
            "profile": None,
            "parsed_objective": None,
            "cleaning_plan": None,
            "cleaned_df": None,
            "experiment_plan": None,
            "critique": None,
            "experiment_results": None,
            "reflection": None,
            "iteration_count": 0,
            "critique_iteration_count": 0,
            "accepted": False,
            "error_message": None,
            "trace": [],
        }
        final = graph.invoke(
            initial_state,config = {"configurable":{"client":get_client()}}

        )
    finally:
        os.remove(path)
    def dump(x):
        return x.model_dump() if x is not None else None
    return {
        "accepted": final["accepted"],
        "error_message": final["error_message"],
        "objective": dump(final["parsed_objective"]),
        "cleaning_plan": dump(final["cleaning_plan"]),
        "experiment_plan": dump(final["experiment_plan"]),
        "results": final["experiment_results"],
        "reflection": dump(final["reflection"]),
        "trace": final["trace"],
    }