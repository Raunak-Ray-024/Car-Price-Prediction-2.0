from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "This is a placeholder entrypoint for Vercel.\n" 
                   "Streamlit code is in main.py; run locally with `streamlit run main.py`."
    }
