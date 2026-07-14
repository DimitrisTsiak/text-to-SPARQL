import os
import json
import uvicorn
import webbrowser
from threading import Timer
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

def start_server(pipeline, port=8080):
    app = FastAPI(title="Wikidata Assistant API")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.join(base_dir, "web")
    
    @app.get("/")
    async def read_root():
        return FileResponse(os.path.join(web_dir, "index.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
        
    @app.get("/index.html")
    async def read_index():
        return FileResponse(os.path.join(web_dir, "index.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
        
    @app.get("/style.css")
    async def read_css():
        return FileResponse(os.path.join(web_dir, "style.css"), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
        
    @app.get("/app.js")
    async def read_js():
        return FileResponse(os.path.join(web_dir, "app.js"), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
        
    @app.get("/api/trajectories")
    async def get_trajectories():
        trajectories = []
        log_path = os.path.join(base_dir, "trajectories.jsonl")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            trajectories.append(json.loads(line))
            except Exception:
                pass
        # Return latest 20 trajectories, reversed
        trajectories = list(reversed(trajectories))[:20]
        return trajectories

    @app.get("/api/config")
    async def get_config():
        config_data = {
            "model_name": pipeline.model_name,
            "seed": pipeline.seed,
            "temperature": pipeline.temperature
        }
        return config_data

    class QueryRequest(BaseModel):
        question: str

    @app.post("/api/query")
    async def post_query(request: QueryRequest):
        question = request.question.strip()
        if not question:
            return JSONResponse(status_code=400, content={"error": "Question cannot be empty"})
            
        try:
            # Run pipeline
            pipeline_result = pipeline.run_pipeline(question)
            
            # Read the latest trajectory for this question (since it was just written)
            trajectory = None
            log_path = os.path.join(base_dir, "trajectories.jsonl")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                t = json.loads(line)
                                if t.get("question") == question:
                                    trajectory = t
                except Exception:
                    pass
            
            response_data = {
                "success": pipeline_result.get("success", False),
                "result": pipeline_result,
                "trajectory": trajectory
            }
            return response_data
            
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Error executing query: {str(e)}"})

    url = f"http://localhost:{port}/"
    print(f"\nStarting FastAPI Web Server on {url} ...")
    print("Press Ctrl+C to stop the server.")
    
    # Auto-open browser
    def open_browser():
        try:
            webbrowser.open(url)
        except Exception:
            pass
            
    Timer(1.0, open_browser).start()
    
    # Start Uvicorn
    uvicorn.run(app, host="localhost", port=port, access_log=False)
