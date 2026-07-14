import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from text_to_sparql_pipeline import TextToSparqlPipeline

class WebServerHandler(BaseHTTPRequestHandler):
    pipeline = None  # Will be set before starting the server
    
    def log_message(self, format, *args):
        # Suppress request logging to keep console output clean
        pass

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path
        
        # Static files serving
        if url_path == "/" or url_path == "/index.html":
            self.serve_static("index.html", "text/html")
        elif url_path == "/style.css":
            self.serve_static("style.css", "text/css")
        elif url_path == "/app.js":
            self.serve_static("app.js", "application/javascript")
        elif url_path == "/api/trajectories":
            self.serve_trajectories()
        elif url_path == "/api/config":
            self.serve_config()
        else:
            self.send_error(404, "File Not Found")
            
    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path
        
        if url_path == "/api/query":
            self.handle_query()
        else:
            self.send_error(404, "Not Found")
            
    def serve_static(self, filename, content_type):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "web", filename)
        
        if not os.path.exists(file_path):
            self.send_error(404, f"File {filename} not found")
            return
            
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")
            
    def serve_trajectories(self):
        # Return recent trajectories from trajectories.jsonl
        trajectories = []
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories.jsonl")
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
        
        response_bytes = json.dumps(trajectories, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def serve_config(self):
        # Return pipeline config
        config_data = {
            "model_name": self.pipeline.model_name,
            "seed": self.pipeline.seed,
            "temperature": self.pipeline.temperature
        }
        response_bytes = json.dumps(config_data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)
            
    def handle_query(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(post_data.decode("utf-8"))
            question = req_data.get("question", "").strip()
            
            if not question:
                self.send_json_error("Question cannot be empty", 400)
                return
                
            # Run pipeline
            pipeline_result = self.pipeline.run_pipeline(question)
            
            # Read the latest trajectory for this question (since it was just written)
            trajectory = None
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories.jsonl")
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
            
            response_bytes = json.dumps(response_data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", len(response_bytes))
            self.end_headers()
            self.wfile.write(response_bytes)
            
        except Exception as e:
            self.send_json_error(f"Error executing query: {str(e)}", 500)
            
    def send_json_error(self, message, status_code=500):
        response_bytes = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

def start_server(pipeline, port=8080):
    WebServerHandler.pipeline = pipeline
    server = HTTPServer(("localhost", port), WebServerHandler)
    url = f"http://localhost:{port}/"
    print(f"\nStarting Web Server on {url} ...")
    print("Press Ctrl+C to stop the server.")
    
    # Auto-open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb Server stopped.")
