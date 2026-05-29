# scripts/dashboard.py

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import VoiceCommandAPI

# Global APIs
APIs: Dict[str, VoiceCommandAPI] = {}

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler to serve dashboard frontend and run classification API."""
    
    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default console logging to keep terminal output clean."""
        pass

    def do_GET(self) -> None:
        """Serves static files (HTML, CSS, JS)."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dashboard_dir = os.path.join(project_root, "src", "dashboard")
        
        # Default route
        path = self.path
        if path == "/":
            path = "/index.html"
            
        file_path = os.path.join(dashboard_dir, path.lstrip("/"))
        
        # Check if file exists and is within dashboard directory to prevent traversal attacks
        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            
            # Content headers
            if file_path.endswith(".html"):
                self.send_header("Content-type", "text/html")
            elif file_path.endswith(".css"):
                self.send_header("Content-type", "text/css")
            elif file_path.endswith(".js"):
                self.send_header("Content-type", "application/javascript")
            elif file_path.endswith(".png"):
                self.send_header("Content-type", "image/png")
            elif file_path.endswith(".svg"):
                self.send_header("Content-type", "image/svg+xml")
            self.end_headers()
            
            # Write file content
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File Not Found")

    def do_POST(self) -> None:
        """Handles POST classification requests."""
        if self.path == "/classify":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "")
                model_type = data.get("model_type", "onnx")
                
                # Dynamic model switching / caching
                if model_type not in APIs:
                    APIs[model_type] = VoiceCommandAPI(model_type=model_type)
                
                api = APIs[model_type]
                result = api.classify(text)
                
                # Send JSON response
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8000) -> None:
    """Launches the HTTP server and opens the dashboard in the default browser."""
    # Ensure models are built
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfidf_path = os.path.join(project_root, "models", "tfidf_model.json")
    onnx_dir = os.path.join(project_root, "models", "onnx_minilm")
    
    if not os.path.exists(tfidf_path) or not os.path.exists(os.path.join(onnx_dir, "model_quantized.onnx")):
        print("Required models not found. Running benchmark to build models first...")
        os.system(f"{sys.executable} scripts/benchmark.py")
        
    # Warm up ONNX API
    print("Loading Quantized ONNX model into memory...")
    APIs["onnx"] = VoiceCommandAPI(model_type="onnx")
    print("ONNX model loaded successfully.")

    server = HTTPServer(("localhost", port), DashboardHandler)
    url = f"http://localhost:{port}/"
    
    print("\n" + "="*60)
    print(f"   AUTOMOTIVE COCKPIT DASHBOARD RUNNING AT: {url}")
    print("="*60)
    print("Press Ctrl+C to terminate the dashboard server.\n")
    
    # Auto-open browser
    webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()

if __name__ == "__main__":
    run_server()
