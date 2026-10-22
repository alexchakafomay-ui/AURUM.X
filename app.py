import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import main

# Server kecil biar Hugging Face tau bot kita hidup
class HealthCheck(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"AURUM-X is alive")

def run_health_server():
    server = HTTPServer(('0.0.0.0', 7860), HealthCheck)
    server.serve_forever()

if __name__ == "__main__":
    # Jalankan server kesehatan di background
    threading.Thread(target=run_health_server, daemon=True).start()
    # Jalankan bot utama
    main.main()
