"""
Vercel Serverless Function — Health Check
"""
from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            'status': 'ok',
            'mode': 'vercel-serverless',
            'timestamp': datetime.now().isoformat(),
            'endpoints': ['/api/prices', '/api/status'],
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()
