import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler

httpd = HTTPServer(('0.0.0.0', 80), SimpleHTTPRequestHandler)
httpd.serve_forever()
