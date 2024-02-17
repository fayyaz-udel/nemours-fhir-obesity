from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import ssl
import os


httpd = HTTPServer(('localhost', 80), SimpleHTTPRequestHandler)

# httpd.socket = ssl.wrap_socket (httpd.socket, keyfile="path/to/key.pem", certfile='path/to/cert.pem', server_side=True)

httpd.serve_forever()