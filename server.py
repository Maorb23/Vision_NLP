import http.server
import ssl

# Define the server address and port
server_address = ('0.0.0.0', 8888)

# Create an HTTP server
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# Wrap the server with SSL
httpd.socket = ssl.wrap_socket(
    httpd.socket,
    certfile="cert.pem",  # Path to your certificate file
    keyfile="key.pem",    # Path to your key file
    server_side=True
)

print("Serving HTTPS on 0.0.0.0 port 8888...")
httpd.serve_forever()
