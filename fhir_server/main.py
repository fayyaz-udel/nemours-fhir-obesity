import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and the port on which to listen
host = '0.0.0.0'
port = 3000

# Bind the socket to the address and port
server_socket.bind((host, port))

# Start listening for incoming connections, with a backlog of 5 connections
server_socket.listen(5)
print(f'Server listening on port {port}...')

try:
    while True:
        # Accept a connection
        client_socket, addr = server_socket.accept()
        print(f'Connected by {addr}')

        # Receive data from the client
        data = client_socket.recv(1024)
        print(f'Received data: {data.decode()}')

        # Close the connection
        client_socket.close()

except KeyboardInterrupt:
    # Close the server socket
    print('Shutting down the server...')
    server_socket.close()
