import asyncio
import websockets
import numpy as np
import joblib
import turtle

class WebSocketServer:
    def __init__(self, port, model_filename):
        self.port = port
        self.server = None
        self.loaded_model = joblib.load(model_filename)

        turtle.speed(0)
        turtle.title("Prediction Display")

    async def start_server(self):
        self.server = await websockets.serve(
            self._handle_connection, "0.0.0.0", self.port
        )
        print(f"Server started on port {self.port}")

    async def _handle_connection(self, websocket, path):
        print("Client connected")
        peer_address = websocket.remote_address
        self.on_client_connected(peer_address)

        try:
            async for message in websocket:
                self.on_data_received(peer_address, message)
        except websockets.exceptions.ConnectionClosedError:
            pass  # client disconnected!!
        finally:
            self.on_client_disconnected(peer_address)

    def on_client_connected(self, peer_address):
        print(f"Client connected: {peer_address}")

    def on_client_disconnected(self, peer_address):
        print(f"Client disconnected: {peer_address}")

    def on_data_received(self, peer_address, data):
        print(f"Received data: {data}")
        # bytes to string
        decoded_data = data.decode('utf-8')
        prediction = self.predict(decoded_data)
        print(f"Predicted data: {prediction}")
        self.save_to_file(decoded_data, prediction)

        #red screen if the prediction is 1
        if prediction == 1:
            self.flash_red_screen()

    def save_to_file(self, data, prediction):
        with open("received_data_with_gravity.txt", "a") as file:
            file.write(f"{data.strip()} - Predicted tag: {prediction}\n")

    def predict(self, data):
        values = [float(val) for val in data.split('(')[1].split(')')[0].split(',')]
        new_data_point = np.array([values])
        prediction = self.loaded_model.predict(new_data_point)
        return prediction[0]

    def flash_red_screen(self):
        turtle.bgcolor("red")
        turtle.update()
        turtle.ontimer(self.reset_screen, 50)

    def reset_screen(self):
        turtle.bgcolor("white")
        turtle.update()

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

async def main():
    PORT = 5000
    MODEL_FILENAME = "random_forest_model.joblib"
    server = WebSocketServer(PORT, MODEL_FILENAME)
    await server.start_server()

    try:
        while True:
            await asyncio.sleep(0.1)  #adjust
    finally:
        await server.close()

if __name__ == "__main__":
    asyncio.run(main())
