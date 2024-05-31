import io
import socketio

def run_client(
    port: int,
    train_callback,
    host: str = 'localhost',
):
    sio = socketio.Client()

    @sio.event
    def connect():
        print('connect')

    @sio.event
    def disconnect():
        print('disconnect')

    @sio.on('start_train')
    def on_start_train(data):
        print('start_train', data)
        sio.start_background_task(train_callback, sio, data)

    sio.connect(f'http://{host}:{port}')
    sio.wait()

def send_data(
    sio: socketio.Client,
    data
):
    sio.emit('train_complete', data)