import socketio
import eventlet
import threading
from utils.console import NonBlockingConsole

class ClientGroup:
    def __init__(self, gid: int, sio: socketio.Server) -> None:
        self.sio = sio
        self.gid = gid
        self.clients = set()
        self.ready_clients = set()
        self.client_data = {} # sid: data
    
    def add_client(self, sid):
        self.clients.add(sid)
    
    def remove_client(self, sid):
        self.clients.remove(sid)
    
    def client_be_ready(self, sid, data) -> bool:
        self.ready_clients.add(sid)
        self.client_data[sid] = data
        return len(self.ready_clients) == len(self.clients)
    
    def send(self, event, data):
        self.emit(event, data, to= list(self.clients))
    
    def start_train(self, data):
        self.ready_clients.clear()
        self.client_data.clear()
        print('Start train on', self.clients)
        self.sio.emit('start_train', 'start_train', to= list(self.clients))


class NetworkServer:
    def __init__(
        self,
        group_complete_callback,
        create_data_callback,
        ) -> None:
        # Creating a new Socket.IO server
        sio = socketio.Server()
        app = socketio.WSGIApp(sio)
        self.groups = {
            0: ClientGroup(0, sio) # the group for unlauched clients
        } # gid: ClientGroup
        self.next_gid = 1
        self.create_data_callback = create_data_callback

        @sio.event
        def connect(sid, environ):
            print('Client connected', sid)
            self.groups[0].add_client(sid)
        
        @sio.event
        def disconnect(sid):
            print('Client disconnected', sid)
            for gid, group in self.groups.items():
                if sid in group.clients:
                    group.remove_client(sid)
                    if len(group.clients) == 0:
                        del self.groups[gid]
                    break
        
        @sio.on('train_complete')
        def on_train_complete(sid, data):
            print('Device train complete', sid)
            group = self.find_client_group(sid)
            if group is None:
                print('Client not found')
                return
            if group.client_be_ready(sid, data):
                sio.start_background_task(group_complete_callback, self, group)
        
        self.sio = sio
        self.app = app

    def split_group(self, group: ClientGroup, clients):
        new_group = ClientGroup(self.next_gid, self.sio)
        self.next_gid += 1
        for client in clients:
            group.remove_client(client)
            new_group.add_client(client)
        self.groups[new_group.gid] = new_group

    def find_client_group(self, sid):
        for gid, group in self.groups.items():
            if sid in group.clients:
                return group
        return None
    
    def print_groups(self):
        for gid, group in self.groups.items():
            print('Group:', gid)
            print('  Clients:', group.clients)
            print('  Ready clients:', group.ready_clients)
    
    def background_job(self):
        with NonBlockingConsole() as nbc:
            while True:
                data = nbc.get_data()
                if data == 'l':
                    grp = self.groups[0]
                    if len(grp.clients) == 0:
                        print('No client to launch')
                        continue
                    print('Clients in unlanch group:', grp.clients)
                    self.groups[0] = ClientGroup(0, self.sio)
                    self.groups[self.next_gid] = grp
                    grp.gid = self.next_gid
                    print('New group:', self.next_gid, 'created')
                    grp.start_train(self.create_data_callback())
                    self.next_gid += 1
                elif data == 'p':
                    self.print_groups()
                elif data == 'h':
                    print('l: launch new group')
                    print('p: print groups')
                eventlet.sleep(0.1)
    
    def start(self, port: int, host: str= 'localhost'):
        self.sio.start_background_task(self.background_job)
        eventlet.wsgi.server(eventlet.listen((host, port)), self.app)

if __name__ == '__main__':
    def group_complete_callback(group: ClientGroup):
        print('Group complete:', group.gid)
    
    server = NetworkServer(group_complete_callback)
    server.start(10294, 'localhost')
