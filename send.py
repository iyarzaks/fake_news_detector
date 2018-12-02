import socket
import socket
from threading import *
class client(Thread):
    def __init__(self, socket, address):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.start()

    def run(self):
        while 1:
            x = 1
            rcvdData = self.sock.recv(1024).decode()
            print ("Server: What is ", rcvdData)
            result = url_query(rcvdData)
            self.sock.send(result.encode())
        # print('Client sent:', self.sock.recv(1024).decode())
        # self.sock.send('Oi you sent something to me')



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "132.69.194.143"
port = 40000
s.connect((host,port))
s.send("new url query,https://www.newyorker.com/humor/borowitz-report/g-20-leaders-vote-unanimously-not-to-give-trump-asylum".encode())
#s.listen(2)
#s.close()

