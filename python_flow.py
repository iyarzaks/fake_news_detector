from project_utils import *
from sklearn.externals import joblib
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
            rcvdData = clientsocket.recv(1024).decode()
            if ("new_url_query," in rcvdData ):
                print "Server: What is ", rcvdData
                result = url_query(rcvdData)
            sendData = "Server: got the messgage"
            clientsocket.send(sendData.encode())
        # print('Client sent:', self.sock.recv(1024).decode())
        # self.sock.send('Oi you sent something to me')

def get_results(df_with_imp_words,clfs_dic):
    X_test = df_with_imp_words.iloc[:, 6:]
    print (X_test)
    result ={}
    for clf in clfs_dic:
        try:
            result[clf] = clfs_dic[clf].predict_proba(X_test)[0][1]
        except:
            print (clf + " can't return probability")
    print (result)


def check_new_article():
    clfs_dic ={}
    clfs_dic["nn_clf_from_file"]= joblib.load('nn_clf.pkl')
    clfs_dic["lr_clf_from_file"] = joblib.load('lr_clf.pkl')
    clfs_dic["adaboost_clf_from_file"] = joblib.load('adaboost_clf.pkl')
    clfs_dic["r_forest_clf_from_file"] = joblib.load('r_forest_clf.pkl')
    clfs_dic["dec_tree_clf_from_file"] = joblib.load('dec_tree_clf.pkl')
    test_df = pd.read_csv('Classified_Data_kaggle.csv', engine='python',nrows=1)
    print (test_df)
    words = read_json("top_50.json.1")
    words = words[:500]
    #print (words)
    df_with_imp_words = build_table_form_words(test_df, words)
    # df_with_imp_words = df_with_imp_words.dropna(thresh=4)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    get_results(df_with_imp_words,clfs_dic)


def url_query(url=''):
    #aritcle_csv = convertUrlToDF()
    check_new_article()


def main():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "ip-172-31-21-98.us-east-2.compute.internal"
    port = 8000
    # print (host)
    # print (port)
    serversocket.bind((host, port))
    serversocket.listen(5)
    print ('server started and listening')
    while 1:
        clientsocket, address = serversocket.accept()
        print ("Server: hi what is up")
        client(clientsocket, address)




if __name__ == "__main__":
    main()
