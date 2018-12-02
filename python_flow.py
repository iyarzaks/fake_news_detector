import os
from project_utils import *
from sklearn.externals import joblib
import socket
from threading import *
from input_receiving import *
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
            rcvdData = self.sock.recv(1024).decode()
            if "url-" in rcvdData:
                url = str(rcvdData)[4:]
                print ("got it:" + url)
                result = url_query(url)
                print (result)
            else:
                print (rcvdData)
            try:
                self.sock.send(result.encode())
                print ("send back")
            except:
                print ("can not send response")
            break
        # print('Client sent:', self.sock.recv(1024).decode())
        # self.sock.send('Oi you sent something to me')

def get_results(df_with_imp_words,clfs_dic):
    X_test = df_with_imp_words.iloc[:, 3:]
    #print (X_test.values)
    result ={}
    for clf in clfs_dic:
        result[clf] = clfs_dic[clf].predict_proba(X_test)[0][1]
        result[clf] = round(result[clf],2)*100
        # except:
        #     print (clf + " can't return probability")
    #print (result)
    return str(result)


def check_new_article(test_df):
    clfs_dic ={}
    clfs_dic["nn_clf_from_file"]= joblib.load('nn_clf.pkl')
    clfs_dic["lr_clf_from_file"] = joblib.load('lr_clf.pkl')
    clfs_dic["svm_clf_from_file"] = joblib.load('svm_clf.pkl')
    clfs_dic["KNeighbors_clf_from_file"] = joblib.load('KNeighbors_clf.pkl')
    # clfs_dic["dec_tree_clf_from_file"] = joblib.load('dec_tree_clf.pkl')
    words = read_json("top_50.json.1")
    words = words[:1000]
    #print (words)
    df_with_imp_words = build_table_form_words(test_df, words)
    # df_with_imp_words = df_with_imp_words.dropna(thresh=4)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    return get_results(df_with_imp_words,clfs_dic)


def url_query(url=''):
    aritcle_csv = convertUrlToDF(url)
    print ()
    return check_new_article(aritcle_csv)


def main():
    url_query("https://www.huzlers.com/tragic-3-new-deaths-reportedly-linked-to-popular-internet-fad-no-nut-november/")
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("132.69.194.143", 40000))
    serversocket.listen(5)
    print ("i listening")
    while 1:
        clientsocket, address = serversocket.accept()
        client(clientsocket, address)




if __name__ == "__main__":
    main()
