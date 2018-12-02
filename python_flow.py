import os
from project_utils import *
from sklearn.externals import joblib
import socket
from threading import *
from input_receiving import *
import socket
from threading import *
import pyodbc



class client(Thread):
    def __init__(self, socket, address,cursor):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.cursor = cursor
        self.start()

    def run(self):
        while 1:
            rcvdData = self.sock.recv(1024).decode()
            if "url-" in rcvdData:
                url = str(rcvdData)[4:]
                print ("got it:" + url)
                result = url_query(url,self.cursor)
                print (result)
            else:
                print (rcvdData)
            try:
                self.sock.send(result.encode())
                print ("send back")
            except:
                print ("can not send response")
        # print('Client sent:', self.sock.recv(1024).decode())
        # self.sock.send('Oi you sent something to me')

def get_results(df_with_imp_words,clfs_dic):
    X_test = df_with_imp_words.iloc[:, 3:]
    #print (X_test.values)
    result ={}
    for clf in clfs_dic:
        result[clf] = clfs_dic[clf].predict_proba(X_test)[0][1]
        result[clf] = float(round(result[clf]*100,2))
        # except:
        #     print (clf + " can't return probability")
    #print (result)
    result = list(result.values())
    return result


def check_new_article(test_df):
    clfs_dic ={}
    clfs_dic["nn_clf_from_file"]= joblib.load('nn_clf.pkl')
    clfs_dic["lr_clf_from_file"] = joblib.load('lr_clf.pkl')
    # clfs_dic["svm_clf_from_file"] = joblib.load('svm_clf.pkl')
    clfs_dic["KNeighbors_clf_from_file"] = joblib.load('KNeighbors_clf.pkl')
    # clfs_dic["dec_tree_clf_from_file"] = joblib.load('dec_tree_clf.pkl')
    words = read_json("top_50.json.1")
    words = words[:1000]
    #print (words)
    df_with_imp_words = build_table_form_words(test_df, words)
    # df_with_imp_words = df_with_imp_words.dropna(thresh=4)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    return get_results(df_with_imp_words,clfs_dic)


def url_query(url, cursor,cnxn):
    aritcle_csv = convertUrlToDF(url)
    if str(aritcle_csv) == 'input error':
        return "input error"
    res = check_new_article(aritcle_csv)
    add_to_db(cursor,url,res,str(aritcle_csv["HeadLine"]),cnxn)
    return str(res)


def check_url_in_db(url,cursor):
    query = "select URL from ARTICLE"
    cursor.execute(query)
    row = cursor.fetchone()
    while row:
        if url == str(row[0]):
            return None
        row = cursor.fetchone()
    query = "select max(aID) as maxID from ARTICLE"
    cursor.execute(query)
    row = cursor.fetchone()
    return int(str(row[0]))+1



def add_to_db(cursor,url,res,headline,cnxn):
    mean_res = round(np.mean(res),2)
    true_article = int(mean_res > 50)
    aid = check_url_in_db(url,cursor)
    if aid is None:
        return
    domain = url.split("//")[-1].split("/")[0]
    query = "INSERT INTO ARTICLE (aID, URL, header, modelResult, domain, alg1, alg2, alg3, modelWeightedResult) VALUES ({}, '{}', '{}', {}, '{}', {}, {}, {}, {});".format(aid,url,headline,true_article,domain,res[0],res[1],res[2],mean_res)
    #query= "SELECT * FROM ARTICLE"
    print (query)
    cursor.execute(query)
    cnxn.commit()

    # row = cursor.fetchone()
    # while row:
    #     print(str(row[0]) + " " + str(row[1]))
    #     row = cursor.fetchone()



def connect_sql_server():
    server = 'iz.database.windows.net'
    database = 'fake_news_DB'
    username = 'izaks'
    password = 'Fakenews!'
    driver = '{SQL Server}'
    cnxn = pyodbc.connect(
        'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    return cursor,cnxn


def main():
    cursor,cnxn = connect_sql_server()
    print(url_query("https://www.sport5.co.il/HTML/Articles/Article.285.297563.html",cursor,cnxn))
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("132.69.194.143", 50000))
    serversocket.listen(5)
    print ("i listening")
    while 1:
        clientsocket, address = serversocket.accept()
        client(clientsocket, address,cursor,cnxn)




if __name__ == "__main__":
    main()
