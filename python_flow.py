import os
from project_utils import *
from sklearn.externals import joblib
import socket
from threading import *
from input_receiving import *
import socket
from threading import *
import pyodbc
import time
import datetime



class client(Thread):
    def __init__(self, socket, address,cursor,cnxn):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.cursor = cursor
        self.cnxn = cnxn
        self.start()

    def run(self):
        while 1:
            rcvdData = self.sock.recv(1024).decode()
            if "url-" in rcvdData:
                url = str(rcvdData)[4:]
                print ("got it:" + url)
                result = url_query(url,self.cursor,self.cnxn)
            elif "sql-"in rcvdData:
                sql_str = str(rcvdData)[4:]
                print("got it:" + sql_str)
                result = sql_response(sql_str,self.cursor,self.cnxn)
            else:
                print (rcvdData)
            try:
                self.sock.send(result.encode())
                print ("send back: "+result)
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
    clfs_dic["svm_clf_from_file"] = joblib.load('svm_clf.pkl')
    words = read_json("top_50.json.1")
    words = words[:1000]
    df_with_imp_words = build_table_form_words(test_df, words)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    return get_results(df_with_imp_words,clfs_dic)


def url_query(url, cursor,cnxn):
    aritcle_csv = convertUrlToDF(url)
    if type(aritcle_csv) == type("alon"):
        return "input error"
    res = check_new_article(aritcle_csv)
    add_to_db(cursor,url,res,str(aritcle_csv.loc[0,"HeadLine"]),cnxn)
    return str(res)


def check_url_in_db(url,cursor,cnxn):
    query = "select aID, URL from ARTICLE"
    cursor.execute(query)
    row = cursor.fetchone()
    while row:
        if url == str(row[1]):
            add_article_to_log(int(str(row[0])),cursor,cnxn)
            return None
        row = cursor.fetchone()
    query = "select max(aID) as maxID from ARTICLE"
    cursor.execute(query)
    row = cursor.fetchone()
    if row[0] is None:
        return 1
    return int(str(row[0]))+1

def add_article_to_log(aid,cursor,cnxn):
    cur_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    query = """INSERT INTO HISTORICAL_INFO (time, HIST_aID)
            VALUES ( '{}', {});""".format(cur_time, aid)
    cursor.execute(query)
    cnxn.commit()


def add_to_db(cursor,url,res,headline,cnxn):
    mean_res = round(np.mean(res),2)
    true_article = int(mean_res > 50)
    aid = check_url_in_db(url,cursor,cnxn)
    if aid is None:
        return
    domain = url.split("//")[-1].split("/")[0]
    query = "INSERT INTO ARTICLE (aID, URL, header, modelResult, domain, alg1, alg2, alg3, modelWeightedResult) VALUES ({}, '{}', '{}', {}, '{}', {}, {}, {}, {});".format(aid,url,headline,true_article,domain,res[0],res[1],res[2],mean_res)
    cursor.execute(query)
    cnxn.commit()
    add_article_to_log(aid, cursor, cnxn)

    # row = cursor.fetchone()
    # while row:
    #     print(str(row[0]) + " " + str(row[1]))
    #     row = cursor.fetchone()

def sql_response(sql_str,cursor,cnxn):
    if sql_str == "top_n_articles_today":
        query = """SELECT header
                    FROM ARTICLE
                    WHERE aID IN (SELECT  top 3 HIST_aID
                    FROM     HISTORICAL_INFO
                    where convert(varchar(10), time, 120) = convert(varchar(10), GETDATE(), 120)
                    GROUP BY HIST_aID
                    ORDER BY COUNT(HIST_aID) DESC);"""
    elif sql_str == "top_n_sites":
        query = """SELECT domain
                    FROM (SELECT top 2 domain,COUNT(domain) AS domainCount
                    FROM ARTICLE WHERE modelResult = 0
                    GROUP BY domain
                    ORDER BY domainCount DESC) ARTICLE_COUNT"""
    cursor.execute(query)
    row = cursor.fetchone()
    result = []
    while row:
        result .append(str(row[0]))
        row = cursor.fetchone()
    return str(result)



def main():
    cursor,cnxn = connect_sql_server()
    print(url_query("https://alternative-science.com/elon-musk-facebook/",cursor,cnxn))
    print(sql_response("top_n_sites",cursor,cnxn))
    # serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # serversocket.bind(("132.69.194.143", 50000))
    # serversocket.listen(5)
    # print ("i listening")
    # while 1:
    #     clientsocket, address = serversocket.accept()
    #     client(clientsocket, address,cursor,cnxn)




if __name__ == "__main__":
    main()
