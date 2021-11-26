# Shamelessly copied from http://flask.pocoo.org/docs/quickstart/

import numpy as np
import os
import tensorflow as tf
from flask import Flask, request, redirect, url_for, render_template, g

app = Flask(__name__)
BP = tf.keras.models.load_model('./Model/BP')
RNN = tf.keras.models.load_model('./Model/RNN')

@app.route('/')
def index():
    return render_template('index.html')
import prepdata
@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    with open(uploaded_file.filename) as f:
        lines = f.readlines()
        testnames = [l.strip() for l in lines[::2]]
        test = [l.strip() for l in lines[1::2]]
        
        BP_VoteWeight = 0.67
        RNN_VoteWeight = 1 - BP_VoteWeight

        testx_BP = np.array([prepdata.work(l) for l in test]).astype('float64')
        testx_RNN= np.array(LoadRNNdata(test)).astype('float64')
        pred_BP = BP.predict(testx_BP)
        pred_RNN = RNN.predict(testx_RNN)

        pred_vote = pred_BP * BP_VoteWeight + pred_RNN * RNN_VoteWeight

        data = ''
        for i in range(len(test)):
            if pred_vote[i][0]>=40: data += testnames[i][1:] + '\t' + 'non-umami\n'
            else: data += testnames[i][1:] + '\t' + 'umami, predicted threshold'+'\t'+ str(pred_vote[i][0]) + 'mmol/L\n'

    os.remove(uploaded_file.filename)
    return  render_template('index.html',data=data)


def SearchAcc(pred,testy,detail=False,show=True):

    if detail:
        print(pred, testy)

    # find the best Threshold and acc
    test_bitter_num = sum(testy == 40)
    test_umami_num = len(testy) - test_bitter_num
    test_num = len(testy)
    posP = sorted([pred[j][0] for j in range(test_umami_num)])
    negP = sorted([pred[j + test_umami_num][0] for j in range(test_bitter_num)])

    bestacc = 0
    bestmax = 0
    bestmin = 0
    accList = []
    maxList = []
    minList = []
    for i in range(test_bitter_num):
        err = 0
        min = 0
        for j in range(test_umami_num - 1, -1, -1):
            if posP[j] < negP[i]:
                min = posP[j]
                break
            else:
                err += 1
        acc = (test_num - err - i) / test_num
        if bestacc < acc:
            bestacc = acc
            bestmax = negP[i]
            bestmin = min
        if acc >= 0.9:
            accList.append(acc)
            maxList.append(negP[i])
            minList.append(min)

    if show:
        for i in range(len(accList)):
            print("acc=", accList[i], "upbound=", maxList[i], "lowbound=", minList[i])
        #print("--------------------------------------------")
        print("Bestacc=", bestacc, "upbound=", bestmax, "lowbound=", bestmin)
        print("---------------------End SearchAcc-------------------")
        #print("EPOCH=", EPOCHS)

    pred_label=pred>bestmin

    return bestacc,pred_label

TrainACC=[]
globaltrainx=1
globaltrainy=1

def SearchBestWeight(pred_BP,pred_RNN,testy,step=100):
    #print("Search best vote weight...")
    Best = 0
    BV = 0
    for i in range(step):
        BP_VoteWeight = i / step
        RNN_VoteWeight = 1 - BP_VoteWeight

        pred_vote = pred_BP * BP_VoteWeight + pred_RNN * RNN_VoteWeight
        acc,_ = SearchAcc(pred_vote, testy,False,False)
        if Best < acc:
            Best = acc
            BV = BP_VoteWeight
    print(Best,BV)

    return Best,BV



animo = 'ACDEFGHIKLMNPQRSTVWY2'
OPF={'A':'0000100110','R':'1101000000','N':'1000000100','D':'1011000100','C':'1000100110','Q':'1000000000',
'E':'1011000000','G':'0000100110','H':'1101101000','I':'0000110000','L':'0000110000','K':'1101100000','M':'0000100000',
'F':'0000101000','P':'0000000101','S':'1000000110','T':'1000100100','W':'1000101000','Y':'1000101000','V':'0000110100','2':'2222222222'
}
embed_weights = [ [int(i) for i in OPF[s]] for s in animo]
def LoadRNNdata(sequences):
#first row is ans
    maxlen=39
    x_train = []
    for l in sequences:
        name = l.split()[0]
        name += '2'*(maxlen-len(name))
        data=[embed_weights[animo.find(c)] for c in name]
        x_train.append(data)


    return x_train

import prepdata

@app.before_first_request
def before_first_request():
    pass


if __name__=='__main__':


    app.run()

