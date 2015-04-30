# encoding=utf8
'''
Created on 2015-4-29

@author: qq
'''
import math
import numpy
import random
import struct
import threading
import time

class MyThread(threading.Thread):    
    def __init__(self,threadname,tANN,idx_start,idx_end):    
        threading.Thread.__init__(self,name=threadname)  
        self.ANN=tANN  
        self.idx_start=idx_start  
        self.idx_end=idx_end  
    def run(self):  
        cDetaW,cDetaB,cError=self.ANN.backwardPropogation(self.ANN.traindata[self.idx_start],0)  
        for idx in range(self.idx_start+1,self.idx_end):  
            DetaWtemp,DetaBtemp,Errortemp=self.ANN.backwardPropogation(self.ANN.traindata[idx],idx)  
            cError += Errortemp  
            #cDetaW += DetaWtemp  
            #cDetaB += DetaBtemp  
            for idx_W in range(0,len(cDetaW)):  
                cDetaW[idx_W] += DetaWtemp[idx_W]  
                  
            for idx_B in range(0,len(cDetaB)):  
                cDetaB[idx_B] += DetaBtemp[idx_B]        
        return cDetaW,cDetaB,cError  
def sigmoid(inX): 
    return 1.0 / (1 + numpy.exp(-inX)) 
    
def difsigmoid(inX):  
    return sigmoid(inX) * (1.0 - sigmoid(inX))


def sigmoidMatrix(inputMatrix):  
    m,n=numpy.shape(inputMatrix)  
    outMatrix=numpy.mat(numpy.zeros((m,n)))  
    for idx_m in range(0,m):  
        for idx_n in range(0,n):  
            outMatrix[idx_m,idx_n]=sigmoid(inputMatrix[idx_m,idx_n])  
    return outMatrix


def softmax(inMatrix):  
    m,n=numpy.shape(inMatrix)  
    outMatrix=numpy.mat(numpy.zeros((m,n)))  
    soft_sum=0  
    for idx in range(0,n):  
        outMatrix[0,idx] = numpy.exp(inMatrix[0,idx])  
        soft_sum += outMatrix[0,idx]  
    for idx in range(0,n):  
        outMatrix[0,idx] /= soft_sum  
    return  outMatrix

def linear(inMatrix):  
    return  inMatrix

def softmaxErrorSum(errorMat,trainlabel,currentDataIdx):
    return -1.0*math.log(errorMat[0,int(trainlabel[currentDataIdx])])

def loadMNISTimage(absFilePathandName,datanum=60000):  
    images=open(absFilePathandName,'rb')  
    buf=images.read()  
    index=0  
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)  
    #print magic, numImages , numRows , numColumns  
    index += struct.calcsize('>IIII')  
    if magic != 2051:  
        raise Exception  
    datasize=int(784*datanum)  
    datablock=">"+str(datasize)+"B"  
    #nextmatrix=struct.unpack_from('>47040000B' ,buf, index)  
    nextmatrix=struct.unpack_from(datablock ,buf, index)  
    nextmatrix=numpy.array(nextmatrix)/255.0  
    #nextmatrix=nextmatrix.reshape(numImages,numRows,numColumns)  
    nextmatrix=nextmatrix.reshape(datanum,1,numRows*numColumns)  
    #for idx in range(0,numImages):  
    #    test=nextmatrix[idx,:,:]  
    #    print idx,numpy.shape(test)  
    #im = struct.unpack_from('>784B' ,buf, index)  
    #move=struct.calcsize('>784B')  
    #print move  
    #index += struct.calcsize('>784B')  
    #im=numpy.array(im)  
    #im = im.reshape(14,56)  
    #row,col=numpy.shape(im)  
    #print row,col  
    #fig = plt.figure()  
    #plotwindow = fig.add_subplot(111)  
    #plt.imshow(im , cmap='gray')  
    #plt.show()  
    #nextsum=59999*28*28  
    #print nextsum  
    #nextmatrix=struct.unpack_from('>47039216B' ,buf, index)  
    #nextmatrix=numpy.array(nextmatrix)  
    #nextmatrix=nextmatrix.reshape(59999,28,28)  
    #for idx in range(1,59999):  
        #temp=nextmatrix[idx,:,:]  
        #plt.imshow(temp,cmap='gray')  
        #plt.show()  
        #print temp  
      
    #print next  
      
    #for lines in images.readlines():  
        #print type(lines),lines  
      
    return nextmatrix, numImages  
      
def loadMNISTlabels(absFilePathandName,datanum=60000):  
    labels=open(absFilePathandName,'rb')  
    buf=labels.read()  
    index=0  
    magic, numLabels  = struct.unpack_from('>II' , buf , index)  
    #print magic, numLabels  
    index += struct.calcsize('>II')  
    if magic != 2049:  
        raise Exception  
      
    datablock=">"+str(datanum)+"B"  
    #nextmatrix=struct.unpack_from('>60000B' ,buf, index)  
    nextmatrix=struct.unpack_from(datablock ,buf, index)  
    nextmatrix=numpy.array(nextmatrix)  
    #for idx in range(0,numLabels):  
    #    test=nextmatrix[idx]  
    #    print idx,type(test),test  
    return nextmatrix, numLabels  
    
    
class MultiLayerNetwork(object):
    
    def __init__(self, inputCount, hiddenLayerCount, nodesInHiddenLayers, outputCount, maxIter=100,errorFunc = softmax,errorSumFunc = softmaxErrorSum,trainDataNum=200,recurrent=False):
        self.trainDataNum = trainDataNum
        self.decayRate = 0.2  
        self.punishFactor = 0.05  # 惩罚系数
        self.eps = 0.00001  # check g(x)
        self.hiddenLayerCount=hiddenLayerCount  
        self.inputCount=inputCount
        self.outputCount=outputCount
        self.numberOfLayers = hiddenLayerCount+2
        self.nodesInHiddenLayers = nodesInHiddenLayers
        self.nodesInHidden=[]
        self.errorFunc = errorFunc
        self.errorSumFunc = errorSumFunc
    
        for element in nodesInHiddenLayers:
            self.nodesInHidden.append(element)
        self.maxIter = maxIter
    
    
    def setSample(self,dataset,trainlabel):
        self.dataset = dataset
        self.trainlabel = trainlabel
    def setTrainDataNum(self,datanum):  
        self.trainDataNum=datanum  
        return
        
    def initWeight(self):
        #initial matrix  
        #nodesinLayers is a list
        self.nodesinLayers=[]  
        self.nodesinLayers.append(int(self.inputCount))
        self.nodesinLayers +=self.nodesInHidden
        self.nodesinLayers.append(int(self.outputCount))
        #weight matrix, it's a list and each element is a numpy matrix  
        #weight matrix, here is Wij, and in BP we may inverse it into wji  
        #here we store the matrix as numpy.array  
        self.weightMatrix=[]  
        self.B=[]
        for idx in range(0,self.numberOfLayers-1):  
            #Xaxier's scaling factor  
            #X. Glorot, Y. Bengio. Understanding the difficulty of training   
            #deep feedforward neural networks. AISTATS 2010.  
            s=math.sqrt(6)/math.sqrt(self.nodesinLayers[idx]+self.nodesinLayers[idx+1])
            tempMatrix=numpy.zeros((self.nodesinLayers[idx],self.nodesinLayers[idx+1]))  
            for row_m in range(0,self.nodesinLayers[idx]):  
                for col_m in range(0,self.nodesinLayers[idx+1]):  
                    tempMatrix[row_m,col_m]=random.random()*2.0*s-s  
            self.weightMatrix.append(numpy.mat(tempMatrix))  
            self.B.append(numpy.mat(numpy.zeros((1,self.nodesinLayers[idx+1]))))              
        return 0
    
    
    
    #前向传播
    def forwardPropogation(self,singleDataInput,currentDataIdx):  
        Ztemp = []
        #首先计算input层的输入
        
        
        
        
        Ztemp.append(numpy.mat(singleDataInput)*self.weightMatrix[0]+self.B[0])
        
        #定义激活函数计算后的输出
        Atemp = []
        #计算所有激活后的输出
        for idx in range(1,self.numberOfLayers-1):
            Atemp.append(sigmoidMatrix(Ztemp[idx-1]))
            Ztemp.append(numpy.mat(Atemp[idx-1])*self.weightMatrix[idx]+self.B[idx])
        
        Atemp.append(Ztemp[self.numberOfLayers-2])
        errorMat = self.errorFunc(Atemp[self.numberOfLayers-2])
        #errorMat2=softmax(Atemp[self.numberOfLayers-2]) 
        #print errorMat
        errorsum=self.errorSumFunc(errorMat,self.trainlabel,currentDataIdx)
        #errorsum2 = -1.0*math.log(errorMat[0,int(self.trainlabel[currentDataIdx])])
        #print 'errorsum',errorMat==errorMat2
        return Atemp,Ztemp,errorsum
    
    #最终输出的的结果与样本的差值
    def calThetaNl(self,Anl,Y,Znl):  
        thetaNl=self.errorFunc(Anl)-Y  
        return thetaNl
    
    
    #反向传播
    def backwardPropogation(self,singleDataInput,currentDataIdx):
        #先对网络做前向传播
        Atemp,Ztemp,temperror=self.forwardPropogation(numpy.mat(singleDataInput),currentDataIdx)
        
        
        Theta=[]  
        outlabels=numpy.mat(numpy.zeros((1,self.outputCount)))  
        outlabels[0,int(self.trainlabel[currentDataIdx])]=1.0
        
        #开始计算残差
        thetaNl=self.calThetaNl(Atemp[self.numberOfLayers-2], outlabels, Ztemp[self.numberOfLayers-2])
        Theta.append(thetaNl)  
        #开始反向传播
        for idx in range(1,self.numberOfLayers-1):
            #获取最后的索引
            inverseidx = self.numberOfLayers-1-idx
            thetaLPlus1=Theta[idx-1]
            #从反向获取那一层的权重矩阵
            W = self.weightMatrix[inverseidx]
            #获取输出层以外最后一层的输入值
            Z1 = Ztemp[inverseidx-1]
            thetal=thetaLPlus1*W.transpose()  
            #print "thetal temp",thetal  
            row_theta,col_theta=numpy.shape(thetal)  
            if row_theta != 1:  
                raise Exception  
            #print col_theta  
            for idx_col in range(0,col_theta):  
                #print idx_col  
                #print "dif",difsigmoid(Zl[0,idx_col])  
                thetal[0,idx_col] =thetal[0,idx_col]*difsigmoid(Z1[0,idx_col]) 
            Theta.append(thetal) 
        DetaW=[]  
        DetaB=[]  
        for idx in range(0,self.numberOfLayers-2):  
            inverse_idx=self.numberOfLayers-2-1-idx  
            #######################################################  
            #???pay great attention to the deminson of matrix???###  
            #######################################################  
            #dW=Theta[idx]*Atemp[inverse_idx].transpose()  
            dW=Atemp[inverse_idx].transpose()*Theta[idx]  
            #print dW  
            dB=Theta[idx]  
            DetaW.append(dW)  
            DetaB.append(dB)  
        DetaW.append(singleDataInput.transpose()*Theta[self.numberOfLayers-2])  
        DetaB.append(Theta[self.numberOfLayers-2])  
        #print "DetaW",DetaW  
        #print "DetaB",DetaB  
  
        return DetaW,DetaB,temperror  
    #计算惩罚
    def calpunish(self):  
        punishment=0.0  
        for idx in range(0,self.numberOfLayers-1):  
            temp=self.weightMatrix[idx]  
            idx_m,idx_n=numpy.shape(temp)  
            for i_m in range(0,idx_m):  
                for i_n in range(0,idx_n):  
                    punishment += temp[i_m,i_n]*temp[i_m,i_n]  
        return 0.5*self.punishFactor*punishment
    #更新权重与偏移
    def updatePara(self,DetaW,DetaB):  
        #update parameters  
        for idx in range(0,self.numberOfLayers-1):  
            #print DetaW[idx]  
            #print DetaB[idx]  
            inverse_idx=self.numberOfLayers-1-1-idx  
            self.weightMatrix[inverse_idx] -= self.decayRate*((1.0/self.trainDataNum)*DetaW[idx]+self.punishFactor*self.weightMatrix[inverse_idx])              
            #self.weightMatrix[inverse_idx] -= (self.decayRate*(DetaW[idx]+self.punishFactor*self.weightMatrix[inverse_idx]))     
            self.B[inverse_idx] -= self.decayRate*(1.0/self.trainDataNum)*DetaB[idx]  
            #self.B[inverse_idx] -= self.decayRate*DetaB[idx]  
        #print self.weightMatrix  
        #print self.B      
    
    def loadtraindata(self,absFilePathandName):  
        self.traindata,self.TotalnumoftrainData=loadMNISTimage(absFilePathandName,self.trainDataNum)  
        #print self.traindata[0]  
        return  
      
    def loadtrainlabel(self,absFilePathandName):  
        self.trainlabel,self.TotalnumofTrainLabels=loadMNISTlabels(absFilePathandName,self.trainDataNum)  
        if self.TotalnumofTrainLabels != self.TotalnumoftrainData:  
            raise Exception  
        return  
    
    def trainwithMultiThread(self):  
        Error_old=10000000000.0  
        iter_idx=0  
        while iter_idx<self.maxIter:  
            print "iter num: ",iter_idx,"==============================="  
            iter_idx += 1  
            cDetaW,cDetaB,cError=self.backwardPropogation(self.traindata[0],0)  
              
            segNum=int(self.trainDataNum/3)  
              
            work1 = MyThread('work1',self,1,segNum)  
            cDetaW1,cDetaB1,cError1=work1.run()  
            work2 = MyThread('work2',self,segNum,int(2*segNum))  
            cDetaW2,cDetaB2,cError2=work2.run()  
            work3 = MyThread('work3',self,int(2*segNum),self.trainDataNum)  
            cDetaW3,cDetaB3,cError3=work3.run()              
              
            while work1.isAlive() or work2.isAlive() or work3.isAlive():  
                time.sleep(0.005)  
                continue  
  
            cDetaW=cDetaW+cDetaW1+cDetaW2+cDetaW3  
            cDetaB=cDetaB+cDetaB1+cDetaB2+cDetaB3  
            cError=cError+cError1+cError2+cError3  
            cError/=self.trainDataNum  
            cError += self.calpunish()  
            print "old error",Error_old  
            print "new error",cError  
            Error_new=cError  
            if Error_old-Error_new < self.eps:  
                break  
            Error_old=Error_new  
            self.updatePara(cDetaW, cDetaB)  
        return  
    def train(self):
        #首先定义一个error
        Error_old=10000000000.0  
        iter_idx = 0
        while iter_idx<self.maxIter:
            iter_idx+=1
            #计算一个初始状态
            cDetaW,cDetaB,cError=self.backwardPropogation(self.traindata[0],0)
            #更新迭代
            #print "iteration : %d"%iter_idx
            for idx in range(1,self.trainDataNum):  
                DetaWtemp,DetaBtemp,Errortemp=self.backwardPropogation(self.traindata[idx],idx)  
                cError += Errortemp  
                #cDetaW += DetaWtemp  
                #cDetaB += DetaBtemp  
                for idx_W in range(0,len(cDetaW)):  
                    cDetaW[idx_W] += DetaWtemp[idx_W]  
                      
                for idx_B in range(0,len(cDetaB)):  
                    cDetaB[idx_B] += DetaBtemp[idx_B]    
            cError/=self.trainDataNum  
            cError += self.calpunish()  
            #print "old error",Error_old  
            #print "new error",cError  
            Error_new=cError  
            if Error_old-Error_new < self.eps:  
                break  
            Error_old=Error_new  
            self.updatePara(cDetaW, cDetaB)  
    
        
    def printWeightMatrix(self):  
        for idx in range(0,int(self.numberOfLayers)-1):  
            print self.weightMatrix[idx]  
            print self.B[idx]  
        return 0  
    
    def getTrainAccuracy(self):  
        accuracycount=0  
        for idx in range(0,self.trainDataNum):  
            Atemp,Ztemp,errorsum=self.forwardPropogation(self.traindata[idx],idx)  
            TrainPredict=softmax(Atemp[self.numberOfLayers-2])  
            print TrainPredict  
            Plist=TrainPredict.tolist()  
            LabelPredict=Plist[0].index(max(Plist[0]))  
            print "LabelPredict",LabelPredict  
            print "trainLabel",self.trainlabel[idx]  
            if int(LabelPredict) == int(self.trainlabel[idx]):  
                accuracycount += 1  
        print "accuracy:", float(accuracycount)/float(self.trainDataNum)  
        return  
    
    

if __name__ == "__main__":
    network=MultiLayerNetwork(784, 1, [50], 10, maxIter=200)
    network.initWeight()
    #print network.weightMatrix
    network.printWeightMatrix()
    network.loadtraindata("C:\\Users\\qq\\Documents\\data mining\\mmnist\\train-images.idx3-ubyte")
    network.loadtrainlabel("C:\\Users\\qq\\Documents\\data mining\\mmnist\\train-labels.idx1-ubyte")
    network.train()
    network.getTrainAccuracy()
    

    
    
    
   
