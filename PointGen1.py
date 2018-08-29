import theano
import theano.tensor as T
import lasagne
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import glob
import scipy.misc
import time
import os

class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)

files = glob.glob("128_crop/*.jpg")
#files = os.listdir('128_crop/*.jpg')

batch_size = 16
CONTROLDIM=64

with open('netBestGeneratorCELEBa_BEGAN3.pickle','rb') as handle:
    nno = pickle.load(handle)



InputTensor = T.matrix(name='InputTensor')
TargetTensor = T.tensor4(name='TargetTensor')

time1 = time.time()
for imageNum in range(0,15619):
    seedsTemp = np.float32(np.random.rand(1, CONTROLDIM)*2-1)
    InputParams = theano.shared(seedsTemp)
    imageTargetTemp = scipy.misc.imread(files[imageNum])
    imageTargetTemp = np.transpose(imageTargetTemp,axes=[2,0,1])
    imageTargetTemp = imageTargetTemp.reshape([1,3,128,128])
    imageTargetTemp = np.float32(imageTargetTemp)/np.float32(255) * 2 - 1
    Target = theano.shared(imageTargetTemp)
    NetInput = InputParams.repeat(16,axis=0)
    NetTarget = Target.repeat(16,axis=0)

    loss = lasagne.objectives.squared_error(lasagne.layers.get_output(nno,NetInput),NetTarget)
    loss = T.mean(loss)
    updates = lasagne.updates.adam(loss,[InputParams],learning_rate=0.1)
    NetOutputFunc = theano.function([NetInput],[lasagne.layers.get_output(nno,NetInput),loss],updates=updates)

    iter = 0
    tolerate = 0
    lossFinal = 9999999999

    while iter<500:
        iter += 1
        Output, loss = NetOutputFunc(NetInput.eval())
        if loss < lossFinal:
            lossFinal = loss
            OutputFinal = Output
            print lossFinal
        if iter > 20 and  tolerate<3 and lossFinal > 0.1:
            iter = 0
            tolerate += 1
            seedsTemp = np.float32(np.random.rand(1, CONTROLDIM) * 2 - 1)
            InputParams = theano.shared(seedsTemp)
            NetInput = InputParams.repeat(16, axis=0)
            loss = lasagne.objectives.squared_error(lasagne.layers.get_output(nno, NetInput), NetTarget)
            loss = T.mean(loss)
            updates = lasagne.updates.adam(loss, [InputParams], learning_rate=0.1)
            NetOutputFunc = theano.function([NetInput], [lasagne.layers.get_output(nno, NetInput), loss], updates=updates)
    print time.time()-time1

    FinalParams = np.float32(InputParams.eval())
    FileName2save = 'Parameters/'+files[imageNum][9:20]
    scipy.io.savemat(FileName2save+'.mat',{'params':FinalParams})
    print imageNum
#f2, ((axx1, axx2)) = plt.subplots(1, 2, sharey=True)

#axx1.imshow(np.transpose(np.squeeze((OutputFinal[0, :, :, :]) + 1) / 2, axes=[1, 2, 0]))
#axx2.imshow(np.transpose(np.squeeze((imageTargetTemp[0, :, :, :]) + 1) / 2, axes=[1, 2, 0]))
#plt.show()
print 'c'