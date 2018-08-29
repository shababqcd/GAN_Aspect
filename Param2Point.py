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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

filesParam = glob.glob("Parameters/*.mat")
filesPoints = glob.glob("points/*.mat")
filesParam.sort()
filesPoints.sort()
#files = os.listdir('128_crop/*.jpg')

batch_size = 16
CONTROLDIM=64
TrainX = np.float32(np.zeros([124952,64]))
TrainY = np.float32(np.zeros([124952,98]))
for i in range(124952):
    paramsTemp = scipy.io.loadmat(filesParam[i])
    paramsTemp = paramsTemp['params']
    TrainX[i,:] =paramsTemp


for i in range(124952):
    pointsTemp = scipy.io.loadmat(filesPoints[i])
    pointsTemp = pointsTemp['test_points']
    pointsTemp=pointsTemp.flatten()
    TrainY[i,:] = pointsTemp
    if i%1000 == 0:
        print i


ValidationX = np.float32(np.zeros([31238,64]))
ValidationY = np.float32(np.zeros([31238,98]))
for i in range(31238):
    paramsTemp = scipy.io.loadmat(filesParam[i+124952])
    paramsTemp = paramsTemp['params']
    ValidationX[i,:] =paramsTemp


for i in range(31238):
    pointsTemp = scipy.io.loadmat(filesPoints[i+124952])
    pointsTemp = pointsTemp['test_points']
    ValidationY[i,:] =pointsTemp.flatten()


TrainY = TrainY/np.float32(128)
ValidationY = ValidationY/np.float32(128)

input_var = T.matrix('inputs')
target_var = T.matrix('targets')
batch_size = 100
networki = lasagne.layers.InputLayer(shape=(None,64),name='inputLayer')
netL1 = lasagne.layers.DenseLayer(networki,num_units=128,name='netL1')
netL2 = lasagne.layers.DenseLayer(netL1,num_units=128,name='netL2')
netL3 = lasagne.layers.DenseLayer(netL2 , num_units=128,name='netL3')
#netL4 = lasagne.layers.DenseLayer(netL3 , num_units=128,name='netL4')
#netL5 = lasagne.layers.DenseLayer(netL4 , num_units=16,name='netL5')
#netL6 = lasagne.layers.DenseLayer(netL5 , num_units=16,name='netL6')
#netL7 = lasagne.layers.DenseLayer(netL6 , num_units=16,name='netL7')
networkOut = lasagne.layers.DenseLayer(netL3 , num_units=98 ,nonlinearity=lasagne.nonlinearities.sigmoid, name='networkOut')

output =lasagne.layers.get_output(networkOut , inputs=input_var)

loss = lasagne.objectives.squared_error(output , target_var)

loss = loss.mean()

params = lasagne.layers.get_all_params(networkOut , trainable = True)
updates = lasagne.updates.adam(loss , params , learning_rate = 0.0003 )
#updates = lasagne.updates.sgd(loss , params , learning_rate = 0.01)
test_output = lasagne.layers.get_output(networkOut ,inputs=input_var, deterministic = True)
test_loss = lasagne.objectives.squared_error(test_output , target_var)
test_loss = test_loss.mean()

#test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), target_var),
  #                dtype=theano.config.floatX)
#test_acc = T.mean(T.eq(T._tensor_py_operators.squeeze(T.gt(test_output,0.5)),target_var),
#                  dtype=theano.config.floatX)

train_func = theano.function([input_var , target_var] , loss , updates=updates)
valid_func = theano.function([input_var , target_var] , test_loss)

print('Training...')
TRAINLOSS = np.array([])
VALIDATIONLOSS = np.array([])
num_epochs = 1000
validationErrorBest = 100000;

for epoch in range(num_epochs):

    train_err = 0
    train_acc = 0
    train_batches = 0

    start_time = time.time()
    for batch in iterate_minibatches(TrainX , TrainY , batch_size , shuffle=True):
        inputs , targets = batch

        train_err += train_func(inputs , targets)
        #err , tacc = valid_func(inputs, targets)
        train_batches += 1
        #train_acc += tacc
        #err = valid_func(inputs , targets)

    val_err = 0
    val_batches = 0
    val_acc = 0
    for batch in iterate_minibatches(ValidationX , ValidationY , batch_size , shuffle=False):
        inputs , targets = batch
        err = valid_func(inputs , targets)
        #xx , xxl = test_func(inputs , targets)
        val_err += err
        #val_acc += acc
        val_batches +=1

    validationError = val_err / val_batches
    if validationError < validationErrorBest:
            validationErrorBest = validationError
            with open('netBestParam2Point1.pickle' , 'wb') as handle:
                print('saving the model....')
                pickle.dump(networkOut , handle)


    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    #print("  training acc:\t\t{:.6f}".format(train_acc / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    #print("  validation acc:\t\t{:.6f}".format(val_acc / val_batches))
    TRAINLOSS = np.append(TRAINLOSS , train_err / train_batches)
    VALIDATIONLOSS = np.append(VALIDATIONLOSS , val_err / val_batches)
    scipy.io.savemat('LossesParam2Point1.mat',mdict={'TrainLoss':TRAINLOSS , 'ValidationLoss':VALIDATIONLOSS})



print 'yup'