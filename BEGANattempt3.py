import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm
import time
import pickle
import sys
from MODELSrgb import GeneratorNetwork128x128, DiscriminatorNetwork128x128

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





sys.setrecursionlimit(50000)


num_epochs = 10000
#my_loss = 100000
k_t=0
lambda_k = 0.001
gamma = 0.5

CONTROLDIM = 64
batch_size = 16
NN = 64

Gen_out_layer = GeneratorNetwork128x128(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Gen')
Disc_out_layer = DiscriminatorNetwork128x128(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Disc')

print 'BUILDING THE MODEL....'
noise_var = T.matrix(name='noise')
input_var = T.tensor4(name='inputs')
K_T = T.scalar(name='K_T')

'''TestFunc = theano.function([noise_var,Carry],lasagne.layers.get_output(GenL3input,inputs=noise_var))
seeds = np.random.rand(batch_size, CONTROLDIM)*2-1
TestFunc2 = theano.function([noise_var],lasagne.layers.get_output(GenL2BN,inputs=noise_var))'''

realOutput = lasagne.layers.get_output(Disc_out_layer,inputs=input_var)
fakeOutput = lasagne.layers.get_output(Disc_out_layer,
                                       inputs=lasagne.layers.get_output(Gen_out_layer,inputs=noise_var))
generator_Output = lasagne.layers.get_output(Gen_out_layer,inputs=noise_var)

Disc_loss_real = T.abs_(realOutput-input_var)
Disc_loss_real = Disc_loss_real.mean()

Disc_loss_fake = T.abs_(generator_Output - fakeOutput)
Disc_loss_fake = Disc_loss_fake.mean()

DiscLoss = Disc_loss_real - K_T * Disc_loss_fake
GenLoss = T.abs_(generator_Output - fakeOutput)
GenLoss = GenLoss.mean()

Gen_params = lasagne.layers.get_all_params(Gen_out_layer, trainable=True)
Disc_params = lasagne.layers.get_all_params(Disc_out_layer, trainable=True)

updates = lasagne.updates.adam(GenLoss,Gen_params,learning_rate=0.0001,beta1=.5,beta2=0.999)
updates_Disc = lasagne.updates.adam(DiscLoss,Disc_params,learning_rate=0.0001,beta1=.5,beta2=0.999)
updates.update(updates_Disc)

print 'COMPILING THE MODEL...'
TrainFunction = theano.function([noise_var,input_var,K_T],[DiscLoss,GenLoss,Disc_loss_real,Disc_loss_real+T.abs_(gamma*Disc_loss_real-GenLoss)],
                                updates=updates,allow_input_downcast=True)

GEN_LOSS = np.array([])
DISC_LOSS = np.array([])
M_GLOBAL = np.array([])

m_global_value_best = 100000000000
print 'TRAINING STARTED...'
for epoch in range(num_epochs):
    train_error = 0
    gen_loss_value = 0
    disc_loss_value = 0
    m_global_value = 0
    start_time = time.time()
    #carry = np.float32(1)
    for DBname in range(4):
        print 'Loading database ' + str(DBname+1)
        Data = scipy.io.loadmat('DB'+str(DBname+1)+'.mat')
        RealDataX = Data['images']
        RealDataX = np.transpose(RealDataX,[0,3,1,2])
        RealDataX = RealDataX / np.float32(255)
        RealDataX = RealDataX * 2 - 1
        for dummy in range(2400):
            seeds = np.float32(np.random.rand(batch_size, CONTROLDIM)*2-1)
            sampleIndices = np.random.permutation(RealDataX.shape[0])
            samples = RealDataX[sampleIndices[0:batch_size],]
            disc_error , gen_error, Disc_Loss_real_Num,M_Global_Num  = TrainFunction(seeds,samples,k_t)
            k_t = np.clip(k_t+lambda_k*(gamma*Disc_Loss_real_Num-gen_error),0,1)
            gen_loss_value += gen_error
            disc_loss_value += disc_error
            m_global_value += M_Global_Num
            #carry = np.float32(carry- 1.0/1022.0)
            #gen_loss_value += genLossFunction(seeds,samples)
            #disc_loss_value += DiscLossFunction(seeds,samples)
            if dummy%100==0:
                print 'Iteration ' + str(dummy+1) + ' for database '+ str(DBname+1) + ' finished successfully.'
    if m_global_value<m_global_value_best:
        m_global_value_best=m_global_value
        with open('netBestDiscriminatorCELEBa_BEGAN3.pickle', 'wb') as handle:
            print('saving the model Discriminator....')
            pickle.dump(Disc_out_layer, handle)

        with open('netBestGeneratorCELEBa_BEGAN3.pickle', 'wb') as handle:
            print('saving the model Generator....')
            pickle.dump(Gen_out_layer, handle)

    '''my_loss_temp = gen_loss_value + disc_loss_value
    if my_loss_temp < my_loss:
        my_loss = my_loss_temp
        with open('netLastDiscriminatorMyLossAERESINJAttemp1BigData.pickle', 'wb') as handle:
            print('saving the model Discriminator MyLoss....')
            pickle.dump(Disc_out_layer, handle)

        with open('netLastGeneratorMyLossAERESINJAttempt1BigData.pickle', 'wb') as handle:
            print('saving the model Generator MyLoss....')
            pickle.dump(Gen_out_layer, handle)'''

    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    #print("  training loss acc:\t\t{}".format(train_error/10))
    print("  training loss generator:\t\t{:.6f}".format(gen_loss_value / 10))
    print("  training loss discriminator:\t\t{:.6f}".format(disc_loss_value / 10))
    print("  training m_global:\t\t{:.6f}".format(m_global_value))
    GEN_LOSS = np.append(GEN_LOSS,gen_loss_value)
    DISC_LOSS = np.append(DISC_LOSS,disc_loss_value)
    M_GLOBAL = np.append(M_GLOBAL,m_global_value)
    scipy.io.savemat('LossesCELEBa_BEGAN3.mat',mdict={'genLoss':GEN_LOSS,'discLoss':DISC_LOSS,'Mglobal':M_GLOBAL})

print'yup'