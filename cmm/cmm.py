# -*- coding: utf-8 -*-  
""" 
Created on Sat May 10 18:55:26 2014 
 
@author: rachel 
 
Function: convolution option of two pictures with same size (width,height) 
input: 3 feature maps (3 channels <RGB> of a picture) 
convolution: two 9*9 convolutional filters 
"""  
  
from theano.tensor.nnet import conv  
import theano.tensor as T  
import numpy, theano  
  
  
rng = numpy.random.RandomState(23455)  
  
# symbol variable  
input = T.tensor4(name = 'input')  
  
# initial weights  
w_shape = (2,3,9,9) #2 convolutional filters, 3 channels, filter shape: 9*9  
w_bound = numpy.sqrt(3*9*9)  
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape),  
                                dtype = input.dtype),name = 'W')  
  
b_shape = (2,)  
b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),  
                                dtype = input.dtype),name = 'b')  
                                  
conv_out = conv.conv2d(input,W)  
  
#T.TensorVariable.dimshuffle() can reshape or broadcast (add dimension)  
#dimshuffle(self,*pattern)  
# >>>b1 = b.dimshuffle('x',0,'x','x')  
# >>>b1.shape.eval()  
# array([1,2,1,1])  
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))  
f = theano.function([input],output)  
  
  
  
  
  
# demo  
import pylab  
from PIL import Image  
#minibatch_img = T.tensor4(name = 'minibatch_img')  
  
#-------------img1---------------  
img1 = Image.open(open('20140515202416843.png'))  
width1,height1 = img1.size  
img1 = numpy.asarray(img1, dtype = 'float32')/256. # (height, width, 3)  
  
# put image in 4D tensor of shape (1,3,height,width)  
img1_rgb = img1.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height1,width1) #(3,height,width)  
  
  
#-------------img2---------------  
img2 = Image.open(open('20140515202416843.png'))  
width2,height2 = img2.size  
img2 = numpy.asarray(img2,dtype = 'float32')/256.  
img2_rgb = img2.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height2,width2) #(3,height,width)  
  
  
  
#minibatch_img = T.join(0,img1_rgb,img2_rgb)  
minibatch_img = numpy.concatenate((img1_rgb,img2_rgb),axis = 0)  
filtered_img = f(minibatch_img)  
  
  
# plot original image and two convoluted results  
pylab.subplot(2,3,1);pylab.axis('off');  
pylab.imshow(img1)  
  
pylab.subplot(2,3,4);pylab.axis('off');  
pylab.imshow(img2)  
  
pylab.gray()  
pylab.subplot(2,3,2); pylab.axis("off")  
pylab.imshow(filtered_img[0,0,:,:]) #0:minibatch_index; 0:1-st filter  
  
pylab.subplot(2,3,3); pylab.axis("off")  
pylab.imshow(filtered_img[0,1,:,:]) #0:minibatch_index; 1:1-st filter  
  
pylab.subplot(2,3,5); pylab.axis("off")  
pylab.imshow(filtered_img[1,0,:,:]) #0:minibatch_index; 0:1-st filter  
  
pylab.subplot(2,3,6); pylab.axis("off")  
pylab.imshow(filtered_img[1,1,:,:]) #0:minibatch_index; 1:1-st filter  
pylab.show()  