import opengm
import vigra
import numpy
import time
import sys

fname = "135069.jpg"
img = vigra.readImage(fname)
img = numpy.sum(img,axis=2)
img = vigra.resize(img,[s/1 for s in img.shape])
noise = numpy.random.random(img.size).reshape(img.shape)*255
print noise.shape
img += noise
img -= img.min()
img /= img.max()
print "shape", img.shape
vigra.imshow(img)
#vigra.show()

threshold = 0.24
labelsNaive = img > threshold
vigra.imshow(labelsNaive)
#vigra.show()

nVar = img.size
nLabelsPerVar = 2
variableSpace = numpy.ones(nVar)*nLabelsPerVar
gm = opengm.gm(variableSpace)


t0 = time.time()
# add unaries
for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        
        energy0 = img[x,y] -threshold
        energy1 = threshold - img[x,y]
        unaryFunction = numpy.array([energy0,energy1])
        
        # add unary function to graphical model
        functionId = gm.addFunction(unaryFunction)
        
        # add unary factor to graphical model
        variableIndex = y +x*img.shape[1]
        
        gm.addFactor(functionId,variableIndex)
        
# add 2. order regularizer
# ``Potts``- regularizer
beta = 0.1
pottsFunction = numpy.zeros([2,2])
pottsFunction[0,1]=beta
pottsFunction[1,0]=beta

# add 2. order function to graphical model
# but only ONCE
pottsFunctionId = gm.addFunction(pottsFunction)


for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        
        # add "horizontal" second order factor
        if x+1 < img.shape[0]:
            variableIndex0 = y + x*img.shape[1]
            variableIndex1 = y + (x+1)*img.shape[1]
            
            gm.addFactor(pottsFunctionId,[variableIndex0,variableIndex1])
        
        # add "vertical"  facator
        if y+1 < img.shape[1]:
            variableIndex0 = y + x*img.shape[1]
            variableIndex1 = (y+1) + x*img.shape[1]
            # add "horizontal" second order factor
            gm.addFactor(pottsFunctionId,[variableIndex0,variableIndex1])
    
t1 = time.time()
print "build model in", t1-t0, "sek"

gm=None

t0=time.time()
beta = 0.3
unaries=numpy.ones(img.shape+(2,))
unaries[:,:,0] = img-threshold
unaries[:,:,1] = -img+threshold
potts=opengm.PottsFunction([2,2],0.0,beta)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)

t1=time.time()
print "build model in", t1-t0, "sek"   

graphCut = opengm.inference.GraphCut(gm=gm)
graphCut.infer()
labels = graphCut.arg()
labels = labels.reshape(img.shape)


vigra.imshow(vigra.taggedView(labels,'xy'))



