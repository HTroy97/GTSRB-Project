from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import time
import glob 
import cv2
from skimage.color import rgb2grey
import pandas as pd
import numpy.matlib as npmat

#Clase que definel el modelo de Red Neuronal
#Class used to define the Neural Network model
class Neural_Network(nn.Module):
    #Función que construye el modelo y define las capas y las neuronas de la red
    #Constructor function of the model and it defines the layers and the amount of neurons
    def __init__(self,input_size, output_size,learning_rate, weight):
        super(Neural_Network, self).__init__()
        self.layer1= nn.BatchNorm1d(4096)
        self.linear1= nn.Linear(input_size,100) 
        self.linear2= nn.Linear(100,50) 
        self.linear3= nn.Linear(50,43)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
    #Función que realiza la predicción
    #Function to make the prediction
    def forward(self, x):
        x=torch.tanh(self.linear1(self.layer1(torch.Tensor.float(x))))
        x=torch.tanh(self.linear2(x))
        x=self.linear3(x)
        return x
    #Función para calcular costo
    #Function to calculate the cost between the labels and the predicted ouputs
    def calculate_cost(self, y_hat, y):
        cost=self.criterion(y_hat,y)
        return cost  
    #To calculate F1score we need the output between 0 and 1, for which we use 'Softmax' and we get a joint probability of 1 for all of the classes
    #We define Softmax on its own because the cost function used (crossEntropyLoss) already applies softmax and we need the matrix prior to the activation step
    def softmax(self,y_hat): 
        y_hat_soft=F.softmax(y_hat,dim=1)
        return y_hat_soft

#Clase que se utilza para definir el dataset, tiene la función de inicialización, que establece las imágenes y los labels del dataset.
#Class used to define the dataset, it sets the images and the labels as attributes of the class
class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

#Definimos la ubicacion de las carpetas con las imágenes para train y para test e inicializamos la tarjeta gráfica
#We define the location of the images folders for training and testing and we initialize the GPU and some arreays
data_path = 'Insert datapath to training images folder'
data_path_t = 'Insert datapath to test images folder'
images = []
image_labels = []

images_t = []
image_labels_t = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Cargamos la data para entrenar el modelo
#We load the data for the training
for i in range(43):
    image_path = data_path + '/' + format(i, '05d') + '/'
    #Se cargan todas las imágenes con la librería cvread
    #cvread library is used to load images, to turn them into black & white, to rescale the image values to 0-1, and we set all the images to 64x64 pixels    for img in glob.glob(image_path + '*.ppm'):
        image = cv2.imread(img)
        image = rgb2grey(image) #convertimos la imagen a blanco y negro
        image = (image / 255.0) # reescalamos la imagen a valores entre 0 y 1
        image = cv2.resize(image, (64, 64)) #cambiamos el tamaño de la imagen a 64x64 
        images.append(image)
        
        #Hacemos la matriz de labels que tiene 43 valores, donde la posición del 1 es igual a la clase y en las demás posiciones hay 0
        #We build the labels matrix with 43 classes, where the position of the 1 represents the class and the other positions hold 0
        labels = np.zeros((43, ), dtype=np.float32)
        labels[i] = 1.0
        image_labels.append(labels)

#Se convierte la matriz de imágenes y de labels en tensores y se envían a la tarjeta gráfica
#Transform the matrix to tensors and then to the GPU
images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
images = torch.tensor(images)
image_labels = np.matrix(image_labels).astype(np.float32)
image_labels = np.reshape(image_labels, (39209,43))
image_labels = torch.LongTensor(image_labels).cuda()
images = np.reshape(images, (39209,4096)).cuda()
dataset=Dataset(images, image_labels)

#Cargamos la data para el test
#We load the data for testing
image_path_t = data_path_t + '/'
for img_t in glob.glob(image_path_t + '*.ppm'):
    image_t = cv2.imread(img_t)
    image_t = rgb2grey(image_t)
    image_t = (image_t / 255.0) # rescale
    image_t = cv2.resize(image_t, (64, 64)) #resize
    images_t.append(image_t)

#Para generar la matriz de labels para el dataset de test utilizamos la última columna del archivo csv que nos dicta el número de clase de cada imagen
#To generate the labels matrix for testing we use the last column of the csv file attached to the images folder
labels_t2 = pd.read_csv('Datapath to csv file containing data of the images', usecols=[7])
labels_t2=np.array(labels_t2.values)
labels_t2=(labels_t2[:]==np.arange(labels_t2.max()+1)).astype(int)

#Hacemos el dataset de validación de los tensores de las imágenes y los tensores de los labels
#We build the test dataset from images tensors and labels tensors
images_t = np.stack([img_t[:, :, np.newaxis] for img_t in images_t], axis=0).astype(np.float32)
images_t = torch.tensor(images_t)
image_labels_t = np.matrix(labels_t2).astype(np.float32)
image_labels_t = np.reshape(labels_t2, (12630,43))
image_labels_t = torch.LongTensor(labels_t2)
images_t = np.reshape(images_t, (12630,4096))
dataset_test=Dataset(images_t, image_labels_t)

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
test_loader =torch.utils.data.DataLoader(dataset_test, batch_size=12630, shuffle=True)

#Vector de pesos para cálculo de función de costo ya que las clases están desbalanceadas
#Weights vector for cost calculation given that classes are unbalanced
pesos = np.array([210, 2220, 2250, 1410, 1980, 1860, 420, 1440, 1410, 1470, 2010, 1320, 2100, 2160, 780, 630, 420, 1110, 1200, 210, 360, 330, 390, 512, 270, 1500, 600, 240, 540, 270, 450, 780, 240, 689, 420, 1200, 390, 210, 2070, 300, 360, 240, 240])
pesos2 = 1/pesos
suma = np.sum(pesos2)
pesos = torch.FloatTensor(pesos2/suma)

#Inicialización de Red Neuronal, definimos optimizador y scheduler para learning rate decay, y el número de épocas
#We initialize the Neural network, define the optimize, and scheduler for learning rate decay, as well as the epochs number
model = Neural_Network(4096, 43, 0.01, pesos)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) #0.001
scheduler = StepLR(optimizer, step_size=15, gamma=0.8) # lr decay 0.8
epocas = 60
#Lists and variables initialized for further calculation
#Listas y variables inicializadas para futuros cálculos
cost_list_train=[]
cost_list_test=[]
COST_test=0
cost=0
y_hat_list=[]
batch_y_list=[]

#Ciclo FOR principal de Entrenamiento
#Main FOR Cycle for training
for i in range(epocas):
    y_hat_list=[]
    batch_y_list=[]
    COST=0
    model.to(device)
    time_inicial=time.time()
    #FOR interno para entrenar con minibatches  
    #Internal FOR Cycle to train minibatches 
    for j, (images, image_labels) in enumerate(train_loader, 0):
        #Los gradientes del optimizador en 0 - We set the optimizer gradients to 0    
        optimizer.zero_grad() 
        batch_x = images
        batch_y = image_labels
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
        y_hat = model.forward(batch_x) 
        #Calculamos la predicción y el costo - We calculate Prediction and Cost
        cost = model.calculate_cost(y_hat, torch.max(batch_y, 1)[1] ).cuda()
        cost.backward()
        COST += cost
        y_pred =model.softmax(y_hat)

        y_hat_list.append(y_pred)
        y_concat=torch.cat((y_hat_list),dim=0)
        batch_y_list.append(batch_y)
        batch_y_concat=torch.cat((batch_y_list),dim=0)
        optimizer.step() #Actualizamos el optimizador - We update the optimizer gradients
    #We calculate the time taken to make the predictions for reference
    #Calculamos el tiempo en hacer las predicciones, para referencia
    time_forward=time.time()-time_inicial
    

    y_hat_train = np.asarray(y_concat.detach().cpu(), dtype=np.float32)
    batch_y_train= np.asarray(batch_y_concat.detach().cpu(),dtype=np.float32)
    #El costo se calcula en cada minibatch así que se debe dividir entre el número de épocas
    #The cost is calculated in each minibatch so it must be divide between the epochs number
    COST=COST/(j+1) 
    cost_list_train.append(COST) #Anexamos el costo a un vector para imprimir al final - We append the cost to a vector to get the progress
    puntaje_f1 = f1_score(batch_y_train,  (y_hat_train > 0.5), average='weighted')
    print("Epoca: ",i)
    print("learning rate: ", scheduler.get_lr())
    print("Time: ", time_forward)
    print("Costo TRAIN: ",COST)
    print("F1 score TRAIN: ",puntaje_f1)
    
    #We start testing with the torch.no_grad option
    #Se inicia la validación con la opción torch.no_grad
    model.cpu()
    with torch.no_grad():
        model.eval()
        for h, (images_t, image_labels_t) in enumerate(test_loader, 0):
            batch_x_t = images_t
            batch_y_t = image_labels_t
            batch_x_t, batch_y_t = Variable(batch_x_t), Variable(batch_y_t)
            y_hat_t = model.forward(batch_x_t) #Calculamos el costo - We calculate the cost
            costtest = model.calculate_cost(y_hat_t, torch.max(batch_y_t, 1)[1] )

    y_pred_t =model.softmax(y_hat_t)    
    y_hat_test = np.asarray(y_pred_t.detach().cpu(), dtype=np.float32)
    batch_y_test= np.asarray(batch_y_t.detach().cpu(),dtype=np.float32)
    cost_list_test.append(costtest)
    puntaje_f1_test = f1_score(batch_y_test,  (y_hat_test > 0.5), average='weighted')
    print("Costo TEST: ", costtest)
    print("F1 score TEST: ",puntaje_f1_test)
    scheduler.step()  


#Graficamos la función de costo para la validacion y el entrenamiento
#We graph the cost function for testing and training
plt.figure()
plt.plot(list(range(epocas)), cost_list_test, '-r',label='Test')
plt.plot(list(range(epocas)), cost_list_train, '-b',label='Train')
legend= plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.xlabel('Epoca')
plt.ylabel('Costo')
plt.show()