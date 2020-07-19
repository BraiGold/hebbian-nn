# general
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# model
from Hebbian import Hebbian

# graficos
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# io
import os
import sys
import pickle

# manejor de overflows
import warnings

def plot3(X_transf,categories,batch_number=0 , speed=10, label='dimensiones'):
    for angle in range(0, 360,speed):
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D( *zip(*X_transf[:,batch_number*3:batch_number*3 +3]), c=categories)
        ax.view_init(30, angle)
        plt.title(f'Dimensiones {batch_number*3 +1}-{batch_number*3 + 3 }')
        ax.legend(*scatter.legend_elements(), title="Categorias")
        plt.draw()
        plt.pause(.001)
        if angle % 90 == 0:
            plt.savefig(f'./plots/{label}_{batch_number*3+1}_to_{batch_number*3 + 3}_angle_30_{angle}.png')
    

    for angle in range(30, 390,speed):
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D( *zip(*X_transf[:,batch_number*3:batch_number*3 +3]), c=categories)
        ax.view_init(angle, 0)
        plt.title(f'Dimensiones {batch_number*3 + 1 }-{batch_number*3 + 3 }')
        ax.legend(*scatter.legend_elements(), title="Categorias")
        plt.draw()
        plt.pause(.001)
        if angle % 90 == 0:
            plt.savefig(f'./plots/{label}_{batch_number*3+1}_to_{batch_number*3 + 3}_angle_{angle}_0.png')



def fit_if_necessary(model_file, data_file, rule='sanger',retrain=False,lr=0):
    # use data_file_name = tp2_training_dataset.csv (second parameter in argv , argv[2])
    
    data = pd.read_csv(f'./datasets/{data_file}',header=None,dtype=np.double)
    X,categories = data.iloc[:,1:],data.iloc[:,0]
    X_scaled = StandardScaler().fit_transform(X)

    
    if os.path.isfile(f'./models/{model_file}') and not retrain:
        print('usando modelo guardado')
        red = pickle.load( open( f'./models/{model_file}', 'rb' ) )
    else:
        if not lr:
            lr = 0.001 if rule == 'sanger' else 0.0001
        print(f'reentrenando regla {rule}:')
        
        # si hay un warning por overflow manejarlo como un error y correr denuevo con un lr mas chico
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            fitted_successfully = False
            while not fitted_successfully:
                try:
                    red = Hebbian(X_scaled.shape[1],9,lr,20000,rule=rule, change_lr='when_stuck')
                    red.fit(X_scaled)
                    fitted_successfully = True
                except Warning as e:
                    print('Un posible overflow ocurrio se volvera a intentar con un LR mas chico.\n el error:', e)
                    lr = lr / 2
                   
        pickle.dump( red, open( f'./models/{model_file}', 'wb' ) )

    o = np.sum(np.abs( np.dot( red.W, red.W.T) - np.identity(len(red.W)) ))/2
    print('ortogonalidad total del modelo:', o)

    return red, X_scaled, categories

if __name__ == "__main__":
    model_file = sys.argv[1]
    data_file = sys.argv[2]
    rule = 'oja' if '-oja' in sys.argv else 'sanger'

    red, X_scaled, categories = fit_if_necessary(model_file, data_file, rule)

    X_transf = red.transform(X_scaled)


    label = sys.argv[sys.argv.index('-label') + 1] if '-label' in sys.argv else 'dimensiones'
    animation_speed = int(sys.argv[sys.argv.index('-anim-speed') + 1]) if '-anim-speed' in sys.argv else 10
    
    for batch_number in range(3):
        plot3(X_transf,categories,batch_number=batch_number, speed=animation_speed,label=label)

    