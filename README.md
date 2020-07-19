# TP2 Redes Neuronales
El tp fue probado en Ubuntu usando python 3.6.7

Es necesario tener instalado: 

- numpy
- pandas
- matplotlib
- scikitlearn
- mpl_toolkits (para graficar 3D)

## para correr el tp:

python3 main.py <modelo.sav> <dataset.csv> [-oja] [-anim-speed int] [-label img_label]

- <modelo.sav> nombre del archivo del modelo que TIENE que estar en el directorio /models ej: sanger_full.sav (este viene precargado en /models/). De no existir un archivo con este nombre en /models se entrenara un nuevo modelo y se guardara con este nombre. Parametro obligatorio.

- <dataset.csv> nombre del archivo csv con la data a transformar que TIENE que estar en /datasets ej: tp2_training_dataset.csv . paramentro obligatorio.

- [-oja] si se aplica este parametro sera usada la regla de Oja en caso de que <modelo.sav> no este en /models, de no usarse este parametro se aplicara la regla de Sanger por defecto. Parametro opcional.

- [-anim-speed n] con este parametro opcional (un entero n  en [1,360]) se especifica una velocidad determinada para la animacion que rota los graficos. si no se usa, por defecto la velocidad es 10 (ni muy rapida ni muy lenta).

- [-label str(img_label)] si se usa este parametro opcional las a las imagenes que se guarden en /plots se les pondra el prefijo img_label  

### Un posible ejemplo para correr el tp:

 **DETALLE**: el formato .md de este readme interpreta los underscores como italic, si estas leyendo esto en formato plano dejo tambien el ejemplo en texto plano:

#### Usando un modelo ya guardado en /models:

(formato Markdown .md:)

$ python3 main.py sanger\_full.sav tp2\_training_dataset.csv

 (texto plano:)

$ python3 main.py sanger_full.sav tp2_training_dataset.csv

#### Etrenando un nuevo modelo (con Sanger):

(formato Markdown .md:)

$ python3 main.py nuevo\_modelo.sav tp2\_training_dataset.csv

 (texto plano:)

$ python3 main.py nuevo_modelo.sav tp2_training_dataset.csv

#### Etrenando un nuevo modelo (con Oja):

(formato Markdown .md:)

$ python3 main.py nuevo\_modelo.sav tp2\_training_dataset.csv -oja

 (texto plano:)

$ python3 main.py nuevo_modelo.sav tp2_training_dataset.csv -oja

 ## Datasets y modelos precargados

 vienen precargados los modelos sanger\_full.sav y oja\_full.sav (sanger_full.sav y oja_full.sav en texto plano) estos fueron entrenados con toda la data.

 Tambien vienen precargados diferentes modelos que se generan con usando split_dataset.py, este splitea de la data (diferentes dimensiones) y con una parte de la data entrena un modelo y con la genera un dataset. Pueden ser usados para ver el comportamiento con diferentes sets de datos asi como tambien para ver como transforma el algoritmo data que nunca vio.