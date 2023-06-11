import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras.layers.activation.leaky_relu import LeakyReLU
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Discriminator(Model):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24,)),
        tf.keras.layers.Reshape((24, 1)),  # Add a Reshape layer to prepare input for Conv1D  
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="LeakyReLU"), 
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="LeakyReLU"), 
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation="LeakyReLU"), 
        tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation="LeakyReLU"), 
        tf.keras.layers.Conv1D(filters=4, kernel_size=3, activation="LeakyReLU"), 
        tf.keras.layers.Flatten(),  # Flatten the output of Conv1D for the next Dense layer
        tf.keras.layers.Dense(2,activation='softmax')
    ])

  def call(self, x):
    return self.model(x)

class Generator(Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Reshape((24, 1)),  # Add a Reshape layer to prepare input for Conv1D
      tf.keras.layers.Conv1D(filters=24, kernel_size=3, activation="LeakyReLU"), 
      tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation="LeakyReLU"), 
      tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation="LeakyReLU"), # Replace Dense layer with Conv1D
      tf.keras.layers.Flatten(), 
      tf.keras.layers.Dense(2, activation="LeakyReLU")
      ])
    

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(2, activation="LeakyReLU"),
      tf.keras.layers.Dense(4, activation="LeakyReLU"),
      tf.keras.layers.Dense(8, activation="LeakyReLU"),
      tf.keras.layers.Dense(16, activation="LeakyReLU"),
      tf.keras.layers.Dense(24, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

generator = Generator()
discriminator = Discriminator()

generator.load_weights("weights/generator_weights.ckpt")

IMP_COL = [	'XOAVelocity',	'XPeakmg',	'XRMSmg',	'XKurtosis',	'XCrestFactor',	'XSkewness'	,'XDeviation',	'XPeaktoPeakDisplacement',
           'YOAVelocity',	'YPeakmg',	'YRMSmg',	'YKurtosis',	'YCrestFactor',	'YSkewness'	,'YDeviation',	'YPeaktoPeakDisplacement',
           'ZOAVelocity',	'ZPeakmg',	'ZRMSmg',	'ZKurtosis',	'ZCrestFactor',	'ZSkewness'	,'ZDeviation',	'ZPeaktoPeakDisplacement',
           'converted', 'target']
TAR_COL = 'target'

def run_model(file_path):
    DATASET_PATH = file_path
    df = pd.read_csv(DATASET_PATH)[IMP_COL]
    df = df. sort_values(by=["converted"])
    df = df.reset_index(drop=True)

    df = df.drop(['converted'], axis=1)
    df['target2'] = df['target'].apply(lambda x: 1 if x == 0 else 0)
    
    filtered_df = df[df['target'] == 1]
    max_values = filtered_df.max()
    min_values = filtered_df.min()
    
    
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns)
    test_data = df[['XOAVelocity',	'XPeakmg',	'XRMSmg',	'XKurtosis',	'XCrestFactor',	'XSkewness'	,'XDeviation',	'XPeaktoPeakDisplacement',
           'YOAVelocity',	'YPeakmg',	'YRMSmg',	'YKurtosis',	'YCrestFactor',	'YSkewness'	,'YDeviation',	'YPeaktoPeakDisplacement',
           'ZOAVelocity',	'ZPeakmg',	'ZRMSmg',	'ZKurtosis',	'ZCrestFactor',	'ZSkewness'	,'ZDeviation',	'ZPeaktoPeakDisplacement']].values
    labels = df[['target', 'target2']].values
    
    seperated_df = pd.DataFrame()
    
    denoised_all = []

    generated_data = generator(np.array(df.iloc[:,:24]), training=False)

    values = generated_data.numpy()
    denoised_all = values


    separated_XOAVelocity = [x[0] for x in denoised_all]
    separated_XPeakmg = [x[1] for x in denoised_all]
    separated_XRMSmg = [x[2] for x in denoised_all]
    separated_XKurtosis = [x[3] for x in denoised_all]
    separated_XCrestFactor = [x[4] for x in denoised_all]
    separated_XSkewness = [x[5] for x in denoised_all]
    separated_XDeviation = [x[6] for x in denoised_all]
    separated_XPeaktoPeakDisplacement = [x[7] for x in denoised_all]
    separated_YOAVelocity = [x[8] for x in denoised_all]
    separated_YPeakmg = [x[9] for x in denoised_all]
    separated_YRMSmg = [x[10] for x in denoised_all]
    separated_YKurtosis = [x[11] for x in denoised_all]
    separated_YCrestFactor = [x[12] for x in denoised_all]
    separated_YSkewness = [x[13] for x in denoised_all]
    separated_YDeviation = [x[14] for x in denoised_all]
    separated_YPeaktoPeakDisplacement = [x[15] for x in denoised_all]
    separated_ZOAVelocity = [x[16] for x in denoised_all]
    separated_ZPeakmg = [x[17] for x in denoised_all]
    separated_ZRMSmg = [x[18] for x in denoised_all]
    separated_ZKurtosis = [x[19] for x in denoised_all]
    separated_ZCrestFactor = [x[20] for x in denoised_all]
    separated_ZSkewness = [x[21] for x in denoised_all]
    separated_ZDeviation = [x[22] for x in denoised_all]
    separated_ZPeaktoPeakDisplacement = [x[23] for x in denoised_all]
    
    seperated_df['XOAVelocity'] = pd.DataFrame(separated_XOAVelocity, columns=['XOAVelocity']) 
    seperated_df['XPeakmg'] = pd.DataFrame(separated_XPeakmg, columns=['XPeakmg'])
    seperated_df['XRMSmg'] = pd.DataFrame(separated_XRMSmg, columns=['XRMSmg']) 
    seperated_df['XKurtosis'] = pd.DataFrame(separated_XKurtosis, columns=['XKurtosis']) 
    seperated_df['XCrestFactor'] = pd.DataFrame(separated_XCrestFactor, columns=['XCrestFactor']) 
    seperated_df['XSkewness'] = pd.DataFrame(separated_XSkewness, columns=['XSkewness']) 
    seperated_df['XDeviation'] = pd.DataFrame(separated_XDeviation, columns=['XDeviation']) 
    seperated_df['XPeaktoPeakDisplacement'] = pd.DataFrame(separated_XPeaktoPeakDisplacement, columns=['XPeaktoPeakDisplacement']) 
    seperated_df['YOAVelocity'] = pd.DataFrame(separated_YOAVelocity, columns=['YOAVelocity']) 
    seperated_df['YPeakmg'] = pd.DataFrame(separated_YPeakmg, columns=['YPeakmg'])
    seperated_df['YRMSmg'] = pd.DataFrame(separated_YRMSmg, columns=['YRMSmg']) 
    seperated_df['YKurtosis'] = pd.DataFrame(separated_YKurtosis, columns=['YKurtosis']) 
    seperated_df['YCrestFactor'] = pd.DataFrame(separated_YCrestFactor, columns=['YCrestFactor']) 
    seperated_df['YSkewness'] = pd.DataFrame(separated_YSkewness, columns=['YSkewness']) 
    seperated_df['YDeviation'] = pd.DataFrame(separated_YDeviation, columns=['YDeviation']) 
    seperated_df['YPeaktoPeakDisplacement'] = pd.DataFrame(separated_YPeaktoPeakDisplacement, columns=['YPeaktoPeakDisplacement']) 
    seperated_df['ZOAVelocity'] = pd.DataFrame(separated_ZOAVelocity, columns=['ZOAVelocity']) 
    seperated_df['ZPeakmg'] = pd.DataFrame(separated_ZPeakmg, columns=['ZPeakmg'])
    seperated_df['ZRMSmg'] = pd.DataFrame(separated_ZRMSmg, columns=['ZRMSmg']) 
    seperated_df['ZKurtosis'] = pd.DataFrame(separated_ZKurtosis, columns=['ZKurtosis']) 
    seperated_df['ZCrestFactor'] = pd.DataFrame(separated_ZCrestFactor, columns=['ZCrestFactor']) 
    seperated_df['ZSkewness'] = pd.DataFrame(separated_ZSkewness, columns=['ZSkewness']) 
    seperated_df['ZDeviation'] = pd.DataFrame(separated_ZDeviation, columns=['ZDeviation']) 
    seperated_df['ZPeaktoPeakDisplacement'] = pd.DataFrame(separated_ZPeaktoPeakDisplacement, columns=['ZPeaktoPeakDisplacement']) 

    
    noise = df[['XOAVelocity']].values.flatten()
    seperated = seperated_df[['XOAVelocity']].values.flatten()
    seperated_csv = pd.DataFrame({column: ((seperated_df[column] * (max_values[column] - min_values[column])) + min_values[column]) for column in seperated_df.columns})
    
    return seperated_csv.to_json()