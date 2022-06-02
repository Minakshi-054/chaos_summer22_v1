from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import argparse
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').disabled = True

import numpy as np
from pathlib import Path
import tensorflow as tf
import sys
import os
from datetime import datetime

from sklearn.metrics import classification_report

#CUDA_VISIBLE_DEVICES=3,4,5,6,7 python train.py 2 24 222 train

THRESHOLD = 0.5
tf.get_logger().setLevel('INFO')

assert os.environ.get('CUDA_VISIBLE_DEVICES'), """Please use CUDA_VISIBLE_DEVICES environment variable to run the experiment
Eg. CUDA_VISIBLE_DEVICES=1,2,3 python train.py --options"""

parser = argparse.ArgumentParser()
parser.add_argument('--version_data', required=True)
parser.add_argument('--version_model', required=True)
parser.add_argument('--version_catalog', required = True)
parser.add_argument('--num_gpu', required=True, type = int)
parser.add_argument('--sanity_check', default=1, type = int, choices=[0,1])
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--sanity_epoch', default = 4, type = int)
parser.add_argument('--train_batch_size', default=128, type = int)
parser.add_argument('--test_batch_size', default=128, type = int)
parser.add_argument('--mode',required=True, choices = ['train', 'test'])
args = parser.parse_args()

version_catalog = args.version_catalog
version_data = args.version_data
version_model = args.version_model
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
sanity_check = args.sanity_check == 1

if sanity_check:
    print("The sanity check is enabled, running validation and training for following configs")
    limit = max(train_batch_size, test_batch_size) * 2
    print(f"Run for two batch, input size: {limit}")
    print(f"Sanity Epoch Check, epoch size:  {args.sanity_epoch}")
    print("!!!!!NOTE")
    print("Dsiable sanity_check flags to start training")
else:
    print(f"The sanity check is disabled, {args.mode} starting soon")
    limit = None

num_gpu = args.num_gpu
epoch = args.epoch if sanity_check else args.sanity_epoch
mode = args.mode

if not sanity_check:    
    cat_root = "catalog"
    rep_root = "reports"
else:
    cat_root = rep_root = "temp"

if not sanity_check:
    assert not Path(f"{cat_root}/version_{version_catalog}").exists(), f"version-{version_catalog} Exists in the catalog"

gpulist = [f"/gpu:{i}" for i in range(num_gpu)]

strategy = tf.distribute.MirroredStrategy(devices=gpulist)
if not sanity_check:    
    checkpoint_root = f"/data/keshav/CACHE/version_{version_model}"
else:
    checkpoint_root = f"temp"

Path(checkpoint_root).mkdir(exist_ok=True)

checkpoint_path = f"{checkpoint_root}/model.ckpt"

root = Path(f"/data/keshav/DATA/version_{version_data}")

assert root.exists(), f"{root} doesn't exists"

with open(f"{cat_root}/version_{version_catalog}.txt", "w") as f:
    header = f"""
    DATE : {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
    version_catalog : {version_catalog}
    version_data : {version_data}
    version_model : {version_model}
    train_batch_size : {train_batch_size}
    test_batch_size : {test_batch_size}
    
    num_gpu : {num_gpu}
    limit : {limit}
    epoch : {epoch}
    mode : {mode}
    
    checkpoint_path : {checkpoint_path}
    
    data_root = {root}

    trainpath = {root} / 'train.npz'
    valpath = {root} / 'val.npz'
    testpath = {root} / 'par.npz'
    predpath = {root} / 'pred.npz'
    
    """
    
    with open("preparedata.py", "r") as p:
        header_ = """
        ########################################################
        ###############       preparedata.py       #############
        ########################################################
        
        """
        data_header = header_ + p.read()
        
    
    with open("train.py", "r") as t:
        header_ = """
        ########################################################
        ###############       train.py       ###################
        ########################################################
        
        """
        train_header = header_ + t.read()
    
    content = header + "\n" + data_header + "\n" + train_header
    
    f.write(content)

trainpath = root / 'train.npz'
valpath = root / 'val.npz'
testpath = root / 'par.npz'
predpath = root / 'pred.npz'

BATCH_SIZE = train_batch_size
TEST_BATCH_SIZE = test_batch_size
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 128

print('Train Loading')
with np.load(trainpath, allow_pickle=True) as data:
    train_examples = data['x'][:limit]
    train_labels = (data['y'][:limit]).astype(np.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(8)
print('Train Loaded')

print('Val Loading')
with np.load(valpath, allow_pickle=True) as data:
    val_examples = data['x']
    val_labels = (data['y']).astype(np.int64)

val_dataset = tf.data.Dataset.from_tensor_slices((val_examples, val_labels)).prefetch(8)
val_dataset = val_dataset.batch(BATCH_SIZE)
print('Val Loaded')


print('Test Loading')
with np.load(testpath, allow_pickle=True) as data:
    test_examples = data['x']
    test_labels = (data['y']).astype(np.int64)

test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels)).prefetch(8)
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
print('Test Loaded')

print('Pred Loading')
with np.load(predpath, allow_pickle=True) as data:
    pred_examples = data['x']
    pred_labels = (data['y']).astype(np.int64)

pred_dataset = tf.data.Dataset.from_tensor_slices((pred_examples, pred_labels)).prefetch(8)
pred_dataset = test_dataset.batch(TEST_BATCH_SIZE)
print('Pred Loaded')


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_data = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)
test_dataset = test_dataset.with_options(options)
pred_dataset = pred_dataset.with_options(options)





print("Creating a model")
tf.keras.backend.set_floatx('float64')



class TestResult(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        y_prob = tf.sigmoid(self.model.predict(test_examples, batch_size=TEST_BATCH_SIZE))
        y_out = (y_prob > 0.5).numpy().astype("int32")
        target_names = ['No-Fall', 'Fall']
        print("Results on Par Set")
        report = classification_report(test_labels, y_out, target_names=target_names, zero_division = 0)
        
        target_path = f"{rep_root}/version_{version_catalog}_report_par_milton.txt"
        
        if Path(target_path).exists():
            curr_acc = float(report.split('\n')[5].split()[1])
            with open(target_path,"r") as f:
                cache_acc = float(f.readlines()[5].split()[1])
            
            if curr_acc>cache_acc:
                with open(target_path,"w") as f:
                    f.write(report)
        else:        
            with open(target_path,"w") as f:
                f.write(report)
        print(report)
        
    def on_epoch_end(self, epoch, logs=None):
        y_prob = tf.sigmoid(self.model.predict(pred_examples, batch_size=TEST_BATCH_SIZE))
        y_out = (y_prob > 0.5).numpy().astype("int32")
        target_names = ['No-Fall', 'Fall']
        print("Results on Pred Set")
        report = classification_report(pred_labels, y_out, target_names=target_names, zero_division = 0)
        
        target_path = f"{rep_root}/version_{version_catalog}_report_pred_milton.txt"
        
        if Path(target_path).exists():
            curr_acc = float(report.split('\n')[5].split()[1])
            with open(target_path,"r") as f:
                cache_acc = float(f.readlines()[5].split()[1])
            
            if curr_acc>cache_acc:
                with open(target_path,"w") as f:
                    f.write(report)
        else:        
            with open(target_path,"w") as f:
                f.write(report)
        print(report)
        
        

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 4)),
        tf.keras.layers.LayerNormalization(
            axis=1, epsilon=1e-10, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
        ),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)),
        
        tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           ),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           ),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           ),
        tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           ),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           ),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy'])
    
    if mode == "test":
        model.load_weights(checkpoint_path)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)

if mode == "train":
    print("model Fitting started")
    print("#"*200)
    print(f"TRAINING STATS : GPU-NUM : {num_gpu}, limit: {limit}, gpulist = {gpulist}, Epoch : {epoch}")
    print("#"*200)
    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs=epoch, 
                        callbacks=[callback, 
                                cp_callback, 
                                TestResult(), 
                                tf.keras.callbacks.TensorBoard(
                                        log_dir=f'logs', histogram_freq=0, write_graph=True,
                                        write_images=False, update_freq='batch', profile_batch=2,
                                        embeddings_freq=0, embeddings_metadata=None
                                    )
                                    ]
                    )

    print("Model Fitting Done")
    print("Model testing Started")
    print(f"Test Performance Loss, Accuracy = {model.evaluate(test_dataset)}")
    print("Model Testing Done")
    
    try:    
        print("Model Evaluation Started Val Set")
        y_prob = tf.sigmoid(model.predict(val_examples))
        y_out = (y_prob > 0.5).numpy().astype("int32")
    except Exception as e:
        sys.exit(e)
        
    target_names = ['No-Fall', 'Fall']
    report = classification_report(val_labels, y_out, target_names=target_names, zero_division = 0)
    with open(f"{rep_root}/version_{version_catalog}_report_val_milton.txt","w") as f:
        f.write(report)
    print(report)
    
    
else:
    print("Model Evaluation Started")
    
    y_prob = tf.sigmoid(model.predict(test_examples))
    y_out = (y_prob > 0.84).numpy().astype("int32")

    target_names = ['No-Fall', 'Fall']
    report = classification_report(test_labels, y_out, target_names=target_names, zero_division = 0)
    
    with open(f"{rep_root}/test_report_lt10_0_85.txt","w") as f:
        f.write(report)
    
    print(report)


with open(f"{cat_root}/version_{version_catalog}.txt", "a") as f:
    results = """
    ########################################################################### 
    #########################        RESULTS       ############################ 
    ########################################################################### 
    """
    
    par_path = f"{rep_root}/version_{version_catalog}_report_par_milton.txt"
    pred_path = f"{rep_root}/version_{version_catalog}_report_pred_milton.txt"
    val_path = f"{rep_root}/version_{version_catalog}_report_val_milton.txt"
    
    with open(par_path, "r") as p:
        par_header = p.read()
    
    with open(pred_path, "r") as p:
        pred_header = p.read()
    
    with open(val_path, "r") as p:
        val_header = p.read()
    
    content = "\n" + results + "\n" + par_header + "\n" + pred_header + "\n" + val_header
    
    f.write(content)

