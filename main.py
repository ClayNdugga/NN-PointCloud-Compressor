import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import callbacks

from models.model import PCNAutoEncoder
from training.train_utils import ReconstructionVisualizationCallback, AlphaSchedulerCallback, SaveReconstructionsCallback, UpdateMuVarianceCallback, LogMuVarianceCallback
from data.data_loader import Model40Dataset, ShapeNetDataset 


parser = argparse.ArgumentParser(description="Testing Script")
parser.add_argument("--dataset",    type=str,   default="modelnet40", help="< modelnet40 | shapenet >")
parser.add_argument("--latent_dim", type=int,   default=512,          help="< 512 | 1024 >")
parser.add_argument("--lmbda",      type=float, default=0.0001,       help="< 0.0000001 - 0.001>")
parser.add_argument("--batch_size", type=int,   default=32,           help="eg 4, 16, 32, ...")
parser.add_argument("--model_name", type=str,   default="",           help="eg PCN Expirement 2")

args = parser.parse_args()
print(f"args: {args}")
print(f"args data: {args.dataset}")

################################# Data Loading #################################

data_path = "/home/user7/NTC Project/pointcloud_data/shapenetcorev2_hdf5_2048/" 
if args.dataset == "shapenet": 
    train_dataset = ShapeNetDataset(data_path, split="train", batch_size=args.batch_size, data_augmentation="True").create_dataset()
    val_dataset = ShapeNetDataset(data_path, split="val", batch_size=args.batch_size, data_augmentation="True").create_dataset()
    test_dataset = ShapeNetDataset(data_path, split="test", batch_size=args.batch_size, data_augmentation="True").create_dataset()

    val_dataset = val_dataset.map(lambda x, _: x)
    label_map = ShapeNetDataset(data_path, split="test", batch_size=args.batch_size, data_augmentation="True").label_map
else:
    train_dataset = Model40Dataset(data_path, split="train", batch_size=args.batch_size, data_augmentation="True").create_dataset()
    test_dataset = Model40Dataset(data_path, split="test", batch_size=args.batch_size, data_augmentation="True").create_dataset()
    label_map = Model40Dataset(data_path, split="test", batch_size=args.batch_size, data_augmentation="True").label_map
    
train_dataset = train_dataset.map(lambda x, _: x)
test_dataset = test_dataset.map(lambda x, _: x)


################################# Training Callbacks #################################

unqiue_run = datetime.datetime.now().strftime(f"{args.model_name}-%Y-%m-%d--%H-%M-%S")
print(unqiue_run)
log_dir = "logs/fit/" + unqiue_run

count = 0
for batch in test_dataset.take(5):
    count += 1
    if count > 4:
        break
reconstruction_cloud = batch[0]

cb = [
    callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),               
    callbacks.EarlyStopping(monitor='val_recon_loss', patience=40, verbose=1, mode='min', restore_best_weights=True),
    # callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, mode='min', save_best_only=True),
    UpdateMuVarianceCallback(train_dataset=train_dataset, update_freq=5),   
    LogMuVarianceCallback(log_dir=log_dir, update_freq=5)  ,                
    AlphaSchedulerCallback([20, 30, 40], [0.01, 0.1, 0.5, 1.0]),           
    ReconstructionVisualizationCallback(log_dir=log_dir, input_tensor=tf.expand_dims(reconstruction_cloud, 0))
]


################################# Training  #################################


# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

model = PCNAutoEncoder(1024, alpha = 0.01, lmbda=args.lmbda)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay=1e-6))
model.fit(train_dataset, epochs=200, validation_data=test_dataset, callbacks=cb)
model.save(f"/home/user/NTC Project/saved_models/{unqiue_run}")
