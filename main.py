import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import callbacks

from models.model import PCNAutoEncoder
from training.train_utils import ReconstructionVisualizationCallback, AlphaSchedulerCallback, SaveReconstructionsCallback
from data.data_loader import Model40Dataset, ShapeNetDataset 




parser = argparse.ArgumentParser(description="Testing Script")
parser.add_argument("--dataset", type=str, default="modelnet40", help="< modelnet40 | shapenet >")
parser.add_argument("--latent_dim", type=int, default=512, help="< 512 | 1024 >")
parser.add_argument("--batch_size", type=int, default=32, help="eg 4, 16, 32, ...")
parser.add_argument("--model_name", type=str, default="", help="eg Foldnet Expirement 2")

args = parser.parse_args()
print(f"args: {args}")
print(f"args data: {args.dataset}")


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


log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"{args.model_name}-%Y-%m-%d--%H-%M-%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='300,350')
early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='min', restore_best_weights=True)


count = 0
for batch in test_dataset.take(5):
    count += 1
    if count > 4:
        break
input_tensor = batch[0]
plotting_callback = ReconstructionVisualizationCallback(log_dir=log_dir, input_tensor=tf.expand_dims(input_tensor, 0))
saving_callback = SaveReconstructionsCallback(input_tensor=tf.expand_dims(input_tensor, 0))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = PCNAutoEncoder(args.latent_dim, 0.01, debug=False)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay=1e-6))
    
    a_callback = AlphaSchedulerCallback(model.alpha, [20, 40, 60], [0.01, 0.1, 0.5, 1.0])    

    model.fit(train_dataset, epochs=250, validation_data=test_dataset, callbacks=[tensorboard_callback, plotting_callback, early_stopping_callback, a_callback]) #early_stopping_callback

# model.save_weights(f"/home/user7/NTC Project/saved_models/pcn/pcn_weights3/{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}")
#removed batch norm layers in decoder, new optimzer, different grid size