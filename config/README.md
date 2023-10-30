## Parameter description

### net_config
Contains all parameters of the network architecture.

| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
input_features | int | 4 | Number of input features to network |
dt_step_s | float | 0.1 | Step size in s of object history |
output_length | int | 50 | Length of predicted trajectory |
output_features | int | 2 | Number of predicted output features (x, y in case of 2)|
linear_hidden_layers | int | 1 | Number of hidden layers of linear input embedding |
linear_hidden_size | int | 50 | Size of hidden layers of linear input embedding |
input_embedding_size | int | 50 | Output size of linear input embedding |
lstm_hidden_size | int | 50 | Hidden size of LSTM-encoder |
lstm_num_layers | int | 1 | Number of hidden layers of LSTM-encoder|
dg_encoder | boolean | true | If true DG_LSTM get its own LSTM-encoder (recommended), otherwise shared with L_LSTM |
gnn_distance | int | 20 | Distance around target object to consider interaction |
gnn_message_size | int | 64 | Size of GNN message passing |
gnn_aggr | str | "max" | Method to some gnn messages ('max' or 'add') |
gnn_embedding_size | int | 128 | Embedding size of gnn after message passing |
gnn_num_hidden_layers | int | 1 | Number of hidden layers of gnn embedding after message passing |
gnn_hidden_size | int | 48 |  Output size of gnn embedding after message passing |
dyn_embedding_size | int | 48 | Embedding size of encoding in latent space |
decoder_size | int | 128 | Size of LSTM decoder |
g_sel_dyn_emb | str| "no_lstm" | Type of selector head ('no_lstm', 'obj_state' or false) |
g_sel_output_linear_size | int | 128 | Size of linear output layer of selection head |
num_img_filters | int | 32 | Number of images filters for scene encoding |
dec_img_size | int | 16 | Size of image extraction |

### train_config
Contains all parameters of the training procedure.
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
model_path | list(str) | - | parent path to trained models, required in case of load_model argument in training |
epochs | int | 50 | Number of training epochs |
num_workers | int | 2 | Number of data loaders |
pin_memory | bool | true | Argument to pin memory for data loaders |
batch_size | int | 64 | Batch size for data loaders |
base_lr | float | 0.0005 | Base learning rate of training |
gamma | float | 0.5 | Decay factor for learning rate scheduler |
step_size | float | 0.0005 | Step size for learning rate scheduler |
clip | bool | true | Argument for gradient clipping during training |
sc_img | bool | true | Argument to consider scene image encoding |
get_borders | bool | true | Argument to extract lane borders instead of center lanes from scene image |
best_val | str | "sel" | Selection of metric to choose the best model ('sel' or 'rmse'). "sel" select the best selection rate. |
optim | str | "add" | Optimizer selection ('add' = ADAMS, 'sgd' = SGD) |
loss_fn | str | "nll" | Loss function of selector ('nll' = NLLLoss, 'ce' = CrossEntropyLoss) |
select_invalid | bool | true | Argument to select also invalid scenarios |
weight | float | 3.64 | Weight of invalid choice during selection training |
err_quantile | float | 0.8 | Relative Threshold to classify a scenario as invalid, RMSE ratio of batch, only taken if 'err_rmse' < 0 |
err_rmse | float | 0.6221 | Absolute Threshold to classify a scenario as invalid, RMSE in meter |