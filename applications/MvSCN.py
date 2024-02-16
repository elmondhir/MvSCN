import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input
from keras import backend as K

from core.Config import load_config
from core.data import get_data
from . import Clustering
from core import networks
import wandb



def run_net():
    

    with wandb.init():
        ## load nus
        configs = load_config('./config/nus.yaml')
        
        ## load noisymnist
        # configs = load_config('./config/noisymnist.yaml')

        # load config for Caltech101-20
        # configs = load_config('./config/Caltech101-20.yaml')

        # use pretrained SiameseNet. 
        # configs['siam_pre_train'] = True

        # LOAD DATA
        data_list = get_data(configs)
        
        # wandb params
        num_epochs = wandb.config.epochs
        lamb = wandb.config.lamb

        # network input shapes stored as view-list
        MvSCN_input_shape = []
        SiameseNet_input_shape = []

        # training and testing data stored as view-list
        x_train_list = []
        x_test_list = []

        # get the labels
        _, y_train, _, y_test = data_list[0]['spectral']

        batch_sizes = {
            'Embedding': configs['batch_size'],
            'Orthogonal': configs['batch_size_orthogonal'],
        }

        
        for i in range(configs['view_size']):
            data_i = data_list[i]
            x_train_i, y_train_i, x_test_i, y_test_i = data_i['spectral']
            x = x_train_i

            x_test_list.append(x_test_i)
            x_train_list.append(x_train_i)

            # SiameseNet training pairs
            pairs_train, dist_train = data_i['siamese']

            # input shape
            input_shape = x.shape[1:]
            inputs = {
                    'Embedding': Input(shape=input_shape,name='EmbeddingInput'+'view'+str(i)),
                    'Orthogonal': Input(shape=input_shape,name='OrthogonalInput'+'view'+str(i)),
                    }
            MvSCN_input_shape.append(inputs)

            print('******** SiameseNet ' + 'view'+str(i+1)+' ********')
            siamese_net = networks.SiameseNet(name=configs['dset']+'_'+'view'+str(i+1),
                    inputs=inputs,
                    arch=configs['arch'], 
                    siam_reg=configs['siam_reg'])
            
            history = siamese_net.train(pairs_train=pairs_train,
                    dist_train=dist_train,
                    lr=configs['siam_lr'], 
                    #lr= siam_lr, #from wandb config
                    drop=configs['siam_drop'], 
                    patience=configs['siam_patience'],
                    num_epochs=num_epochs, # from wandb config
                    batch_size=configs['siam_batch_size'], 
                    pre_train=configs['siam_pre_train'],
                    )
                    

            SiameseNet_input_shape.append(siamese_net)

        # MvSCN
        mvscn = networks.MvSCN(input_list=MvSCN_input_shape,
                            arch=configs['arch'],
                            spec_reg=configs['spectral_reg'],
                            n_clusters=configs['n_clusters'],
                            scale_nbr=configs['scale_nbr'],
                            n_nbrs=configs['n_nbrs'],
                            batch_sizes=batch_sizes,
                            view_size=configs['view_size'],
                            siamese_list=SiameseNet_input_shape,
                            x_train_siam=x_train_list,
                            lamb=lamb # from wandb.config
                            )

        # training
        print('********', 'Training', '********')
            # configs = wandb.configs
            
        mvscn.train(x_train=x_train_list,
            lr=configs["spectral_lr"], 
            drop=configs['spectral_drop'], 
            patience=configs['spectral_patience'],
            num_epochs = num_epochs)

        print("Training finished ")

        print('********', 'Prediction', '********')
        print('Testing data')
        # get final testing embeddings
        x_test_final_list = mvscn.predict(x_test_list)
        print('Learned representations for testing (view size, nsamples, dimension)', x_test_final_list.shape)
        
        # clustering
        y_preds, scores= Clustering.Clustering(x_test_final_list, y_test)

        # K.clear_session()

        return x_test_final_list, scores


