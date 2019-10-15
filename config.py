import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Document Classification')
    #
    parser.add_argument('--data', default='./data',
                        help='path to dataset')

    #path to Glove embeddings

    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints')


    parser.add_argument('--expno', type=int, default=0,
                        help='Experiment number')
    parser.add_argument('--expname', type=str, default='vgg16',
                        help='Name to identify experiment')
    parser.add_argument('--pretrained_model', default='vgg16',
                        help='Pretrained model')
    parser.add_argument('--pretrained_holistic', type=int, default=0,
                        help='Pretrained model')

    parser.add_argument('--num_classes', default=16, type=int,
                        help='Number of classes in dataset')

    # training arguments
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--batchsize', default=2, type=int,
                        help='batchsize for optimizer updates')

    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')

    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
