import argparse


def get_args():
    parser = argparse.ArgumentParser("EfficientLIF")
    parser.add_argument('--exp_name', type=str, default='EfficientLIF', help='experiment name')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10]')
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--tau', type=float, default=2, help='neuron decay time factor')
    parser.add_argument('--threshold', type=float, default=1.0, help='neuron firing threshold')

    parser.add_argument('--arch', type=str, default='resnet19', help='[resnet19]')
    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=2e-1, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument('--valid_freq', type=int, default=10, help='test for SNN')
    parser.add_argument("--print_freq", default=20, type=int)

    parser.add_argument('--lifshare', type=str, default='layer', help='[layer, channel]')
    parser.add_argument('--ch_group_num', type=int, default=2, help='define channel group number')



    args = parser.parse_args()
    print(args)

    return args
