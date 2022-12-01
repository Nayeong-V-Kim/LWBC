import argparse

parser = argparse.ArgumentParser()

### COMMON
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--temperature', type=float, default=200)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--dataset', type=str, help='BAR, celebA, imagenet, NICO, bffhq, cifar10c')

parser.add_argument('--local', dest='local', action='store_true', help='disable wandb')
parser.add_argument('--save', dest='save', action='store_true', help='save')

parser.add_argument('--alpha', default=0.02, type=float, help='alpha')

### bootstrap
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--loss_scale', type=float, default=0.2)
parser.add_argument('--num_classifier', type=int, default=5)
parser.add_argument('--set_size', type=int, default=40, help='total_size = set_size * num_class')
parser.add_argument('--warmup_init', dest='warmup_init', type=int, default=0)

parser.add_argument('--linear_bias', dest='linear_bias', action='store_true', help='linear_bias')
parser.add_argument('--classifier', type=str, default='2linear')

### KD
parser.add_argument('--kd_lambda', type=float, default=0.6)
parser.add_argument('--kd_temperature', type=float, default=1)


### path
parser.add_argument('--save_path', type=str, default='/root/checkpoints')
parser.add_argument('--data_path', type=str, default='/root/checkpoints')


### COMMONS
parser.set_defaults(local=False)
config = parser.parse_args()
