import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Brain Alzheimeir Classification')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50'],
                                                                default='resnet18')
    
    parser.add_argument('--out', '--out_dir', type=str, default='session')
    
    parser.add_argument('csv', '--csv_dir', default='data/CSVs')

    parser.add_argument('bs', '--batch_size', default=16, type=int, choices=[16, 32, 64])

    parser.add_argument('learning_rate', '--lr', default=0.001, type=float, choices=[0.0001, 0.00001])

    parser.add_argument('epochs', '--epochs', default=100, type=int)
    
    args = parser.parse_args()

    return args
