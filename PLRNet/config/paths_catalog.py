import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'data_train_small': {
            'img_dir': 'data/train/images',
            'ann_file': 'data/train/annotation-small.json'
        },
        'data_test_small': {
            'img_dir': 'data/val/images',
            'ann_file': 'data/val/annotation-small.json'
        },
        'data_train': {
            'img_dir': 'data/train/images',
            'ann_file': 'data/train/annotation.json'
        },
        'data_test': {
            'img_dir': 'data/val/images',
            'ann_file': 'data/val/annotation.json'
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name] 
       
        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            #{factory："TrainDataset"，args：{....}}
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args) 
        raise NotImplementedError()
