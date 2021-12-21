class DictDot(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opts = DictDot({
    'model_list': ['vgg13', 'vgg16', 'vgg19', 'resnet18', 'own'], # 'alexnet', 'resnet18'
    'data_list': ['imagenet'],
    'app_width': '1200px',
    'app_height': '600px',
    'url': DictDot({
        'imagenet_classes': 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt',
        'image_demo_1': 'https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale',
        'image_demo_2': 'https://nashzelenyimir.ru/wp-content/uploads/2016/06/%D0%A3%D0%BB%D0%B8%D1%82%D0%BA%D0%B0-%D1%84%D0%BE%D1%82%D0%BE.jpg'
    })
})
