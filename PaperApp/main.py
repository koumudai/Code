# -- coding: utf-8 --

import hashlib
import config as cfg
from utils import *
from easydict import EasyDict
import atexit
from collections import defaultdict
from markdown_strings import header, table


def md5(x):
    return hashlib.md5(x.encode('utf-8')).hexdigest()


class Robot:
    def __init__(self, ):
        self.db = EasyDict(load_json(f"{cfg.root_path}/{cfg.db_name}.json"))
        if 'glb' not in self.db: self.db = self.new_db()
        self.relation_set = set([(e.paper_md5, e.label_md5) for e in self.db.relation_list])
        atexit.register(self.save_db)

    def new_db(self):
        return EasyDict({
            'glb': {'label_num': 0, 'paper_num': 0},
            'label_list': [],
            'paper_list': [],
            'relation_list': [],
        })

    def new_label(self, name):
        self.db.glb.label_num += 1
        self.db.label_list.append(EasyDict({'id': self.db.glb.label_num - 1, 'name': name, 'md5': md5(name)}))

    def new_paper(self, title, method, web, path=None, year=None):
        self.db.glb.paper_num += 1
        if path is None:
            path = f'{title}.pdf'
        if year is None:
            assert 'arxiv' in web
            year = f"20{web.split('/')[-1][:2]}"
        self.db.paper_list.append(EasyDict({'id': self.db.glb.paper_num - 1, 'title': title,
                'method': method, 'web': web, 'path': f'{cfg.PAPER_ROOT_PATH}/{path}', 'year': year, 'md5': md5(title)}))

    def new_relation(self, paper_title, label_name_list):
        paper_md5 = md5(paper_title)
        label_md5_list = [md5(e) for e in label_name_list]
        self.relation_set.update([(paper_md5, label_md5) for label_md5 in label_md5_list])

    def gen_markdown(self):
        with open(f"./PaperList.md", 'w', encoding="utf8") as f:
            f.write("---\n")
            f.write("title: Paper List\n")
            f.write("mathjax: true\n")
            f.write("abbrlink: 20191231000001\n")
            f.write("date: 2019-12-31 00:00:01\n")
            f.write("top: true\n")
            f.write("---\n\n")

            f.write(header("Paper List", 1) + "\n\n")
            f.write(header("Preamble", 2) + "\n\n")
            f.write("本文档由 Python 自动生成.\n\n")
            
            for label in self.db.label_list:
                f.write(header(f"{label.name}", 2) + "\n\n")
                paper_proj = {e.md5: e for e in self.db.paper_list}
                paper_list = [paper_proj[e.paper_md5] for e in self.db.relation_list if e.label_md5 == label.md5]
                paper_list = sorted(paper_list, key=lambda e: (int(e.year), e.web))
                f.write(table([
                    ["论文", *[f'[{e.title}]({e.web})' for e in paper_list]],
                    ["年份", *[f'{e.year}' for e in paper_list]],
                    ["阅读", *['Y' for e in paper_list]],
                ]) + "\n\n")


    def save_db(self):
        label_name_dict = [e.name for e in self.db.label_list]
        label_name_dict = {md5(k): v for v, k in enumerate(list(sorted(set(label_name_dict))))}

        paper_name_dict = [e.title for e in self.db.paper_list]
        paper_name_dict = {md5(k): v for v, k in enumerate(list(sorted(set(paper_name_dict))))}

        self.db.glb.label_num = len(label_name_dict.keys())
        self.db.glb.paper_num = len(paper_name_dict.keys())

        old_label_proj = defaultdict(list)
        for idx, e in enumerate(self.db.label_list):
            old_label_proj[md5(e.name)].append(idx)

        label_list = []
        for k, v in label_name_dict.items():
            self.db.label_list[old_label_proj[k][0]].id = v
            label_list.append(self.db.label_list[old_label_proj[k][0]])

        self.db.label_list = label_list
        
        old_paper_proj = defaultdict(list)
        for idx, e in enumerate(self.db.paper_list):
            old_paper_proj[md5(e.title)].append(idx)
        
        paper_list = []
        for k, v in paper_name_dict.items():
            self.db.paper_list[old_paper_proj[k][0]].id = v
            paper_list.append(self.db.paper_list[old_paper_proj[k][0]])
        
        self.db.paper_list = paper_list
        self.db.relation_list = sorted([{'paper_md5': e[0], 'label_md5': e[1]} for e in self.relation_set], key=lambda e: (e['paper_md5'], e['label_md5']))
        self.gen_markdown()
        save_json(dict(self.db), f"{cfg.root_path}/{cfg.db_name}.json")


if __name__ == "__main__":
    rbt = Robot()
    
    # ----------------------------------------------------------- #
    rbt.new_label('Image-based Object Detection')
    rbt.new_label('Lidar-based 3D Object Detection')
    rbt.new_label('Camera-based 3D Object Detection')
    rbt.new_label('One-stage Object Detection')
    rbt.new_label('Two-stage Object Detection')
    rbt.new_label('Anchor-based Object Detection')
    rbt.new_label('Anchor-free Object Detection')
    rbt.new_label('Dense Prediction Object Detection')
    rbt.new_label('Sparse Prediction Object Detection')
    rbt.new_label('Knowledge Distillation')
    rbt.new_label('Self-Distillation Scheme')
    

    # ----------------------------------------------------------- #
    rbt.new_paper(method='Fast R-CNN'       , web='https://arxiv.org/abs/1504.08083v2'  , title='Fast R-CNN')
    rbt.new_paper(method='Faster R-CNN'     , web='https://arxiv.org/abs/1506.01497v3'  , title='Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks')
    rbt.new_paper(method='FPN'              , web='http://arxiv.org/abs/1612.03144'     , title='Feature Pyramid Networks for Object Detection')
    rbt.new_paper(method='RetinaNet'        , web='http://arxiv.org/abs/1708.02002'     , title='Focal Loss for Dense Object Detection')
    rbt.new_paper(method='R-CNN'            , web='http://arxiv.org/abs/1311.2524'      , title='Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation')
    rbt.new_paper(method='Scaled-YOLOv4'    , web='http://arxiv.org/abs/2011.08036'     , title='Scaled-YOLOv4: Scaling Cross Stage Partial Network')
    rbt.new_paper(method='SSD'              , web='http://arxiv.org/abs/1512.02325'     , title='SSD: Single Shot MultiBox Detector')
    rbt.new_paper(method='YOLOv2'           , web='http://arxiv.org/abs/1612.08242'     , title='YOLO9000: Better, Faster, Stronger')
    rbt.new_paper(method='YOLOv3'           , web='http://arxiv.org/abs/1804.02767'     , title='YOLOv3: An Incremental Improvement')
    rbt.new_paper(method='YOLOv4'           , web='http://arxiv.org/abs/2004.10934'     , title='YOLOv4: Optimal Speed and Accuracy of Object Detection')
    rbt.new_paper(method='YOLOv6'           , web='http://arxiv.org/abs/2209.02976'     , title='YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications')
    rbt.new_paper(method='YOLOv7'           , web='http://arxiv.org/abs/2207.02696'     , title='YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors')
    rbt.new_paper(method='YOLOv9'           , web='http://arxiv.org/abs/2402.13616'     , title='YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information')
    rbt.new_paper(method='YOLOX'            , web='http://arxiv.org/abs/2107.08430'     , title='YOLOX: Exceeding YOLO Series in 2021')
    rbt.new_paper(method='YOLOR'            , web='http://arxiv.org/abs/2105.04206'     , title='You Only Learn One Representation: Unified Network for Multiple Tasks')
    rbt.new_paper(method='YOLOv1'           , web='http://arxiv.org/abs/1506.02640'     , title='You Only Look Once: Unified, Real-Time Object Detection')

    rbt.new_paper(method='CenterPoint'      , web='http://arxiv.org/abs/2006.11275'     , title='Center-based 3D Object Detection and Tracking')
    rbt.new_paper(method='VoxelNet'         , web='http://arxiv.org/abs/1711.06396'     , title='VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection')
    rbt.new_paper(method='PointPillars'     , web='http://arxiv.org/abs/1812.05784'     , title='PointPillars: Fast Encoders for Object Detection from Point Clouds')
    # rbt.new_paper(method='SECOND'           , web='')
    # rbt.new_paper(method='')
    # rbt.new_paper(method='')
    # rbt.new_paper(method='')
    # rbt.new_paper(method='')
    # rbt.new_paper(method='')
    # rbt.new_paper(method='')

    # ----------------------------------------------------------- #
    





    rbt.new_relation(paper_title='Fast R-CNN' , label_name_list=['Anchor-based Object Detection', 'Image-based Object Detection', 'Sparse Prediction Object Detection', 'Two-stage Object Detection'])
    rbt.new_relation(paper_title='Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks' , label_name_list=['Anchor-based Object Detection', 'Image-based Object Detection', 'Sparse Prediction Object Detection', 'Two-stage Object Detection'])
    rbt.new_relation(paper_title='Feature Pyramid Networks for Object Detection' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='Focal Loss for Dense Object Detection' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation' , label_name_list=['Anchor-based Object Detection', 'Image-based Object Detection', 'Sparse Prediction Object Detection', 'Two-stage Object Detection'])
    rbt.new_relation(paper_title='Scaled-YOLOv4: Scaling Cross Stage Partial Network' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='SSD: Single Shot MultiBox Detector' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLO9000: Better, Faster, Stronger' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLOv3: An Incremental Improvement' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLOv4: Optimal Speed and Accuracy of Object Detection' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications' , label_name_list=['Anchor-free Object Detection', 'Image-based Object Detection', 'Knowledge Distillation', 'One-stage Object Detection', 'Self-Distillation Scheme', 'Sparse Prediction Object Detection'])
    rbt.new_relation(paper_title='YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='YOLOX: Exceeding YOLO Series in 2021' , label_name_list=['Anchor-free Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='You Only Learn One Representation: Unified Network for Multiple Tasks' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])
    rbt.new_relation(paper_title='You Only Look Once: Unified, Real-Time Object Detection' , label_name_list=['Anchor-based Object Detection', 'Dense Prediction Object Detection', 'Image-based Object Detection', 'One-stage Object Detection'])

