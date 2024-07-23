# -- coding: utf-8 --
import os
import shutil
import hashlib
import config as cfg
from utils import *
from easydict import EasyDict
import atexit
from collections import defaultdict
from markdown_strings import header, table


def md5(x):
    return hashlib.md5(x.encode("utf-8")).hexdigest()


class Robot:
    def __init__(self, ):
        self.db = EasyDict(load_json(f"{cfg.ROOT_PATH}/{cfg.db_name}.json"))
        if "glb" not in self.db: self.db = self.new_db()
        self.relation_set = set([(e.paper_md5, e.label_md5) for e in self.db.relation_list])
        atexit.register(self.save_db)

    def new_db(self):
        return EasyDict({
            "glb": {"label_num": 0, "paper_num": 0, "paper_root_path": cfg.PAPER_ROOT_PATH},
            "label_list": [],
            "paper_list": [],
            "relation_list": [],
        })

    def new_label(self, name):
        self.db.glb.label_num += 1
        self.db.label_list.append(EasyDict({"id": self.db.glb.label_num - 1, "name": name, "md5": md5(name)}))

    def new_paper(self, title, method, web, path=None, year=None, label_name_list=None):
        self.db.glb.paper_num += 1
        if path is None:
            path = f"{md5(title)}.pdf"
        if year is None:
            assert "arxiv" in web
            year = f"20{web.split('/')[-1][:2]}"
        self.db.paper_list.append(EasyDict({"id": self.db.glb.paper_num - 1, "title": title,
                "method": method, "web": web, "path": f"{path}", "year": year, "md5": md5(title)}))

        if label_name_list is not None:
            assert type(label_name_list) is list
            self.new_relation(title, label_name_list)


    def new_relation(self, paper_title, label_name_list):
        paper_md5 = md5(paper_title)
        label_md5_list = [md5(e) for e in label_name_list]
        self.relation_set.update([(paper_md5, label_md5) for label_md5 in label_md5_list])

    def gen_markdown(self):
        with open(f"./PaperList.md", "w", encoding="utf8") as f:
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

            f.write(header("Paper", 2) + "\n\n")

            pair = set([(e["paper_md5"], e["label_md5"]) for e in self.db.relation_list])
            tpl = sorted(self.db.paper_list, key = lambda e: e.method)
            tll = sorted(self.db.label_list, key = lambda e: e.name)
            f.write(table([
                ["论文", *[f"[{p.method}]({p.web})" for p in tpl]],
                *[[l.name] + ["Y" if (p.md5, l.md5) in pair else " " for p in tpl] for l in tll],
            ], column_format=["L", *["C" for e in tll]]) + "\n\n")

            # for label in self.db.label_list:
            #     f.write(header(f"{label.name}", 2) + "\n\n")
            #     paper_proj = {e.md5: e for e in self.db.paper_list}
            #     paper_list = [paper_proj[e.paper_md5] for e in self.db.relation_list if e.label_md5 == label.md5]
            #     paper_list = sorted(paper_list, key=lambda e: (int(e.year), e.web))
            #     f.write(table([
            #         ["论文", *[f"[{e.title}]({e.web})" for e in paper_list]],
            #         ["年份", *[f"{e.year}" for e in paper_list]],
            #         ["阅读", *["Y" for e in paper_list]],
            #     ]) + "\n\n")



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
        self.db.relation_list = sorted([{"paper_md5": e[0], "label_md5": e[1]} for e in self.relation_set], key=lambda e: (e["paper_md5"], e["label_md5"]))
        self.gen_markdown()
        save_json(dict(self.db), f"{cfg.ROOT_PATH}/{cfg.db_name}.json")


if __name__ == "__main__":
    r = Robot()

    # ----------------------------------------------------------- #
    r.new_label("Language-based Backbone")
    r.new_label("Large Language Model")

    r.new_label("Image-based Backbone")
    r.new_label("Image-based Object Classification")
    r.new_label("Image-based Object Detection")
    r.new_label("Image-based Semantic Segmentation")
    r.new_label("Image-based Instance Segmentation")
    r.new_label("Image-based Keypoint Detection")
    r.new_label("Lidar-based 3D Object Classification")
    r.new_label("Lidar-based 3D Object Detection")
    r.new_label("Lidar-based 3D Part Segmentation")
    r.new_label("Lidar-based 3D Scene Segmentation")
    r.new_label("Lidar-based 3D Object Tracking")
    r.new_label("Camera-based 3D Object Detection")
    r.new_label("Camera-based 3D Map Segmentation")
    r.new_label("One-stage Object Detection")
    r.new_label("Two-stage Object Detection")
    r.new_label("Anchor-based Object Detection")
    r.new_label("Anchor-free Object Detection")
    r.new_label("Dense Prediction Object Detection")
    r.new_label("Sparse Prediction Object Detection")
    r.new_label("Knowledge Distillation")
    r.new_label("Self-Distillation Scheme")
    r.new_label("Intersection over Union Optimization")

    BbLb = "Language-based Backbone"
    LLM_ = "Large Language Model"
    BbIb = "Image-based Backbone"
    OCIb = "Image-based Object Classification"
    ODIb = "Image-based Object Detection"
    SSIb = "Image-based Semantic Segmentation"
    ISIb = "Image-based Instance Segmentation"
    KDIb = "Image-based Keypoint Detection"
    OCLb = "Lidar-based 3D Object Classification"
    ODLb = "Lidar-based 3D Object Detection"
    PSLb = "Lidar-based 3D Part Segmentation"
    SSLb = "Lidar-based 3D Scene Segmentation"
    OTLb = "Lidar-based 3D Object Tracking"
    ODCb = "Camera-based 3D Object Detection"
    MSCb = "Camera-based 3D Map Segmentation"
    ODOs = "One-stage Object Detection"
    ODTs = "Two-stage Object Detection"
    ODAb = "Anchor-based Object Detection"
    ODAf = "Anchor-free Object Detection"
    ODDP = "Dense Prediction Object Detection"
    ODSP = "Sparse Prediction Object Detection"
    KD__ = "Knowledge Distillation"
    KDSd = "Self-distillation Scheme"
    IoUO = "Intersection over Union Optimization"


    # ----------------------------------------------------------- #
    r.new_paper(method="GPT-1"              , web="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf", title="Improving Language Understanding by Generative Pre-Training", year=2018 , label_name_list=[LLM_])
    r.new_paper(method="GPT-2"              , web="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", title="Language Models are Unsupervised Multitask Learners", year=2019                           , label_name_list=[LLM_])
    r.new_paper(method="GPT-3"              , web="http://arxiv.org/abs/2005.14165"     , title="Language Models are Few-Shot Learners"                                                                                                                     , label_name_list=[LLM_])

    r.new_paper(method="DCNv1"              , web="http://arxiv.org/abs/1703.06211"     , title="Deformable Convolutional Networks"                                                                             , label_name_list=[BbIb, ODIb, SSIb])
    r.new_paper(method="DCNv2"              , web="http://arxiv.org/abs/1811.11168"     , title="Deformable ConvNets v2: More Deformable, Better Results"                                                       , label_name_list=[BbIb, ODIb, ISIb])
    r.new_paper(method="InternImage/DCNv3"  , web="http://arxiv.org/abs/2211.05778"     , title="InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions"                      , label_name_list=[BbIb, OCIb, ODIb, SSIb])
    r.new_paper(method="DCNv4"              , web="http://arxiv.org/abs/2401.06197"     , title="Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications"                 , label_name_list=[BbIb, OCIb, ODIb, SSIb, ISIb])

    r.new_paper(method="SS"                 , web="https://link.springer.com/article/10.1007/s11263-013-0620-5", title="Selective Search for Object Recognition", year=2013                                     , label_name_list=[ODIb])
    r.new_paper(method="R-CNN"              , web="http://arxiv.org/abs/1311.2524"      , title="Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation"                              , label_name_list=[ODIb, ODTs, ODAb, ODSP])
    r.new_paper(method="Fast R-CNN"         , web="https://arxiv.org/abs/1504.08083v2"  , title="Fast R-CNN"                                                                                                    , label_name_list=[ODIb, ODTs, ODAb, ODSP])
    r.new_paper(method="Faster R-CNN"       , web="https://arxiv.org/abs/1506.01497v3"  , title="Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks"                                 , label_name_list=[ODIb, ODTs, ODAb, ODSP])
    r.new_paper(method="Mask R-CNN"         , web="http://arxiv.org/abs/1703.06870"     , title="Mask R-CNN"                                                                                                    , label_name_list=[ODIb, SSIb, KDIb, ODTs, ODAb])
    r.new_paper(method="FPN"                , web="http://arxiv.org/abs/1612.03144"     , title="Feature Pyramid Networks for Object Detection"                                                                 , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="RetinaNet"          , web="http://arxiv.org/abs/1708.02002"     , title="Focal Loss for Dense Object Detection"                                                                         , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="SSD"                , web="http://arxiv.org/abs/1512.02325"     , title="SSD: Single Shot MultiBox Detector"                                                                            , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv1"             , web="http://arxiv.org/abs/1506.02640"     , title="You Only Look Once: Unified, Real-Time Object Detection"                                                       , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv2"             , web="http://arxiv.org/abs/1612.08242"     , title="YOLO9000: Better, Faster, Stronger"                                                                            , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv3"             , web="http://arxiv.org/abs/1804.02767"     , title="YOLOv3: An Incremental Improvement"                                                                            , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv4"             , web="http://arxiv.org/abs/2004.10934"     , title="YOLOv4: Optimal Speed and Accuracy of Object Detection"                                                        , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="Scaled-YOLOv4"      , web="http://arxiv.org/abs/2011.08036"     , title="Scaled-YOLOv4: Scaling Cross Stage Partial Network"                                                            , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOR"              , web="http://arxiv.org/abs/2105.04206"     , title="You Only Learn One Representation: Unified Network for Multiple Tasks"                                         , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOX"              , web="http://arxiv.org/abs/2107.08430"     , title="YOLOX: Exceeding YOLO Series in 2021"                                                                          , label_name_list=[ODIb, ODOs, ODAf, ODDP])
    r.new_paper(method="YOLOv6"             , web="http://arxiv.org/abs/2209.02976"     , title="YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications"                                 , label_name_list=[ODIb, ODOs, ODAf, ODSP, KD__, KDSd])
    r.new_paper(method="YOLOv7"             , web="http://arxiv.org/abs/2207.02696"     , title="YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors"                    , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv9"             , web="http://arxiv.org/abs/2402.13616"     , title="YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"                               , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="YOLOv10"            , web="http://arxiv.org/abs/2405.14458"     , title="YOLOv10: Real-Time End-to-End Object Detection"                                                                , label_name_list=[ODIb, ODOs, ODAb, ODDP])
    r.new_paper(method="FCN"                , web="http://arxiv.org/abs/1411.4038"      , title="Fully Convolutional Networks for Semantic Segmentation"                                                        , label_name_list=[SSIb])
    r.new_paper(method="Double-Head"        , web="http://arxiv.org/abs/1904.06493"     , title="Rethinking Classification and Localization for Object Detection"                                               , label_name_list=[ODIb])
    r.new_paper(method="TSD"                , web="http://arxiv.org/abs/2003.07540"     , title="Revisiting the Sibling Head in Object Detector"                                                                , label_name_list=[ODIb])
    r.new_paper(method="DETR"               , web="http://arxiv.org/abs/2005.12872"     , title="End-to-End Object Detection with Transformers"                                                                 , label_name_list=[ODIb])
    r.new_paper(method="Deformable-DETR"    , web="http://arxiv.org/abs/2010.04159"     , title="Deformable DETR: Deformable Transformers for End-to-End Object Detection"                                      , label_name_list=[ODIb])
    r.new_paper(method="RT-DETR"            , web="http://arxiv.org/abs/2304.08069"     , title="DETRs Beat YOLOs on Real-time Object Detection"                                                                , label_name_list=[ODIb])

    r.new_paper(method="CloserLook3D"       , web="http://arxiv.org/abs/2007.01294"     , title="A Closer Look at Local Aggregation Operators in Point Cloud Analysis"                                          , label_name_list=[OCLb, PSLb, SSLb])

    r.new_paper(method="CenterPoint"        , web="http://arxiv.org/abs/2006.11275"     , title="Center-based 3D Object Detection and Tracking"                                                                 , label_name_list=[ODLb, OTLb, ODTs, ODAb])
    r.new_paper(method="VoxelNet"           , web="http://arxiv.org/abs/1711.06396"     , title="VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"                                       , label_name_list=[ODLb, ODOs, ODAb])
    r.new_paper(method="PointPillars"       , web="http://arxiv.org/abs/1812.05784"     , title="PointPillars: Fast Encoders for Object Detection from Point Clouds"                                            , label_name_list=[ODLb, ODOs, ODAb])
    r.new_paper(method="SECOND"             , web="https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf", title="SECOND: Sparsely Embedded Convolutional Detection", year="2018"      , label_name_list=[ODLb, ODOs, ODAb])
    r.new_paper(method="PillarNet"          , web="http://arxiv.org/abs/2205.07403"     , title="PillarNet: Real-Time and High-Performance Pillar-based 3D Object Detection"                                    , label_name_list=[ODLb, ODOs, ODAb])
    r.new_paper(method="PV-RCNN"            , web="http://arxiv.org/abs/1912.13192"     , title="PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection"                                          , label_name_list=[ODLb, ODTs, ODAb])
    r.new_paper(method="PV-RCNN++"          , web="http://arxiv.org/abs/2102.00463"     , title="PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection"       , label_name_list=[ODLb, ODTs, ODAb])
    r.new_paper(method="Voxel R-CNN"        , web="http://arxiv.org/abs/2012.15712"     , title="Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection"                                         , label_name_list=[ODLb, ODTs, ODAb])
    r.new_paper(method="IA-SSD"             , web="http://arxiv.org/abs/2203.11139"     , title="Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds"           , label_name_list=[ODLb, ODOs, ODAb])
    r.new_paper(method="AFDetv1"            , web="http://arxiv.org/abs/2006.12671"     , title="AFDet: Anchor Free One Stage 3D Object Detection"                                                              , label_name_list=[ODLb, ODOs, ODAf])
    r.new_paper(method="AFDetv2"            , web="http://arxiv.org/abs/2112.09205"     , title="AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds"                  , label_name_list=[ODLb, ODOs, ODAf])
    r.new_paper(method="3DSSD"              , web="http://arxiv.org/abs/2002.10187"     , title="3DSSD: Point-based 3D Single Stage Object Detector"                                                            , label_name_list=[ODLb, ODOs, ODAf])
    r.new_paper(method="Real-Aug"           , web="http://arxiv.org/abs/2305.12853"     , title="Real-Aug: Realistic Scene Synthesis for LiDAR Augmentation in 3D Object Detection"                             , label_name_list=[ODLb])
    r.new_paper(method="FSTR"               , web="https://ieeexplore.ieee.org/document/10302363", title="Fully Sparse Transformer 3-D Detector for LiDAR Point Cloud", year=2023                               , label_name_list=[ODLb, ODOs, ODAf])

    r.new_paper(method="BEVFormer"          , web="http://arxiv.org/abs/2203.17270"     , title="BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"   , label_name_list=[ODCb, MSCb])

    r.new_paper(method="GIoU"               , web="http://arxiv.org/abs/1902.09630"     , title="Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"                          , label_name_list=[ODIb, IoUO])
    r.new_paper(method="DIoU/CIoU"          , web="http://arxiv.org/abs/1911.08287"     , title="Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"                                     , label_name_list=[ODIb, IoUO])
    



    paper_name_list = [e.title for e in r.db.paper_list]
    for e in paper_name_list:
        if not os.path.exists(f"{cfg.PAPER_ROOT_PATH}/{e}.pdf".replace(":", "")):
            continue
        shutil.move(f"{cfg.PAPER_ROOT_PATH}/{e}.pdf".replace(":", ""), f"{cfg.PAPER_ROOT_PATH}/{md5(e)}.pdf")

    