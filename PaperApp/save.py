import json
import os
import shutil
import time
import hashlib
import pickle as p
 
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f)
 
# 将文件转换为md5文件
def md5check(fname):
    m = hashlib.md5()
    with open(fname) as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data.encode())
    return m.hexdigest()

# print(md5check('./list.json'))




# # 全量备份，每次备份都是备份完整的文件夹所有内容
# def full_backup(src_dir_list, dst_dir_total, selected_type, no_selected_dir=[]):
#     time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
#     md5dict = {}                                                 # 字典，存储每个文件的md5文件
#     md5file = os.path.join(dst_dir_total, "md5.data")
#     for i, src_dir in enumerate(src_dir_list):                   # ！！分别备份列表中的每个文件夹
#         base_dir = src_dir.rsplit("/")[-1]                       # 当前文件夹的名称
#         back_name = "full_%s/%d_%s" % (time_stamp, i, base_dir)  # 当前文件夹在备份目录下的名称
#         dst_dir = os.path.join(dst_dir_total, back_name)         # 当前文件夹要备份到的最终目的目录
#         if os.path.exists(dst_dir): shutil.rmtree(dst_dir)       # 所以src_dir是要备份的文件夹的地址
#         os.makedirs(dst_dir)                                     # dst_dir是要备份的文件夹的目的地址
#         print("=="*80)
#         print("backup '%s' to '%s'" % (src_dir, dst_dir))
#         for root, dirs, files in os.walk(src_dir):               # ！！递归遍历src_dir的每一级目录结构
#             if root.split("/")[-1] in no_selected_dir: continue  # 剔除不要的文件夹，只是名称不是地址
#             print("--"*70)
#             print("backup '%s' to '%s'" % (root, root.replace(src_dir, dst_dir)))
#             for dir in dirs:                                     # ！！构建当前一级的文件夹结构
#                 dst_sub_dir = os.path.join(root.replace(src_dir, dst_dir), dir)
#                 if dir not in no_selected_dir:
#                     os.makedirs(dst_sub_dir)
#                     print("bulid dir '%s'" % dst_sub_dir)
#                 else:
#                     print("-d-d-d- give up dir '%s' !!!" % dst_sub_dir)
#             for file in files:                                   # ！！构建当前一级的文件结构
#                 src_file = os.path.join(root, file)         
#                 if file.split(".")[-1] in selected_type:         # 剔除不要的文件类型
#                     src_flie_dst_dir = root.replace(src_dir, dst_dir)  # shutil.copy(A,B) A是文件，B是文件夹
#                     shutil.copy(src_file, src_flie_dst_dir)      # shutil.copyfile(A,B) A是文件，B是文件
#                     md5dict[src_file] = [md5check(src_file), ""] # 记录每个要备份文件的md5文件, shutil.copytree(A,B) A是文件夹，B是文件夹
#                     print("copy '%s' to '%s'" % (src_file, src_flie_dst_dir))
#                 else:
#                     print("-f-f-f- give up file '%s' !!!" % src_file)
 
#     if os.path.exists(md5file):
#         with open(md5file, 'wb') as f0:  # w，只写模式【不可读；不存在则创建；存在则清空内容；】
#             p.dump(md5dict, f0)          # r ，只读模式【默认】
#     else:
#         with open(md5file, 'xb') as f1:  # x， 只写模式【不可读；不存在则创建，存在则报错】
#             p.dump(md5dict,f1)           # a， 追加模式【不可读；   不存在则创建；存在则只追加内容；】
 
# # 增量备份，每次备份都是仅仅备份与上一次相比增加或修改的文件，这里如果存在文件删除，不做任何记录，调用后md5文件更新为此时完整目录的记录
# def incr_backup(src_dir_list, dst_dir_total, selected_type, no_selected_dir=[]):
#     time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
#     md5new = {}                                                  # 字典，存储每个文件的md5文件
#     md5file = os.path.join(dst_dir_total, "md5.data")
#     for i, src_dir in enumerate(src_dir_list):                   # ！！分别备份列表中的每个文件夹
#         base_dir = src_dir.rsplit("/")[-1]
#         back_name = "incr_%s/%d_%s" % (time_stamp, i, base_dir)
#         dst_dir = os.path.join(dst_dir_total, back_name)
#         if os.path.exists(dst_dir): shutil.rmtree(dst_dir)
#         for root, dirs, files in os.walk(src_dir):               # ！！递归遍历src_dir的每一级目录结构
#             if root.split("/")[-1] in no_selected_dir: continue
#             for file in files:                                   # ！！遍历当前一级的文件结构，无需建立当前文件夹结构，因为可能文件夹没啥变化
#                 src_file = os.path.join(root, file)
#                 if file.split(".")[-1] in selected_type:
#                     src_flie_dst_dir = root.replace(src_dir, dst_dir)
#                     md5new[src_file] = [md5check(src_file), src_flie_dst_dir]  # 记录当前时刻总目录里每个要做备份检查的文件的md5文件
 
#     if os.path.exists(md5file):           # 打开之前的md5文件记录
#         with open(md5file,'rb') as fobj:
#             md5old = p.load(fobj)
#     else:
#         md5old = {}
 
#     with open(md5file, 'wb') as fobj:     # 存储当前的md5文件记录
#         p.dump(md5new, fobj)
 
#     for key in md5new:                    # 检查当前总目录结构相比之前总目录结构发生了什么变化，如果有增加或修改文件，则记录下来，删除文件不做记录
#         if key in md5old:
#             if md5old[key][0] == md5new[key][0]:
#                 continue                  # 如果文件没有任何变化则不备份
#         key_flie_dst_dir = md5new[key][1]
#         if not os.path.exists(key_flie_dst_dir):
#             os.makedirs(key_flie_dst_dir)
#             print("bulid '%s'" % key_flie_dst_dir)
#         shutil.copy(key, key_flie_dst_dir)
#         print("copy '%s' to '%s'" % (key, key_flie_dst_dir))
 
 
# # if __name__ == '__main__':
# #     src_dir_list = ["/nfs/cache-902-2/xiawenze/VehicleClassification"]  # 需要备份的文件夹(子文件夹默认都一起备份)，可以是多个文件夹
# #     dst_dir_total = "/nfs/cache-902-2/xiawenze/backup"                  # 备份目的地
# #     selected_type = ["py", "md"]                                        # 需要备份的文件后缀
# #     no_selected_dir = []                                                # 不需要备份的文件夹名称，这里只能是某文件夹名称，而不是路径，例如所有名为tmp的文件夹不备份
# #     if time.strftime("%a") == "Fri":                                    # 每周五进行全量备份，即完整备份当前的所有内容
# #         full_backup(src_dir_list, dst_dir_total, selected_type, no_selected_dir)
# #     else:                                                               # 其它时间仅仅备份相比上次备份新增加或被修改的文件，相比上次备份删除的文件不管
# #         incr_backup(src_dir_list, dst_dir_total, selected_type, no_selected_dir)


# if __name__ == '__main__':
    