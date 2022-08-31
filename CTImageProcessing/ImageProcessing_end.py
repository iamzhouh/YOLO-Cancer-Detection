import SimpleITK as sitk
from matplotlib import pyplot as plt
import os
import pydicom
from matplotlib import pyplot
import cv2
import numpy as np
import csv

numbers = 0

def showNii(img):  # 显示图像
    plt.imshow(img, cmap='gray')
    plt.show()

# 初始化csv文件
# #################################################################################################################################################
with open('label.csv', 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['number', 'start_x', 'start_y', 'end_x', 'end_y', 'imgPath'])

def read_dicom_and_nii_to_label(nii_file_path,img_file_path):    # 传入 标注路径 和 单个病人的图像目录
    global numbers
    # 读取nii文件的label
    # #################################################################################################################################################
    itk_img = sitk.ReadImage(nii_file_path)
    img = sitk.GetArrayFromImage(itk_img)  # 得出的数组切片索引顺序为 人体的自下向上
    mark_index_and_coordinates = {}   # 用来 存储标注的切片的索引：所对应的坐标 的字典

    img_number = img.shape[0]
    print("该病人共有%d张横截面CT图像。" % img_number, end="")  # 表示各个维度的切片数量

    for i in range(0, img.shape[0]):      # 数组中1为红色（病灶） 2是绿色（腹壁皮下脂肪） 3是蓝色（正常胃周脂肪间隙） 4是黄色（病灶周边脂肪间隙）
        coordinates = []  # 用来保存坐标
        if (img[i, :, :].sum()>0) & (img[i,:,:].shape == (512, 512)):  # 提取标注过的图像的具体索引
            # showNii(img[i, :, :])  # 显示图像

            for x in range(len(img[i, :, :])):        # 遍历所有像素点
                for y in range(len(img[i, x, :])):
                    if (img[i, x, y] == 1) | (img[i, x, y] == 4):  # 统一1和4的标注
                        img[i, x, y] = 255
                    else:
                        img[i, x, y] = 0

        # 取得标注区域的方框坐标（利用OpenCV二值化取外接矩形）
            label_img_solo = np.array(img[i, :, :], dtype=np.uint8)

            # opencv膨胀操作 使其 只有一个轮廓 ，以便后面求出外接矩形
            contours, hierarchy = cv2.findContours(label_img_solo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_len = len(contours)
            if contours_len == 0:
                continue
            while contours_len != 1:
                ke = np.ones((3, 3), np.uint8)    # 3*3对比
                label_img_solo = cv2.dilate(label_img_solo, ke, iterations=2)    # opencv膨胀操作

                contours, hierarchy = cv2.findContours(label_img_solo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_len = len(contours)

            # 得到 外接矩形 的坐标
            x, y, w, h = cv2.boundingRect(contours[0])


            # 绘画外接矩阵
            # img_solo = cv2.rectangle(img_solo, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # cv2.imshow('img', img_solo)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            mark_index_and_coordinates[i] = [x, y, x + w, y + h]  # 得到标注过的图像的具体索引 及 其坐标
            # showNii(img[i, :, :])  # 显示图像
    # print(mark_index_and_coordinates)   # 形如 {68: [172, 150, 274, 246], 69: [179, 151, 273, 234], 70: [179, 142, 272, 226]}
    #                                     #     {索引:[开始x，开始y， 结束x， 结束y] }


    # 读取单个病人的DICOM文件
    # #################################################################################################################################################
    file_name_list = os.listdir(img_file_path)  # 单个病人的图像列表

    j = 0
    for file_name in file_name_list:
        if len(file_name.split('.')) == 1:
            file_path = os.path.join(img_file_path,file_name)
            # print(file_path)
            ds = pydicom.dcmread(file_path)

            if ((ds.SeriesDescription == "Venous Phase  5.0  B30f") | (ds.SeriesDescription == "Venous Phase  7.0  B30f") \
                    | (ds.SeriesDescription == "") | (ds.SeriesDescription == "Venous Phase  5.0  B20f") | (ds.SeriesDescription == "Venous Phase  7.0  B20f"))\
                    & (ds.Columns == 512) | (ds.Rows == 512):
                j += 1
                if img_number-j in mark_index_and_coordinates:
                    mark_index_and_coordinates[img_number-j].append(file_path)

                    # 绘制方框
                    CT_img_solo = np.array(ds.pixel_array, dtype=np.uint8)
                    CT_img_solo = cv2.rectangle(CT_img_solo, (mark_index_and_coordinates[img_number-j][0], mark_index_and_coordinates[img_number-j][1]),\
                                  (mark_index_and_coordinates[img_number-j][2], mark_index_and_coordinates[img_number-j][3]), (255, 255, 255), 2)

                    cv2.imshow('img', CT_img_solo)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()



                    numbers = numbers + 1  # 计数
                    mark_index_and_coordinates[img_number - j].insert(0, numbers)

                    # 写入csv文件
                    with open('label.csv', 'a+', newline='') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(mark_index_and_coordinates[img_number-j])

    # print(mark_index_and_coordinates)  # 输出标注的详细信息
    print("有%d张是带有病灶的。" % len(mark_index_and_coordinates))


if __name__ == "__main__":
    folder_path = r"D:\YOLO-Cancer-Detection\MedicalDataSet"
    patient_list = os.listdir(folder_path)   # 得到病人的目录列表
    for patient_list_name in patient_list:
        try:
            patient_path = os.path.join(folder_path, patient_list_name)   # 取得单个病人的目录路径
            period_list = os.listdir(patient_path)  # 得到单个病人脉期目录列表
            for period_name in period_list:
                img_path = os.path.join(patient_path, period_name)   # 得到单个病人的图像路径
                img_list = os.listdir(img_path)   # 得到单个病人图像列表
                if "Untitled.nii" in img_list:
                    nii_file_path = img_path + "/Untitled.nii"
                    img_path = img_path
                    read_dicom_and_nii_to_label(nii_file_path, img_path)
                else:
                    continue
        except Exception as e:
            print(e)





