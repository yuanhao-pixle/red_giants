import os.path
import threading
import urllib
from time import sleep
import pandas as pd
from astropy.wcs import WCS
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.font_manager as mpt
from PIL import ImageDraw, ImageFont
from PIL import Image
import random
from shutil import copy, rmtree
from torchvision import transforms, datasets
from sklearn.metrics import precision_recall_curve
import warnings
import math
import xml.dom.minidom
import requests
from requests.packages import urllib3
import bz2
from astropy.io import fits
import shutil
import gzip
# from astroquery.gaia import Gaia
from astropy.visualization import make_lupton_rgb
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from download_utils import *


# 选出每个part中后续符合预测条件的星
def filtrate(dir_name, basePath):
    origin_path = r'F:\yh\data3\gz\SkyMapper.DR1.1.master.part{0:03d}.csv'.format(dir_name)

    df = pd.read_csv(origin_path)

    name_list = ['object_id', 'raj2000', 'dej2000', 'flags', 'class_star',
                 'u_psf', 'e_u_psf', 'u_nvisit', 'v_psf', 'e_v_psf', 'v_nvisit',
                 'g_psf', 'e_g_psf', 'g_nvisit', 'i_psf', 'e_i_psf', 'i_nvisit']

    df = df[name_list]

    new_name_list = ['ObjectId', 'ra', 'dec', 'flags', 'ClassStar', 'uPSF', 'e_uPSF', 'u_nvisit', 'vPSF',
                     'e_vPSF', 'v_nvisit', 'gPSF', 'e_gPSF', 'g_nvisit', 'iPSF', 'e_iPSF', 'i_nvisit']

    df.columns = new_name_list

    # 排除没有图像的点
    df = df[
        (df['u_nvisit'] > 0) & (df['v_nvisit'] > 0) & (df['g_nvisit'] > 0) & (
                df['i_nvisit'] > 0)]

    # 选择flags = 0 & class_star>=0.6
    df = df[(df['flags'] == 0) & (df['ClassStar'] >= 0.6) &
            ((df['e_uPSF'] <= 0.05) & (df['e_vPSF'] <= 0.05) &
             (df['e_gPSF'] <= 0.05) & (df['e_iPSF'] <= 0.05))
            ]

    # # 将需要四个波段星等不存在的值删除
    df = df.fillna(-9999.9999)
    df = df[
        (df['uPSF'] != -9999.9999) & (df['vPSF'] != -9999.9999) & (df['gPSF'] != -9999.9999) & (
                df['iPSF'] != -9999.9999)]

    # basePath = r'E:\giants_samples\part{0:03d}'.format(dir_name)
    # if not os.path.exists(basePath):
    #     os.mkdir(basePath)

    # df.to_csv(os.path.join(basePath, 'part{0:03d}.csv'.format(dir_name)), index=False)

    # if not os.path.exists(os.path.join(basePath, 'fits')):
    #     os.mkdir(os.path.join(basePath, 'fits'))
    #
    # if not os.path.exists(os.path.join(basePath, 'npy')):
    #     os.mkdir(os.path.join(basePath, 'npy'))

    # # 增加（g-i）/ （u-v）
    # LS_rest['g-i'] = LS_rest.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
    # LS_rest['u-v'] = LS_rest.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

    # print()
    return df


# 将所有符合条件的星整合成一个文件
def get_partAll(basePath):
    df_empty = pd.DataFrame()
    for i in range(1, 952):
        print('正在处理{0}/{1}'.format(i, 951))
        dir_name = i
        Path = r'E:\giants_samples\part{0:03d}'.format(dir_name)
        try:
            df_part = filtrate(dir_name, Path)
        except Exception as e:
            print(e, type(e))
            if (isinstance(e, pd.errors.EmptyDataError)):
                print("这里对空行文件进行处理")
        # df判断是否为空
        if df_part.empty:
            continue
        df_empty = pd.concat([df_empty, df_part], axis=0)

    df_empty.to_csv(os.path.join(basePath, 'partAll.csv'), index=False)
    print()


# 挑选指定天空子区域的星
def Region_filter(ra_list, dec_list, basePath):
    df = pd.read_csv(os.path.join(basePath, 'partAll.csv'))

    df_ra_dec = df[((df['ra'] >= ra_list[0]) & (df['ra'] <= ra_list[1])) &
                   ((df['dec'] >= dec_list[0]) & (df['dec'] <= dec_list[1]))]

    df_ra_dec.to_csv(os.path.join(basePath, 'giant_sample_2', 'partAll_RaDec.csv'), index=False)

    print()


def xml(basePath):
    # 读取巨星
    stars = pd.read_csv(os.path.join(basePath, 'giant_sample_2', 'partAll_RaDec.csv'))
    # stars = pd.read_csv(r'E:\giants_samples\part{0:03d}\part{0:03d}.csv'.format(dir_name))

    # stars = stars.iloc[1:50, :]

    run_download_xml(basePath, stars)
    print()


# npy文件的合成
#
def complex_npy(basePath):
    # 将每个波段的额数据读取并且存储到numpy中
    def bands_to_numpy(df, type, bands=['u', 'v', 'g', 'i']):
        # 索引重置
        df.index = range(len(df))

        # 确定图像大小
        size = 96

        # bands = ['u', 'v', 'g', 'i']
        array = np.zeros((df.shape[0], len(bands), size, size))

        path_1 = os.path.join(basePath, 'giant_sample_2_fits')
        # 遍历df
        for index, data in df.iterrows():
            print('{} : {}/{}'.format(type, index, df.shape[0]))
            path_2 = os.path.join(path_1, str(int(data.ObjectId)))
            for i, band in enumerate(bands):
                path_3 = os.path.join(path_2, '{}_{}.fits'.format(str(int(data.ObjectId)), band))
                hdulist = fits.open(path_3)
                image = hdulist[0].data
                # 对有偏差的进行裁剪（有部分fits数据下载的有偏差 例如68266862的u波段 48*49）
                image = crop_center(image, size)
                array[index, i, :, :] = image

            # aaa = array[0, 0, 0:48, 0:48]
        return array

    # 按图像中心裁剪为指定大小
    def crop_center(image, size):
        # size = 46
        # image = (fits.open('../data1/fits/giants/68266862/68266862_u.fits'))[0].data
        # plt.imshow(image)
        # plt.show()
        # 找到图像的中心
        center_x = int(image.shape[0] / 2)
        center_y = int(image.shape[1] / 2)

        # 裁剪范围
        x_start = int(center_x - (size / 2))
        x_end = int(center_x + (size / 2))
        y_start = int(center_y - (size / 2))
        y_end = int(center_y + (size / 2))

        # 裁剪
        image = image[x_start:x_end, y_start:y_end]
        # plt.imshow(image)
        # plt.show()
        # print()
        return image

    # 得到巨星和矮星训练的总样本
    stars = pd.read_csv(os.path.join(basePath, 'giant_sample_2', 'partAll_RaDec.csv'))

    stars = stars.iloc[150000:, :]

    # 将要训练的数据保存成numpy格式
    data = bands_to_numpy(stars, type='train')
    label = np.array(stars)

    # 定义存储目录
    save_path_1 = os.path.join(basePath, 'giant_sample_2', 'npy')
    if not os.path.exists(save_path_1):
        os.mkdir(save_path_1)
    np.save(os.path.join(save_path_1, 'partAll_data_2.npy'), data)
    np.save(os.path.join(save_path_1, 'partAll_label_2.npy'), label)
    print()


# 寻找候选体
def Candidate_star():
    # 预测数据合并
    all_star_1 = pd.read_csv(r'E:\giants_samples\partAll\giant_sample_2\partAll_RaDec_predict1.csv')
    all_star_2 = pd.read_csv(r'E:\giants_samples\partAll\giant_sample_2\partAll_RaDec_predict2.csv')
    all_star = pd.concat([all_star_1, all_star_2], axis=0)

    # 计算预测为红巨星的数量
    all_star["new"] = all_star["our_pre1"] - all_star["our_pre2"]
    red_giants = (all_star[all_star['new'] >= 0])
    red_giants.iloc[:, 0:-1].to_csv(r'E:\giants_samples\partAll\giant_sample_2\partAll_RaDec_redGiants.csv',
                                    index=False)

    print()


# 数据分析
def data_analyse():
    redG_path = r'E:\giant_sample\partAll\giant_sample_2\partAll_RaDec_redGiants.csv'
    redG = pd.read_csv(redG_path)
    match_redG_path = r'E:\giant_sample\partAll\giant_sample_2\GAIA_teff_parallar\redGiants_gaia.csv'
    match_redG = pd.read_csv(match_redG_path)

    # 列筛选
    def columns(data, dataPath):
        # 筛选出有用的列
        name_list = ['ObjectId', 'source_id', 'ra', 'dec', 'teff_gaia', 'parallax_gaia', 'parallax_error_gaia',
                     'radius_gaia', 'rest',
                     'flags', 'ClassStar', 'uPSF', 'e_uPSF', 'u_nvisit',
                     'vPSF', 'e_vPSF', 'v_nvisit', 'gPSF', 'e_vPSF', 'g_nvisit', 'iPSF', 'e_iPSF', 'i_nvisit',
                     'our_pre1', 'our_pre2']  # # 来源于gaia DR2以及gaia DR2 distance
        data = data[name_list]
        # 将列改名规范
        data = data.rename(columns={'rest': 'rest_gaia'})
        # data = data.rename(columns={'teff_val': 'teff_gaia'})
        # data = data.rename(columns={'parallax': 'parallax_gaia'})
        # data = data.rename(columns={'parallax_error': 'parallax_error_gaia'})
        # data = data.rename(columns={'radius_val': 'radius_gaia'})
        data.to_csv(dataPath, index=False)

    # 表合并
    # table2加入table1
    def table_contact(tabel1, tabel2, table1_path):
        # 计算绝对星等
        tabel2['d0/d'] = tabel2.apply(lambda x: 10 / x['rest_gaia'], axis=1)
        tabel2['absolute_mag'] = tabel2.apply(lambda x: (x['iPSF'] + 5 * (1 + math.log(x['d0/d'], 10))), axis=1)

        list = [-9999.9999 for i in range(tabel1.shape[0])]

        # 给tabel1添加列
        tabel1['gaia_id'] = list
        tabel1['teff_gaia'] = list
        # tabel1['teff_err_apogee'] = list
        # tabel1['logg_apogee'] = list
        # tabel1['logg_err_apogee'] = list
        tabel1['parallax_gaia'] = list
        tabel1['parallax_error_gaia'] = list
        tabel1['radius_gaia'] = list
        tabel1['rest_gaia'] = list
        tabel1['absolute_mag'] = list

        for index, data in tabel1.iterrows():
            print(index)
            # 按照objectId来进行匹配
            table2_ = tabel2[tabel2['ObjectId'] == data.ObjectId]

            if table2_.shape[0] > 0:
                table2_.index = range(1)
                # df.loc[参数1，参数2]=1
                tabel1.loc[index, 'gaia_id'] = str(table2_.loc[0, 'source_id'])
                tabel1.loc[index, 'teff_gaia'] = table2_.loc[0, 'teff_gaia']
                tabel1.loc[index, 'parallax_gaia'] = table2_.loc[0, 'parallax_gaia']
                tabel1.loc[index, 'parallax_error_gaia'] = table2_.loc[0, 'parallax_error_gaia']
                tabel1.loc[index, 'radius_gaia'] = table2_.loc[0, 'radius_gaia']
                tabel1.loc[index, 'rest_gaia'] = table2_.loc[0, 'rest_gaia']
                tabel1.loc[index, 'absolute_mag'] = table2_.loc[0, 'absolute_mag']
                # tabel1.loc[index, :]['parallax_gaia'] = table2_['parallax_gaia']
                # tabel1.loc[index, :]['parallax_error_gaia'] = table2_['parallax_error_gaia']

        tabel1 = tabel1.fillna(-9999.9999)
        tabel1.to_csv(table1_path, index=False)

    # CBAMResnes找的红巨星和训练集红巨星趋势散点图
    def red_giants_plot():
        # 加载训练集中的点
        LA_VAC_redGiants = pd.read_csv(r'D:\Workspace\pycharmWorkspace\red_giants\data1_6\red_giants.csv')
        x1 = LA_VAC_redGiants.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
        y1 = LA_VAC_redGiants.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

        # 加载CBAMResnes找的红巨星
        x2 = redG.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
        y2 = redG.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

        # 画散点图
        colors = ['aquamarine', 'orange']  # 建立颜色列表
        labels = ['red giants by CBAMResnets search', 'red giants in LA_VAC']  # 建立标签类别列表
        plt.scatter(x2, y2, s=10, c=colors[0], label=labels[0], alpha=0.8, marker='v')
        plt.scatter(x1, y1, s=10, c=colors[1], label=labels[1], alpha=1, marker='o')
        plt.xlabel('$g - i$', fontsize=10)
        plt.ylabel('$u - v$', fontsize=10)
        # plt.xlim(0, 1.65)
        # plt.ylim(0.1, 0.87)
        # plt.xticks(np.arange(0, 1.6, step=0.5), fontsize=40)
        # plt.yticks(np.arange(0.2, 0.8, step=0.2), fontsize=40)
        plt.legend()  # 显示图例
        plt.savefig('../notebooks/figures/compare_lavac_candidate.png', dpi=500)
        plt.show()

    # 判断是否被其他老师标注好
    def label_by_others(tabel, table_path):
        # 标注列
        list = [-9999.9999 for i in range(tabel.shape[0])]
        tabel['tag'] = list
        tabel['database_tag'] = list

        for index, data in tabel.iterrows():
            print(index)
            # 是否被lamost标注过
            if str(data.lamost_id) != str(-9999.9999):
                teff = data.teff_lamost
                logg = data.logg_lamost
                if (teff <= 5600.00) & (logg <= 3.8):
                    tabel.loc[index, 'tag'] = 0
            # 是否被apogee标注过
            if str(data.apogee_id) != str(-9999.9999):
                teff = data.teff_apogee
                logg = data.logg_apogee
                if (teff <= 5600.00) & (logg <= 3.8):
                    if str(data.tag) == str(-9999.9999):
                        tabel.loc[index, 'tag'] = 1
            # 是否被galah标注过
            if str(data.galah_id) != str(-9999.9999):
                teff = data.teff_galah
                logg = data.logg_galah
                if (teff <= 5600.00) & (logg <= 3.8):
                    if str(data.tag) == str(-9999.9999):
                        tabel.loc[index, 'tag'] = 2
            # 是否被huangY老师标注过
            gi = data.gPSF - data.iPSF
            uv = data.uPSF - data.vPSF
            abm = data.absolute_mag
            # 如果在huangY老师划分的范围内
            if (uv >= 0.93 - 0.76 * gi) & (uv >= 0.36 + 0.09 * gi) & (gi >= 0.4) & (gi <= 1.35):
                # 如果存在绝对星等
                if str(abm) != str(-9999.9999):
                    # 如果绝对星等小于8
                    if abm <= 8:
                        if str(data.tag) == str(-9999.9999):
                            tabel.loc[index, 'tag'] = 3
            # 如果gaia能划分
            if (data.teff_gaia != -9999.9999) and (data.radius_gaia != -9999.9999):
                if (data.teff_gaia <= 5600) and (data.radius_gaia >= 10):
                    if str(data.tag) == str(-9999.9999):
                        tabel.loc[index, 'tag'] = 4

            # 有光谱数据的
            if (str(data.lamost_id) != str(-9999.9999)) or (str(data.galah_id) != str(-9999.9999)) or \
                    (str(data.apogee_id) != str(-9999.9999)):
                tabel.loc[index, 'database_tag'] = 0

        tabel.to_csv(table_path, index=False)

    # 根据标注画图
    def red_giants_plot_2(tabel):
        spec = tabel[tabel['database_tag'] == 0]
        # 数据清洗
        aa = spec[spec['logg_lamost'] >= 4]
        bb = aa.sample(frac=0.8, replace=False, axis=0)
        spec = spec[~spec.ObjectId.isin(bb.ObjectId)]
        spec = spec[spec['lamost_id'] != -9999.9999]
        # CBAMResnets置信度挑选
        # spec = spec[spec['our_pre1'] >= 0.75]

        spec_rg = spec[spec['tag'] != -9999.9999]

        # 预测正确百分比
        right_por = spec_rg.shape[0] / spec.shape[0]

        # spec_lamost_redGiants = spec_lamost[spec_lamost['tag'] != -9999.9999]

        # lamost承认的点
        # t = tabel[tabel['tag'] == 0]
        x1 = spec.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
        y1 = spec.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

        # apogee承认的点
        # t = tabel[tabel['tag'] == 1]
        x2 = spec_rg.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
        y2 = spec_rg.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

        # apogee承认的点
        # t = tabel[tabel['tag'] == 1]
        # x3 = spec_da_hp.apply(lambda x: x['gPSF'] - x['iPSF'], axis=1)
        # y3 = spec_da_hp.apply(lambda x: x['uPSF'] - x['vPSF'], axis=1)

        # 画散点图
        colors = ['aquamarine', 'orange', 'lightgreen', 'cornflowerblue', 'cyan']  # 建立颜色列表
        labels = ['LARGR', 'LARGC', 'GALAH', 'huangY', 'no']  # 建立标签类别列表
        # plt.scatter(x5, y5, s=10, c=colors[4], label=labels[4], alpha=0.8, marker='v')
        # plt.scatter(x4, y4, s=10, c=colors[3], label=labels[3], alpha=0.5, marker='o')
        plt.scatter(x1, y1, s=10, c=colors[1], label=labels[1], alpha=0.6, marker='o')
        plt.scatter(x2, y2, s=10, c=colors[0], label=labels[0], alpha=1, marker='v')
        # plt.scatter(x3, y3, s=10, c=colors[3], label=labels[3], alpha=1, marker='v')
        # plt.scatter(x3, y3, s=10, c=colors[2], label=labels[2], alpha=0.8, marker='v')

        plt.xlabel('$g - i$', fontsize=10)
        plt.ylabel('$u - v$', fontsize=10)
        # plt.xlim(0, 1.65)
        # plt.ylim(0.1, 0.87)
        # plt.xticks(np.arange(0, 1.6, step=0.5), fontsize=40)
        # plt.yticks(np.arange(0.2, 0.8, step=0.2), fontsize=40)
        plt.legend()  # 显示图例
        plt.savefig('../notebooks/figures/compare_lamost_candiate.png', dpi=1000)
        plt.show()
        print()

    # columns(match_redG, match_redG_path)
    # table_contact(redG, match_redG, redG_path)
    red_giants_plot()
    # label_by_others(redG, redG_path)
    # red_giants_plot_2(redG)
    print()


if __name__ == '__main__':
    ##################
    ## 以下函数依次执行
    ##################

    #
    # dir_name = 'All'
    basePath = r'E:\giants_samples\partAll'

    # 将SMSS DR1.1所有符合要求的恒星全部筛选出来
    # get_partAll(basePath)

    # 挑选子区域的星
    # ra_list = [200, 221]
    # dec_list = [-55, -34]
    # LAMOST和SMSS中共同的恒星 ra[0:400] dec[-10:12] 我们计划选用部分共同点
    ra_list = [200, 221]
    dec_list = [-38, 20]
    # Region_filter(ra_list, dec_list, basePath)

    # 下载xml文件
    # xml(basePath)

    # 处理xml文件生成波段的csv文件(默认四个波段)
    # deal_xml(basePath)

    # 下载fits文件
    run_download_fits(basePath)

    # 合成
    # complex_npy(basePath)

    # 挑选候选体
    # Candidate_star()

    # 数据分析
    data_analyse()
