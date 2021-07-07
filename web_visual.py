# -*-coding:utf-8-*-

"""============================================
    Time   : 2021/6/10  3:31 下午
    Author : Xiaoying Bai
    Brief  : 
============================================"""
import streamlit as st
from PIL import Image
from predict import predict_web
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.write('# 基于X-ray影像的肺炎辅助诊断')
image_file = st.file_uploader("请上传一张肺部X-ray影像")

if image_file is not None:
    img = Image.open(image_file).convert('RGB')
    img.save(image_file.name)
    img_show = img.resize((224, int(img.size[1] / (img.size[0] / 224))))  # img.size[0]是宽
    st.image(img, use_column_width=True)

    st.write("# 诊断信息：")

    prediction = predict_web(image_file.name)
    st.write(prediction)
    os.remove(image_file.name)