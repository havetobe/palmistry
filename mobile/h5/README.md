# PalmTrace H5 Demo

打开 `mobile/h5/index.html` 即可在手机浏览器运行。

- 点击“拍照”或“导入照片”
- 自动校正 EXIF 方向并绘制三条主线
- 低光或分辨率不足会显示失败原因

## 本地模型 API

1. 在 `code` 目录安装依赖：`pip install -r requirements.txt`
2. 启动服务：`python api_server.py`
3. 打开 H5 页面后会自动请求 `http://localhost:8000/api/predict`

未启动服务时会回退到示意线条模式。
