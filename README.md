# 基于深度学习的人脸识别考勤系统

本仓库用于计算机综合项目实践的分工提交与展示材料整理。项目主题是“基于深度学习的人脸识别系统”，完整系统采用本地 Python Web 应用实现，静态展示页可部署到 Netlify 作为项目说明入口。

## 项目简介

系统围绕校园考勤场景设计，提供人员管理、人脸录入、动态识别签到、考勤记录查询与 CSV 导出等功能。核心识别流程基于 OpenCV DNN 与 ONNX 预训练模型完成，不需要重新训练大规模模型，适合课程实践和答辩演示。

## 技术路线

- 后端：Python 3.11、Flask、SQLite
- 前端：HTML、CSS、JavaScript、浏览器摄像头 API
- 人脸检测：YuNet ONNX 模型
- 人脸识别：SFace ONNX 模型
- 反欺骗检测：MiniFASNet ONNX 模型
- 部署展示：Netlify 静态站点，仅展示项目介绍和运行说明

## 本地运行

完整功能需要在本地电脑运行 Flask 服务：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_models.py
python app.py
```

启动后访问：

```text
http://127.0.0.1:5000
```

如果模型下载失败，可手动下载后放入 `models/` 目录：

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`
- `face_antispoof_minifasnetv2.onnx`

## Netlify 说明

Netlify 主要部署 `public/` 下的静态展示页。由于完整系统依赖 Flask、SQLite、本地摄像头和 OpenCV 模型推理，不能直接在 Netlify 静态环境中运行完整签到功能。公网展示页用于说明项目目标、技术架构和本地运行方式。

## 隐私说明

人脸图片属于敏感个人信息。课程演示时应只采集小组成员或授权测试人员的人脸样本，采集前说明用途，并在演示结束后及时清理本地数据库和图片文件。

## 分工说明

本分支主要整理第 4 部分“文档、部署与展示材料”，包括 README、部署说明、模型运行说明、Netlify 配置和静态展示页。
