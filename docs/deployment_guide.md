# 部署与运行说明

## 1. 本地完整系统运行

完整的人脸识别考勤系统是 Flask Web 应用，需要在本地电脑运行。推荐使用 Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_models.py
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

`127.0.0.1` 表示本机地址，只能在启动服务的电脑上访问。如果浏览器提示连接被拒绝，通常说明 Flask 服务没有启动、端口不是 `5000`，或终端里的程序已经停止。

## 2. Netlify 静态展示页

Netlify 用于部署项目介绍页，配置文件是 `netlify.toml`：

```toml
[build]
  publish = "public"
  command = ""
```

部署后，Netlify 会把 `public/index.html` 和 `public/styles.css` 发布为公网网页。这个页面适合放项目介绍、技术路线、运行步骤和 GitHub 链接。

## 3. 为什么 Netlify 不能直接运行完整系统

完整系统包含以下本地能力：

- Flask 后端路由和 API
- SQLite 数据库读写
- 本地摄像头采集
- OpenCV 加载 ONNX 模型进行推理
- 本地图片和数据库文件存储

Netlify 的静态站点环境不能长期运行 Flask 服务，也不能直接访问用户电脑摄像头后把数据写入本地 SQLite。因此 Netlify 只承担“公网展示页”的角色，完整功能仍需要本地启动。

## 4. 演示建议

答辩时建议同时准备两个入口：

- 公网展示页：说明项目目标、架构和部署材料。
- 本地 Flask 页面：现场演示新增人员、人脸录入、动态识别签到和考勤记录查询。

这样既能体现部署成果，也能完整展示系统功能。
