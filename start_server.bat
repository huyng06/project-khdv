@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo =========================================================================
echo    HỆ THỐNG ĐIỀU PHỐI VÀ GHÉP CHUYẾN GỌI XE - HÀ NỘI (GRAB SIMULATOR)
echo =========================================================================
echo.

rem Uu tien py (Python launcher cua Windows), neu khong co thi dung python
where py > nul 2> nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")

echo  - Đang kiểm tra môi trường Python...
%PY% --version
if errorlevel 1 goto no_python

%PY% -c "import fastapi, uvicorn, websockets, osmnx" > nul 2> nul
if errorlevel 1 (
    echo  - Chưa đủ thư viện, đang cài từ requirements.txt, có thể mất vài phút...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 goto pip_failed
)

echo.
echo  - Khởi động FastAPI WebSocket Server...
echo  - Mở trình duyệt tại: http://localhost:8000
echo  - Nhấn Ctrl+C để dừng server.
echo =========================================================================
echo.
set PYTHONIOENCODING=utf-8
%PY% server.py
pause
exit /b 0

:no_python
echo.
echo  [LỖI] Không tìm thấy Python. Hãy cài Python 3.10 trở lên rồi chạy lại.
pause
exit /b 1

:pip_failed
echo.
echo  [LỖI] Cài thư viện thất bại. Hãy kiểm tra kết nối mạng rồi chạy lại.
pause
exit /b 1
