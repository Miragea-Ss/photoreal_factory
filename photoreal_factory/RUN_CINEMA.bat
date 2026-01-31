@echo off
cd /d %~dp0
call venv\Scripts\activate.bat

echo ==========================================
echo  Photoreal Factory - Cinema Mode Starting
echo ==========================================

REM 実行コマンド (長いコマンドをここに書いておく)
python factory_run.py --job cinema_log --input "I:\ComfUI_G_42\ComfyUI\output\GL" --output "I:\ComfUI_G_42\ComfyUI\output\GL_Upscaled"

REM 終わったらキー入力待ちにする（勝手に閉じないように）
pause