@echo off
title Pearl (PRL) Miner - Live Display
mode con: cols=100 lines=30
color 0A
cls
echo.
echo ============================================
echo   Pearl (PRL) Miner - Live Display
echo ============================================
echo.
echo  Wallet: prl1peyn72nhze69sju9g6nkjehqmtrdzgjqgg8excyp805uuch0thzqqyav93y
echo  Pool: br.pearl.herominers.com:1200
echo  Worker: pearlhost_live
echo.
echo  Minero en ejecucion - ver datos LIVE abajo
echo  Presiona Ctrl+C para detener.
echo.
echo ============================================
echo.
.\out\pearl-host.exe --wallet prl1peyn72nhze69sju9g6nkjehqmtrdzgjqgg8excyp805uuch0thzqqyav93y --lowmem --worker pearlhost_live
echo.
echo Minero detenido.
pause