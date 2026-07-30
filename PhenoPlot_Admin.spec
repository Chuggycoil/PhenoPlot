# PhenoPlot_Admin.spec
# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

a = Analysis(
    ['gui_admin.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app_logo.png', '.'),
        ('phenotypes.ppc', '.')
    ],
    hiddenimports=['PIL._tkinter_finder', 'crypto_utils', 'pheno_core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhenoPlot_Admin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhenoPlot_Admin',
)
