# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import get_package_paths

block_cipher = None

# --- Path B: Vendor the entire webview package ---
webview_path = get_package_paths('webview')[0]
# ---

a = Analysis(
    ['src/server.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('config', 'config'),
        ('credentials', 'credentials'),
        (webview_path, 'webview')  # Add the webview package as data
    ],
    hiddenimports=[
        'webview.platforms.cocoa',
        'objc',
        'Foundation',
        'AppKit',
        'WebKit',
        'gspread',
        'google.auth',
        'google_auth_oauthlib',
        'googleapiclient',
        'google.generativeai'
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AnyAI_Video_Analysis',
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
    icon='static/assets/AnyAI_icon.icns'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AnyAI_Video_Analysis',
)
app = BUNDLE(
    coll,
    name='AnyAI_Video_Analysis.app',
    icon='static/assets/AnyAI_icon.icns',
    bundle_identifier=None,
)