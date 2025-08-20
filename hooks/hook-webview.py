from PyInstaller.utils.hooks import collect_all

def hook(hook_api):
    """
    Explicitly tell PyInstaller how to bundle pywebview.
    """
    # The 'pywebview' package contains the necessary modules.
    # We collect all data files, binaries, and hidden imports from it.
    datas, binaries, hiddenimports = collect_all('webview')
    
    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
