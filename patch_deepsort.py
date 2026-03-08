import pathlib

f = pathlib.Path(r'C:\Users\badie\AppData\Local\Programs\Python\Python312\Lib\site-packages\deep_sort_realtime\embedder\embedder_pytorch.py')
txt = f.read_text()

# Replace the broken import
txt = txt.replace(
    'import importlib.resources as pkg_resources',
    'import importlib.resources as _ir'
)

# Replace both resource_filename calls with the modern equivalent
txt = txt.replace(
    'pkg_resources.resource_filename(\n    "deep_sort_realtime", "embedder/weights/mobilenetv2_bottleneck_wts.pt"\n)',
    'str(_ir.files("deep_sort_realtime") / "embedder/weights/mobilenetv2_bottleneck_wts.pt")'
)
txt = txt.replace(
    'pkg_resources.resource_filename(\n    "deep_sort_realtime", "embedder/weights/osnet_ain_ms_d_c_wtsonly.pth"\n)',
    'str(_ir.files("deep_sort_realtime") / "embedder/weights/osnet_ain_ms_d_c_wtsonly.pth")'
)

f.write_text(txt)
print("Patched OK")
