from pathlib import Path
root=Path(r'D:\a\bahbench_repro_package')
for p in root.rglob('*.sh'):
    b=p.read_bytes(); p.write_bytes(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n'))
print('shell scripts normalized to LF')
