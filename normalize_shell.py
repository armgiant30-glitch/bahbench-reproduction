from pathlib import Path
for p in Path(r'D:\a\bahbench_repro_package').rglob('*.sh'):
    b=p.read_bytes(); p.write_bytes(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n'))
